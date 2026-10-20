#!/usr/bin/env python3
"""14B + Ulysses SP：全 rank 真实构造（不用 meta/broadcast），测 fwd(+bwd) 显存与耗时。

对照组（已测，DP=SP1）：
  DP(no ckpt) : F=9/17/25/33 全 OOM
  DP(+ckpt)   : F=9 76.6GB 勉强, F=17+ OOM

用法：
  python -m torch.distributed.run --nproc_per_node=4 test_sp_mem2.py --frames 5,9,17,33 --bwd 0
"""
import argparse, os, sys, time, traceback
import torch
import torch.distributed as dist
from datetime import timedelta

Z = os.environ.get("ZROOT", "/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengmenghan")
sys.path.insert(0, Z + "/dreamzero_sp/shim")
sys.path.insert(0, Z + "/dreamzero_sp/dreamzero")
import groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk as DZMOD
from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import CausalWanModel

# ---- 关 ckpt 时上游 block 返回 tuple 未解包，必须 patch ----
_orig_blk = DZMOD.CausalWanAttentionBlock.forward
def _unwrap(self, x, *a, **k):
    out = _orig_blk(self, x, *a, **k)
    return out[0] if isinstance(out, tuple) else out
DZMOD.CausalWanAttentionBlock.forward = _unwrap

# ---- 集合通信计数器：检测各 rank 调用次数是否一致 ----
A2A = {"n": 0}
_orig_a2a = DZMOD._sp_all_to_all_4d
def _counted_a2a(*a, **k):
    A2A["n"] += 1
    return _orig_a2a(*a, **k)
DZMOD._sp_all_to_all_4d = _counted_a2a

ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=str, default="5,9,17,33")
ap.add_argument("--bwd", type=int, default=0)
ap.add_argument("--iters", type=int, default=1)
ap.add_argument("--timeout", type=int, default=240)
args = ap.parse_args()

dist.init_process_group("nccl", timeout=timedelta(seconds=args.timeout))
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)
dev = "cuda:%d" % rank
is0 = rank == 0

H, W = 40, 88
W_LOCAL = W // world
FSL_FULL = (H // 2) * (W // 2)

CFG = dict(model_type="ti2v", patch_size=(1, 2, 2), frame_seqlen=FSL_FULL, text_len=512,
           in_dim=36, dim=5120, ffn_dim=13824, freq_dim=256, text_dim=4096, out_dim=16,
           num_heads=40, num_layers=40, num_frame_per_block=2, action_dim=32,
           num_action_per_block=24, num_state_per_block=1, max_state_dim=64,
           max_num_embodiments=32, hidden_size=1024, concat_first_frame_latent=True,
           qk_norm=True, cross_attn_norm=True, max_chunk_size=4, sink_size=0)

# ---- 真实构造（CPU bf16 -> GPU），四卡各自独立、不 broadcast ----
t0 = time.time()
torch.manual_seed(0)
torch.set_default_dtype(torch.bfloat16)
m = CausalWanModel(**CFG)
nparam = sum(p.numel() for p in m.parameters())
m = m.to(dev)
torch.set_default_dtype(torch.float32)
m.eval()
m.gradient_checkpointing = False
for p in m.parameters():
    p.requires_grad_(True)
if is0:
    print("[build] %.1fs  params=%.2fB  resident=%.1fGB" %
          (time.time() - t0, nparam / 1e9, torch.cuda.memory_allocated() / 1e9), flush=True)
dist.barrier()

DZMOD.set_sp_state(group=dist.group.WORLD, world=world, rank=rank)
if is0:
    print("[sp] world=%d W_local=%d" % (world, W_LOCAL), flush=True)

B = 1
for F in [int(v) for v in args.frames.split(",")]:
    N_GLOBAL = F * FSL_FULL
    x = torch.randn(B, 36, F, H, W_LOCAL, device=dev, dtype=torch.bfloat16)
    ctx = torch.randn(B, 512, 4096, device=dev, dtype=torch.bfloat16)
    t = torch.randint(0, 1000, (B, F), device=dev)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    A2A["n"] = 0
    tag = "SP%d F=%d%s" % (world, F, " +bwd" if args.bwd else "")
    try:
        tf = tb = 0.0
        for it in range(args.iters):
            t1 = time.time()
            out = m(x, t, None, ctx, seq_len=N_GLOBAL, clean_x=x,
                    action=None, state=None, embodiment_id=None)
            torch.cuda.synchronize()
            tf += time.time() - t1
            o = out[0] if isinstance(out, (list, tuple)) else out
            loss = o.float().mean()
            if args.bwd:
                t1 = time.time()
                loss.backward()
                torch.cuda.synchronize()
                tb += time.time() - t1
                m.zero_grad(set_to_none=True)
        tf /= args.iters
        tb /= args.iters
        peak = torch.cuda.max_memory_allocated() / 1e9
        if is0:
            print("[OK ] %-18s fwd=%.2fs bwd=%.2fs peak=%.1fGB a2a=%d" %
                  (tag, tf, tb, peak, A2A["n"]), flush=True)
    except torch.cuda.OutOfMemoryError:
        peak = torch.cuda.max_memory_allocated() / 1e9
        if is0:
            print("[OOM] %-18s peak=%.1fGB" % (tag, peak), flush=True)
    except Exception as e:
        if is0:
            print("[ERR] %-18s %s: %s" % (tag, type(e).__name__, str(e)[:300]), flush=True)
            traceback.print_exc()
    # 各 rank 的 a2a 次数必须一致，否则下一轮必然错位
    cnt = torch.tensor([A2A["n"]], device=dev)
    allc = [torch.zeros_like(cnt) for _ in range(world)]
    dist.all_gather(allc, cnt)
    if is0:
        print("      a2a per rank = %s" % [int(c.item()) for c in allc], flush=True)
    del x, ctx, t
    torch.cuda.empty_cache()

dist.barrier()
dist.destroy_process_group()
