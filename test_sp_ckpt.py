#!/usr/bin/env python3
"""14B + SP v3：峰值显存 & 步时（DP 对照 vs SP=2/4）。

对照组已测（v1 实现，显存没降下来是因为 e0 没被切）：
  DP(no ckpt) F=9/17/25/33 全 OOM
  DP(+ckpt)   F=9 76.6GB, F=17+ OOM

用法：
  python -m torch.distributed.run --nproc_per_node=4 test_sp_mem3.py --frames 9,17,33 --bwd 1 --dp 1
"""
import argparse, os, sys, time, traceback
from datetime import timedelta
import torch
import torch.distributed as dist

Z = os.environ.get("ZROOT", "/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengmenghan")
sys.path.insert(0, Z + "/dreamzero_sp/shim")
sys.path.insert(0, Z + "/dreamzero_sp/dreamzero")
import groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk as DZMOD
from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import CausalWanModel

ap0 = argparse.ArgumentParser(add_help=False)
ap0.add_argument("--ckpt", type=int, default=1)
_pa, _ = ap0.parse_known_args()
if not _pa.ckpt:
    # 只有关 ckpt 时才需要打这个解包补丁；开 ckpt 时上游 create_custom_forward 已经解包，
    # 再 patch 就会 "not enough values to unpack (expected 2, got 1)"。
    _orig_blk = DZMOD.CausalWanAttentionBlock.forward
    def _unwrap(self, x, *a, **k):
        out = _orig_blk(self, x, *a, **k)
        return out[0] if isinstance(out, tuple) else out
    DZMOD.CausalWanAttentionBlock.forward = _unwrap

A2A = {"n": 0}
_orig_a2a = DZMOD._sp_all_to_all_4d
def _counted(*a, **k):
    A2A["n"] += 1
    return _orig_a2a(*a, **k)
DZMOD._sp_all_to_all_4d = _counted

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", type=int, default=1)
ap.add_argument("--spec", type=str, default="9:0,17:0,33:0",
                help="逗号分隔的 F:bwd，例如 9:0,9:1")
ap.add_argument("--dp", type=int, default=0)      # 1 = 额外跑一遍 world=1 的 DP 对照
ap.add_argument("--iters", type=int, default=1)
ap.add_argument("--warmup", type=int, default=0)
ap.add_argument("--gradsync", type=int, default=0)
ap.add_argument("--timeout", type=int, default=300)
args = ap.parse_args()

dist.init_process_group("nccl", timeout=timedelta(seconds=args.timeout))
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)
dev = "cuda:%d" % rank
is0 = rank == 0

H, W = 40, 88
FSL = (H // 2) * (W // 2)          # 880

CFG = dict(model_type="ti2v", patch_size=(1, 2, 2), frame_seqlen=FSL, text_len=512,
           in_dim=36, dim=5120, ffn_dim=13824, freq_dim=256, text_dim=4096, out_dim=16,
           num_heads=40, num_layers=40, num_frame_per_block=2, action_dim=32,
           num_action_per_block=24, num_state_per_block=1, max_state_dim=64,
           max_num_embodiments=32, hidden_size=1024, concat_first_frame_latent=True,
           qk_norm=True, cross_attn_norm=True, max_chunk_size=4, sink_size=0)

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
    print("[build] %.1fs params=%.2fB resident=%.1fGB" %
          (time.time() - t0, nparam / 1e9, torch.cuda.memory_allocated() / 1e9), flush=True)
dist.barrier()

m.gradient_checkpointing = bool(args.ckpt)

for item in args.spec.split(","):
        _p = item.split(":")
        sp_w = int(_p[0]); F = int(_p[1])
        do_bwd = int(_p[2]) if len(_p) > 2 else 0
        DZMOD.set_sp_state(group=dist.group.WORLD if sp_w > 1 else None,
                           world=sp_w, rank=rank if sp_w > 1 else 0)
        N = F * FSL
        NB = (F - 1) // 2
        torch.manual_seed(11)
        x = torch.randn(1, 36, F, H, W, device=dev, dtype=torch.bfloat16)
        ctx = torch.randn(1, 512, 4096, device=dev, dtype=torch.bfloat16)
        t = torch.randint(0, 1000, (1, F), device=dev)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        A2A["n"] = 0
        out = None
        tag = "SP%d F=%-2d ckpt=%d%s" % (sp_w, F, args.ckpt, " +bwd" if do_bwd else "")
        try:
            tf = tb = tg = 0.0
            for _w in range(args.warmup + args.iters):
                t1 = time.time()
                out = m(x, t, None, ctx, seq_len=N, clean_x=x.clone(),
                        action=None, state=None, embodiment_id=None)
                torch.cuda.synchronize()
                if _w >= args.warmup:
                    tf += time.time() - t1
                o = out[0] if isinstance(out, (list, tuple)) else out
                if do_bwd:
                    loss = o.float().mean()
                    t1 = time.time()
                    loss.backward()
                    torch.cuda.synchronize()
                    if _w >= args.warmup:
                        tb += time.time() - t1
                    if args.gradsync:
                        t1 = time.time()
                        for _p in m.parameters():
                            if _p.grad is not None:
                                dist.all_reduce(_p.grad)
                        torch.cuda.synchronize()
                        if _w >= args.warmup:
                            tg += time.time() - t1
                    m.zero_grad(set_to_none=True)
            tf /= args.iters
            tb /= args.iters
            peak = torch.cuda.max_memory_allocated() / 1e9
            if is0:
                print("[OK ] %s fwd=%.2fs bwd=%.2fs sync=%.2fs peak=%.1fGB a2a=%d" %
                      (tag, tf, tb, tg, peak, A2A["n"]), flush=True)
        except torch.cuda.OutOfMemoryError:
            peak = torch.cuda.max_memory_allocated() / 1e9
            if is0:
                print("[OOM] %s peak=%.1fGB" % (tag, peak), flush=True)
        except Exception as e:
            if is0:
                print("[ERR] %s %s: %s" % (tag, type(e).__name__, str(e)[:240]), flush=True)
                traceback.print_exc()
        cnt = torch.tensor([A2A["n"]], device=dev)
        allc = [torch.zeros_like(cnt) for _ in range(world)]
        dist.all_gather(allc, cnt)
        if is0:
            print("      a2a/rank=%s" % [int(c.item()) for c in allc], flush=True)
        del x, ctx, t
        out = None
        torch.cuda.empty_cache()

dist.barrier()
dist.destroy_process_group()
