#!/usr/bin/env python3
"""baseline：DP + gradient checkpointing + activation offload（CPU 换出）。

回答审稿人一定会问的那句：「DP 是不是真的不行，还是你没试够？」

做法：用 `torch.autograd.graph.saved_tensors_hooks` 把 backward 需要的 saved tensors
搬到 CPU（pin memory），backward 时再搬回来。配合 gradient checkpointing，
saved 的就是 block 输入（40 × [S,5120]），offload 能省下这部分显存。

注意：这条 baseline 的代价是 PCIe 往返，所以**只有当它慢很多时，
SP 的价值才成立**——这正是我们要量的东西。

用法：bash run_off.sh
"""
import argparse, os, sys, time
from datetime import timedelta
import torch
import torch.distributed as dist

Z = os.environ.get("ZROOT", "/home/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengmenghan")
if not os.path.isdir(Z):
    Z = "/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengmenghan"
sys.path.insert(0, Z + "/dreamzero_sp/shim")
sys.path.insert(0, Z + "/dreamzero_sp/dreamzero")
import groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk as DZMOD
from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import CausalWanModel

ap = argparse.ArgumentParser()
ap.add_argument("--spec", type=str, default="1:17,1:33")   # world:F
ap.add_argument("--offload", type=int, default=1)
ap.add_argument("--iters", type=int, default=2)
ap.add_argument("--warmup", type=int, default=1)
args = ap.parse_args()

dist.init_process_group("nccl", timeout=timedelta(seconds=600))
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank); dev = "cuda:%d" % rank; is0 = rank == 0
H, W, FSL = 40, 88, 880

CFG = dict(model_type="ti2v", patch_size=(1, 2, 2), frame_seqlen=FSL, text_len=512,
           in_dim=36, dim=5120, ffn_dim=13824, freq_dim=256, text_dim=4096, out_dim=16,
           num_heads=40, num_layers=40, num_frame_per_block=2, action_dim=32,
           num_action_per_block=24, num_state_per_block=1, max_state_dim=64,
           max_num_embodiments=32, hidden_size=1024, concat_first_frame_latent=True,
           qk_norm=True, cross_attn_norm=True, max_chunk_size=4, sink_size=0)

t0 = time.time()
torch.manual_seed(0); torch.set_default_dtype(torch.bfloat16)
m = CausalWanModel(**CFG).to(dev); torch.set_default_dtype(torch.float32)
m.eval(); m.gradient_checkpointing = True          # 和主表一致
for p in m.parameters():
    p.requires_grad_(True)
if is0:
    print("[build] %.0fs resident=%.1fGB" % (time.time() - t0, torch.cuda.memory_allocated() / 1e9), flush=True)
dist.barrier()

H2 = {"n": 0}


def pack(t):
    H2["n"] += 1
    return t.detach().to('cpu', copy=True).pin_memory()   # pin_memory 只能作用于 CPU 张量


def unpack(t):
    return t.to(dev, non_blocking=True)


for item in args.spec.split(","):
    p_, f_ = item.split(":")
    sp_w, F = int(p_), int(f_)
    N = F * FSL
    NB = (F - 1) // 2
    DZMOD.set_sp_state(group=dist.group.WORLD if sp_w > 1 else None,
                       world=sp_w, rank=rank if sp_w > 1 else 0)
    torch.manual_seed(11)
    x = torch.randn(1, 36, F, H, W, device=dev, dtype=torch.bfloat16) * 0.5
    ctx = torch.randn(1, 512, 4096, device=dev, dtype=torch.bfloat16) * 0.1
    t = torch.randint(0, 1000, (1, F), device=dev)
    act = torch.randn(1, NB * 24, 32, device=dev, dtype=torch.bfloat16) * 0.3
    st = torch.randn(1, NB * 1, 64, device=dev, dtype=torch.bfloat16) * 0.3
    ta = torch.randint(0, 1000, (1, NB * 24), device=dev)
    emb = torch.zeros(1, dtype=torch.long, device=dev)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    H2["n"] = 0
    tag = "DP+ckpt+offload P=%d F=%-2d" % (sp_w, F)
    if not args.offload:
        tag = "DP+ckpt         P=%d F=%-2d" % (sp_w, F)
    try:
        tt = 0.0
        for w in range(args.warmup + args.iters):
            t1 = time.time()
            if args.offload:
                with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                    out = m(x, t, ta, ctx, seq_len=N, clean_x=x.clone(),
                            action=act, state=st, embodiment_id=emb)
                    loss = out[0].float().pow(2).mean()
                    loss.backward()
            else:
                out = m(x, t, ta, ctx, seq_len=N, clean_x=x.clone(),
                        action=act, state=st, embodiment_id=emb)
                loss = out[0].float().pow(2).mean()
                loss.backward()
            torch.cuda.synchronize()
            if w >= args.warmup:
                tt += time.time() - t1
            m.zero_grad(set_to_none=True)
        tt /= args.iters
        peak = torch.cuda.max_memory_allocated() / 1e9
        if is0:
            print("[OK ] %s step=%.2fs peak=%.1fGB offloaded=%d" % (tag, tt, peak, H2["n"]), flush=True)
    except torch.cuda.OutOfMemoryError:
        if is0:
            print("[OOM] %s peak=%.1fGB" % (tag, torch.cuda.max_memory_allocated() / 1e9), flush=True)
    except Exception as e:
        if is0:
            print("[ERR] %s %s: %s" % (tag, type(e).__name__, str(e)[:180]), flush=True)
    del x, ctx, t, act, st, ta
    torch.cuda.empty_cache()

dist.barrier(); dist.destroy_process_group()
