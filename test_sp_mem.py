#!/usr/bin/env python3
"""14B + 4 卡 Ulysses SP 的 fwd+bwd：能否跑通官方 33 帧？

对照组（已测）：
  DP(无 ckpt): F=9/17/25/33 全 OOM
  DP(+ckpt)  : F=9 76.6GB 勉强, F=17+ OOM
本实验：SP=4，每卡激活 1/4。

用法：python -m torch.distributed.run --nproc_per_node=4 test_sp_mem.py --frames 33
"""
import argparse, gc, sys, time
import torch
import torch.distributed as dist

Z = "/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengmenghan"
sys.path.insert(0, Z + "/dreamzero_sp/shim")
sys.path.insert(0, Z + "/dreamzero_sp/dreamzero")
import groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk as DZMOD
from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import CausalWanModel

_orig_blk = DZMOD.CausalWanAttentionBlock.forward
def _unwrap(self, x, *a, **k):
    out = _orig_blk(self, x, *a, **k)
    return out[0] if isinstance(out, tuple) else out
DZMOD.CausalWanAttentionBlock.forward = _unwrap

ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, default=33)
ap.add_argument("--ckpt", type=int, default=0)
args = ap.parse_args()

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)
dev = "cuda:%d" % rank

F, H, W = args.frames, 40, 88
W_LOCAL = W // world
FSL_FULL = (H // 2) * (W // 2)      # 20*22 = 440? 不对：全量 patch 后 20*44=880
FSL_FULL = (H // 2) * (W // 2)      # 20 * 44 = 880   (W/2=44)
N_GLOBAL = F * FSL_FULL

CFG = dict(model_type="ti2v", patch_size=(1, 2, 2), frame_seqlen=FSL_FULL, text_len=512,
           in_dim=36, dim=5120, ffn_dim=13824, freq_dim=256, text_dim=4096, out_dim=16,
           num_heads=40, num_layers=40, num_frame_per_block=2, action_dim=32,
           num_action_per_block=24, num_state_per_block=1, max_state_dim=64,
           max_num_embodiments=32, hidden_size=1024, concat_first_frame_latent=True,
           qk_norm=True, cross_attn_norm=True, max_chunk_size=4, sink_size=0)

if rank == 0:
    print("[cfg] world=%d F=%d N_global=%d ckpt=%s" % (world, F, N_GLOBAL, bool(args.ckpt)), flush=True)

t0 = time.time()
# 主机内存只够构造一份 14B：rank0 真实构造，其余 rank 用 meta device 占位后 to_empty，
# 再分块 broadcast 权重（逐 tensor broadcast 2000+ 次太慢，按 512MB 打包）。
if rank == 0:
    torch.manual_seed(0)
    m = CausalWanModel(**CFG).to(torch.bfloat16).to(dev)
else:
    with torch.device("meta"):
        m = CausalWanModel(**CFG)
    m = m.to_empty(device=dev).to(torch.bfloat16)

if world > 1:
    flat = [p.data.view(-1) for p in m.parameters()]
    total = sum(t.numel() for t in flat)
    buf = torch.empty(total, device=dev, dtype=torch.bfloat16)
    off = 0
    for t in flat:
        buf[off:off + t.numel()].copy_(t)
        off += t.numel()
    dist.broadcast(buf, src=0)
    off = 0
    for t in flat:
        t.copy_(buf[off:off + t.numel()])
        off += t.numel()
    del buf
    torch.cuda.empty_cache()

m.eval()
m.gradient_checkpointing = bool(args.ckpt)
for p in m.parameters():
    p.requires_grad_(True)
if rank == 0:
    print("[build] %.1fs resident=%.1fGB" % (time.time() - t0,
                                             torch.cuda.memory_allocated() / 1e9), flush=True)

DZMOD.set_sp_state(group=dist.group.WORLD, world=world, rank=rank)

B = 1
NB = (F - 1) // 2
x_full = torch.randn(B, 36, F, H, W_LOCAL, device=dev, dtype=torch.bfloat16)
ctx = torch.randn(B, 512, 4096, device=dev, dtype=torch.bfloat16)
t = torch.randint(0, 1000, (B, F), device=dev)

torch.cuda.reset_peak_memory_stats()
try:
    t0 = time.time()
    out = m(x_full, t, None, ctx, seq_len=N_GLOBAL, clean_x=x_full,
            action=None, state=None, embodiment_id=None)
    torch.cuda.synchronize()
    tf = time.time() - t0
    o = out[0] if isinstance(out, (list, tuple)) else out
    loss = o.float().mean()
    torch.cuda.synchronize()
    t0 = time.time()
    loss.backward()
    torch.cuda.synchronize()
    tb = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    if rank == 0:
        print("[OK ] SP=%d F=%d  fwd=%.2fs bwd=%.2fs  peak=%.1fGB" % (world, F, tf, tb, peak), flush=True)
except torch.cuda.OutOfMemoryError as e:
    peak = torch.cuda.max_memory_allocated() / 1e9
    if rank == 0:
        print("[OOM] SP=%d F=%d peak=%.1fGB" % (world, F, peak), flush=True)
except Exception as e:
    if rank == 0:
        print("[ERR] %s: %s" % (type(e).__name__, str(e)[:200]), flush=True)

dist.barrier()
dist.destroy_process_group()
