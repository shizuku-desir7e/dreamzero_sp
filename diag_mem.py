#!/usr/bin/env python3
"""量 14B 每层激活显存增量：DP vs SP，定位大头在哪。

背景：v3 之后 SP4 仍然 OOM（82GB），且 F 越大死得越早 =>
单层激活随 token 线性增长。需要知道 (a) 每层涨多少 (b) SP 是否真的把它除以 4。

用法：python -m torch.distributed.run --nproc_per_node=4 diag_mem.py
"""
import os, sys, time, traceback
from datetime import timedelta
import torch
import torch.distributed as dist

Z = os.environ.get("ZROOT", "/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengmenghan")
sys.path.insert(0, Z + "/dreamzero_sp/shim")
sys.path.insert(0, Z + "/dreamzero_sp/dreamzero")
import groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk as DZMOD
from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import CausalWanModel

_orig_blk = DZMOD.CausalWanAttentionBlock.forward
def _unwrap(self, x, *a, **k):
    out = _orig_blk(self, x, *a, **k)
    return out[0] if isinstance(out, tuple) else out
DZMOD.CausalWanAttentionBlock.forward = _unwrap

dist.init_process_group("nccl", timeout=timedelta(seconds=300))
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
m = CausalWanModel(**CFG); m = m.to(dev); torch.set_default_dtype(torch.float32)
m.eval(); m.gradient_checkpointing = False
for p in m.parameters(): p.requires_grad_(True)
if is0: print("[build] %.0fs resident=%.1fGB" % (time.time()-t0, torch.cuda.memory_allocated()/1e9), flush=True)
dist.barrier()

LOG = []
def mk_hook(i):
    def h(mod, inp, out):
        LOG.append((i, torch.cuda.memory_allocated() / 1e9))
    return h
for i, b in enumerate(m.blocks):
    b.register_forward_hook(mk_hook(i))

def run(sp_w, F, ckpt):
    DZMOD.set_sp_state(group=dist.group.WORLD if sp_w > 1 else None,
                       world=sp_w, rank=rank if sp_w > 1 else 0)
    m.gradient_checkpointing = bool(ckpt)
    N = F * FSL
    torch.manual_seed(11)
    x = torch.randn(1, 36, F, H, W, device=dev, dtype=torch.bfloat16)
    ctx = torch.randn(1, 512, 4096, device=dev, dtype=torch.bfloat16)
    t = torch.randint(0, 1000, (1, F), device=dev)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    LOG.clear()
    tag = "SP%d F=%-2d ckpt=%d" % (sp_w, F, ckpt)
    try:
        out = m(x, t, None, ctx, seq_len=N, clean_x=x.clone(),
                action=None, state=None, embodiment_id=None)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1e9
        if is0:
            prof = {i: v for i, v in LOG}
            pts = [0, 4, 9, 19, 29, 39]
            line = " ".join("L%d=%.1f" % (i + 1, prof[i]) for i in pts if i in prof)
            slope = (prof[39] - prof[0]) / 39 if (39 in prof and 0 in prof) else float("nan")
            print("[OK ] %s peak=%.1fGB  d/layer=%.3fGB  %s" % (tag, peak, slope, line), flush=True)
    except torch.cuda.OutOfMemoryError:
        if is0:
            print("[OOM] %s peak=%.1fGB  撑到 L%d" % (tag, torch.cuda.max_memory_allocated()/1e9,
                                                      (LOG[-1][0] + 1) if LOG else 0), flush=True)
    except Exception as e:
        if is0:
            print("[ERR] %s %s: %s" % (tag, type(e).__name__, str(e)[:200]), flush=True)
    del x, ctx, t
    torch.cuda.empty_cache()

for cfg in [(1, 3, 0), (4, 3, 0), (1, 9, 0), (4, 9, 0), (4, 33, 1), (1, 33, 1)]:
    run(*cfg)
    dist.barrier()

dist.barrier(); dist.destroy_process_group()
