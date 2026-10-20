#!/usr/bin/env python3
"""SP 探针 Step 4：gradient checkpointing 对照。

Step 3 已证明：**不开 checkpointing，连 9 帧（7,920 token）都 OOM**（peak 83.6GB > 79.25GB）。
本步回答：开了 checkpointing 之后，DP 能跑到多大？代价是多少时间？

这决定 SP 的定位：
  - 若 ckpt 也跑不动 F=33 ⇒ SP 是「能跑」的唯一手段
  - 若 ckpt 能跑但很慢 ⇒ SP 的卖点是「同等显存下更快」
"""
import argparse, gc, sys, time
import torch

Z = "/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengmenghan"
sys.path.insert(0, Z + "/dreamzero_sp/shim")
sys.path.insert(0, Z + "/dreamzero_sp/dreamzero")
import groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk as DZMOD
from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import CausalWanModel

ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=str, default="9,17,25,33")
ap.add_argument("--ckpt", type=int, default=1)
args = ap.parse_args()
USE_CKPT = bool(args.ckpt)

# 只有「不开 checkpointing」时才需要解包补丁；开了之后模型自己会解包。
if not USE_CKPT:
    _orig = DZMOD.CausalWanAttentionBlock.forward
    def _unwrap(self, x, *a, **k):
        out = _orig(self, x, *a, **k)
        return out[0] if isinstance(out, tuple) else out
    DZMOD.CausalWanAttentionBlock.forward = _unwrap

FSL = 880
CFG = dict(model_type="ti2v", patch_size=(1, 2, 2), frame_seqlen=FSL, text_len=512,
           in_dim=36, dim=5120, ffn_dim=13824, freq_dim=256, text_dim=4096, out_dim=16,
           num_heads=40, num_layers=40, num_frame_per_block=2, action_dim=32,
           num_action_per_block=24, num_state_per_block=1, max_state_dim=64,
           max_num_embodiments=32, hidden_size=1024, concat_first_frame_latent=True,
           qk_norm=True, cross_attn_norm=True, max_chunk_size=4, sink_size=0)

print("[cfg] ckpt=%s" % USE_CKPT, flush=True)
t0 = time.time()
m = CausalWanModel(**CFG).to("cuda:0").to(torch.bfloat16)
m.eval()
m.gradient_checkpointing = USE_CKPT
for p in m.parameters():
    p.requires_grad_(True)
n = sum(p.numel() for p in m.parameters())
print("[build] %.1fs params=%.2fB resident=%.1fGB" % (time.time() - t0, n / 1e9,
                                                      torch.cuda.memory_allocated() / 1e9), flush=True)

rows = []
for F in [int(x) for x in args.frames.split(",")]:
    B, NB, N = 1, (F - 1) // 2, F * FSL
    try:
        torch.cuda.empty_cache(); gc.collect(); torch.cuda.reset_peak_memory_stats()
        dev, dt = "cuda:0", torch.bfloat16
        x = torch.randn(B, 36, F, 40, 88, device=dev, dtype=dt)
        clean_x = torch.randn(B, 36, F, 40, 88, device=dev, dtype=dt)
        ctx = torch.randn(B, 512, 4096, device=dev, dtype=dt)
        t = torch.randint(0, 1000, (B, F), device=dev)
        ta = torch.randint(0, 1000, (B, NB * 24), device=dev)
        act = torch.randn(B, NB * 24, 32, device=dev, dtype=dt)
        st = torch.randn(B, NB, 64, device=dev, dtype=dt)
        emb = torch.zeros(B, dtype=torch.long, device=dev)

        t0 = time.time()
        out = m(x, t, ta, ctx, seq_len=N, clean_x=clean_x, action=act, state=st, embodiment_id=emb)
        torch.cuda.synchronize(); tf = time.time() - t0
        o = out[0] if isinstance(out, (list, tuple)) else out
        loss = o.float().mean()
        t0 = time.time(); loss.backward(); torch.cuda.synchronize(); tb = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        print("[OK ] F=%2d N=%6d  fwd=%5.1fs bwd=%5.1fs  peak=%.1fGB" % (F, N, tf, tb, peak), flush=True)
        rows.append((F, N, peak, tf, tb, "ok"))
        del out, o, loss, x, clean_x, ctx, t, ta, act, st
        m.zero_grad(set_to_none=True); torch.cuda.empty_cache(); gc.collect()
    except torch.cuda.OutOfMemoryError as e:
        peak = torch.cuda.max_memory_allocated() / 1e9
        print("[OOM] F=%2d N=%6d peak=%.1fGB" % (F, N, peak), flush=True)
        rows.append((F, N, peak, -1, -1, "oom"))
        m.zero_grad(set_to_none=True); torch.cuda.empty_cache(); gc.collect()

print("\n=== summary ckpt=%s ===" % USE_CKPT, flush=True)
print("  F   N      peak(GB)  fwd(s)  bwd(s)  status", flush=True)
for F, N, pk, tf, tb, stt in rows:
    print("  %2d  %6d  %8.1f  %6.1f  %6.1f  %s" % (F, N, pk, tf, tb, stt), flush=True)
