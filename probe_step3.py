#!/usr/bin/env python3
"""SP 探针 Step 3：fwd+bwd 峰值显存 vs 序列长度（**不开 gradient checkpointing**）。

目的：给出「DP 在什么规模下 OOM」的硬边界，作为 SP 必需性的直接证据。

做法：一次构造 14B，从短序列向上扫，每个长度跑一次 fwd+bwd，记录峰值；
OOM 则捕获并停止（已 OOM 的规模即为 DP 的显存墙）。
"""
import argparse, gc, sys, time, traceback
import torch

Z = "/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengmenghan"
sys.path.insert(0, Z + "/dreamzero_sp/shim")
sys.path.insert(0, Z + "/dreamzero_sp/dreamzero")

import groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk as DZMOD
from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import CausalWanModel

# 上游 bug：block 返回 (out, kv_cache)，_forward_train 非-checkpointing 分支不解包
_orig_blk = DZMOD.CausalWanAttentionBlock.forward
def _unwrap_blk(self, x, *a, **k):
    out = _orig_blk(self, x, *a, **k)
    return out[0] if isinstance(out, tuple) else out
DZMOD.CausalWanAttentionBlock.forward = _unwrap_blk

ap = argparse.ArgumentParser()
ap.add_argument("--layers", type=int, default=40)
ap.add_argument("--dim", type=int, default=5120)
ap.add_argument("--heads", type=int, default=40)
ap.add_argument("--ffn", type=int, default=13824)
ap.add_argument("--frames", type=str, default="9,17,25,33")
ap.add_argument("--bs", type=int, default=1)
args = ap.parse_args()

FSL = 880
CFG = dict(
    model_type="ti2v", patch_size=(1, 2, 2), frame_seqlen=FSL, text_len=512,
    in_dim=36, dim=args.dim, ffn_dim=args.ffn, freq_dim=256, text_dim=4096, out_dim=16,
    num_heads=args.heads, num_layers=args.layers, num_frame_per_block=2, action_dim=32,
    num_action_per_block=24, num_state_per_block=1, max_state_dim=64,
    max_num_embodiments=32, hidden_size=1024, concat_first_frame_latent=True,
    qk_norm=True, cross_attn_norm=True, max_chunk_size=4, sink_size=0,
)

print("[cfg] layers=%d dim=%d heads=%d ffn=%d bs=%d" %
      (args.layers, args.dim, args.heads, args.ffn, args.bs), flush=True)

t0 = time.time()
m = CausalWanModel(**CFG).to("cuda:0").to(torch.bfloat16)   # ← 必须是 bf16（fp32 要 66GB）
# 不能用 m.train()：SDAR/Wan 的 forward 在 training 模式会走 block-diffusion loss 分支并 assert labels。
# eval() 只关 dropout/batchnorm，梯度照常反传。
m.eval()
# 明确关闭 gradient checkpointing —— 我们要测的正是「不做任何省显存手段」的 DP 最坏情况。
# 注意：关掉后走 _forward_train 的非-checkpointing 分支，那条分支不解包 (out, kv)，
# 所以上面的 monkey-patch 解包是必需的；开启 checkpointing 时反而不能 patch（会 unpack 失败）。
m.gradient_checkpointing = False
for p in m.parameters():
    p.requires_grad_(True)
n = sum(p.numel() for p in m.parameters())
base_mem = torch.cuda.memory_allocated() / 1e9
print("[build] %.1fs params=%.2fB resident=%.1fGB" % (time.time() - t0, n / 1e9, base_mem), flush=True)

F_LIST = [int(x) for x in args.frames.split(",")]
results = []

for F in F_LIST:
    B = args.bs
    NB = (F - 1) // 2
    N = F * FSL
    dev, dt = "cuda:0", torch.bfloat16
    try:
        torch.cuda.empty_cache(); gc.collect()
        torch.cuda.reset_peak_memory_stats()

        x = torch.randn(B, 36, F, 40, 88, device=dev, dtype=dt, requires_grad=False)
        clean_x = torch.randn(B, 36, F, 40, 88, device=dev, dtype=dt)
        ctx = torch.randn(B, 512, 4096, device=dev, dtype=dt)
        t = torch.randint(0, 1000, (B, F), device=dev)
        ta = torch.randint(0, 1000, (B, NB * 24), device=dev)
        act = torch.randn(B, NB * 24, 32, device=dev, dtype=dt)
        st = torch.randn(B, NB * 1, 64, device=dev, dtype=dt)
        emb = torch.zeros(B, dtype=torch.long, device=dev)

        t0 = time.time()
        out = m(x, t, ta, ctx, seq_len=N, clean_x=clean_x, action=act, state=st, embodiment_id=emb)
        torch.cuda.synchronize()
        t_fwd = time.time() - t0

        o = out[0] if isinstance(out, (list, tuple)) else out
        loss = o.float().mean()
        t0 = time.time()
        loss.backward()
        torch.cuda.synchronize()
        t_bwd = time.time() - t0

        peak = torch.cuda.max_memory_allocated() / 1e9
        # 优化器状态按 AdamW fp32 估算（ZeRO-2 下每卡仍持有全部参数的 m/v，但只在 step 时分配）
        opt_est = n * 4 * 2 / 1e9
        print("[OK ] F=%2d N=%6d  fwd=%5.1fs bwd=%5.1fs  peak=%.1fGB  (+AdamW m/v %.0fGB => %.0fGB)"
              % (F, N, t_fwd, t_bwd, peak, opt_est, peak + opt_est), flush=True)
        results.append((F, N, peak, t_fwd, t_bwd, "ok"))
        del out, o, loss, x, clean_x, ctx, t, ta, act, st
        m.zero_grad(set_to_none=True)
        torch.cuda.empty_cache(); gc.collect()
    except torch.cuda.OutOfMemoryError as e:
        peak = torch.cuda.max_memory_allocated() / 1e9
        print("[OOM] F=%2d N=%6d  peak_before_oom=%.1fGB  msg=%s"
              % (F, N, peak, str(e).split("\n")[0][:90]), flush=True)
        results.append((F, N, peak, -1, -1, "oom"))
        m.zero_grad(set_to_none=True)
        torch.cuda.empty_cache(); gc.collect()
    except RuntimeError as e:
        peak = torch.cuda.max_memory_allocated() / 1e9
        print("[ERR] F=%2d N=%6d  peak=%.1fGB  %s" % (F, N, peak, str(e)[:120]), flush=True)
        results.append((F, N, peak, -1, -1, "err"))
        m.zero_grad(set_to_none=True)
        torch.cuda.empty_cache(); gc.collect()

print("\n=== summary (no gradient checkpointing) ===", flush=True)
print("  F   N      peak(GB)  fwd(s)  bwd(s)  status", flush=True)
for F, N, peak, tf, tb, st in results:
    print("  %2d  %6d  %8.1f  %6.1f  %6.1f  %s" % (F, N, peak, tf, tb, st), flush=True)
print("\n  A100-80G 可用约 79GB；若 peak > 79 即 OOM。", flush=True)
