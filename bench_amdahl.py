#!/usr/bin/env python3
"""SP 收益上限（Amdahl）：在真实形状下测 attention vs MLP 的墙钟时间。

N=29040, heads=40, head_dim=128, dim=5120, ffn=13824（DreamZero 14B 的真实配置）。
如果 attention 只占 X%，那么 SP 再完美也只能加速 1/(1-X+X/P)。

这一步不需要模型 forward 跑通，也不依赖权重。
"""
import time, torch, torch.nn.functional as F

dev, dt = "cuda:0", torch.bfloat16
N, H, D, DIM, FFN = 29040, 40, 128, 5120, 13824
B = 1


def bench(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1e3      # ms


print("shape: B=%d N=%d H=%d D=%d dim=%d ffn=%d" % (B, N, H, D, DIM, FFN), flush=True)

q = torch.randn(B, N, H, D, device=dev, dtype=dt)
k = torch.randn(B, N, H, D, device=dev, dtype=dt)
v = torch.randn(B, N, H, D, device=dev, dtype=dt)

# attention：用 SDPA（flash 后端）代表；A100 上 flash_attn2 亦可
def attn():
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False)

# MLP：两个线性 + GELU
w1 = torch.randn(DIM, FFN, device=dev, dtype=dt)
w2 = torch.randn(FFN, DIM, device=dev, dtype=dt)
x = torch.randn(B, N, DIM, device=dev, dtype=dt)
def mlp():
    return torch.nn.functional.gelu(x @ w1) @ w2

# QKV/O 投影（也属于 attention 模块的线性部分）
wq = torch.randn(DIM, DIM, device=dev, dtype=dt)
def proj():
    return (x @ wq, x @ wq)

t_attn = bench(attn)
t_mlp = bench(mlp)
t_proj = bench(proj)

# 每层：1 次 attn + 1 次 mlp + 4 个投影（q,k,v,o）
per_layer = t_attn + t_mlp + 2 * t_proj          # proj 计时 2 次 matmul，×2 = 4 个投影
print("\n--- per-layer wall clock (ms) ---", flush=True)
print("  attention(SDPA)  %8.2f" % t_attn, flush=True)
print("  MLP              %8.2f" % t_mlp, flush=True)
print("  QKVO projections %8.2f" % (2 * t_proj), flush=True)
print("  layer total      %8.2f" % per_layer, flush=True)
share = t_attn / per_layer
print("\n  attention share = %.1f%%" % (share * 100), flush=True)

L = 40
print("\n--- Amdahl: SP 只切 attention ---", flush=True)
for P in (2, 4, 8):
    ideal = 1.0 / ((1 - share) + share / P)
    print("  P=%d  ideal speedup = %.2fx" % (P, ideal), flush=True)

print("\n--- 若 SP 同时切 attention + MLP（sequence parallel 的常见做法）---", flush=True)
for P in (2, 4, 8):
    print("  P=%d  ideal speedup = %.2fx" % (P, P), flush=True)

print("\n--- 通信开销估算（Ulysses: 每层 2 次 all-to-all）---", flush=True)
bytes_per_all2all = N * DIM * 2                  # bf16
print("  payload/layer = %.1f MB, 40 layers = %.1f MB" % (
    bytes_per_all2all / 1e6, bytes_per_all2all * 40 / 1e6), flush=True)
print("  注：4 卡 NVLink 有效带宽实测约 100-200 GB/s（A100），理论上 40 层约 %.1f-%.1f ms" % (
    bytes_per_all2all * 40 / 150e9 * 1e3, bytes_per_all2all * 40 / 80e9 * 1e3), flush=True)
