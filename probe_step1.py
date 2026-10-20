#!/usr/bin/env python3
"""SP 可行性探针 Step 1：用 checkpoint 的真实配置构造 14B 模型，测单卡 forward 显存与时间。

不加载权重（随机初始化），只回答一个问题：
  29k token + 14B + A100-80G，DP 能不能跑？SP 是不是必需？
"""
import os, sys, time, json
import torch

Z = "/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengmenghan"
sys.path.insert(0, Z + "/dreamzero_sp/shim")
sys.path.insert(0, Z + "/dreamzero_sp/dreamzero")

from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import CausalWanModel

CFG = dict(
    model_type="ti2v", patch_size=(1, 2, 2), frame_seqlen=880, text_len=512,
    in_dim=36, dim=5120, ffn_dim=13824, freq_dim=256, text_dim=4096, out_dim=16,
    num_heads=40, num_layers=40, num_frame_per_block=2, action_dim=32,
    num_action_per_block=24, num_state_per_block=1, max_state_dim=64,
    max_num_embodiments=32, hidden_size=1024, concat_first_frame_latent=True,
    qk_norm=True, cross_attn_norm=True, max_chunk_size=4, sink_size=0,
)

print("[1] build 14B config (random init, bf16)", flush=True)
t0 = time.time()
try:
    m = CausalWanModel(**CFG).to("cuda:0").to(torch.bfloat16).eval()
except Exception as e:
    print("[fail] build:", type(e).__name__, e, flush=True)
    raise SystemExit(1)
n = sum(p.numel() for p in m.parameters())
print("    built in %.1fs  params=%.2fB  GPU=%.1fGB" % (
    time.time() - t0, n / 1e9, torch.cuda.memory_allocated() / 1e9), flush=True)

# 视觉 token 账
# block layout 要求 (num_image_blocks * num_frame_per_block) = 帧数，
# 16 blocks x 2 = 32 帧；首帧已 concat 进 in_dim=36 的通道里（concat_first_frame_latent）
F, H, W = 33, 40, 88          # 4 views 沿 W 拼：4*22=88
tok_per_frame = (H // 2) * (W // 2)
N = F * tok_per_frame
print("\n[2] token accounting: F=%d H=%d W=%d -> %d tok/frame, N=%d" % (F, H, W, tok_per_frame, N), flush=True)
d, ffn, L = CFG["dim"], CFG["ffn_dim"], CFG["num_layers"]
attn = 4 * N * N * d
mlp = 6 * N * d * ffn
print("    attn/layer=%.2f TFLOPs  mlp/layer=%.2f TFLOPs  ratio=%.2f" % (
    attn / 1e12, mlp / 1e12, attn / mlp), flush=True)
print("    total fwd ~= %.2f PFLOPs" % ((attn + mlp) * L / 1e15), flush=True)

print("\n[3] forward with realistic shapes", flush=True)
B = 1
x = torch.randn(B, CFG["in_dim"], F, H, W, device="cuda:0", dtype=torch.bfloat16)
clean_x = torch.randn(B, CFG["in_dim"], F, H, W, device="cuda:0", dtype=torch.bfloat16)
ctx = torch.randn(B, CFG["text_len"], CFG["text_dim"], device="cuda:0", dtype=torch.bfloat16)
# 契约（从 block layout 校验逻辑反推）：
#   num_blocks = (F-1)//num_frame_per_block = (33-1)//2 = 16
#   action_register_length 必须 = num_blocks*(num_action_per_block+num_state_per_block) = 16*25 = 400
NB = (F - 1) // CFG["num_frame_per_block"]
A_PER = CFG["num_action_per_block"]; S_PER = CFG["num_state_per_block"]
t = torch.randint(0, 1000, (B, F), device="cuda:0")
ta = torch.randint(0, 1000, (B, NB * A_PER), device="cuda:0")
act = torch.randn(B, NB * A_PER, CFG["action_dim"], device="cuda:0", dtype=torch.bfloat16)
st = torch.randn(B, NB * S_PER, CFG["max_state_dim"], device="cuda:0", dtype=torch.bfloat16)
emb = torch.zeros(B, dtype=torch.long, device="cuda:0")

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
try:
    with torch.no_grad():
        out = m(x, t, ta, ctx, seq_len=N, clean_x=clean_x, action=act, state=st, embodiment_id=emb)
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    print("    [ok] forward %.2fs  peak=%.1f GB" % (dt, peak), flush=True)
    if isinstance(out, (list, tuple)):
        print("    out: list of", len(out), "tensor(s), first shape", tuple(out[0].shape), flush=True)
    else:
        print("    out shape", tuple(out.shape), flush=True)
except Exception as e:
    print("    [fail] forward:", type(e).__name__, str(e)[:300], flush=True)
    print("    peak=%.1f GB" % (torch.cuda.max_memory_allocated() / 1e9), flush=True)

print("\n[DONE]", flush=True)
