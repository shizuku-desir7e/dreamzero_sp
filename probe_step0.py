#!/usr/bin/env python3
"""SP 可行性探针 Step 0：能否在 A100 上构造 DreamZero 的 CausalWanModel 并前向。

只测「模型能不能跑、单步多贵」，不测生成质量 —— 所以随机权重即可，不需要 34GB checkpoint。
"""
import os, sys, time, json
import torch

Z = "/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengmenghan"
REPO = f"{Z}/dreamzero_sp/dreamzero"
sys.path.insert(0, REPO)

def log(*a):
    print(*a, flush=True)

try:
    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import CausalWanModel
    log("[ok] import CausalWanModel")
except Exception as e:
    log("[fail] import:", type(e).__name__, e)
    raise SystemExit(1)

# 小配置：只为验证结构与显存量级（真实 5B 是 dim=3072 / layers=30 左右）
CFG_SMALL = dict(model_type="ti2v", patch_size=(1, 2, 2), frame_seqlen=220, text_len=512,
                 in_dim=16, dim=512, ffn_dim=2048, num_heads=8, num_layers=4,
                 num_frame_per_block=2, action_dim=32, num_action_per_block=24,
                 num_state_per_block=1, max_state_dim=64, max_num_embodiments=32,
                 hidden_size=1024, concat_first_frame_latent=False, qk_norm=True,
                 cross_attn_norm=True, max_chunk_size=4, sink_size=0)

log("\n=== 构造小配置模型 ===")
t0 = time.time()
try:
    m = CausalWanModel(**CFG_SMALL)
    n = sum(p.numel() for p in m.parameters())
    log(f"[ok] built in {time.time()-t0:.1f}s  params={n/1e6:.1f}M")
except Exception as e:
    log("[fail] build:", type(e).__name__, e)
    raise SystemExit(1)

# 统计 attention 代码里的分段逻辑，供 SP 设计参考
log("\n=== 真实 5B 规模估算 ===")
for name, cfg in [("Wan2.2-5B(expert)", dict(dim=3072, ffn_dim=14336, num_heads=24, num_layers=30)),
                  ("Wan2.1-14B", dict(dim=5120, ffn_dim=13824, num_heads=40, num_layers=40))]:
    D, F_, L = cfg["dim"], cfg["ffn_dim"], cfg["num_layers"]
    # 每层: attn 4*D^2 + mlp 2*D*F  (粗略)
    per_layer = 4 * D * D + 2 * D * F_
    tot = L * per_layer
    log(f"  {name:<20} dim={D} layers={L}  ~{tot/1e9:.1f}G params(linear only)")

log("\n=== 序列长度账（320x176, 33 frames, VAE 8x16x16? 用 frame_seqlen=220 近似）===")
for T in (33, 17, 9):
    for fsl in (220, 440, 880):
        n_tok = T * fsl
        attn_flops_per_layer = 4 * n_tok * n_tok * 3072      # ~4*N^2*d
        mlp_flops_per_layer = 2 * n_tok * 3072 * 14336 * 3   # ~6*N*D*F
        log(f"  T={T:>2} fsl={fsl:>4} -> N={n_tok:>6}  attn/layer={attn_flops_per_layer/1e12:.2f}T "
            f" mlp/layer={mlp_flops_per_layer/1e12:.2f}T  ratio={attn_flops_per_layer/mlp_flops_per_layer:.2f}")

log("\n[done]")
