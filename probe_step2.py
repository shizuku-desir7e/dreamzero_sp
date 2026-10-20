#!/usr/bin/env python3
"""SP 探针 Step 2：用小模型（默认 4 层）快速调通 _forward_train 的输入契约，再放大到 14B。

用法：
  python probe_step2.py --layers 4        # 快速调试（~5s 构造）
  python probe_step2.py --layers 40 --full # 真实 14B（~175s 构造，33GB）
"""
import argparse, json, sys, time, traceback
import torch

Z = "/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengmenghan"
sys.path.insert(0, Z + "/dreamzero_sp/shim")
sys.path.insert(0, Z + "/dreamzero_sp/dreamzero")
import groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk as DZMOD
from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import CausalWanModel

# ---- 临时探针：看 block 到底收到什么 ----
_orig_blk = DZMOD.CausalWanAttentionBlock.forward
_SEEN = {"n": 0}
def _probe_blk(self, x, *a, **k):
    out = _orig_blk(self, x, *a, **k)
    # 上游 bug：forward 返回 (outputs, kv_cache)，但 _forward_train 的非 gradient-checkpointing
    # 分支写的是 `x = block(x, **kwargs)`，没有解包 ⇒ 第二个 block 收到 tuple。
    # 这里解包，等价于官方训练时走的 checkpointing 分支。
    if isinstance(out, tuple):
        return out[0]
    return out
DZMOD.CausalWanAttentionBlock.forward = _probe_blk

ap = argparse.ArgumentParser()
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--dim", type=int, default=512)
ap.add_argument("--heads", type=int, default=8)
ap.add_argument("--ffn", type=int, default=2048)
ap.add_argument("--frames", type=int, default=33)
ap.add_argument("--views", type=int, default=4)
ap.add_argument("--lat-h", type=int, default=40)
ap.add_argument("--lat-w", type=int, default=22)
ap.add_argument("--full", action="store_true")
args = ap.parse_args()

if args.full:
    args.layers, args.dim, args.heads, args.ffn = 40, 5120, 40, 13824

# patch(1,2,2) 之后：每个视角 (lat_h/2) x (lat_w/2)，四个视角拼在 W 维
FSL = (args.lat_h // 2) * (args.lat_w // 2) * args.views   # 20*11*4 = 880
CFG = dict(
    model_type="ti2v", patch_size=(1, 2, 2), frame_seqlen=FSL, text_len=512,
    in_dim=36, dim=args.dim, ffn_dim=args.ffn, freq_dim=256, text_dim=4096, out_dim=16,
    num_heads=args.heads, num_layers=args.layers, num_frame_per_block=2, action_dim=32,
    num_action_per_block=24, num_state_per_block=1, max_state_dim=64,
    max_num_embodiments=32, hidden_size=1024, concat_first_frame_latent=True,
    qk_norm=True, cross_attn_norm=True, max_chunk_size=4, sink_size=0,
)

print("[cfg] layers=%d dim=%d heads=%d ffn=%d frame_seqlen=%d" %
      (args.layers, args.dim, args.heads, args.ffn, FSL), flush=True)

t0 = time.time()
m = CausalWanModel(**CFG).to("cuda:0").to(torch.bfloat16).eval()
n = sum(p.numel() for p in m.parameters())
print("[build] %.1fs params=%.2fB GPU=%.1fGB" % (time.time() - t0, n / 1e9,
                                                torch.cuda.memory_allocated() / 1e9), flush=True)

# 单视角 latent 是 lat_h x lat_w；4 视角沿 W 拼
H = args.lat_h
W = args.lat_w * args.views             # 4 视角沿 W 拼：22*4 = 88 -> patch 后 44
F = args.frames
NB = (F - 1) // CFG["num_frame_per_block"]
N = F * FSL
print("[shape] F=%d H=%d W=%d -> tok/frame=%d N=%d num_blocks=%d" %
      (F, H, W, (H // 2) * (W // 2), N, NB), flush=True)

B = 1
dev, dt = "cuda:0", torch.bfloat16
x = torch.randn(B, CFG["in_dim"], F, H, W, device=dev, dtype=dt)
clean_x = torch.randn(B, CFG["in_dim"], F, H, W, device=dev, dtype=dt)
ctx = torch.randn(B, CFG["text_len"], CFG["text_dim"], device=dev, dtype=dt)
t = torch.randint(0, 1000, (B, F), device=dev)
ta = torch.randint(0, 1000, (B, NB * CFG["num_action_per_block"]), device=dev)
act = torch.randn(B, NB * CFG["num_action_per_block"], CFG["action_dim"], device=dev, dtype=dt)
st = torch.randn(B, NB * CFG["num_state_per_block"], CFG["max_state_dim"], device=dev, dtype=dt)
emb = torch.zeros(B, dtype=torch.long, device=dev)

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
try:
    with torch.no_grad():
        out = m(x, t, ta, ctx, seq_len=N, clean_x=clean_x, action=act, state=st, embodiment_id=emb)
    torch.cuda.synchronize()
    print("[ok] forward %.2fs peak=%.1fGB" % (time.time() - t0,
                                              torch.cuda.max_memory_allocated() / 1e9), flush=True)
    if isinstance(out, (list, tuple)):
        print("     out: %d tensor(s), first %s" % (len(out), tuple(out[0].shape)), flush=True)
    else:
        print("     out %s" % (tuple(out.shape),), flush=True)
except Exception:
    print("[FAIL] peak=%.1fGB" % (torch.cuda.max_memory_allocated() / 1e9), flush=True)
    traceback.print_exc()
