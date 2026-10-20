#!/usr/bin/env python3
"""真权重语义验证：把 $Z/model 的 14B DiT 灌进 CausalWanModel，比较 DP 与 SP=4 的输出和梯度。

这是"系统测量"和"端到端训练"的分界线：随机权重只能证明显存/时间，
真权重才能证明 SP 的 forward/backward 在语义上是等价的。

用法：
  python -m torch.distributed.run --nproc_per_node=4 test_real_weights.py --frames 9
"""
import argparse, json, os, sys, time
from datetime import timedelta
import torch
import torch.distributed as dist
from safetensors import safe_open

Z = os.environ.get("ZROOT", "/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengmenghan")
sys.path.insert(0, Z + "/dreamzero_sp/shim")
sys.path.insert(0, Z + "/dreamzero_sp/dreamzero")
import groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk as DZMOD
from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import CausalWanModel

ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, default=9)
ap.add_argument("--ckpt_dir", type=str, default=Z + "/model")
ap.add_argument("--prefix", type=str, default="action_head.model.")
ap.add_argument("--ckpt", type=int, default=1)
args = ap.parse_args()

dist.init_process_group("nccl", timeout=timedelta(seconds=600))
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank); dev = "cuda:%d" % rank; is0 = rank == 0

if not 1:
    pass

H, W, FSL = 40, 88, 880
CFG = dict(model_type="ti2v", patch_size=(1, 2, 2), frame_seqlen=FSL, text_len=512,
           in_dim=36, dim=5120, ffn_dim=13824, freq_dim=256, text_dim=4096, out_dim=16,
           num_heads=40, num_layers=40, num_frame_per_block=2, action_dim=32,
           num_action_per_block=24, num_state_per_block=1, max_state_dim=64,
           max_num_embodiments=32, hidden_size=1024, concat_first_frame_latent=True,
           qk_norm=True, cross_attn_norm=True, max_chunk_size=4, sink_size=0)

# ---------- 构造 + 逐分片灌权重（safetensors 直接 load 到 GPU，避免 CPU 峰值 66GB）----------
t0 = time.time()
torch.manual_seed(0)
torch.set_default_dtype(torch.bfloat16)
m = CausalWanModel(**CFG).to(dev)
torch.set_default_dtype(torch.float32)
m.eval(); m.gradient_checkpointing = bool(args.ckpt)
if not args.ckpt:
    # 关 ckpt 时必须补上游的解包 bug；开 ckpt 时上游已解包，再 patch 会报 unpack 错
    _orig_blk = DZMOD.CausalWanAttentionBlock.forward
    def _unwrap(self, x, *a, **k):
        out = _orig_blk(self, x, *a, **k)
        return out[0] if isinstance(out, tuple) else out
    DZMOD.CausalWanAttentionBlock.forward = _unwrap
for p in m.parameters():
    p.requires_grad_(True)
if is0:
    print("[build] %.0fs resident=%.1fGB" % (time.time() - t0, torch.cuda.memory_allocated() / 1e9), flush=True)

idx = json.load(open(os.path.join(args.ckpt_dir, "model.safetensors.index.json")))
wm = idx["weight_map"]
shards = sorted(set(wm.values()))
own = {k[len(args.prefix):] for k in wm if k.startswith(args.prefix)}
model_keys = set(dict(m.named_parameters()).keys()) | set(dict(m.named_buffers()).keys())
missing = sorted(model_keys - own)
unexpected = sorted(own - model_keys)
if is0:
    print("[keys] ckpt=%d model=%d missing=%d unexpected=%d" %
          (len(own), len(model_keys), len(missing), len(unexpected)), flush=True)
    for k in missing[:8]:
        print("        missing: %s" % k, flush=True)
    for k in unexpected[:8]:
        print("        unexpected: %s" % k, flush=True)

bad_shape = []
for sh in shards:
    from safetensors.torch import load_file
    sd = load_file(os.path.join(args.ckpt_dir, sh), device=dev)
    sub = {k[len(args.prefix):]: v for k, v in sd.items() if k.startswith(args.prefix)}
    pmap = dict(m.named_parameters()); bmap = dict(m.named_buffers())
    for k, v in sub.items():
        tgt = pmap[k] if k in pmap else bmap.get(k)
        if tgt is not None and tgt.shape != v.shape:
            bad_shape.append((k, tuple(tgt.shape), tuple(v.shape)))
        elif tgt is not None:
            tgt.data.copy_(v)
    del sd, sub
    torch.cuda.empty_cache()
if is0:
    print("[load] %.0fs  shape_mismatch=%d" % (time.time() - t0, len(bad_shape)), flush=True)
    for k, a, b in bad_shape[:8]:
        print("        %s: model%s vs ckpt%s" % (k, a, b), flush=True)
dist.barrier()

# ---------- 输入（真实权重要求 input 是 latent 量级，scale 不能乱来）----------
F, N = args.frames, args.frames * FSL
NB = (F - 1) // 2
torch.manual_seed(1234)
x = (torch.randn(1, 36, F, H, W, device=dev, dtype=torch.bfloat16) * 0.5)
ctx = torch.randn(1, 512, 4096, device=dev, dtype=torch.bfloat16) * 0.1
t = torch.randint(0, 1000, (1, F), device=dev)
act = torch.randn(1, NB * 24, 32, device=dev, dtype=torch.bfloat16) * 0.3
st = torch.randn(1, NB * 1, 64, device=dev, dtype=torch.bfloat16) * 0.3
ta = torch.randint(0, 1000, (1, NB * 24), device=dev)
emb = torch.zeros(1, dtype=torch.long, device=dev)


def run(sp_w, do_bwd=True):
    DZMOD.set_sp_state(group=dist.group.WORLD if sp_w > 1 else None,
                       world=sp_w, rank=rank if sp_w > 1 else 0)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m.zero_grad(set_to_none=True)
    out = m(x, t, ta, ctx, seq_len=N, clean_x=x.clone(),
            action=act, state=st, embodiment_id=emb)
    v, a = out
    loss = v.float().pow(2).mean() + a.float().pow(2).mean()
    if do_bwd:
        loss.backward()
    gn = torch.sqrt(sum((p.grad.float().pow(2).sum() for p in m.parameters()
                         if p.grad is not None), torch.zeros((), device=dev)))
    stats = torch.tensor([float(loss), float(gn), float(torch.cuda.max_memory_allocated() / 1e9)],
                         device=dev)
    if not do_bwd:
        m.zero_grad(set_to_none=True)
    return v.detach().float(), a.detach().float(), stats


# SP=4 先跑（显存友好），再跑 DP
v_sp, a_sp, s_sp = run(4)
if is0:
    print("[SP4] loss=%.6f gradnorm=%.4f peak=%.1fGB" % (s_sp[0], s_sp[1], s_sp[2]), flush=True)
torch.cuda.empty_cache()
v_dp, a_dp, s_dp = run(1)
if is0:
    print("[DP ] loss=%.6f gradnorm=%.4f peak=%.1fGB" % (s_dp[0], s_dp[1], s_dp[2]), flush=True)

if is0:
    print("\n[语义等价性] 真权重，F=%d，DP vs SP=4" % F, flush=True)
    for name, u, r in (("video", v_sp, v_dp), ("action", a_sp, a_dp)):
        d = (u - r).abs()
        denom = r.abs().max().item() + 1e-8
        print("  %-7s ref_absmax=%.4f  max_abs_err=%.4e  rel=%.3e" %
              (name, r.abs().max().item(), d.max().item(), d.max().item() / denom), flush=True)
    print("  loss  : SP=%.6f  DP=%.6f  rel=%.3e" %
          (s_sp[0], s_dp[0], abs(s_sp[0] - s_dp[0]) / (abs(s_dp[0]) + 1e-8)), flush=True)
    print("  gradnorm: SP=%.4f  DP=%.4f  rel=%.3e" %
          (s_sp[1], s_dp[1], abs(s_sp[1] - s_dp[1]) / (abs(s_dp[1]) + 1e-8)), flush=True)

dist.barrier(); dist.destroy_process_group()
