#!/usr/bin/env python3
"""真权重 + 梯度正确性：SP 的梯度经 all_reduce(SUM) 之后必须与 DP 的梯度一致。

为什么要专门测这个
------------------
SP 里每张卡只持有 1/P 的 token，所以单卡 gradnorm 天然只有 DP 的 1/2 左右 —— 这不是错。
但要拿到完整梯度必须跨卡**求和**。而本实现里有 2.2% 的 token（首帧 + action_register）是
**复制**在每张卡上的：
  * attention 的 q/k/v/o 按 head 切片 => 各卡贡献互不重叠，求和正确 ✓
  * MLP / o-proj 处理全 dim 的本地 token => 复制段的贡献被算 P 次 => 求和后会偏大 ✗
所以本实验要量出这个偏差到底有多大。

用法：bash run_grad.sh
"""
import argparse, json, os, sys, time
from datetime import timedelta
import torch
import torch.distributed as dist

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

if not args.ckpt:
    _ob = DZMOD.CausalWanAttentionBlock.forward
    def _unwrap(self, x, *a, **k):
        o = _ob(self, x, *a, **k)
        return o[0] if isinstance(o, tuple) else o
    DZMOD.CausalWanAttentionBlock.forward = _unwrap

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
m.eval(); m.gradient_checkpointing = bool(args.ckpt)
for p in m.parameters():
    p.requires_grad_(True)

# 灌权重
from safetensors.torch import load_file
idx = json.load(open(os.path.join(args.ckpt_dir, "model.safetensors.index.json")))
for sh in sorted(set(idx["weight_map"].values())):
    sd = load_file(os.path.join(args.ckpt_dir, sh), device=dev)
    pmap = dict(m.named_parameters()); bmap = dict(m.named_buffers())
    for k, v in sd.items():
        if not k.startswith(args.prefix):
            continue
        kk = k[len(args.prefix):]
        tgt = pmap[kk] if kk in pmap else bmap.get(kk)
        if tgt is not None and tgt.shape == v.shape:
            tgt.data.copy_(v)
    del sd; torch.cuda.empty_cache()
if is0:
    print("[load] %.0fs" % (time.time() - t0), flush=True)
dist.barrier()

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

SAMPLE = ["blocks.0.ffn.0.weight", "blocks.0.self_attn.q.weight",
          "blocks.20.ffn.2.weight", "patch_embedding.weight", "head.head.weight"]


def grads(sp_w, reduce_sum):
    DZMOD.set_sp_state(group=dist.group.WORLD if sp_w > 1 else None,
                       world=sp_w, rank=rank if sp_w > 1 else 0)
    m.zero_grad(set_to_none=True); torch.cuda.empty_cache()
    out = m(x, t, ta, ctx, seq_len=N, clean_x=x.clone(),
            action=act, state=st, embodiment_id=emb)
    v, a = out
    # 复制输出（action 头）在每个 rank 上都算了一遍完整的 loss，反向时必须除以 P，
    # 否则梯度 all_reduce(SUM) 后被放大 P 倍。上报的 loss 仍用未缩放值。
    # v5 零复制：action 输出已 all_gather 回全量，两侧都不需要缩放
    loss = v.float().pow(2).mean() + a.float().pow(2).mean()
    loss.backward()
    if reduce_sum and world > 1:
        for p in m.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
    pmap = dict(m.named_parameters())
    gn = torch.sqrt(sum((p.grad.float().pow(2).sum() for p in m.parameters()
                         if p.grad is not None), torch.zeros((), device=dev)))
    picks = {k: pmap[k].grad.detach().float().clone() for k in SAMPLE if pmap[k].grad is not None}
    return float(loss), float(gn), picks


ls, gs, ps = grads(4, True)
ld, gd, pd = grads(1, False)

if is0:
    print("[SP4+allreduce] loss=%.6f gradnorm=%.4f" % (ls, gs), flush=True)
    print("[DP          ] loss=%.6f gradnorm=%.4f" % (ld, gd), flush=True)
    print("\n[梯度一致性] SP(all_reduce SUM) vs DP，真权重，F=%d" % F, flush=True)
    print("  gradnorm rel = %.3e" % (abs(gs - gd) / (gd + 1e-12)), flush=True)
    for k in SAMPLE:
        if k in ps and k in pd:
            a_, b_ = ps[k], pd[k]
            rel = (a_ - b_).norm().item() / (b_.norm().item() + 1e-12)
            print("  %-28s |SP|=%.4f |DP|=%.4f  relL2=%.3e" %
                  (k, a_.norm().item(), b_.norm().item(), rel), flush=True)

dist.barrier(); dist.destroy_process_group()
