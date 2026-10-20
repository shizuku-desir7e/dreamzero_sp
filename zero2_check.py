#!/usr/bin/env python3
"""验：ZeRO-2 分片梯度 == all_reduce(SUM) 全量梯度（逐元素）。

技巧：只做 backward、不做 optimizer step，两次的权重自然完全一致，不用备份/恢复 33GB。
ZeRO-2 的分片要 all_gather 回全量才能与 all_reduce 的结果逐元素比对。
采样 4 个有代表性的参数（attention q / ffn / patch_embedding / head），全量比对。

用法：python -m torch.distributed.run --nproc_per_node=4 zero2_check.py
"""
import os, sys, time
from datetime import timedelta
import torch
import torch.distributed as dist

Z = os.environ.get("ZROOT", "/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengmenghan")
sys.path.insert(0, Z + "/dreamzero_sp/shim")
sys.path.insert(0, Z + "/dreamzero_sp/dreamzero")
import groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk as DZMOD
from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import CausalWanModel

dist.init_process_group("nccl", timeout=timedelta(seconds=600))
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank); dev = "cuda:%d" % rank; is0 = rank == 0
H, W, FSL, F = 40, 88, 880, 9
N = F * FSL
NB = (F - 1) // 2

CFG = dict(model_type="ti2v", patch_size=(1, 2, 2), frame_seqlen=FSL, text_len=512,
           in_dim=36, dim=5120, ffn_dim=13824, freq_dim=256, text_dim=4096, out_dim=16,
           num_heads=40, num_layers=40, num_frame_per_block=2, action_dim=32,
           num_action_per_block=24, num_state_per_block=1, max_state_dim=64,
           max_num_embodiments=32, hidden_size=1024, concat_first_frame_latent=True,
           qk_norm=True, cross_attn_norm=True, max_chunk_size=4, sink_size=0)

t0 = time.time()
torch.manual_seed(0); torch.set_default_dtype(torch.bfloat16)
m = CausalWanModel(**CFG).to(dev); torch.set_default_dtype(torch.float32)
m.eval(); m.gradient_checkpointing = True
for p in m.parameters():
    p.requires_grad_(True)
if is0:
    print("[build] %.0fs" % (time.time() - t0), flush=True)
dist.barrier()

DZMOD.set_sp_state(group=dist.group.WORLD, world=world, rank=rank)

torch.manual_seed(11)
x = torch.randn(1, 36, F, H, W, device=dev, dtype=torch.bfloat16) * 0.5
ctx = torch.randn(1, 512, 4096, device=dev, dtype=torch.bfloat16) * 0.1
t = torch.randint(0, 1000, (1, F), device=dev)
act = torch.randn(1, NB * 24, 32, device=dev, dtype=torch.bfloat16) * 0.3
st = torch.randn(1, NB * 1, 64, device=dev, dtype=torch.bfloat16) * 0.3
ta = torch.randint(0, 1000, (1, NB * 24), device=dev)
emb = torch.zeros(1, dtype=torch.long, device=dev)

SAMPLE = ["blocks.0.self_attn.q.weight", "blocks.0.ffn.0.weight",
          "patch_embedding.weight", "head.head.weight"]
pmap = dict(m.named_parameters())


def fwd_bwd():
    m.zero_grad(set_to_none=True)
    out = m(x, t, ta, ctx, seq_len=N, clean_x=x.clone(),
            action=act, state=st, embodiment_id=emb)
    loss = out[0].float().pow(2).mean() + out[1].float().pow(2).mean()
    loss.backward()
    return float(loss)


# ---------- A) all_reduce(SUM) 全量梯度 ----------
la = fwd_bwd()
for q in m.parameters():
    if q.grad is not None:
        dist.all_reduce(q.grad, op=dist.ReduceOp.SUM)
gA = {k: pmap[k].grad.detach().float().clone() for k in SAMPLE}
mA = {k: pmap[k].grad.detach().float().norm().item() for k in SAMPLE}
gnA = torch.sqrt(sum((q.grad.float().pow(2).sum() for q in m.parameters()
                      if q.grad is not None), torch.zeros((), device=dev)))
if is0:
    print("[A all_reduce] loss=%.6f gradnorm=%.4f" % (la, gnA), flush=True)

# ---------- B) ZeRO-2：hook 里 reduce_scatter，每卡只留 1/P ----------
shards = {}
full_cnt = [0]


def mk(i):
    def hook(param):
        g = param.grad
        if g is None:
            return
        n = g.numel()
        assert n % world == 0, ("不可整除", n, world)
        out = torch.empty(n // world, dtype=g.dtype, device=g.device)
        dist.reduce_scatter_tensor(out, g.reshape(-1).contiguous(), op=dist.ReduceOp.SUM)
        shards[i] = out
        full_cnt[0] += 1
        param.grad = None
    return hook


for i, p in enumerate(m.parameters()):
    p.register_post_accumulate_grad_hook(mk(i))
lb = fwd_bwd()
# 把采样的分片 all_gather 回全量
gB = {}
for k in SAMPLE:
    p = pmap[k]
    idx = [i for i, q in enumerate(m.parameters()) if q is p][0]
    sh = shards[idx]
    out = torch.empty(p.numel(), dtype=sh.dtype, device=dev)
    dist.all_gather_into_tensor(out, sh)
    gB[k] = out.float().reshape(p.shape)
if is0:
    print("[B zero2    ] loss=%.6f 分片参数=%d" % (lb, full_cnt[0]), flush=True)
    print("\n[逐元素比对] ZeRO-2 分片梯度 vs all_reduce 全量梯度（F=%d, SP=%d）" % (F, world), flush=True)
    print("  %-30s %10s %10s %12s" % ("param", "|A|", "|B|", "relL2"), flush=True)
    for k in SAMPLE:
        a, b = gA[k], gB[k]
        d = (a - b).norm().item() / (a.norm().item() + 1e-12)
        maxabs = (a - b).abs().max().item()
        print("  %-30s %10.5f %10.5f %12.3e  (max_abs=%.3e)  %s" %
              (k, mA[k], b.norm().item(), d, maxabs, "PASS" if d < 5e-3 else "FAIL"), flush=True)
    print("\n  loss A=%.6f B=%.6f 差=%.2e" % (la, lb, abs(la - lb)), flush=True)

dist.barrier(); dist.destroy_process_group()
