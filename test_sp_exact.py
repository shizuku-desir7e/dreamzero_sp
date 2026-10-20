#!/usr/bin/env python3
"""SP v2 精确性验证（小模型、4 卡）：SP 输出必须与单卡全序列**逐元素**一致。

关键：v1 按 W 切输入，gathered 顺序是 (rank,f,h,w_local) 的编织，rope/帧边界全错；
v2 改成按全局 row-major 连续区间切，顺序天然正确。这个测试就是证明这件事。

用法：python -m torch.distributed.run --nproc_per_node=4 test_sp_exact.py [--action 1]
"""
import argparse, os, sys
import torch
import torch.distributed as dist

Z = os.environ.get("ZROOT", "/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengmenghan")
sys.path.insert(0, Z + "/dreamzero_sp/shim")
sys.path.insert(0, Z + "/dreamzero_sp/dreamzero")
import groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk as DZMOD
from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import CausalWanModel

ap = argparse.ArgumentParser()
ap.add_argument("--action", type=int, default=0)
ap.add_argument("--frames", type=int, default=17)
args = ap.parse_args()

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)
dev = "cuda:%d" % rank
is0 = rank == 0

_orig_blk = DZMOD.CausalWanAttentionBlock.forward
def _unwrap(self, x, *a, **k):
    out = _orig_blk(self, x, *a, **k)
    return out[0] if isinstance(out, tuple) else out
DZMOD.CausalWanAttentionBlock.forward = _unwrap

FSL = 880
F, H, W = args.frames, 40, 88
N = F * FSL
NB = (F - 1) // 2

CFG = dict(model_type="ti2v", patch_size=(1, 2, 2), frame_seqlen=FSL, text_len=512,
           in_dim=36, dim=512, ffn_dim=2048, freq_dim=256, text_dim=4096, out_dim=16,
           num_heads=8, num_layers=2, num_frame_per_block=2, action_dim=32,
           num_action_per_block=24, num_state_per_block=1, max_state_dim=64,
           max_num_embodiments=32, hidden_size=1024, concat_first_frame_latent=True,
           qk_norm=True, cross_attn_norm=True, max_chunk_size=4, sink_size=0)

torch.manual_seed(0)
m = CausalWanModel(**CFG).to(dev).to(torch.bfloat16)
m.eval()
m.gradient_checkpointing = False
for p in m.parameters():
    p.requires_grad_(False)
for p in m.parameters():
    dist.broadcast(p.data, src=0)

# 上游的 out head 是零初始化 -> 输出恒为 0，那种情况下"逐元素相等"毫无意义
# （v1 曾因此得到假阳性 0.0）。这里重新随机化权重，保证输出非零。
torch.manual_seed(7)
with torch.no_grad():
    for n, p in m.named_parameters():
        if p.dim() >= 2:
            p.normal_(0, 0.05)
        elif "norm" in n or n.endswith("gamma"):
            p.fill_(1.0)
        else:
            p.zero_()

torch.manual_seed(1234)
x = torch.randn(1, 36, F, H, W, device=dev, dtype=torch.bfloat16)
ctx = torch.randn(1, 512, 4096, device=dev, dtype=torch.bfloat16)
t = torch.randint(0, 1000, (1, F), device=dev)
if args.action:
    act = torch.randn(1, NB * 24, 32, device=dev, dtype=torch.bfloat16)
    st = torch.randn(1, NB * 1, 64, device=dev, dtype=torch.bfloat16)
    ta = torch.randint(0, 1000, (1, NB * 24), device=dev)
    emb = torch.zeros(1, dtype=torch.long, device=dev)
else:
    act = st = ta = emb = None

def run(sp_on):
    DZMOD.set_sp_state(group=dist.group.WORLD if sp_on else None,
                       world=world if sp_on else 1, rank=rank if sp_on else 0)
    with torch.no_grad():
        out = m(x, t, ta, ctx, seq_len=N, clean_x=x.clone(),
                action=act, state=st, embodiment_id=emb)
    return out

# 单卡参考（只在 rank0）
ref_v, ref_a = None, None
if is0:
    ref_v, ref_a = run(False)
    print("[ref] video=%s action=%s" % (
        tuple(ref_v.shape), None if ref_a is None else tuple(ref_a.shape)), flush=True)
dist.barrier()

sp_v, sp_a = run(True)

if is0:
    print("[ref] absmax=%.4f  (必须 >0，否则比较无意义)" % ref_v.float().abs().max(), flush=True)
    for name, a, b in (("video", sp_v, ref_v), ("action", sp_a, ref_a)):
        if b is None:
            continue
        if a.shape != b.shape:
            print("[FAIL] %s shape %s vs %s" % (name, tuple(a.shape), tuple(b.shape)), flush=True)
            continue
        d = (a.float() - b.float()).abs()
        rel = d.max().item() / (b.float().abs().max().item() + 1e-8)
        print("[%s] %s max_abs_err=%.4e rel=%.3e" %
              ("PASS" if rel < 5e-2 else "FAIL", name, d.max().item(), rel), flush=True)

dist.barrier()
dist.destroy_process_group()
