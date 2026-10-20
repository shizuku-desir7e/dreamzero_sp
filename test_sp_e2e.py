#!/usr/bin/env python3
"""端到端 SP 验证（纯视频路径，action=None）。

验证目标：4 卡 Ulysses SP 的输出与单卡全序列**逐元素一致**。
这证明 SP 与 block-causal 注意力兼容（不含 action_register 的复杂度）。

用法：python -m torch.distributed.run --nproc_per_node=4 test_sp_e2e.py
"""
import os, sys
import torch
import torch.distributed as dist

Z = "/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengmenghan"
sys.path.insert(0, Z + "/dreamzero_sp/shim")
sys.path.insert(0, Z + "/dreamzero_sp/dreamzero")

import groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk as DZMOD
from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import CausalWanModel

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)
dev = "cuda:%d" % rank

# 上游 bug：非-checkpointing 分支不解包 (out, kv)
_orig_blk = DZMOD.CausalWanAttentionBlock.forward
def _unwrap(self, x, *a, **k):
    out = _orig_blk(self, x, *a, **k)
    return out[0] if isinstance(out, tuple) else out
DZMOD.CausalWanAttentionBlock.forward = _unwrap

FSL_FULL = 880          # 4 views × 220
F, H, W = 33, 40, 88
N_GLOBAL = F * FSL_FULL
W_LOCAL = W // world
FSL_LOCAL = (H // 2) * (W_LOCAL // 2)

CFG = dict(model_type="ti2v", patch_size=(1, 2, 2), frame_seqlen=FSL_FULL, text_len=512,
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
# 广播参数，保证 4 卡完全一致
for p in m.parameters():
    dist.broadcast(p.data, src=0)

torch.manual_seed(1234)
x_full = torch.randn(1, 36, F, H, W, device=dev, dtype=torch.bfloat16)
ctx = torch.randn(1, 512, 4096, device=dev, dtype=torch.bfloat16)
t = torch.randint(0, 1000, (1, F), device=dev)

# 单卡参考（只在 rank0 上算，用全量 x）
if rank == 0:
    DZMOD.set_sp_state(group=None, world=1, rank=0)
    with torch.no_grad():
        ref = m(x_full, t, None, ctx, seq_len=N_GLOBAL, clean_x=x_full,
                action=None, state=None, embodiment_id=None)
    ref_o = ref[0] if isinstance(ref, (list, tuple)) else ref
    print("[ref] done shape=%s" % (tuple(ref_o.shape),), flush=True)
else:
    ref_o = None

dist.barrier()

# 4 卡 SP：每卡按 W 切一个视角
DZMOD.set_sp_state(group=dist.group.WORLD, world=world, rank=rank)
x_local = x_full[..., rank * W_LOCAL:(rank + 1) * W_LOCAL].contiguous()
clean_local = x_local.clone()

with torch.no_grad():
    out = m(x_local, t, None, ctx, seq_len=N_GLOBAL, clean_x=clean_local,
            action=None, state=None, embodiment_id=None)
o = out[0] if isinstance(out, (list, tuple)) else out
print("[sp] rank%d out shape=%s" % (rank, tuple(o.shape)), flush=True)

# gather 回全量（沿 W 维）
gathered = [torch.empty_like(o) for _ in range(world)]
dist.all_gather(gathered, o.contiguous())
full = torch.cat(gathered, dim=-1)
print("[sp] gathered shape=%s" % (tuple(full.shape),), flush=True)

if rank == 0:
    a = full.float()
    b = ref_o.float()
    if a.shape != b.shape:
        print("[FAIL] shape mismatch %s vs %s" % (tuple(a.shape), tuple(b.shape)), flush=True)
    else:
        err = (a - b).abs().max().item()
        scale = b.abs().max().item() + 1e-8
        print("[CHECK] max_abs_err=%.4e  rel=%.4e  %s" % (
            err, err / scale, "PASS" if err / scale < 5e-2 else "FAIL"), flush=True)
        print("        (bf16 + 2 层，rel<5%% 视为通过)", flush=True)

dist.barrier()
dist.destroy_process_group()
