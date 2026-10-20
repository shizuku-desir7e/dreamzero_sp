#!/usr/bin/env python3
"""给 DreamZero 的 CausalWanModel 打 Ulysses SP 补丁（不改上游文件，运行时注入）。

设计要点
--------
1. **按视角切分**：latent W = 88 = 4 views × 22，正好 world=4 时每 rank 一个视角；
   token 数 29040 / 4 = 7260，整除，且与 block 结构（每 block 2 帧 × 880）对齐。
2. **attention 内做 Ulysses**：qkv 后 all-to-all（head 切 / token 拼），
   出口再 all-to-all 换回。block 循环、mask、rope 全部不变 —— 因为 attention 只依赖 s/n/d 三个局部量。
3. **`seq_len` 保持全量语义**：`_forward_train` 里 assert 改为 `x.shape[1] * world == seq_len`；
   `freqs` 用全量 grid_size 生成（否则 rope 位置错）。
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F

_SP = {"group": None, "world": 1, "rank": 0}


def set_sp(group=None, world=1, rank=0):
    _SP["group"], _SP["world"], _SP["rank"] = group, world, rank


def all_to_all_4d(x, scatter_dim, gather_dim):
    """[B,S,H,D] 在 scatter_dim 切 P 份、在 gather_dim 拼回 P 份。"""
    world = _SP["world"]
    if world == 1:
        return x
    x = x.contiguous()
    shape = list(x.shape)
    assert shape[scatter_dim] % world == 0, (shape, scatter_dim, world)
    split = shape[scatter_dim] // world
    idx = [slice(None)] * x.dim()
    chunks = []
    for i in range(world):
        idx[scatter_dim] = slice(i * split, (i + 1) * split)
        chunks.append(x[tuple(idx)])
    send = torch.stack(chunks, dim=0).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=_SP["group"])
    return torch.cat([recv[i] for i in range(world)], dim=gather_dim)


def patch_model(DZMOD):
    """注入 SP 到 CausalWanSelfAttention。必须在构造模型之前调用。"""
    Attn = DZMOD.CausalWanSelfAttention
    _orig_fwd = Attn.forward

    def sp_forward(self, x, *args, **kwargs):
        world = _SP["world"]
        if world == 1:
            return _orig_fwd(self, x, *args, **kwargs)

        # ---- 1) 本地 qkv（x 已是本地 token 段）----
        b, s_local, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s_local, n, d)
        k = self.norm_k(self.k(x)).view(b, s_local, n, d)
        v = self.v(x).view(b, s_local, n, d)

        # ---- 2) all-to-all：head 切 / token 拼 ----
        q = all_to_all_4d(q, scatter_dim=2, gather_dim=1)   # [B, S_full, H/P, D]
        k = all_to_all_4d(k, scatter_dim=2, gather_dim=1)
        v = all_to_all_4d(v, scatter_dim=2, gather_dim=1)
        s_full, n_local = q.shape[1], q.shape[2]

        # ---- 3) 复用上游逻辑：把 qkv_fn 换成"直接返回已切好的 q/k/v" ----
        class _QKV:
            def __init__(self, q, k, v):
                self.q, self.k, self.v = q, k, v
            def __call__(self, _x):
                return self.q, self.k, self.v

        # 关键：调用原 forward 时传一个 fake x，并临时替换 qkv_fn 结果
        # 更干净的做法是复制上游 body；这里用 hook 变量注入。
        prev = getattr(Attn, "_SP_INJECT", None)
        Attn._SP_INJECT = (s_full, n_local, d, q, k, v)
        try:
            out, kv = _orig_fwd(self, x, *args, **kwargs)
        finally:
            if prev is None:
                delattr(Attn, "_SP_INJECT")
            else:
                Attn._SP_INJECT = prev

        # ---- 4) 出口 all-to-all（token 切回 / head 拼回）----
        if isinstance(out, tuple):
            out = out[0]
        out = all_to_all_4d(out, scatter_dim=1, gather_dim=2)
        return out, kv

    Attn.forward = sp_forward
    return Attn
