#!/usr/bin/env python3
"""v6：把 v5 的多段 a2a 合并成**一次扁平 all_to_all_single + 一次重排**。

v5 的问题
---------
零复制之后序列被拆成 5 段（clean f0 / clean body / noisy f0 / noisy body / act），
每段各自调一次 a2a。5 段 × (q,k,v 入口 + 出口) 让每层 a2a 从 16 次涨到 20 次，
每次都要 stack/empty/cat 三份 buffer ⇒ 峰值多 5.5 GB（实测 51.4 → 56.9 GB，F=33 前向），
再加上 33 GB 梯度就超了 85.9 GB。

v6 的做法
---------
段与段之间只是**顺序问题**，不是数据问题。所以：

  入口（本地 → 全局）：一次 a2a（scatter head / gather token），拿到
      recv[peer] = 该 peer 的全部本地 token（本 rank 的 head 组）；
      然后按段重排：对每段 L，把 recv[:, :, :, off:off+L] 从 [P,B,nh,L,d]
      排成 [B, P*L, nh, d] —— 段内按 peer 序拼，正好等于全局序。
  出口（全局 → 本地）：按段把全局 token 按 P 切成片，第 i 片拼成 send[i]，
      一次 a2a（scatter token / gather head），recv[j] 就是 peer j 在每段的第 r 片。

每次只剩 1 份 send + 1 份 recv（原来每段 3 份），collective 从 20 次降到 8 次/层。
autograd 仍然手写：'in' 的反向是 'out'，反之亦然（纯重排，不是归约）。
"""
import sys

F = sys.argv[1] if len(sys.argv) > 1 else \
    "/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengmenghan/dreamzero_sp/dreamzero/groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py"
src = open(F).read()

if "_SPFlatA2AFn" in src:
    print("[skip] v6 已打过")
    sys.exit(0)


def rep(old, new, tag):
    global src
    assert src.count(old) == 1, "[%s] 命中 %d 次" % (tag, src.count(old))
    src = src.replace(old, new)
    print("[ok] %s" % tag)


# ---------- 1) 扁平 a2a + autograd ----------
rep('''def _sp_gather_qkv(q, k, v, segs):''',
'''def _sp_a2a_seg(x, segs, mode):
    """一次 a2a 完成全部分段的重排。x: [B, ΣL, n, d]。

    mode='in' : 本地 -> 全局。segs 是**本地**段长；输出 [B, ΣL*P, n/P, d]
    mode='out': 全局 -> 本地。segs 是**全局**段长；输出 [B, ΣL/P, n, d]
    """
    world, rank = _SP_STATE["world"], _SP_STATE["rank"]
    if world == 1:
        return x
    B, T, n, d = x.shape
    Ls = [L for _, L in segs]
    assert T == sum(Ls), ("段长与张量不匹配", T, Ls)

    if mode == "in":
        assert n % world == 0, (n, world)
        nh = n // world
        # send[i] = 本 rank 的第 i 组 head、全部本地 token
        send = x.view(B, T, world, nh, d).permute(2, 0, 3, 1, 4).contiguous()  # [P,B,nh,T,d]
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=_SP_STATE["group"])
        # recv[p] = peer p 的本地 token（本 rank 的 head 组）
        parts, off = [], 0
        for L in Ls:
            blk = recv[:, :, :, off:off + L, :]                     # [P,B,nh,L,d]
            parts.append(blk.permute(1, 0, 3, 2, 4).reshape(B, world * L, nh, d))
            off += L
        return torch.cat(parts, dim=1)

    # mode == "out"
    parts, off = [], 0
    for Lg in Ls:
        assert Lg % world == 0, (Lg, world)
        c = x[:, off:off + Lg]                                      # [B,Lg,n,d]
        parts.append(c.view(B, world, Lg // world, n, d))           # [B,P,Lloc,n,d]
        off += Lg
    s = torch.cat(parts, dim=2).permute(1, 0, 2, 3, 4).contiguous() # [P,B,Tloc,n,d]
    recv = torch.empty_like(s)
    dist.all_to_all_single(recv, s, group=_SP_STATE["group"])
    # head 维是 dim=2！写成 dim=3 会拼到 d 维上；因为下游 flatten(2) 之后总维度
    # 碰巧还是 dim，前向不报错，只在反向才炸（踩过，见 blog §6③）
    return torch.cat([recv[i] for i in range(world)], dim=2)        # 拼 head -> [B,Tloc,n*P,d]


class _SPFlatA2AFn(torch.autograd.Function):
    """in 的反向是 out，out 的反向是 in —— 都是纯重排，不做归约。"""

    @staticmethod
    def forward(ctx, x, mode, segs_local, segs_global):
        ctx.mode, ctx.sl, ctx.sg = mode, segs_local, segs_global
        return _sp_a2a_seg(x, segs_local if mode == "in" else segs_global, mode)

    @staticmethod
    def backward(ctx, g):
        if ctx.mode == "in":
            return _sp_a2a_seg(g.contiguous(), ctx.sg, "out"), None, None, None
        return _sp_a2a_seg(g.contiguous(), ctx.sl, "in"), None, None, None


def _sp_gather_qkv(q, k, v, segs):''', "flat-a2a")

# ---------- 2) 入口改成一次扁平 a2a ----------
rep('''    world, rank = _SP_STATE["world"], _SP_STATE["rank"]
    res = []
    for t in (q, k, v):
        parts = []
        for kind, seg in _sp_cut(t, segs):
            parts.append(_sp_heads(seg, world, rank) if kind == "rep"
                         else _sp_all_to_all_4d(seg, scatter_dim=2, gather_dim=1))
        res.append(torch.cat(parts, dim=1))
    return tuple(res)''',
'''    world = _SP_STATE["world"]
    if world == 1:
        return q, k, v
    sg = [(k_, L * world) for k_, L in segs]
    return tuple(_SPFlatA2AFn.apply(t, "in", segs, sg) for t in (q, k, v))''', "gather")

# ---------- 3) 出口改成一次扁平 a2a ----------
rep('''def _sp_scatter_out(x, segs):
    world = _SP_STATE["world"]
    parts = []
    for kind, seg in _sp_cut(x, segs):
        parts.append(_sp_head_allgather(seg, world) if kind == "rep"
                     else _sp_all_to_all_4d(seg, scatter_dim=1, gather_dim=2))
    return torch.cat(parts, dim=1)''',
'''def _sp_scatter_out(x, segs):
    world = _SP_STATE["world"]
    if world == 1:
        return x
    sl = [(k_, L // world) for k_, L in segs]
    return _SPFlatA2AFn.apply(x, "out", sl, segs)''', "scatter")

open(F, "w").write(src)
print("[done] v6 已写入 %s" % F)
