#!/usr/bin/env python3
"""SP v4：给所有集合通信加 autograd（v3 的反向是坏的）。

实测证据（真权重，F=9，SP=4 梯度 all_reduce 后）
------------------------------------------------
  blocks.0.self_attn.q.weight   |SP|=0.0000  |DP|=0.0318   <-- 梯度精确为 0

根因：`dist.all_to_all_single` 和 `dist.all_gather_into_tensor` **都不支持 autograd**，
产出的张量没有 grad_fn。于是 q/k/v 这条必须经过 a2a 的路径梯度直接断掉；
ffn 还有非零梯度，只是因为 action 输出没走 gather，靠 action loss 那一项漏回来。

修法：把三种集合通信包成 autograd.Function，反向用**对应的逆置换**（注意不是 sum/reduce）：
  a2a(scatter=sd, gather=gd)     反向 -> a2a(scatter=gd, gather=sd)
  all_gather(token 维)            反向 -> 取自己那片（各 rank token 互不重叠）
  all_gather(head 维，复制段)     反向 -> 取自己那片 head
梯度跨卡求和交给优化器前的一次 all_reduce(SUM)，这是 SP 的标准做法。
"""
import shutil, sys

F = sys.argv[1] if len(sys.argv) > 1 else \
    "/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengmenghan/dreamzero_sp/dreamzero/groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py"
src = open(F).read()

if "_SPA2AFn" in src:
    print("[skip] v4 已打过")
    sys.exit(0)


def rep(old, new, tag):
    global src
    assert src.count(old) == 1, "[%s] 命中 %d 次" % (tag, src.count(old))
    src = src.replace(old, new)
    print("[ok] %s" % tag)


# ---------- 1) a2a：原生逻辑改名，外面套 autograd ----------
rep('''def _sp_all_to_all_4d(x, scatter_dim, gather_dim):
    world = _SP_STATE["world"]
    if world == 1:
        return x''',
'''def _sp_a2a_raw(x, scatter_dim, gather_dim):
    """纯集合通信，无 autograd。只在 _SPA2AFn 里调用。"""
    world = _SP_STATE["world"]
    if world == 1:
        return x''', "a2a-rename")

rep('''    send = torch.stack(chunks, dim=0).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=_SP_STATE["group"])
    return torch.cat([recv[_i] for _i in range(world)], dim=gather_dim)''',
'''    send = torch.stack(chunks, dim=0).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=_SP_STATE["group"])
    return torch.cat([recv[_i] for _i in range(world)], dim=gather_dim)


class _SPA2AFn(torch.autograd.Function):
    """a2a 的反向是"反过来的 a2a"：scatter 与 gather 维度互换。"""

    @staticmethod
    def forward(ctx, x, sd, gd):
        ctx.sd, ctx.gd = sd, gd
        return _sp_a2a_raw(x, sd, gd)

    @staticmethod
    def backward(ctx, g):
        return _sp_a2a_raw(g.contiguous(), ctx.gd, ctx.sd), None, None


def _sp_all_to_all_4d(x, scatter_dim, gather_dim):
    if _SP_STATE["world"] == 1:
        return x
    return _SPA2AFn.apply(x, scatter_dim, gather_dim)


class _SPTokenGather(torch.autograd.Function):
    """token 维 all_gather。反向 = 取自己那一段（各 rank 的 token 互不重叠，不能求和）。"""

    @staticmethod
    def forward(ctx, t):
        ctx.T = t.shape[1]
        outs = [torch.empty_like(t) for _ in range(_SP_STATE["world"])]
        dist.all_gather(outs, t.contiguous(), group=_SP_STATE["group"])
        return torch.cat(outs, dim=1)

    @staticmethod
    def backward(ctx, g):
        r, T = _SP_STATE["rank"], ctx.T
        return g[:, r * T:(r + 1) * T].contiguous()''', "a2a-autograd")

# ---------- 2) 复制段的 head all_gather ----------
rep('''def _sp_head_allgather(t, world):
    """复制段出口：把 head 维 all_gather 拼回全部 head（token 维本来就是复制的）。"""
    nh = t.shape[2]
    a = t.permute(2, 0, 1, 3).contiguous()                 # [nh, B, S, D]
    out = torch.empty((nh * world,) + tuple(a.shape[1:]), device=a.device, dtype=a.dtype)
    dist.all_gather_into_tensor(out, a, group=_SP_STATE["group"])
    return out.permute(1, 2, 0, 3).contiguous()            # [B, S, nh*world, D]''',
'''class _SPHeadGatherFn(torch.autograd.Function):
    """复制段的 head all_gather。反向 = 取自己那片 head（head 互不重叠，不能求和）。"""

    @staticmethod
    def forward(ctx, t):
        world = _SP_STATE["world"]
        ctx.nh = t.shape[2]
        a = t.permute(2, 0, 1, 3).contiguous()             # [nh, B, S, D]
        out = torch.empty((ctx.nh * world,) + tuple(a.shape[1:]),
                          device=a.device, dtype=a.dtype)
        dist.all_gather_into_tensor(out, a, group=_SP_STATE["group"])
        return out.permute(1, 2, 0, 3).contiguous()        # [B, S, nh*world, D]

    @staticmethod
    def backward(ctx, g):
        world, rank, nh = _SP_STATE["world"], _SP_STATE["rank"], ctx.nh
        a = g.permute(2, 0, 1, 3).contiguous()             # [nh*world, B, S, D]
        return a[rank * nh:(rank + 1) * nh].permute(1, 2, 0, 3).contiguous()


def _sp_head_allgather(t, world):
    return _SPHeadGatherFn.apply(t)''', "headgather-autograd")

# ---------- 3) 尾部 video token gather ----------
rep('''                else:
                    _gs = [torch.empty_like(_seg) for _ in range(_spw)]
                    dist.all_gather(_gs, _seg.contiguous(), group=_SP_STATE["group"])
                    _vp.append(torch.cat(_gs, dim=1))''',
'''                else:
                    _vp.append(_SPTokenGather.apply(_seg))''', "videogather-autograd")

open(F, "w").write(src)
print("[done] v4 已写入 %s" % F)
