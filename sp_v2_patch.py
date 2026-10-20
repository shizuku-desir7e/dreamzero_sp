#!/usr/bin/env python3
"""SP v2 补丁：把「按 W 切输入」改成「全局构造、按 token 连续区间切」。

为什么必须改
------------
v1（按 W 切）有两个致命问题：
1. **rope 顺序错**：局部 token 顺序是 (f, h, w_local)，all-to-all 按 rank 拼接后得到
   (rank, f, h, w_local)，而全局正确顺序是 (f, h, w) —— 两者是 stride-W_local 的编织关系，
   不是简单分块。于是 rope 索引和 block-mask 的帧边界全部错位。
2. **显存没降下来**：e0（[B, 2N+400, 6, dim]）是 per-token 的全局张量，每个 block 的
   modulation 都会在它上面做加法生成新张量并留到 backward。它没被切 → 每层 ~565MB(F=5)，
   40 层就是 22GB。实测 SP4@F=5 peak=76.6GB，DP 反而更低，就是这块漏的。

v2 做法
-------
- 输入 x 保持**全量**（各 rank 相同），patch_embedding 之后按 row-major(f,h,w) 取连续区间
  [r*q, (r+1)*q)。这保证 a2a 拼接后的顺序 == 全局顺序，rope/mask 天然正确。
- e0 / e / clean_x 同步按区间切，显存真正降 1/P。
- action_register（400 个 token，各 rank 复制一份）不进 a2a：
  入口按 head 本地切片，出口 all_gather 拼回头。
- 末尾 unpatchify 需要全局矩形网格，先把视频 token all_gather 拼回全量。
"""
import re, shutil, sys

F = sys.argv[1] if len(sys.argv) > 1 else \
    "/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengmenghan/dreamzero_sp/dreamzero/groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py"
src = open(F).read()
shutil.copy(F, F + ".bak_v1")

if "_sp_gather_qkv" in src:
    print("[skip] 已经打过 v2 补丁")
    sys.exit(0)

def rep(old, new, tag):
    global src
    assert src.count(old) == 1, "[%s] 命中 %d 次" % (tag, src.count(old))
    src = src.replace(old, new)
    print("[ok] %s" % tag)

# ---------- 1) 新增 helper ----------
rep('''def _sp_all_to_all_4d(x, scatter_dim, gather_dim):''',
'''def _sp_gather_qkv(q, k, v, ar):
    """q/k/v 全量 token 段 -> a2a 拼成全量 token；action_register 段复制、按 head 本地切片。"""
    world, rank = _SP_STATE["world"], _SP_STATE["rank"]
    if ar == 0:
        return (_sp_all_to_all_4d(q, 2, 1), _sp_all_to_all_4d(k, 2, 1),
                _sp_all_to_all_4d(v, 2, 1))
    n = q.shape[2]
    nh = n // world
    outs = []
    for t in (q, k, v):
        img, act = t[:, :-ar], t[:, -ar:]
        img = _sp_all_to_all_4d(img, scatter_dim=2, gather_dim=1)
        act = act[:, :, rank * nh:(rank + 1) * nh, :].contiguous()
        outs.append(torch.cat([img, act], dim=1))
    return tuple(outs)


def _sp_scatter_out(x, ar):
    """全量 token + 本地 head -> 本地 token 段 + 全部 head（还原 o-proj 的输入）。"""
    world = _SP_STATE["world"]
    if ar == 0:
        return _sp_all_to_all_4d(x, scatter_dim=1, gather_dim=2)
    img, act = x[:, :-ar], x[:, -ar:]
    img = _sp_all_to_all_4d(img, scatter_dim=1, gather_dim=2)
    # act 段：token 维本来就是复制的，只需把 head 维 all_gather 拼回去
    nh = act.shape[2]
    a = act.permute(2, 0, 1, 3).contiguous()          # [nh, B, ar, d]
    out = torch.empty((nh * world,) + tuple(a.shape[1:]),
                      device=a.device, dtype=a.dtype)
    dist.all_gather_into_tensor(out, a, group=_SP_STATE["group"])
    act = out.permute(1, 2, 0, 3).contiguous()        # [B, ar, nh*world, d]
    return torch.cat([img, act], dim=1)


def _sp_all_to_all_4d(x, scatter_dim, gather_dim):''', "helpers")

# ---------- 2) attention 入口 ----------
rep('''        if _SP_STATE["world"] > 1:
            q = _sp_all_to_all_4d(q, scatter_dim=2, gather_dim=1)
            k = _sp_all_to_all_4d(k, scatter_dim=2, gather_dim=1)
            v = _sp_all_to_all_4d(v, scatter_dim=2, gather_dim=1)
            b, s, n, d = q.shape[0], q.shape[1], q.shape[2], q.shape[3]''',
'''        _sp_ar = 0
        if _SP_STATE["world"] > 1:
            _sp_ar = action_register_length if action_register_length is not None else 0
            q, k, v = _sp_gather_qkv(q, k, v, _sp_ar)
            b, s, n, d = q.shape[0], q.shape[1], q.shape[2], q.shape[3]''',
    "attn-in")

# ---------- 3) attention 出口 ----------
rep('''        # output
        # SP: 先换回本地 token 段（x 此时是 [B, S_global, H_local, D]），再 flatten + o-proj
        if _SP_STATE["world"] > 1:
            x = _sp_all_to_all_4d(x, scatter_dim=1, gather_dim=2)''',
'''        # output
        # SP: 先换回本地 token 段（x 此时是 [B, S_global, H_local, D]），再 flatten + o-proj
        if _SP_STATE["world"] > 1:
            x = _sp_scatter_out(x, _sp_ar)''', "attn-out")

# ---------- 4) _forward_train：patch_embed 之后按区间切 ----------
rep('''        # grid_size 必须保持**本地**尺寸（unpatchify 用它还原本地视频），
        # 只有 rope 的 freqs 需要全局空间尺寸（否则每个 rank 的 rope 都从 0 开始）。
        grid_size = torch.tensor(x.shape[2:], dtype=torch.long)
        _gz_rope = list(x.shape[2:])
        if _SP_STATE["world"] > 1:
            _gz_rope[-1] = _gz_rope[-1] * _SP_STATE["world"]
        freqs = self._create_freqs(
            grid_size=torch.tensor(_gz_rope, dtype=torch.long),
            start_frame=0,
        )

        x = x.flatten(start_dim=2).transpose(1, 2)
        assert x.shape[1] * _SP_STATE["world"] == seq_len''',
'''        # SP v2：全局构造，按 row-major(f,h,w) 的**连续区间**切 token。
        # 顺序天然等于全局顺序 => rope / block-mask 不用改；e0 等 per-token 张量同步切。
        grid_size = torch.tensor(x.shape[2:], dtype=torch.long)
        freqs = self._create_freqs(grid_size=grid_size, start_frame=0)

        x = x.flatten(start_dim=2).transpose(1, 2)
        _spw, _spr = _SP_STATE["world"], _SP_STATE["rank"]
        assert x.shape[1] == seq_len, (x.shape, seq_len)
        _q = seq_len // _spw
        _lo, _hi = _spr * _q, (_spr + 1) * _q
        if _spw > 1:
            x = x[:, _lo:_hi].contiguous()''', "ft-split-x")

# ---------- 5) clean_x ----------
rep('''            clean_x = clean_x.flatten(start_dim=2).transpose(1, 2)
            assert clean_x.shape[1] * _SP_STATE["world"] == seq_len''',
'''            clean_x = clean_x.flatten(start_dim=2).transpose(1, 2)
            assert clean_x.shape[1] == seq_len
            if _spw > 1:
                clean_x = clean_x[:, _lo:_hi].contiguous()''', "ft-clean")

# ---------- 6) e0 / e 切到本地（放 kwargs 之前，两条分支都要）----------
rep('''        # arguments
        kwargs = dict(
            e=e0,''',
'''        # SP v2：e0 是 per-token 的 [B, S, 6, dim]，每个 block 的 modulation 都会在它上面
        # 生成新张量并留到 backward —— 不切的话 40 层能吃掉 20GB+，这是 v1 显存没降的根因。
        if _spw > 1:
            if clean_x is not None:
                e0 = torch.cat([e0[:, _lo:_hi],
                                e0[:, seq_len + _lo: seq_len + _hi],
                                e0[:, 2 * seq_len:]], dim=1)
            else:
                e0 = torch.cat([e0[:, _lo:_hi], e0[:, seq_len:]], dim=1)
            e = torch.cat([e[:, _lo:_hi], e[:, seq_len:]], dim=1)

        # arguments
        kwargs = dict(
            e=e0,''', "ft-e0")

# ---------- 7) 尾部：本地切片 + 视频 token 拼回全局再 unpatchify ----------
rep('''            x = x[:, clean_x.shape[1]:]

        if action is not None:
            action_noise_pred = x[:, seq_len: seq_len + action_length]''',
'''            x = x[:, clean_x.shape[1]:]

        _nl = seq_len // _SP_STATE["world"] if _SP_STATE["world"] > 1 else seq_len
        if action is not None:
            action_noise_pred = x[:, _nl: _nl + action_length]''', "ft-actpred")

def rep_after(anchor, old, new, tag):
    """只在 anchor 之后替换（train / inference 两处同名代码，靠 anchor 区分）。"""
    global src
    i = src.index(anchor)
    head, tail = src[:i], src[i:]
    assert tail.count(old) == 1, "[%s] anchor 后命中 %d 次" % (tag, tail.count(old))
    src = head + tail.replace(old, new)
    print("[ok] %s" % tag)

rep_after('_nl = seq_len // _SP_STATE["world"]', '''        # Build a tensor that contains only video tokens per sample with length = max(video_lens)
        x_video = x[:, :seq_len]
        e_video = e[:, :seq_len]

        # Unpatchify video-only tokens
        x_video = self.head(x_video, e_video.unsqueeze(2))''',
'''        # Build a tensor that contains only video tokens per sample with length = max(video_lens)
        x_video = x[:, :_nl]
        e_video = e[:, :_nl]

        # Unpatchify video-only tokens
        x_video = self.head(x_video, e_video.unsqueeze(2))

        # SP：unpatchify 需要全局矩形网格，先把各 rank 的连续 token 段拼回全量
        if _SP_STATE["world"] > 1:
            _w = _SP_STATE["world"]
            _parts = [torch.empty_like(x_video) for _ in range(_w)]
            dist.all_gather(_parts, x_video.contiguous(), group=_SP_STATE["group"])
            x_video = torch.cat(_parts, dim=1)''', "ft-unpatch")

open(F, "w").write(src)
print("[done] 补丁已写入 %s（备份 %s.bak_v1）" % (F, F))
