#!/usr/bin/env python3
"""SP v3：分段式 sharding（首帧复制 + 其余帧按帧均匀切 + action_register 复制）。

为什么是"分段"
--------------
v2 把序列当成一条连续的 token 流来切 q=N/P，但 attention 的 block-mask 以
`frame_seqlen=880` 为界。N/P 不是 880 的整数倍时（F=33, P=4 -> 8.25 帧）就切在帧中间，
mask 边界全错 -> 输出 NaN。这是 v2 实测到的失败。

v3 保证每个 a2a chunk 都是**整帧对齐**的：
  本地序列 = [首帧(复制) | 我的 8 帧 | 首帧(复制) | 我的 8 帧 | action_register(复制)]
              ^ clean 部分的 img        ^ noisy 部分的 img
首帧复制数 nrep = (F % P) * frame_seqlen，此时 (F - nrep) % P == 0，剩下的帧能整除。
F=33,P=4: nrep=1 帧, 32/4=8 帧/卡 ✓   F=17: 1+16/4=4 ✓   F=9: 1+2 ✓

复制段的处理方式和 action_register 完全一样：入口按 head 本地切片，出口 all_gather 拼回。
复制开销 = (nrep+ar)/S ~ 2%。
"""
import shutil, sys

F = sys.argv[1] if len(sys.argv) > 1 else \
    "/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengmenghan/dreamzero_sp/dreamzero/groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py"
BAK = F + ".bak_v1"

if len(sys.argv) > 2 and sys.argv[2] == "--restore":
    shutil.copy(BAK, F)
    print("[restore] 已从 %s 还原" % BAK)
    sys.exit(0)

src = open(BAK).read()          # 总是从干净基线开始
print("[base] 使用干净基线 %s" % BAK)


def rep(old, new, tag):
    global src
    assert src.count(old) == 1, "[%s] 命中 %d 次" % (tag, src.count(old))
    src = src.replace(old, new)
    print("[ok] %s" % tag)


def rep_after(anchor, old, new, tag):
    global src
    i = src.index(anchor)
    head, tail = src[:i], src[i:]
    assert tail.count(old) == 1, "[%s] anchor 后命中 %d 次" % (tag, tail.count(old))
    src = head + tail.replace(old, new)
    print("[ok] %s" % tag)


# ============ 1) helpers ============
rep('''def _sp_all_to_all_4d(x, scatter_dim, gather_dim):''',
'''def _sp_segs(F, world, fsl, ar):
    """返回 (imgs, full, vid) 三种段结构。

    段 = ('rep', n) 复制段（各 rank 内容相同） | ('chunk', n) 均匀切分段。
    imgs: 单份 img  -> [('rep',nrep), ('chunk',iq)]
    full: attention 输入 -> imgs + imgs + [('rep',ar)]
    vid : 尾部按 [_nl:] 取视频 -> imgs + [('rep',ar)]
    """
    total = F * fsl
    nrep = (F % world) * fsl
    iq = (total - nrep) // world
    assert iq * world + nrep == total, (F, world, fsl)
    assert (total - nrep) % world == 0
    imgs = ([("rep", nrep)] if nrep else []) + [("chunk", iq)]
    imgs_g = ([("rep", nrep)] if nrep else []) + [("chunk", iq * world)]
    tail = [("rep", ar)] if ar else []
    return imgs, imgs + imgs + tail, imgs_g + imgs_g + tail, imgs + tail, nrep, iq


def _sp_cut(t, segs):
    out, i = [], 0
    for kind, ln in segs:
        out.append((kind, t[:, i:i + ln]))
        i += ln
    assert i == t.shape[1], (i, t.shape)
    return out


def _sp_heads(t, world, rank):
    """复制段：只保留本 rank 负责的 head 切片（各 rank 内容相同，无需通信）。"""
    nh = t.shape[2] // world
    return t[:, :, rank * nh:(rank + 1) * nh, :].contiguous()


def _sp_head_allgather(t, world):
    """复制段出口：把 head 维 all_gather 拼回全部 head（token 维本来就是复制的）。"""
    nh = t.shape[2]
    a = t.permute(2, 0, 1, 3).contiguous()                 # [nh, B, S, D]
    out = torch.empty((nh * world,) + tuple(a.shape[1:]), device=a.device, dtype=a.dtype)
    dist.all_gather_into_tensor(out, a, group=_SP_STATE["group"])
    return out.permute(1, 2, 0, 3).contiguous()            # [B, S, nh*world, D]


def _sp_gather_qkv(q, k, v, segs):
    world, rank = _SP_STATE["world"], _SP_STATE["rank"]
    res = []
    for t in (q, k, v):
        parts = []
        for kind, seg in _sp_cut(t, segs):
            parts.append(_sp_heads(seg, world, rank) if kind == "rep"
                         else _sp_all_to_all_4d(seg, scatter_dim=2, gather_dim=1))
        res.append(torch.cat(parts, dim=1))
    return tuple(res)


def _sp_scatter_out(x, segs):
    world = _SP_STATE["world"]
    parts = []
    for kind, seg in _sp_cut(x, segs):
        parts.append(_sp_head_allgather(seg, world) if kind == "rep"
                     else _sp_all_to_all_4d(seg, scatter_dim=1, gather_dim=2))
    return torch.cat(parts, dim=1)


def _sp_all_to_all_4d(x, scatter_dim, gather_dim):''', "helpers")

# 让复制段直接复用 world=1 的快路径
rep('''def _sp_all_to_all_4d(x, scatter_dim, gather_dim):
    world = _SP_STATE["world"]''',
'''def _sp_all_to_all_4d(x, scatter_dim, gather_dim):
    world = _SP_STATE["world"]''', "a2a-keep")

# ============ 2) attention 入口 ============
rep('''        q, k, v = qkv_fn(x)

        if _SP_STATE["world"] > 1:
            q = _sp_all_to_all_4d(q, scatter_dim=2, gather_dim=1)
            k = _sp_all_to_all_4d(k, scatter_dim=2, gather_dim=1)
            v = _sp_all_to_all_4d(v, scatter_dim=2, gather_dim=1)
            b, s, n, d = q.shape[0], q.shape[1], q.shape[2], q.shape[3]''',
'''        q, k, v = qkv_fn(x)

        if _SP_STATE["world"] > 1:
            q, k, v = _sp_gather_qkv(q, k, v, _SP_STATE["segs"])
            b, s, n, d = q.shape[0], q.shape[1], q.shape[2], q.shape[3]''',
    "attn-in")

# ============ 3) attention 出口 ============
rep('''        # output
        # SP: 先换回本地 token 段（x 此时是 [B, S_global, H_local, D]），再 flatten + o-proj
        if _SP_STATE["world"] > 1:
            x = _sp_all_to_all_4d(x, scatter_dim=1, gather_dim=2)''',
'''        # output
        # SP: 先换回本地 token 段（x 此时是 [B, S_global, H_local, D]），再 flatten + o-proj
        if _SP_STATE["world"] > 1:
            x = _sp_scatter_out(x, _SP_STATE["segs_g"])''', "attn-out")

# ============ 4) _forward_train: patch_embed 之后算段结构并切 ============
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
'''        # SP v3：输入保持全量，patch_embed 之后按"首帧复制 + 其余帧整帧均分"切 token。
        grid_size = torch.tensor(x.shape[2:], dtype=torch.long)
        freqs = self._create_freqs(grid_size=grid_size, start_frame=0)

        x = x.flatten(start_dim=2).transpose(1, 2)
        _spw, _spr = _SP_STATE["world"], _SP_STATE["rank"]
        assert x.shape[1] == seq_len, (x.shape, seq_len)
        _nfr, _fsl = int(grid_size[0]), self.frame_seqlen
        _nrep = (_nfr % _spw) * _fsl
        _iq = (seq_len - _nrep) // _spw
        if _spw > 1:
            assert _iq * _spw + _nrep == seq_len
            x = torch.cat([x[:, :_nrep], x[:, _nrep + _spr * _iq: _nrep + (_spr + 1) * _iq]],
                          dim=1).contiguous()
        _sp_imgs, _, _, _sp_vid, _, _ = _sp_segs(_nfr, _spw, _fsl, 0)
        _SP_STATE["segs_vid"] = _sp_vid''', "ft-split-x")

# ============ 5) clean_x ============
rep('''            clean_x = clean_x.flatten(start_dim=2).transpose(1, 2)
            assert clean_x.shape[1] * _SP_STATE["world"] == seq_len''',
'''            clean_x = clean_x.flatten(start_dim=2).transpose(1, 2)
            assert clean_x.shape[1] == seq_len
            if _spw > 1:
                clean_x = torch.cat([clean_x[:, :_nrep],
                                     clean_x[:, _nrep + _spr * _iq:
                                                _nrep + (_spr + 1) * _iq]], dim=1).contiguous()''',
    "ft-clean")

# ============ 6) e0 / e 同步切（v1 显存没降的根因就是 e0 没切） ============
rep('''        # arguments
        kwargs = dict(
            e=e0,''',
'''        # SP v3：e0 是 per-token 的 [B, S, 6, dim]，每个 block 的 modulation 都在上面生成
        # 新张量并留到 backward —— 不切的话 40 层要吃 20GB+，这正是 v1 显存没降的根因。
        _sp_ar = action_register_length if action_register_length is not None else 0
        _a, _sp_segs_set, _sp_segs_g, _b, _, _ = _sp_segs(_nfr, _spw, _fsl, _sp_ar)
        _SP_STATE["segs"] = _sp_segs_set      # 入口：本地长度
        _SP_STATE["segs_g"] = _sp_segs_g      # 出口：全局长度
        _nl = _nrep + _iq
        if _spw > 1:
            def _sp_take(t, base):
                return torch.cat([t[:, base:base + _nrep],
                                  t[:, base + _nrep + _spr * _iq:
                                       base + _nrep + (_spr + 1) * _iq]], dim=1)
            if clean_x is not None:
                e0 = torch.cat([_sp_take(e0, 0), _sp_take(e0, seq_len),
                                e0[:, 2 * seq_len:]], dim=1)
            else:
                e0 = torch.cat([_sp_take(e0, 0), e0[:, seq_len:]], dim=1)
            e = torch.cat([_sp_take(e, 0), e[:, seq_len:]], dim=1)

        # arguments
        kwargs = dict(
            e=e0,''', "ft-e0")

# ============ 7) 尾部：用本地长度 _nl，并把视频 token 拼回全局顺序再 unpatchify ============
rep('''            x = x[:, clean_x.shape[1]:]

        if action is not None:
            action_noise_pred = x[:, seq_len: seq_len + action_length]''',
'''            x = x[:, clean_x.shape[1]:]

        if action is not None:
            action_noise_pred = x[:, _nl: _nl + action_length]''', "ft-actpred")

rep_after('if action is not None:\n            action_noise_pred = x[:, _nl: _nl + action_length]',
'''        # Build a tensor that contains only video tokens per sample with length = max(video_lens)
        x_video = x[:, :seq_len]
        e_video = e[:, :seq_len]

        # Unpatchify video-only tokens
        x_video = self.head(x_video, e_video.unsqueeze(2))''',
'''        # Build a tensor that contains only video tokens per sample with length = max(video_lens)
        x_video = x[:, :_nl]
        e_video = e[:, :_nl]

        # Unpatchify video-only tokens
        x_video = self.head(x_video, e_video.unsqueeze(2))

        # SP v3：本地是 [首帧 | 我的帧]，不是全局连续区间，不能直接 all_gather。
        # 逐段拼：复制段各 rank 相同（取本地即可），chunk 段按 rank 顺序拼。
        if _spw > 1:
            _vp = []
            for _kind, _seg in _sp_cut(x_video, _SP_STATE["segs_vid"]):
                if _kind == "rep":
                    _vp.append(_seg)
                else:
                    _gs = [torch.empty_like(_seg) for _ in range(_spw)]
                    dist.all_gather(_gs, _seg.contiguous(), group=_SP_STATE["group"])
                    _vp.append(torch.cat(_gs, dim=1))
            x_video = torch.cat(_vp, dim=1)[:, :seq_len]''', "ft-unpatch")

open(F, "w").write(src)
print("[done] v3 已写入 %s" % F)
