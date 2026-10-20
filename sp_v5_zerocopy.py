#!/usr/bin/env python3
"""v5：把 v3 的"复制段"改成"零复制、按段均分"。

为什么
------
v3 把首帧(880 token)和 action_register(400 token)在每张卡上**复制**，
attention 的 q/k/v/o 因为按 head 切片所以求和正确，但 **MLP / o-proj 处理全 dim 的本地 token**，
复制段的贡献会被算 P 次。实测（真权重 F=9）：patch_embedding 1.92×、总 gradnorm +12.6%。
F=9 时首帧占 11%，不是我以为的 2.2%——之前的灵敏度测试被 action 路径的梯度掩盖了。

v5 做法
-------
把序列拆成 3 段，每段内都是连续区间 ⇒ a2a 按 rank 序拼接恰好等于全局序，零复制：
  段1 首帧     880  -> 每卡 220
  段2 body     (F-1)*880 -> 每卡 (F-1)*220
  段3 action   400 -> 每卡 100
attention 侧仍是 5 段：[clean f0|clean body|noisy f0|noisy body|act]，
(s - ar_local)//2 仍等于 clean 的本地长度，`clean_frames` 在 gather 后仍等于 F ✓

副作用（好的）：不再需要"复制输出的 loss 除以 P"这条规则，
尾部把 action 输出也 all_gather 回全量后，各 rank 的 loss 完全一致。
"""
import sys

F = sys.argv[1] if len(sys.argv) > 1 else \
    "/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengmenghan/dreamzero_sp/dreamzero/groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py"
src = open(F).read()

if "_SPTokenGather.apply(action_noise_pred)" in src:
    print("[skip] v5 已打过")
    sys.exit(0)


def rep(old, new, tag):
    global src
    assert src.count(old) == 1, "[%s] 命中 %d 次" % (tag, src.count(old))
    src = src.replace(old, new)
    print("[ok] %s" % tag)


# 1) 段结构 -> 零复制
rep('''    total = F * fsl
    nrep = (F % world) * fsl
    iq = (total - nrep) // world
    assert iq * world + nrep == total, (F, world, fsl)
    assert (total - nrep) % world == 0
    imgs = ([("rep", nrep)] if nrep else []) + [("chunk", iq)]
    imgs_g = ([("rep", nrep)] if nrep else []) + [("chunk", iq * world)]
    tail = [("rep", ar)] if ar else []
    return imgs, imgs + imgs + tail, imgs_g + imgs_g + tail, imgs + tail, nrep, iq''',
'''    # v5：零复制。首帧一段、body 一段、action_register 一段，每段内部都是连续区间，
    # 所以 a2a 按 rank 序拼接 == 全局序。段长要能被 P 整除：880=4*220、400=4*100 ✓
    f0, body = fsl, (F - 1) * fsl
    for L in (f0, body, ar):
        assert L % world == 0, ("段长不能被 P 整除", L, world)
    imgs = [("chunk", f0 // world), ("chunk", body // world)]
    imgs_g = [("chunk", f0), ("chunk", body)]
    tail = [("chunk", ar // world)] if ar else []
    tail_g = [("chunk", ar)] if ar else []
    return imgs, imgs + imgs + tail, imgs_g + imgs_g + tail_g, imgs, \\
        (f0 // world) + (body // world), 0''', "segs")

# 2) x 切片
rep('''        _nfr, _fsl = int(grid_size[0]), self.frame_seqlen
        _nrep = (_nfr % _spw) * _fsl
        _iq = (seq_len - _nrep) // _spw
        if _spw > 1:
            assert _iq * _spw + _nrep == seq_len
            x = torch.cat([x[:, :_nrep], x[:, _nrep + _spr * _iq: _nrep + (_spr + 1) * _iq]],
                          dim=1).contiguous()''',
'''        _nfr, _fsl = int(grid_size[0]), self.frame_seqlen
        _bd_ = (_nfr - 1) * _fsl                     # body（除首帧）全局长度
        _f0l, _bdl = _fsl // _spw, _bd_ // _spw
        if _spw > 1:
            assert _fsl % _spw == 0 and _bd_ % _spw == 0, (_fsl, _bd_, _spw)
            x = torch.cat([x[:, _spr * _f0l: (_spr + 1) * _f0l],
                           x[:, _fsl + _spr * _bdl: _fsl + (_spr + 1) * _bdl]],
                          dim=1).contiguous()''', "x")

# 3) clean_x
rep('''            if _spw > 1:
                clean_x = torch.cat([clean_x[:, :_nrep],
                                     clean_x[:, _nrep + _spr * _iq:
                                                _nrep + (_spr + 1) * _iq]], dim=1).contiguous()''',
'''            if _spw > 1:
                clean_x = torch.cat([clean_x[:, _spr * _f0l: (_spr + 1) * _f0l],
                                     clean_x[:, _fsl + _spr * _bdl:
                                                _fsl + (_spr + 1) * _bdl]], dim=1).contiguous()''',
    "clean")

# 4) e0 / e
rep('''        _a, _sp_segs_set, _sp_segs_g, _b, _, _ = _sp_segs(_nfr, _spw, _fsl, _sp_ar)
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
            e = torch.cat([_sp_take(e, 0), e[:, seq_len:]], dim=1)''',
'''        _arl = _sp_ar // _spw if _spw > 1 else _sp_ar
        # _sp_segs 返回 (imgs, full, full_g, vid, nl, 0)：nl 在 index 4，不在 [1:4] 里
        _sp_segs_set, _sp_segs_g = _sp_segs(_nfr, _spw, _fsl, _sp_ar)[1:3]
        _nl = _sp_segs(_nfr, _spw, _fsl, _sp_ar)[4]
        _SP_STATE["segs"] = _sp_segs_set      # 入口：本地长度
        _SP_STATE["segs_g"] = _sp_segs_g      # 出口：全局长度
        if _spw > 1:
            def _sp_loc(t, base, Lg):
                L = Lg // _spw
                return t[:, base + _spr * L: base + (_spr + 1) * L]
            def _sp_img(t, base):
                return torch.cat([_sp_loc(t, base, _fsl),
                                  _sp_loc(t, base + _fsl, _bd_)], dim=1)
            if clean_x is not None:
                e0 = torch.cat([_sp_img(e0, 0), _sp_img(e0, seq_len),
                                _sp_loc(e0, 2 * seq_len, _sp_ar)], dim=1)
            else:
                e0 = torch.cat([_sp_img(e0, 0), _sp_loc(e0, seq_len, _sp_ar)], dim=1)
            e = torch.cat([_sp_img(e, 0), _sp_loc(e, seq_len, _sp_ar)], dim=1)''', "e0")

# 5) action_register 分成 1/P
rep('''            action_register = torch.cat([action_features, state_features], dim=1)
            action_register_length = action_register.shape[1]
            x = torch.cat([x, action_register], dim=1)''',
'''            action_register = torch.cat([action_features, state_features], dim=1)
            action_register_length = action_register.shape[1]
            if _SP_STATE["world"] > 1:
                # v5：action_register 不再是复制段，每卡只留 1/P 个 token
                _al = action_register_length // _SP_STATE["world"]
                action_register = action_register[:, _SP_STATE["rank"] * _al:
                                                  (_SP_STATE["rank"] + 1) * _al].contiguous()
            x = torch.cat([x, action_register], dim=1)''', "actreg")

# 6) 尾部：本地 act 长度 + 拼回全量
rep('''        if action is not None:
            action_noise_pred = x[:, _nl: _nl + action_length]
            action_noise_pred = self.action_decoder(action_noise_pred, embodiment_id)''',
'''        if action is not None:
            action_noise_pred = x[:, _nl: _nl + _arl]
            action_noise_pred = self.action_decoder(action_noise_pred, embodiment_id)
            if _SP_STATE["world"] > 1:
                # 拼回全量后各 rank 的 loss 完全一致，且不需要任何 1/P 缩放规则
                action_noise_pred = _SPTokenGather.apply(action_noise_pred)''', "actpred")

open(F, "w").write(src)
print("[done] v5 已写入 %s" % F)
