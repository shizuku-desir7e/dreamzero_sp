#!/usr/bin/env python3
"""DROID 真实数据加载器（DreamZero / Wan2.1-14B 输入契约）。

事实来源（都已实测，不猜）：
  * meta/info.json      : 180x320 RGB, 15 fps, action 28 维、state 14 维
  * meta/modality.json  : action[joint_position]=14:21, action[gripper_position]=12:13
                          state[joint_position]=7:14, state[gripper_position]=6:7
  * 官方 DROID 配置     : state/action 只取 joint_position(7) + gripper_position(1) = 8 维，
                          余下补零到 64 / 32；joint_position 用相对动作 + q99 归一化
  * VAE                 : WanVideoVAE 空间 stride 8、时间 stride 4
                          ⇒ F 个 latent 帧需要 (F-1)*4+1 个像素帧（129 for F=33）

分辨率：源 180x320，目标 320x704。选「等比放大到 W=704 (x2.2) → 高 396 → 上下中心
裁到 320」，**不拉伸**（直接 resize 会把 0.5625 的宽高比压成 0.4545，几何失真 24%）。
代价：上下各丢 38 px（垂直视野 19.2%）。
"""
import json
import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as TF

ROOT = "/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengmenghan"
DROID = ROOT + "/droid_lerobot"
VIEW = "observation.images.exterior_image_1_left"
H_LAT, W_LAT = 40, 88          # 320x704 / stride 8
PIX_H, PIX_W = 320, 704
T_DOWN = 4                     # VAE 时间下采样


# ---------------------------------------------------------------- 帧列表 / 划分
def list_episodes(root=DROID):
    vids = sorted(glob.glob("%s/videos/chunk-000/%s/*.mp4" % (root, VIEW)))
    pqs = sorted(glob.glob("%s/data/**/*.parquet" % root, recursive=True))
    eps = []
    for v in vids:
        e = int(os.path.basename(v).split("_")[-1].split(".")[0])
        pq = [p for p in pqs if "episode_%06d" % e in os.path.basename(p)]
        eps.append(dict(episode=e, video=v, parquet=pq[0] if pq else None))
    return eps


def split_episodes(eps, holdout=(15, 16, 17)):
    """按 episode 划分，同一 episode 的帧绝不跨集合。"""
    tr = [e for e in eps if e["episode"] not in holdout]
    ho = [e for e in eps if e["episode"] in holdout]
    return tr, ho


# ---------------------------------------------------------------- 归一化
def _q99(x, q01, q99):
    """官方 StateActionTransform 的 q99 模式：2*(x-q01)/(q99-q01)-1，clip 到 [-1,1]。"""
    q01, q99 = torch.as_tensor(q01, dtype=torch.float32), torch.as_tensor(q99, dtype=torch.float32)
    rng = q99 - q01
    ok = rng.abs() > 1e-8
    y = torch.zeros_like(x)
    y[..., ok] = 2 * (x[..., ok] - q01[ok]) / rng[ok] - 1
    y[..., ~ok] = x[..., ~ok]
    return y.clamp_(-1, 1)


class Stats:
    def __init__(self, root=DROID):
        st = json.load(open(os.path.join(root, "meta/stats.json")))
        rel = json.load(open(os.path.join(root, "meta/relative_stats_dreamzero.json")))
        self.state_q01 = np.array(st["observation.state"]["q01"], dtype=np.float32)
        self.state_q99 = np.array(st["observation.state"]["q99"], dtype=np.float32)
        self.act_q01 = np.array(st["action"]["q01"], dtype=np.float32)
        self.act_q99 = np.array(st["action"]["q99"], dtype=np.float32)
        self.rel_q01 = np.array(rel["joint_position"]["q01"], dtype=np.float32)
        self.rel_q99 = np.array(rel["joint_position"]["q99"], dtype=np.float32)


# ---------------------------------------------------------------- 图像预处理
def resize_crop(frames):
    """[T,3,h,w] uint8 -> [T,3,320,704] float32 in [-1,1]，等比放大 + 中心裁。"""
    x = frames.float().div_(255.0)
    h, w = x.shape[-2:]
    s = PIX_W / w
    nh, nw = int(round(h * s)), PIX_W
    x = TF.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False, antialias=True)
    top = (nh - PIX_H) // 2
    x = x[:, :, top:top + PIX_H, :]
    assert x.shape[-2:] == (PIX_H, PIX_W), x.shape
    return x.mul_(2.0).sub_(1.0)


class DroidWindows(torch.utils.data.Dataset):
    """每个样本 = 一个 (episode, 起始像素帧) 窗口。返回像素帧 + action + state。

    F 个 latent 帧 ← T_pix = (F-1)*4+1 个像素帧。
    action 槽位：NB=(F-1)//2 块 × 24 = NB*24 个槽位，铺满同一时间跨度（最近邻对齐）。
    state：每块 1 个，放在该块起始像素帧。
    """

    def __init__(self, eps, F_lat=33, stride_pix=16, stats=None, cache=True):
        self.F = F_lat
        self.T_pix = (F_lat - 1) * T_DOWN + 1
        self.NB = (F_lat - 1) // 2
        self.stats = stats or Stats()
        self._cache = {} if cache else None
        self.items = []
        self.skipped = []
        for e in eps:
            n = self._num_frames(e)
            if n < self.T_pix:
                self.skipped.append((e["episode"], n))
                continue
            for st in range(0, n - self.T_pix + 1, stride_pix):
                self.items.append((e, st, n))
        self.eps = eps

    @staticmethod
    def _num_frames(e):
        if e.get("_n") is None:
            import torchvision.io as tio
            e["_n"] = int(tio.read_video(e["video"], pts_unit="sec", output_format="TCHW")[0].shape[0])
        return e["_n"]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.raw(i)

    def video(self, e):
        if self._cache is not None and e["episode"] in self._cache:
            return self._cache[e["episode"]]
        import torchvision.io as tio
        fr, _, _ = tio.read_video(e["video"], pts_unit="sec", output_format="TCHW")
        if self._cache is not None:
            self._cache[e["episode"]] = fr
        return fr

    def table(self, e):
        key = "_df"
        if e.get(key) is None:
            e[key] = pd.read_parquet(e["parquet"])
        return e[key]

    def raw(self, i):
        """像素帧 [3,T,320,704] in [-1,1] + action [NB*24,32] + state [NB,64]（全部真实值）"""
        e, st, n = self.items[i]
        fr = self.video(e)[st:st + self.T_pix]
        assert len(fr) == self.T_pix
        px = resize_crop(fr).permute(1, 0, 2, 3).contiguous()      # [3,T,H,W]

        df = self.table(e)
        a = np.stack(df["action"].values).astype(np.float32)       # [n,28]
        s = np.stack(df["observation.state"].values).astype(np.float32)

        S = self.stats
        # 8 维 state: joint(7) + gripper(1)
        s8 = np.concatenate([s[st:st + self.T_pix, 7:14], s[st:st + self.T_pix, 6:7]], axis=1)
        s8[st * 0:0] = s8[st * 0:0]                                 # no-op, keep shape explicit
        s8 = _q99(torch.from_numpy(s8),
                  np.concatenate([S.state_q01[7:14], S.state_q01[6:7]]),
                  np.concatenate([S.state_q99[7:14], S.state_q99[6:7]]))
        # 8 维 action: 相对 joint(7) + gripper 绝对值(1)
        aj = a[st:st + self.T_pix, 14:21] - s[st:st + self.T_pix, 7:14]      # 相对当前 state
        ag = a[st:st + self.T_pix, 12:13]
        a8 = torch.cat([_q99(torch.from_numpy(aj), S.rel_q01, S.rel_q99),
                        _q99(torch.from_numpy(ag), S.act_q01[12:13], S.act_q99[12:13])], dim=1)  # [T,8]

        ns = self.NB * 24
        slot = torch.floor(torch.arange(ns, dtype=torch.float64) * self.T_pix / ns).long()
        act = torch.zeros(ns, 32)
        act[:, :8] = a8[slot]

        blk = torch.floor(torch.arange(self.NB, dtype=torch.float64) * self.T_pix / self.NB).long()
        stt = torch.zeros(self.NB, 64)
        stt[:, :8] = s8[blk]
        return px, act, stt


# ---------------------------------------------------------------- VAE
def load_vae(ckpt_dir=ROOT + "/model", dev="cuda:0", dtype=torch.bfloat16):
    from groot.vla.model.dreamzero.modules.wan_video_vae import WanVideoVAE
    from safetensors.torch import load_file
    vae = WanVideoVAE(z_dim=16).to(dev)
    idx = json.load(open(os.path.join(ckpt_dir, "model.safetensors.index.json")))
    shards = sorted({sh for k, sh in idx["weight_map"].items() if k.startswith("action_head.vae.")})
    sd = {}
    for sh in shards:
        d = load_file(os.path.join(ckpt_dir, sh), device="cpu")
        for k, v in d.items():
            if k.startswith("action_head.vae."):
                sd[k[len("action_head.vae."):]] = v
        del d
    miss, unexp = vae.load_state_dict(sd, strict=False)
    assert len(miss) == 0 and len(unexp) == 0, (miss[:5], unexp[:5])
    del sd
    vae.eval().requires_grad_(False)
    return vae.to(dtype)


@torch.no_grad()
def encode(vae, px):
    """px [3,T,H,W] in [-1,1] -> latent [1,16,F,40,88] bf16"""
    x = px.unsqueeze(0).to(vae.model.conv1.weight.device, vae.model.conv1.weight.dtype)
    return vae.encode(x)


@torch.no_grad()
def decode(vae, z):
    return vae.decode(z.float().to(vae.model.conv1.weight.dtype))


# ---------------------------------------------------------------- 自检
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=33)
    ap.add_argument("--check_psnr", type=int, default=1)
    a = ap.parse_args()

    eps = list_episodes()
    tr, ho = split_episodes(eps)
    Wtr = DroidWindows(tr, a.frames)
    Who = DroidWindows(ho, a.frames)
    print("[episodes] total=%d train=%d holdout=%d" % (len(eps), len(tr), len(ho)))
    print("[frames] per-episode:", [(e["episode"], DroidWindows._num_frames(e)) for e in eps])
    print("[skipped] (too short):", Wtr.skipped, Who.skipped)
    print("[windows] train=%d holdout=%d  T_pix=%d F=%d NB=%d" %
          (len(Wtr), len(Who), Wtr.T_pix, Wtr.F, Wtr.NB))

    dev = "cuda:0"
    px, act, st = Wtr[0]
    print("\n--- sample 0 (episode %d, start %d) ---" % (Wtr.items[0][0]["episode"], Wtr.items[0][1]))
    print("pixel   ", tuple(px.shape), "range [%.2f, %.2f]" % (px.min(), px.max()))
    print("action  ", tuple(act.shape), "range [%.2f, %.2f]  nonzero-dim=8" % (act.min(), act.max()))
    print("state   ", tuple(st.shape), "range [%.2f, %.2f]" % (st.min(), st.max()))
    print("action real-dim stats: mean %.3f std %.3f  | padded cols absmax %.1e" %
          (act[:, :8].mean(), act[:, :8].std(), act[:, 8:].abs().max()))

    vae = load_vae(dev=dev)
    z = encode(vae, px.to(dev))
    zf = z.float()
    print("latent  ", tuple(z.shape), "dtype", z.dtype)
    print("latent stats mean %.4f std %.4f min %.3f max %.3f" %
          (zf.mean(), zf.std(), zf.min(), zf.max()))
    assert tuple(z.shape) == (1, 16, a.frames, H_LAT, W_LAT), z.shape

    if a.check_psnr:
        rec = decode(vae, z).float().clamp(-1, 1)
        mse = (rec - px.to(dev).float()).pow(2).mean()
        psnr = 10 * torch.log10(torch.tensor(4.0, device=dev) / mse)
        per = (rec - px.to(dev).float()).pow(2).mean(dim=[0, 1, 3, 4])
        ps = 10 * torch.log10(4.0 / per.clamp_min(1e-8))
        print("recon   ", tuple(rec.shape), "PSNR %.2f dB  per-frame p10 %.2f p50 %.2f" %
              (psnr, ps.quantile(0.1), ps.quantile(0.5)))

    print("\n--- 契约对照 ---")
    print("latent 16ch           :", z.shape[1] == 16)
    print("latent 40x88          :", tuple(z.shape[3:]) == (H_LAT, W_LAT))
    print("F=%d latent 帧        :" % a.frames, z.shape[2] == a.frames)
    print("action [%d,32]        :" % (Wtr.NB * 24), tuple(act.shape) == (Wtr.NB * 24, 32))
    print("state  [%d,64]        :" % Wtr.NB, tuple(st.shape) == (Wtr.NB, 64))
    print("frame_seqlen 880      :", H_LAT // 2 * W_LAT // 2)
    print("action_register_length:", Wtr.NB * 25)
    print("OK")
