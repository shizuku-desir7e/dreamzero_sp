"""留出集评测：flow-matching loss / 单步去噪重建 / VAE 上下界 / fvd_r3d18。

两种用法
--------
1) 独立跑（推荐，SP=4）：
   python -m torch.distributed.run --nproc_per_node=4 eval_holdout.py \
       --ckpt results/real_data_500_adamw.pt --frames 33 --clips 12

2) 被 train_real_data.py 在训练结束时 import 调用（rank0 单卡）：
   ev = eval_holdout.run_eval(m, vae, F_, dev, args, log=print)
   注意：单卡跑 F=33 会 OOM，只在 F<=9 时可用。

指标口径（写论文时按这个说，别改口）
----------------------------------
- holdout_loss   : 留出 episode 上的 flow-matching MSE。**主指标**。
- psnr/ssim_vae  : 真实帧 encode→decode 的 PSNR/SSIM。这是 VAE 上限，不是模型质量。
- psnr/ssim_1step: 单步去噪（ẑ0 = z_t − t·v̂）再解码，与真实帧比。便宜但有意义。
- fvd_r3d18      : 真实 clip 与「单步去噪重建 clip」在 R3D-18(Kinetics-400) 特征上的
                   Fréchet 距离。**不是 I3D-FVD，不可与文献数字横向比较**。
"""
import argparse
import json
import os
import sys
import time
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as Fn

Z = os.environ.get("ZROOT", "/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengmenghan")
sys.path.insert(0, Z + "/dreamzero_sp/shim")
sys.path.insert(0, Z + "/dreamzero_sp/dreamzero")

import groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk as DZMOD
import dl_droid

FSL, H, W = 880, 40, 88


# ------------------------------------------------------------------ 指标
def psnr(a, b, eps=1e-8):
    """a,b: [-1,1] 的 float 张量，任意形状；返回标量 dB。"""
    mse = (a.float() - b.float()).pow(2).mean().clamp_min(eps)
    return float(10.0 * torch.log10(torch.tensor(4.0)) - 10.0 * torch.log10(mse))


def ssim(a, b, win=7, eps=1e-6):
    """简化 SSIM：对 (…,H,W) 用均值/方差窗（uniform），只要趋势可比即可。

    不用 skimage（集群里没装）。win 用 7 而不是 11，因为 320x704 下 11 太粗。
    """
    a = a.float()
    b = b.float()
    k = win
    pad = k // 2
    mu_a = Fn.avg_pool2d(a, k, 1, pad)
    mu_b = Fn.avg_pool2d(b, k, 1, pad)
    sa = Fn.avg_pool2d(a * a, k, 1, pad) - mu_a * mu_a
    sb = Fn.avg_pool2d(b * b, k, 1, pad) - mu_b * mu_b
    sab = Fn.avg_pool2d(a * b, k, 1, pad) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu_a * mu_b + c1) * (2 * sab + c2)
    den = (mu_a.pow(2) + mu_b.pow(2) + c1) * (sa + sb + c2)
    return float((num / den).mean())


def _r3d18():
    """Kinetics-400 预训练的 R3D-18，返回去掉 fc 的特征提取器。"""
    from torchvision.models.video import r3d_18
    m = r3d_18(weights=None)
    p = Z + "/tools_wheels/r3d_18-b3b3357e.pth"
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    m.load_state_dict(torch.load(p, map_location="cpu"))
    m.fc = torch.nn.Identity()
    m.eval()
    return m


@torch.no_grad()
def r3d_features(net, clips, dev, n_frames=16):
    """clips: [B,3,T,H,W] in [-1,1] → [B,512]"""
    x = clips.float()
    if x.shape[2] != n_frames:                      # 时间维插值到 16 帧
        x = Fn.interpolate(x, size=(n_frames, x.shape[-2], x.shape[-1]),
                           mode="trilinear", align_corners=False)
    if x.shape[-1] != 112:                          # 空间缩到 112（R3D 常用）
        x = Fn.interpolate(x, size=(n_frames, 112, 112), mode="trilinear", align_corners=False)
    x = ((x + 1) / 2).clamp(0, 1)
    # 注意：此处 x 还在 CPU 上（下面才 .to(dev)），归一化常数必须跟着 x 的设备走
    mean = torch.tensor([0.43216, 0.394666, 0.37645], device=x.device).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.22803, 0.22145, 0.216989], device=x.device).view(1, 3, 1, 1, 1)
    x = ((x - mean) / std).to(torch.float32)
    feats = []
    for i in range(0, x.shape[0], 4):               # 分块，避免爆显存
        feats.append(net(x[i:i + 4].to(dev)).float().cpu())
    return torch.cat(feats, 0)


def frechet(fa, fb, eps=1e-6):
    """Fr\'echet distance（两个高斯）。fa/fb: [N,D]"""
    fa = fa.double(); fb = fb.double()
    mu1, mu2 = fa.mean(0), fb.mean(0)
    c1 = torch.cov(fa.T); c2 = torch.cov(fb.T)
    d = mu1 - mu2
    # sqrtm 用 eigh：C1^{1/2} C2 C1^{1/2} 的特征值开方
    def sqrtm(a):
        w, v = torch.linalg.eigh(a)
        return (v * w.clamp_min(0).sqrt()) @ v.T
    s1 = sqrtm(c1)
    mid = sqrtm(s1 @ c2 @ s1)
    tr = torch.trace(c1) + torch.trace(c2) - 2 * torch.trace(mid)
    return float((d @ d + tr).item())


# ------------------------------------------------------------------ 一次前向
def make_holdout_batch(ds, i, F_, dev, seed):
    """与训练脚本同构的 batch，但数据来自留出集。全部 rank 必须一致。"""
    z0, act, st = model_latent(ds, i, dev, F_)
    N = F_ * FSL
    NB = (F_ - 1) // 2
    g = torch.Generator().manual_seed(seed * 7919 + i)
    eps = torch.randn(1, 16, F_, H, W, generator=g).to(dev, dtype=torch.bfloat16)
    t01 = torch.rand(1, F_, generator=g).to(dev)
    zf = z0[:, :, :1].expand(-1, -1, F_, -1, -1)
    mask = torch.ones(1, 4, F_, H, W, device=dev, dtype=torch.bfloat16)
    mask[:, :, 1:] = 0.0
    tf_ = t01.view(1, 1, F_, 1, 1).to(torch.bfloat16)
    zv = ((1 - tf_) * z0 + tf_ * eps).to(torch.bfloat16)
    ctx = torch.randn(1, 512, 4096, generator=g).to(dev, dtype=torch.bfloat16) * 0.1
    return dict(
        N=N, tt=t01 * 1000.0, z0=z0, t01=t01,
        x_in=torch.cat([zv, mask, zf], dim=1),
        x_cl=torch.cat([z0, mask, zf], dim=1),
        tgt=(eps - z0).float(),
        ctx=ctx,
        act=act.unsqueeze(0), st=st.unsqueeze(0),
        ta=torch.randint(0, 1000, (1, NB * 24), device=dev),
        emb=torch.zeros(1, dtype=torch.long, device=dev))


def model_latent(ds, i, dev, F_):
    """ds[i] → VAE 编码。注意 dl_droid.encode 的入参是 [3,T,H,W]，它内部自己 unsqueeze。

    返回 z [1,16,F,40,88] / act [NB*24,32] / state [NB,64]
    """
    px, act, stt = ds[i]                             # px: [3,T,H,W] in [-1,1]
    with torch.no_grad():
        z = dl_droid.encode(_VAE, px.to(dev))         # → [1,16,F,40,88]
    assert tuple(z.shape) == (1, 16, F_, H, W), z.shape
    # action/state 在 dl_droid.raw() 里是 float32；模型是 bf16，必须转
    # （训练脚本在 precompute 里转了，评测漏掉会报 expected Float but found BFloat16）
    return (z.to(dtype=torch.bfloat16),
            act.to(dev, dtype=torch.bfloat16),
            stt.to(dev, dtype=torch.bfloat16))


def _to_bcthw(x):
    """把 decode 的输出统一成 [B,3,T,H,W]。"""
    if x.dim() == 5:
        return x
    if x.dim() == 4:                                  # [3,T,H,W]
        return x.unsqueeze(0)
    raise ValueError("unexpected decoded shape %s" % (tuple(x.shape),))


_VAE = None


def _eval_core(m, vae, ds, F_, dev, args, log, rank=0, world=1):
    global _VAE
    _VAE = vae
    n = len(ds) if args.clips <= 0 else min(args.clips, len(ds))
    log("[eval] holdout windows=%d (取前 %d), F=%d" % (len(ds), n, F_))

    net = None
    try:
        net = _r3d18().to(dev)
        log("[eval] R3D-18 已加载")
    except Exception as e:
        log("[eval] R3D-18 不可用（%s），跳过 fvd_r3d18" % str(e)[:120])

    losses, pv, sv, p1, s1 = [], [], [], [], []
    real_clips, pred_clips, vae_clips = [], [], []
    per_ep = {}
    for i in range(n):
        b = make_holdout_batch(ds, i, F_, dev, args.seed)
        # 必须是 enable_grad 而不是 no_grad：
        # 模型里 checkpointing 分支的条件是 `torch.is_grad_enabled() and self.gradient_checkpointing`
        # （见 wan_video_dit_action_casual_chunk.py:2382）。用 no_grad 会走非 ckpt 分支，
        # 既触发上游「block 返回 tuple 未解包」的 bug，又让 33 帧激活直接爆显存。
        # 参数已 requires_grad=False，所以不会建图、不会有梯度。
        with torch.enable_grad():
            out = m(b["x_in"], b["tt"], b["ta"], b["ctx"], seq_len=b["N"],
                    clean_x=b["x_cl"], action=b["act"], state=b["st"],
                    embodiment_id=b["emb"])
            v, a = out
            last_loss = float((v.float() - b["tgt"]).pow(2).mean())
            losses.append(last_loss)

            # 单步去噪：z_t = (1-t) z0 + t eps, v* = eps - z0 ⇒ z0 = z_t - t v
            t = b["t01"].view(1, 1, F_, 1, 1)
            zv = b["x_in"][:, :16]
            z_hat = (zv.float() - t * v.float()).clamp(-6, 6)
            z0 = b["z0"].float()

            px_real = ds.raw(i)[0].unsqueeze(0).to(dev)          # [1,3,T,H,W]
            px_pred = _to_bcthw(dl_droid.decode(vae, z_hat.to(torch.bfloat16)))
            px_vae = _to_bcthw(dl_droid.decode(vae, z0.to(torch.bfloat16)))

            _ep = int(ds.items[i][0]["episode"])
            per_ep.setdefault(_ep, []).append(last_loss)
            pv.append(psnr(px_real, px_vae)); sv.append(ssim(px_real[0], px_vae[0]))
            if net is not None and len(vae_clips) < args.fvd_clips:
                vae_clips.append(px_vae.cpu())
            p1.append(psnr(px_real, px_pred)); s1.append(ssim(px_real[0], px_pred[0]))
            if net is not None and len(real_clips) < args.fvd_clips:
                real_clips.append(px_real.cpu())
                pred_clips.append(px_pred.cpu())

        if rank == 0 and (i + 1) % 4 == 0:
            log("[eval] %d/%d  loss=%.5f  1step_psnr=%.2f" % (i + 1, n, losses[-1], p1[-1]))

    ev = dict(
        n_clips=n, frames=F_,
        holdout_loss=float(np.mean(losses)),
        holdout_loss_per_clip=[round(x, 6) for x in losses],
        psnr_vae=float(np.mean(pv)), ssim_vae=float(np.mean(sv)),
        psnr_1step=float(np.mean(p1)), ssim_1step=float(np.mean(s1)),
        note="psnr/ssim_vae 是 VAE 上限，不是模型质量；psnr/ssim_1step 才是模型指标",
        per_episode={str(k): dict(n=len(x), loss=float(np.mean(x))) for k, x in sorted(per_ep.items())},
        episodes_covered=sorted(per_ep.keys()),
    )
    if net is not None and len(real_clips) >= 2:
        fa = r3d_features(net, torch.cat(real_clips, 0), dev)
        fb = r3d_features(net, torch.cat(pred_clips, 0), dev)
        ev["fvd_r3d18"] = frechet(fa, fb)
        if len(vae_clips) >= 2:
            fc = r3d_features(net, torch.cat(vae_clips, 0), dev)
            ev["fvd_r3d18_vae_floor"] = frechet(fa, fc)
            ev["fvd_floor_note"] = "同指标下『真实 vs VAE往返』的距离，作为地板"
        ev["fvd_note"] = ("R3D-18/Kinetics-400 特征上的 Frechet 距离，"
                          "不是 I3D-FVD，不可与文献数字横向比较")
    return ev


def run_eval(m, vae, F_, dev, args, log=print):
    """给 train_real_data.py 调用的入口（单卡；F<=9 才安全）。"""
    hold = [int(x) for x in str(args.holdout).split(",")]
    eps = dl_droid.list_episodes()
    ho = [e for e in eps if e["episode"] in hold]
    ds = dl_droid.DroidWindows(ho, F_lat=F_, stats=dl_droid.Stats(), cache=True)
    return _eval_core(m, vae, ds, F_, dev, args, log)


# ------------------------------------------------------------------ 独立入口
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--frames", type=int, default=33)
    ap.add_argument("--clips", type=int, default=12)
    ap.add_argument("--fvd_clips", type=int, default=8)
    ap.add_argument("--holdout", type=str, default="15,16,17")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--out", type=str, default=Z + "/dreamzero_sp/results/real_data_eval.json")
    args = ap.parse_args()

    dist.init_process_group("nccl", timeout=timedelta(seconds=1800))
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = "cuda:%d" % rank
    is0 = rank == 0
    F_ = args.frames

    CFG = dict(model_type="ti2v", patch_size=(1, 2, 2), frame_seqlen=FSL, text_len=512,
               in_dim=36, dim=5120, ffn_dim=13824, freq_dim=256, text_dim=4096, out_dim=16,
               num_heads=40, num_layers=40, num_frame_per_block=2, action_dim=32,
               num_action_per_block=24, num_state_per_block=1, max_state_dim=64,
               max_num_embodiments=32, hidden_size=1024, concat_first_frame_latent=True,
               qk_norm=True, cross_attn_norm=True, max_chunk_size=4, sink_size=0)
    torch.manual_seed(0)
    torch.set_default_dtype(torch.bfloat16)
    m = DZMOD.CausalWanModel(**CFG).to(dev)
    torch.set_default_dtype(torch.float32)
    m.eval(); m.gradient_checkpointing = True
    for p in m.parameters():
        p.requires_grad_(False)

    t0 = time.time()
    if args.ckpt == "base":
        # 未微调基线：直接读 $Z/model 的原始 DiT 权重（与训练脚本同一套键名/前缀）
        import json as _json
        from safetensors.torch import load_file
        _idx = _json.load(open(os.path.join(Z, "model", "model.safetensors.index.json")))
        _pmap, _bmap = dict(m.named_parameters()), dict(m.named_buffers())
        _n, _mm = 0, 0
        for _sh in sorted(set(_idx["weight_map"].values())):
            _sd = load_file(os.path.join(Z, "model", _sh), device=dev)
            for _k, _v in _sd.items():
                if not _k.startswith("action_head.model."):
                    continue
                _kk = _k[len("action_head.model."):]
                # 不能写 pmap.get(k) or bmap.get(k)：Tensor 的布尔值是歧义操作
                _t = _pmap[_kk] if _kk in _pmap else _bmap.get(_kk)
                if _t is not None and _t.shape == _v.shape:
                    _t.data.copy_(_v); _n += 1
                else:
                    _mm += 1
            del _sd
            torch.cuda.empty_cache()
        if is0:
            print("[load] base(未微调) %.0fs  拷入 %d 张量, 跳过 %d" %
                  (time.time() - t0, _n, _mm), flush=True)
    else:
        sd = torch.load(args.ckpt, map_location="cpu")
        miss = m.load_state_dict(sd, strict=False)
        if is0:
            print("[load] %s  %.0fs  missing=%d unexpected=%d" %
                  (args.ckpt, time.time() - t0, len(miss.missing_keys), len(miss.unexpected_keys)),
                  flush=True)
        del sd

    vae = dl_droid.load_vae(ckpt_dir=Z + "/model", dev=dev)
    DZMOD.set_sp_state(group=dist.group.WORLD, world=world, rank=rank)

    hold = [int(x) for x in args.holdout.split(",")]
    eps = dl_droid.list_episodes()
    ho = [e for e in eps if e["episode"] in hold]
    ds = dl_droid.DroidWindows(ho, F_lat=F_, stats=dl_droid.Stats(), cache=True)

    ev = _eval_core(m, vae, ds, F_, dev, args, print, rank=rank, world=world)
    ev["ckpt"] = args.ckpt
    if is0:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(ev, open(args.out, "w"), indent=1)
        print("[eval] saved " + args.out, flush=True)
        for k in ("holdout_loss", "psnr_vae", "ssim_vae", "psnr_1step", "ssim_1step", "fvd_r3d18"):
            if k in ev:
                print("  %-14s %s" % (k, ev[k]), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
