#!/usr/bin/env python3
"""真实 DROID 视频 + 真实 action 训练（SP=4 + ZeRO-2 + bf16 AdamW）。

骨架与 train_real.py 完全一致，只把 make_batch() 从「高斯 latent + 随机 action」
换成 dl_droid 的真实 VAE 编码 latent + 真实 action/state。

与 train_real.py 的差异（全部有意为之）：
  1. z0 来自真实视频（VAE encode），不再是 torch.randn
  2. action/state 来自真实 parquet（8 维真实维度 + 零填充到 32/64）
  3. 每步用不同窗口（140 个训练窗口循环），loss 曲线因此有 sampling 噪声
  4. action 头仍用 0.1 * a^2 正则（与 train_real.py 相同），**不是**动作预测目标

用法：bash run_real_data.sh  （launcher 里带 SP=4 的 torch.distributed.run）
"""
import argparse, json, os, sys, time
from datetime import timedelta
import torch
import torch.distributed as dist

Z = os.environ.get("ZROOT", "/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengmenghan")
sys.path.insert(0, Z + "/dreamzero_sp/shim")
sys.path.insert(0, Z + "/dreamzero_sp/dreamzero")
sys.path.insert(0, Z + "/dreamzero_sp")
import groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk as DZMOD
from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import CausalWanModel
import dl_droid

ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, default=33)     # latent 帧数（官方 33）
ap.add_argument("--steps", type=int, default=500)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--ckpt_dir", type=str, default=Z + "/model")
ap.add_argument("--prefix", type=str, default="action_head.model.")
ap.add_argument("--order", type=str, default="4")
ap.add_argument("--zero2", type=int, default=1)
ap.add_argument("--opt", type=str, default="adamw_bf16", choices=["sgd", "adamw_bf16"])
ap.add_argument("--no_load", type=int, default=0)
ap.add_argument("--holdout", type=str, default="15,16,17")
ap.add_argument("--log_every", type=int, default=10)
ap.add_argument("--out", type=str, default=Z + "/dreamzero_sp/results/real_data_train.json")
ap.add_argument("--eval_after", type=int, default=1)
ap.add_argument("--eval_out", type=str, default=Z + "/dreamzero_sp/results/real_data_eval.json")
ap.add_argument("--fvd_clips", type=int, default=10)
ap.add_argument("--ode_steps", type=int, default=8)
ap.add_argument("--seed", type=int, default=2024)
args = ap.parse_args()

dist.init_process_group("nccl", timeout=timedelta(seconds=1800))
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank); dev = "cuda:%d" % rank; is0 = rank == 0
F_ = args.frames
H, W, FSL = 40, 88, 880
NB = (F_ - 1) // 2
N = F_ * FSL

CFG = dict(model_type="ti2v", patch_size=(1, 2, 2), frame_seqlen=FSL, text_len=512,
           in_dim=36, dim=5120, ffn_dim=13824, freq_dim=256, text_dim=4096, out_dim=16,
           num_heads=40, num_layers=40, num_frame_per_block=2, action_dim=32,
           num_action_per_block=24, num_state_per_block=1, max_state_dim=64,
           max_num_embodiments=32, hidden_size=1024, concat_first_frame_latent=True,
           qk_norm=True, cross_attn_norm=True, max_chunk_size=4, sink_size=0)

t0 = time.time()
torch.manual_seed(0); torch.set_default_dtype(torch.bfloat16)
m = CausalWanModel(**CFG).to(dev); torch.set_default_dtype(torch.float32)
m.eval(); m.gradient_checkpointing = True
for p in m.parameters():
    p.requires_grad_(True)

if not args.no_load:
    from safetensors.torch import load_file
    idx = json.load(open(os.path.join(args.ckpt_dir, "model.safetensors.index.json")))
    for sh in sorted(set(idx["weight_map"].values())):
        sd = load_file(os.path.join(args.ckpt_dir, sh), device=dev)
        pmap = dict(m.named_parameters()); bmap = dict(m.named_buffers())
        for k, v in sd.items():
            if not k.startswith(args.prefix):
                continue
            kk = k[len(args.prefix):]
            tgt = pmap[kk] if kk in pmap else bmap.get(kk)
            if tgt is not None and tgt.shape == v.shape:
                tgt.data.copy_(v)
        del sd; torch.cuda.empty_cache()
if is0:
    print("[load] model %.0fs" % (time.time() - t0), flush=True)

# ---------------- VAE + 真实数据 ----------------
t1 = time.time()
vae = dl_droid.load_vae(ckpt_dir=args.ckpt_dir, dev=dev)
torch.cuda.empty_cache()
eps_all = dl_droid.list_episodes()
holdout = tuple(int(v) for v in args.holdout.split(","))
tr_eps, ho_eps = dl_droid.split_episodes(eps_all, holdout)
train_ds = dl_droid.DroidWindows(tr_eps, F_, stride_pix=16)
if is0:
    print("[data] train windows=%d (episodes %s) holdout episodes=%s  T_pix=%d  VAE load %.0fs" %
          (len(train_ds), [e["episode"] for e in tr_eps], list(holdout), train_ds.T_pix,
           time.time() - t1), flush=True)

_zcache = {}

def precompute():
    """把所有训练窗口的 latent 一次性编码好放 GPU（140 × 3.7 MB ≈ 0.5 GB），
    然后释放 VAE —— 否则 VAE 常驻 0.5 GB，在别人占 9.7 GB/卡 时会 OOM。"""
    t = time.time()
    for i in range(len(train_ds)):
        px, act, st = train_ds[i]
        z = dl_droid.encode(vae, px.to(dev))
        assert tuple(z.shape) == (1, 16, F_, H, W), z.shape
        _zcache[i] = (z, act.to(dev, dtype=torch.bfloat16), st.to(dev, dtype=torch.bfloat16))
        del px
        if is0 and (i + 1) % 20 == 0:
            print("[precompute] %d/%d %.0fs" % (i + 1, len(train_ds), time.time() - t), flush=True)
    torch.cuda.empty_cache()


def latent_for(i):
    return _zcache[i]


def make_batch(step):
    """确定性（所有 rank 必须拿到同一份数据，SP 会按帧切分）。"""
    order = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(args.seed))
    i = int(order[step % len(train_ds)])
    z0, act, st = latent_for(i)                       # 真实 latent / action / state
    g = torch.Generator().manual_seed(args.seed * 7919 + step)
    eps = torch.randn(1, 16, F_, H, W, generator=g).to(dev, dtype=torch.bfloat16)
    t01 = torch.rand(1, F_, generator=g).to(dev)
    tt = t01 * 1000.0
    zf = z0[:, :, :1].expand(-1, -1, F_, -1, -1)
    mask = torch.ones(1, 4, F_, H, W, device=dev, dtype=torch.bfloat16)
    mask[:, :, 1:] = 0.0
    tf_ = t01.view(1, 1, F_, 1, 1).to(torch.bfloat16)
    zv = ((1 - tf_) * z0 + tf_ * eps).to(torch.bfloat16)
    return dict(
        N=N, tt=tt, idx=i,
        x_in=torch.cat([zv, mask, zf], dim=1),
        x_cl=torch.cat([z0, mask, zf], dim=1),
        tgt=(eps - z0).float(),
        ctx=torch.randn(1, 512, 4096, generator=g).to(dev, dtype=torch.bfloat16) * 0.1,
        act=act.unsqueeze(0),
        st=st.unsqueeze(0),
        ta=torch.randint(0, 1000, (1, NB * 24), device=dev),
        emb=torch.zeros(1, dtype=torch.long, device=dev))


class Zero2:
    def __init__(self, params, world, rank):
        self.ps = [p for p in params if p.requires_grad]
        self.world, self.rank = world, rank
        self.shards = {}
        self.full = 0
        self.partial = 0
        for i, p in enumerate(self.ps):
            p.register_post_accumulate_grad_hook(self._mk(i))

    def _mk(self, i):
        def hook(param):
            g = param.grad
            if g is None:
                return
            n = g.numel()
            if self.world > 1 and n % self.world == 0:
                out = torch.empty(n // self.world, dtype=g.dtype, device=g.device)
                dist.reduce_scatter_tensor(out, g.reshape(-1).contiguous(), op=dist.ReduceOp.SUM)
                self.shards[i] = out
                self.full += 1
            else:
                self.shards[i] = g.reshape(-1).clone()
                self.partial += 1
            param.grad = None
        return hook

    def begin_step(self):
        self.full = self.partial = 0


class Zero2AdamW(Zero2):
    def __init__(self, params, world, rank, lr, betas=(0.9, 0.999), eps=1e-8, wd=0.01, **kw):
        super().__init__(params, world, rank)
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, betas[0], betas[1], eps, wd
        self.st, self.t = {}, 0

    def state_bytes(self):
        return sum(s["m"].numel() * s["m"].element_size() * 2 for s in self.st.values())

    @torch.no_grad()
    def step(self, lr=None):
        lr = self.lr if lr is None else lr
        self.t += 1
        bc1, bc2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for i, p in enumerate(self.ps):
            sh = self.shards.get(i)
            if sh is None:
                continue
            n = p.numel()
            m_ = sh.numel() if sh.numel() == n else n // self.world
            lo = 0 if sh.numel() == n else self.rank * m_
            s = self.st.get(i)
            if s is None:
                s = dict(m=torch.zeros(m_, dtype=torch.bfloat16, device=sh.device),
                         v=torch.zeros(m_, dtype=torch.bfloat16, device=sh.device))
                self.st[i] = s
            g16 = sh.float()[:m_].to(torch.bfloat16)
            s["m"].mul_(self.b1).add_(g16, alpha=1 - self.b1)
            s["v"].mul_(self.b2).addcmul_(g16, g16, value=1 - self.b2)
            upd = (s["m"].float() / bc1) / ((s["v"].float() / bc2).sqrt() + self.eps)
            seg = p.data.view(-1)[lo:lo + m_]
            if self.wd > 0:
                seg.mul_(1 - lr * self.wd)
            seg.add_(upd.to(p.dtype), alpha=-lr)
            sh.copy_(seg)
        if self.world > 1:
            for i, p in enumerate(self.ps):
                sh = self.shards.get(i)
                if sh is None:
                    continue
                if sh.numel() != p.numel():
                    dist.all_gather_into_tensor(p.data.view(-1), sh.to(p.dtype))
        self.shards.clear()


if args.zero2:
    Z2 = (Zero2AdamW(m.parameters(), world, rank, lr=args.lr) if args.opt == "adamw_bf16"
          else Zero2(m.parameters(), world, rank))
else:
    Z2 = None

DZMOD.set_sp_state(group=dist.group.WORLD if world > 1 else None, world=world, rank=rank)

precompute()
del vae
torch.cuda.empty_cache()
if is0:
    print("[mem] VAE released, %.1f GB in use before training" %
          (torch.cuda.memory_allocated() / 1e9), flush=True)

m.zero_grad(set_to_none=True); torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
hist = []
t_start = time.time()
for s in range(args.steps):
    t1 = time.time()
    b = make_batch(s)
    Z2.begin_step()
    out = m(b["x_in"], b["tt"], b["ta"], b["ctx"], seq_len=b["N"], clean_x=b["x_cl"],
            action=b["act"], state=b["st"], embodiment_id=b["emb"])
    v, a = out
    loss = (v.float() - b["tgt"]).pow(2).mean() + 0.1 * a.float().pow(2).mean()
    loss.backward()
    if world > 1:
        for q in m.parameters():
            if q.grad is not None:
                dist.all_reduce(q.grad, op=dist.ReduceOp.SUM)
    Z2.step(args.lr)
    m.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = time.time() - t1
    lv = float(loss)
    hist.append(dict(step=s, window=b["idx"], loss=lv, dt=dt, t=time.time() - t_start))
    if is0 and (s % args.log_every == 0 or s == args.steps - 1):
        print("[step %3d] loss=%.5f win=%3d %.1fs peak=%.1fGB" %
              (s, lv, b["idx"], dt, torch.cuda.max_memory_allocated() / 1e9), flush=True)
    if lv != lv:
        print("[FATAL] NaN at step %d" % s, flush=True)
        break

peak = torch.cuda.max_memory_allocated() / 1e9
res = dict(
    frames=args.frames, t_pix=train_ds.T_pix, steps_run=len(hist), steps_target=args.steps,
    lr=args.lr, opt=args.opt, zero2=args.zero2, world=world,
    train_windows=len(train_ds), holdout_episodes=list(holdout),
    train_episodes=[e["episode"] for e in tr_eps],
    peak_gb=peak, state_gb=Z2.state_bytes() / 1e9 if hasattr(Z2, "state_bytes") else 0.0,
    wall_s=time.time() - t_start,
    hist=hist)
if is0:
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    # 关键点：前 10 步均值 vs 后 10 步均值（窗口不同，单点噪声大）
    l = [h["loss"] for h in hist]
    print("\n[RESULT] steps=%d first10=%.5f last10=%.5f min=%.5f max=%.5f peak=%.1fGB "
          "step_time=%.1fs nan=%d" %
          (len(l), sum(l[:10]) / max(1, len(l[:10])), sum(l[-10:]) / max(1, len(l[-10:])),
           min(l), max(l), peak, res["wall_s"] / max(1, len(l)), sum(1 for x in l if x != x)),
          flush=True)
    print("[RESULT] saved " + args.out, flush=True)

# ---------------- 留出集评测（rank0 单卡，关掉 SP） ----------------
if args.eval_after and len(hist) > 0:
    DZMOD.set_sp_state(group=None, world=1, rank=0)
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    if is0:
        try:
            import eval_holdout
            vae = dl_droid.load_vae(ckpt_dir=args.ckpt_dir, dev=dev)   # 训练时释放过，重载 ~5s
            ev = eval_holdout.run_eval(m, vae, F_, dev, args, log=lambda s: print(s, flush=True))
        except Exception as e:
            import traceback; traceback.print_exc()
            ev = dict(error="%s: %s" % (type(e).__name__, str(e)[:500]))
        l = [h["loss"] for h in hist]
        ev.update(train_steps=len(hist),
                  train_loss_first10=sum(l[:10]) / max(1, len(l[:10])),
                  train_loss_last10=sum(l[-10:]) / max(1, len(l[-10:])),
                  train_peak_gb=peak, holdout_episodes=list(holdout),
                  frames=args.frames, lr=args.lr, opt=args.opt)
        os.makedirs(os.path.dirname(args.eval_out), exist_ok=True)
        json.dump(ev, open(args.eval_out, "w"), indent=1)
        print("[EVAL] saved " + args.eval_out, flush=True)
    dist.barrier()

dist.barrier(); dist.destroy_process_group()
