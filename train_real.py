#!/usr/bin/env python3
"""真权重 + 真 flow-matching 目标，跑 N 步，证明 loss 下降、无 NaN、DP 与 SP 同步。

输入布局（这是 Wan2.1 i2v 的标准布局，in_dim=36 的来源）：
    x_in = cat([ z_t (16ch, 加噪 latent) , mask (4ch) , first_frame (16ch) ], dim=1)
目标（flow matching / 直线插值速度场）：
    z_t = (1-t)*z0 + t*eps
    v*  = eps - z0
    loss = MSE(model(x_in, t*1000, ...), v*)
teacher forcing：clean_x 用同一布局但 latent 换成 z0（这是 _forward_train 的 is_tf 分支）。

诚实说明：z0 是高斯采样而不是 VAE 编码的真实视频（服务器上没有视频数据）。
所以这条曲线证明的是**在真权重、真目标形式、真 SP 路径下的优化稳定性与 DP/SP 一致性**，
不是复现官方 dataloader 的 loss 曲线。

用法：bash run_train.sh
"""
import argparse, json, os, sys, time
from datetime import timedelta
import torch
import torch.distributed as dist

Z = os.environ.get("ZROOT", "/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengmenghan")
sys.path.insert(0, Z + "/dreamzero_sp/shim")
sys.path.insert(0, Z + "/dreamzero_sp/dreamzero")
import groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk as DZMOD
from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import CausalWanModel

ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, default=9)
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--clip", type=float, default=1.0)
ap.add_argument("--ckpt_dir", type=str, default=Z + "/model")
ap.add_argument("--prefix", type=str, default="action_head.model.")
ap.add_argument("--order", type=str, default="4,1")
ap.add_argument("--dp_steps", type=int, default=10)
ap.add_argument("--zero2", type=int, default=0)
ap.add_argument("--no_load", type=int, default=0)
ap.add_argument("--frames_list", type=str, default="")     # 先 SP 后 DP
args = ap.parse_args()

dist.init_process_group("nccl", timeout=timedelta(seconds=900))
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank); dev = "cuda:%d" % rank; is0 = rank == 0

H, W, FSL = 40, 88, 880
CFG = dict(model_type="ti2v", patch_size=(1, 2, 2), frame_seqlen=FSL, text_len=512,
           in_dim=36, dim=5120, ffn_dim=13824, freq_dim=256, text_dim=4096, out_dim=16,
           num_heads=40, num_layers=40, num_frame_per_block=2, action_dim=32,
           num_action_per_block=24, num_state_per_block=1, max_state_dim=64,
           max_num_embodiments=32, hidden_size=1024, concat_first_frame_latent=True,
           qk_norm=True, cross_attn_norm=True, max_chunk_size=4, sink_size=0)

t0 = time.time()
torch.manual_seed(0); torch.set_default_dtype(torch.bfloat16)
m = CausalWanModel(**CFG).to(dev); torch.set_default_dtype(torch.float32)
m.eval(); m.gradient_checkpointing = True      # 官方 config 就是 True
for p in m.parameters():
    p.requires_grad_(True)

if not args.no_load:
    from safetensors.torch import load_file
if not args.no_load:
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
    print("[load] %.0fs" % (time.time() - t0), flush=True)
dist.barrier()

# ---- 固定 batch：所有对比必须喂一模一样的数据 ----
def make_batch(F):
    N = F * FSL
    NB = (F - 1) // 2
    torch.manual_seed(2024)
    z0 = torch.randn(1, 16, F, H, W, device=dev, dtype=torch.bfloat16) * 0.5
    eps = torch.randn(1, 16, F, H, W, device=dev, dtype=torch.bfloat16)
    t01 = torch.rand(1, F, device=dev)
    tt = t01 * 1000.0
    zf = z0[:, :, :1].expand(-1, -1, F, -1, -1)
    mask = torch.ones(1, 4, F, H, W, device=dev, dtype=torch.bfloat16)
    mask[:, :, 1:] = 0.0
    tf_ = t01.view(1, 1, F, 1, 1).to(torch.bfloat16)
    zv = ((1 - tf_) * z0 + tf_ * eps).to(torch.bfloat16)
    return dict(
        N=N, tt=tt,
        x_in=torch.cat([zv, mask, zf], dim=1),
        x_cl=torch.cat([z0, mask, zf], dim=1),
        tgt=(eps - z0).float(),
        ctx=torch.randn(1, 512, 4096, device=dev, dtype=torch.bfloat16) * 0.1,
        act=torch.randn(1, NB * 24, 32, device=dev, dtype=torch.bfloat16) * 0.3,
        st=torch.randn(1, NB * 1, 64, device=dev, dtype=torch.bfloat16) * 0.3,
        ta=torch.randint(0, 1000, (1, NB * 24), device=dev),
        emb=torch.zeros(1, dtype=torch.long, device=dev))


globals().update(make_batch(args.frames))

class Zero2:
    """ZeRO-2 式梯度分片：backward 中逐参数 reduce_scatter 成 1/P，立刻释放 p.grad。

    为什么必须在 backward 里做：等 backward 跑完再分片，33 GB 已经全部分配出来了，
    分片就没意义了。`post_accumulate_grad_hook` 让我们在该参数梯度就绪的瞬间归约 + 释放。

    更新：每卡只更新自己那 1/P 片；然后用 `all_gather_into_tensor` **就地**写回全量参数
    （out 就是 p.data.view(-1)，不需要额外 buffer）。
    """

    def __init__(self, params, world, rank):
        self.ps = [p for p in params if p.requires_grad]
        self.world, self.rank = world, rank
        self.shards = {}
        self.full = 0            # 可分片的参数个数
        self.partial = 0         # 不能整除、退回全量的参数个数
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
                dist.reduce_scatter_tensor(out, g.reshape(-1).contiguous(),
                                           op=dist.ReduceOp.SUM)
                self.shards[i] = out
                self.full += 1
            else:
                self.shards[i] = g.reshape(-1).clone()
                self.partial += 1
            param.grad = None          # <- 关键：立刻释放，33GB 不会累积
        return hook

    @torch.no_grad()
    def step(self, lr):
        for i, p in enumerate(self.ps):
            sh = self.shards.get(i)
            if sh is None:
                continue
            n = p.numel()
            flat = p.data.view(-1)
            if sh.numel() == n:
                flat.add_(sh.to(p.dtype), alpha=-lr)
                sh.copy_(flat)
            else:
                lo = self.rank * (n // self.world)
                flat[lo:lo + n // self.world].add_(sh.to(p.dtype), alpha=-lr)
                # 关键：把更新后的分片放回 sh，否则下面 all_gather 会把"梯度"当参数写回去
                sh.copy_(flat[lo:lo + n // self.world])
        if self.world > 1:
            for i, p in enumerate(self.ps):
                sh = self.shards.get(i)
                if sh is None:
                    continue
                n = p.numel()
                if sh.numel() != n:
                    dist.all_gather_into_tensor(p.data.view(-1), sh.to(p.dtype))
        self.shards.clear()


class Adafactor:
    """精简版 Adafactor：矩阵用行/列二阶矩分解，1-D 用全二阶矩。
    相比 Adam 省掉 2x fp32 状态（14B 要 132GB，放不下），比 SGD 的步长更有意义。"""

    def __init__(self, params, lr=1e-4, beta2=0.999):
        self.ps = [p for p in params if p.requires_grad]
        self.lr, self.b2 = lr, beta2
        self.st = {}

    def zero_grad(self):
        for p in self.ps:
            p.grad = None

    @torch.no_grad()
    def step(self):
        tot = 0.0
        for i, p in enumerate(self.ps):
            g = p.grad
            if g is None:
                continue
            g32 = g.float()
            tot += float(g32.pow(2).sum())
            s = self.st.setdefault(i, {})
            if p.dim() >= 2:
                r = g32.pow(2).mean(dim=-1).add_(1e-30)
                c = g32.pow(2).mean(dim=-2).add_(1e-30)
                s["r"] = r if "r" not in s else self.b2 * s["r"] + (1 - self.b2) * r
                s["c"] = c if "c" not in s else self.b2 * s["c"] + (1 - self.b2) * c
                v = s["r"].unsqueeze(-1) * s["c"].unsqueeze(-2) / (s["r"].mean() + 1e-30)
            else:
                g2 = g32.pow(2).add_(1e-30)
                s["v"] = g2 if "v" not in s else self.b2 * s["v"] + (1 - self.b2) * g2
                v = s["v"]
            p.data.add_(g32 / v.sqrt().add_(1e-30), alpha=-self.lr)
        return tot ** 0.5


OPT = Adafactor(m.parameters(), lr=args.lr)
Z2 = Zero2(m.parameters(), world, rank) if args.zero2 else None

_needs_bak = len(args.order.split(",")) > 1
bak = {k: v.detach().to("cpu", copy=True) for k, v in m.state_dict().items()} if _needs_bak else None


def train(p, steps):
    DZMOD.set_sp_state(group=dist.group.WORLD if p > 1 else None,
                       world=p, rank=rank if p > 1 else 0)
    m.zero_grad(set_to_none=True); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    hist = []
    for s in range(steps):
        t1 = time.time()
        out = m(x_in, tt, ta, ctx, seq_len=N, clean_x=x_cl,
                action=act, state=st, embodiment_id=emb)
        v, a = out
        # flow-matching 主目标；action 头也参与（它是同一个 head 的一部分）
        loss = (v.float() - tgt).pow(2).mean() + 0.1 * a.float().pow(2).mean()
        loss.backward()
        if p > 1:                                   # SP：参数梯度必须跨卡求和
            for q in m.parameters():
                if q.grad is not None:
                    dist.all_reduce(q.grad, op=dist.ReduceOp.SUM)
        if Z2 is not None:
            Z2.step(args.lr)
            gn = torch.zeros((), device=dev)
            m.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            dt = time.time() - t1
            lv = float(loss)
            hist.append((lv, float(gn), dt))
            if is0 and (s % 1 == 0):
                print("[%s step %2d] loss=%.6f %.2fs peak=%.1fGB (z2 full=%d partial=%d)" %
                      ("SP" if p > 1 else "DP", s, lv, dt,
                       torch.cuda.max_memory_allocated() / 1e9, Z2.full, Z2.partial), flush=True)
            continue
        if args.clip > 0:
            gn = torch.nn.utils.clip_grad_norm_(m.parameters(), args.clip)
        else:
            gn = torch.sqrt(sum((q.grad.float().pow(2).sum() for q in m.parameters()
                                 if q.grad is not None), torch.zeros((), device=dev)))
        with torch.no_grad():
            for q in m.parameters():
                if q.grad is not None:
                    q.add_(q.grad, alpha=-args.lr)
        m.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.time() - t1
        lv = float(loss)
        hist.append((lv, float(gn), dt))
        if is0 and (s % 5 == 0 or s == steps - 1):
            print("[%s step %2d] loss=%.6f gradnorm=%.4f %.2fs" %
                  ("SP" if p > 1 else "DP", s, lv, float(gn), dt), flush=True)
    peak = torch.cuda.max_memory_allocated() / 1e9
    return hist, peak


_frames = [int(v) for v in (args.frames_list or str(args.frames)).split(",")]
for _F in _frames:
    if _F != args.frames:
        globals().update(make_batch(_F))
        if is0:
            print("\n=== F=%d ===" % _F, flush=True)
    for p in [int(v) for v in args.order.split(",")]:
        if bak is not None:
            with torch.no_grad():
                for k, v in m.state_dict().items():
                    v.copy_(bak[k])
        torch.cuda.empty_cache()
        hist, peak = train(p, args.steps if p > 1 else args.dp_steps)
    if is0:
        print("[%s] first=%.6f last=%.6f 降幅=%.2f%% peak=%.1fGB nan=%d" %
              ("SP" if p > 1 else "DP", hist[0][0], hist[-1][0],
               100 * (1 - hist[-1][0] / hist[0][0]), peak,
               sum(1 for h in hist if h[0] != h[0])), flush=True)
        print("     loss 轨迹: " + " ".join("%.4f" % h[0] for h in hist[::10]), flush=True)
    dist.barrier()

dist.barrier(); dist.destroy_process_group()
