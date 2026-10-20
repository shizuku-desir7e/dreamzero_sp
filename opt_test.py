#!/usr/bin/env python3
"""优化器测试：SGD / AdamW（状态 bf16 + 1/P 分片）/ 完整版 Adafactor / AdaGrad-v。

背景
----
显存账（16.48B 参数，F=33，SP=4，ZeRO-2 梯度已 1/P）：
    权重 33.0 + 梯度 8.25 + 激活 14.65 = 55.9 GB（剩余约 27 GB 给优化器状态）

    fp32 Adam ：2×4B/param = 132 GB，分片 1/P 后 33 GB → 总 88.9 ✗ 放不下
    8bit Adam ：2×1B/param =  33 GB，分片 1/P 后 8.25 GB → 总 64.2 ✓（无 bitsandbytes，用 bf16 状态等价）
    fp32 AdamW（不分片） 187.9 ✗

本脚本实现（都按 1/P 分片，与 ZeRO-2 的梯度分片对齐）：
    sgd           无状态            → 55.9 GB
    adamw_bf16    exp_avg/sqrt bf16 → 16.5 GB 状态 → 72.4 GB
    adagrad_bf16  只存 v（bf16）    →  8.25 GB 状态 → 64.2 GB  ← 最接近"8-bit Adam"的显存
    adafactor     标量二阶矩 + 相对步长 + 更新裁剪 → <1 GB → 56.9 GB

用法：python -m torch.distributed.run --nproc_per_node=4 opt_test.py --frames 33 --steps 8
"""
import argparse, math, os, sys, time
from datetime import timedelta
import torch
import torch.distributed as dist

Z = os.environ.get("ZROOT", "/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengmenghan")
sys.path.insert(0, Z + "/dreamzero_sp/shim")
sys.path.insert(0, Z + "/dreamzero_sp/dreamzero")
import groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk as DZMOD
from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import CausalWanModel

ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, default=33)
ap.add_argument("--steps", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--opts", type=str, default="sgd,adamw_bf16,adagrad_bf16,adafactor")
ap.add_argument("--ckpt_dir", type=str, default=Z + "/model")
ap.add_argument("--no_load", type=int, default=0)
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
m.eval(); m.gradient_checkpointing = True
for p in m.parameters():
    p.requires_grad_(True)

if not args.no_load:
    from safetensors.torch import load_file
    import json
    idx = json.load(open(os.path.join(args.ckpt_dir, "model.safetensors.index.json")))
    for sh in sorted(set(idx["weight_map"].values())):
        sd = load_file(os.path.join(args.ckpt_dir, sh), device=dev)
        pmap = dict(m.named_parameters()); bmap = dict(m.named_buffers())
        for k, v in sd.items():
            kk = k[len("action_head.model."):]
            if not k.startswith("action_head.model."):
                continue
            tgt = pmap[kk] if kk in pmap else bmap.get(kk)
            if tgt is not None and tgt.shape == v.shape:
                tgt.data.copy_(v)
        del sd; torch.cuda.empty_cache()
if is0:
    print("[build] %.0fs" % (time.time() - t0), flush=True)
dist.barrier()

DZMOD.set_sp_state(group=dist.group.WORLD, world=world, rank=rank)
F, N = args.frames, args.frames * FSL
NB = (F - 1) // 2
torch.manual_seed(2024)
z0 = torch.randn(1, 16, F, H, W, device=dev, dtype=torch.bfloat16) * 0.5
eps = torch.randn(1, 16, F, H, W, device=dev, dtype=torch.bfloat16)
t01 = torch.rand(1, F, device=dev)
tt = t01 * 1000.0
zf = z0[:, :, :1].expand(-1, -1, F, -1, -1)
mask = torch.ones(1, 4, F, H, W, device=dev, dtype=torch.bfloat16); mask[:, :, 1:] = 0.0
tf_ = t01.view(1, 1, F, 1, 1).to(torch.bfloat16)
zv = ((1 - tf_) * z0 + tf_ * eps).to(torch.bfloat16)
x_in = torch.cat([zv, mask, zf], dim=1)
x_cl = torch.cat([z0, mask, zf], dim=1)
tgt = (eps - z0).float()
ctx = torch.randn(1, 512, 4096, device=dev, dtype=torch.bfloat16) * 0.1
act = torch.randn(1, NB * 24, 32, device=dev, dtype=torch.bfloat16) * 0.3
st = torch.randn(1, NB * 1, 64, device=dev, dtype=torch.bfloat16) * 0.3
ta = torch.randint(0, 1000, (1, NB * 24), device=dev)
emb = torch.zeros(1, dtype=torch.long, device=dev)

PS = [p for p in m.parameters() if p.requires_grad]
PIDX = {id(p): i for i, p in enumerate(PS)}
SHARDS = {}


def _hook(i):
    def hook(param):
        g = param.grad
        if g is None:
            return
        n = g.numel()
        assert n % world == 0, (n, world)
        out = torch.empty(n // world, dtype=g.dtype, device=g.device)
        dist.reduce_scatter_tensor(out, g.reshape(-1).contiguous(), op=dist.ReduceOp.SUM)
        SHARDS[i] = out
        param.grad = None
    return hook


for i, p in enumerate(PS):
    p.register_post_accumulate_grad_hook(_hook(i))


class ShardedSGD:
    name = "sgd"; state_dtype = None
    def __init__(self, ps, lr, **kw):
        self.ps, self.lr = ps, lr
    def step(self):
        for i, p in enumerate(self.ps):
            sh = SHARDS.get(i)
            if sh is None:
                continue
            n = p.numel(); m_ = n // world
            lo = rank * m_
            flat = p.data.view(-1)[lo:lo + m_]
            flat.add_(sh, alpha=-self.lr)
            sh.copy_(flat)
    def state_bytes(self):
        return 0


class ShardedAdamW:
    """exp_avg / exp_avg_sq 都用 bf16（等价于 8-bit Adam 的显存占用：2 字节/参数），并按 1/P 分片。"""
    name = "adamw_bf16"; state_dtype = torch.bfloat16
    def __init__(self, ps, lr, betas=(0.9, 0.999), eps=1e-8, wd=0.01, **kw):
        self.ps, self.lr, self.b1, self.b2, self.eps, self.wd = ps, lr, betas[0], betas[1], eps, wd
        self.st = {}
    def step(self):
        for i, p in enumerate(self.ps):
            sh = SHARDS.get(i)
            if sh is None:
                continue
            n = p.numel(); m_ = n // world; lo = rank * m_
            s = self.st.get(i)
            if s is None:
                s = dict(exp_avg=torch.zeros(m_, dtype=self.state_dtype, device=dev),
                         exp_avg_sq=torch.zeros(m_, dtype=self.state_dtype, device=dev), t=0)
                self.st[i] = s
            s["t"] += 1
            g = sh.float()
            s["exp_avg"].mul_(self.b1).add_(g.to(self.state_dtype), alpha=1 - self.b1)
            s["exp_avg_sq"].mul_(self.b2).addcmul_(g.to(self.state_dtype), g.to(self.state_dtype),
                                                   value=1 - self.b2)
            bc1 = 1 - self.b1 ** s["t"]; bc2 = 1 - self.b2 ** s["t"]
            upd = (s["exp_avg"].float() / bc1) / ((s["exp_avg_sq"].float() / bc2).sqrt() + self.eps)
            flat = p.data.view(-1)[lo:lo + m_]
            if self.wd > 0:
                flat.mul_(1 - self.lr * self.wd)
            flat.add_(upd.to(p.dtype), alpha=-self.lr)
            sh.copy_(flat)
    def state_bytes(self):
        return sum(s["exp_avg"].numel() * s["exp_avg"].element_size() * 2 for s in self.st.values())


class ShardedAdaGrad:
    """只存二阶矩（bf16，1/P）—— 最接近 8-bit Adam 的显存（8.25 GB/卡），但没有一阶动量。"""
    name = "adagrad_bf16"; state_dtype = torch.bfloat16
    def __init__(self, ps, lr, eps=1e-8, **kw):
        self.ps, self.lr, self.eps = ps, lr, eps
        self.st = {}
    def step(self):
        for i, p in enumerate(self.ps):
            sh = SHARDS.get(i)
            if sh is None:
                continue
            n = p.numel(); m_ = n // world; lo = rank * m_
            v = self.st.get(i)
            if v is None:
                v = torch.zeros(m_, dtype=self.state_dtype, device=dev)
                self.st[i] = v
            g = sh.float()
            v.addcmul_(g.to(self.state_dtype), g.to(self.state_dtype), value=1.0)
            upd = g / (v.float().sqrt() + self.eps)
            flat = p.data.view(-1)[lo:lo + m_]
            flat.add_(upd.to(p.dtype), alpha=-self.lr)
            sh.copy_(flat)
    def state_bytes(self):
        return sum(v.numel() * v.element_size() for v in self.st.values())


class ShardedAdafactor:
    """完整版 Adafactor 的三个稳定项：
       ① 相对步长 α_t = max(eps2, RMS(g)) · lr   （不是固定 lr）
       ② 更新裁剪：||u||_2 > d 时等比缩放
       ③ eps 放进 sqrt 里
    分片后无法保留矩阵的行/列结构（切片是扁平的），所以用**每参数标量**二阶矩（scalar Adafactor），
    状态 < 1 GB。这是 Adafactor 对 1-D 参数的退化形式，稳定性来自 ①②，不依赖分解。"""
    name = "adafactor"; state_dtype = torch.float32
    def __init__(self, ps, lr, beta2=0.999, eps=1e-30, eps2=1e-3, clip=1.0, **kw):
        self.ps, self.lr, self.b2 = ps, lr, beta2
        self.eps, self.eps2, self.clip = eps, eps2, clip
        self.st = {}
    def step(self):
        for i, p in enumerate(self.ps):
            sh = SHARDS.get(i)
            if sh is None:
                continue
            n = p.numel(); m_ = n // world; lo = rank * m_
            s = self.st.get(i)
            if s is None:
                s = dict(v=torch.zeros((), device=dev), t=0)   # 标量
                self.st[i] = s
            s["t"] += 1
            g = sh.float()
            g2m = g.pow(2).mean()
            s["v"] = self.b2 * s["v"] + (1 - self.b2) * g2m
            v = s["v"] / (1 - self.b2 ** s["t"])
            u = g / (v.sqrt() + self.eps)                     # ③
            un = u.norm()
            if un > self.clip:                                 # ②
                u = u * (self.clip / un)
            rms = g2m.sqrt()
            alpha = max(self.eps2, float(rms)) * self.lr       # ①  相对步长
            flat = p.data.view(-1)[lo:lo + m_]
            flat.add_(u.to(p.dtype), alpha=-alpha)
            sh.copy_(flat)
    def state_bytes(self):
        return sum(4 for _ in self.st.values())


OPTS = dict(sgd=ShardedSGD, adamw_bf16=ShardedAdamW,
            adagrad_bf16=ShardedAdaGrad, adafactor=ShardedAdafactor)

for name in args.opts.split(","):
    opt = OPTS[name](PS, args.lr)
    m.zero_grad(set_to_none=True); SHARDS.clear()
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    hist = []
    for s in range(args.steps):
        t1 = time.time()
        out = m(x_in, tt, ta, ctx, seq_len=N, clean_x=x_cl,
                action=act, state=st, embodiment_id=emb)
        loss = (out[0].float() - tgt).pow(2).mean() + 0.1 * out[1].float().pow(2).mean()
        loss.backward()
        opt.step()
        # 同步：把各自更新过的分片拼回全量参数
        for i, p in enumerate(PS):
            sh = SHARDS.get(i)
            if sh is not None:
                dist.all_gather_into_tensor(p.data.view(-1), sh)
        m.zero_grad(set_to_none=True); SHARDS.clear()
        torch.cuda.synchronize()
        hist.append((float(loss), time.time() - t1))
        if is0 and (s % 2 == 0 or s == args.steps - 1):
            print("[%s] step %2d loss=%.6f %.1fs peak=%.1fGB state=%.1fGB" %
                  (name, s, hist[-1][0], hist[-1][1],
                   torch.cuda.max_memory_allocated() / 1e9, opt.state_bytes() / 1e9), flush=True)
    if is0:
        los = [h[0] for h in hist]
        nan = any(x != x for x in los)
        print("[%s] 汇总: first=%.6f last=%.6f 降幅=%.1f%% nan=%s 中位步时=%.1fs" %
              (name, los[0], los[-1], 100 * (1 - los[-1] / los[0]), nan,
               sorted(h[1] for h in hist)[len(hist) // 2]), flush=True)
    del opt
    torch.cuda.empty_cache()

dist.barrier(); dist.destroy_process_group()
