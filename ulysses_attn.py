#!/usr/bin/env python3
"""Ulysses 序列并行 attention：最小实现 + 正确性/吞吐自检。

用法：
  torchrun --nproc_per_node=4 ulysses_attn.py --mode check
  torchrun --nproc_per_node=4 ulysses_attn.py --mode bench --seq 29040 --heads 40 --dim 5120

原理（Ulysses / DeepSpeed-Ulysses）：
  输入 q,k,v: [B, S_local, H, D]，每 rank 持有一段连续 token。
  第 1 次 all-to-all：把 head 维切开、把 token 维拼起来
      [B, S_local, H, D] -> [B, S_full, H/P, D]
  局部 attention（每个 rank 只算自己那 H/P 个 head，但看全部 token）
  第 2 次 all-to-all：换回来
      [B, S_full, H/P, D] -> [B, S_local, H, D]
  通信量 = 2 × B·S·H·D × 2B（bf16），与 P 无关。
"""
import argparse, os, time
import torch
import torch.distributed as dist
import torch.nn.functional as F


def _init():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world


def all_to_all_4d(x, group, scatter_dim, gather_dim):
    """[B, S, H, D] 在 scatter_dim 上切 P 份、在 gather_dim 上拼回 P 份。"""
    world = dist.get_world_size(group)
    x = x.contiguous()
    shape = list(x.shape)
    assert shape[scatter_dim] % world == 0, (shape, scatter_dim, world)

    # [B,S,H,D] -> 把 scatter_dim 切成 world 份
    split = shape[scatter_dim] // world
    # 构造 [world, ...] 的第一维用于 all_to_all_single
    idx = [slice(None)] * x.dim()
    chunks = []
    for i in range(world):
        idx[scatter_dim] = slice(i * split, (i + 1) * split)
        chunks.append(x[tuple(idx)])
    # [world, B, S, H/P, D] -> [world, ...] 作为 all_to_all 的输入
    send = torch.stack(chunks, dim=0).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    # 在 gather_dim 上拼：recv 的最后一维顺序是各 rank 的块
    out = torch.cat([recv[i] for i in range(world)], dim=gather_dim)
    return out


def ulysses_attn(q, k, v, group, is_causal=False):
    """q,k,v: [B, S_local, H, D] -> [B, S_local, H, D]"""
    # heads 切分（scatter_dim=2），token 拼接（gather_dim=1）
    q = all_to_all_4d(q, group, scatter_dim=2, gather_dim=1)
    k = all_to_all_4d(k, group, scatter_dim=2, gather_dim=1)
    v = all_to_all_4d(v, group, scatter_dim=2, gather_dim=1)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=is_causal
    ).transpose(1, 2).contiguous()
    out = all_to_all_4d(out, group, scatter_dim=1, gather_dim=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="check", choices=["check", "bench"])
    ap.add_argument("--seq", type=int, default=29040)
    ap.add_argument("--heads", type=int, default=40)
    ap.add_argument("--dim", type=int, default=5120)
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    rank, world = _init()
    group = dist.group.WORLD
    B = 1
    H, D = args.heads, args.dim // args.heads
    S_local = args.seq // world
    assert args.seq % world == 0

    torch.manual_seed(0)
    dt = torch.bfloat16

    if args.mode == "check":
        # 小规模：用 float32 做精确对比（bf16 的容差太大，容易假通过）
        dt2 = torch.float32
        B2, S2, H2, D2 = 1, 64 * world, 8, 32
        S2_local = S2 // world
        q = torch.randn(B2, S2_local, H2, D2, device=rank, dtype=dt2)
        k = torch.randn(B2, S2_local, H2, D2, device=rank, dtype=dt2)
        v = torch.randn(B2, S2_local, H2, D2, device=rank, dtype=dt2)
        out_sp = ulysses_attn(q, k, v, group)

        # 参考：把全局 q,k,v 收齐后单卡算
        qg = [torch.empty_like(q) for _ in range(world)]
        kg = [torch.empty_like(k) for _ in range(world)]
        vg = [torch.empty_like(v) for _ in range(world)]
        dist.all_gather(qg, q.contiguous()); dist.all_gather(kg, k.contiguous()); dist.all_gather(vg, v.contiguous())
        qc = torch.cat(qg, dim=1); kc = torch.cat(kg, dim=1); vc = torch.cat(vg, dim=1)
        ref = F.scaled_dot_product_attention(
            qc.transpose(1, 2), kc.transpose(1, 2), vc.transpose(1, 2), is_causal=False
        ).transpose(1, 2)
        # 取本 rank 对应的那一段
        ref_local = ref[:, rank * S2_local:(rank + 1) * S2_local]
        err = (out_sp.float() - ref_local.float()).abs().max().item()
        rel = err / ref_local.abs().max().item()
        if rank == 0:
            print("[CHECK] world=%d  max_abs_err=%.3e  rel=%.3e  %s" %
                  (world, err, rel, "PASS" if rel < 1e-5 else "FAIL"), flush=True)
        dist.barrier()
        return

    # bench
    q = torch.randn(B, S_local, H, D, device=rank, dtype=dt)
    k = torch.randn(B, S_local, H, D, device=rank, dtype=dt)
    v = torch.randn(B, S_local, H, D, device=rank, dtype=dt)

    def local_dense():
        # 本地部分 attention（不看远端 token），代表 SP 前每卡的 attention 量
        return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))

    for _ in range(2):
        local_dense(); ulysses_attn(q, k, v, group)
    torch.cuda.synchronize(); dist.barrier()

    t0 = time.time()
    for _ in range(args.iters):
        local_dense()
    torch.cuda.synchronize(); dist.barrier()
    t_local = (time.time() - t0) / args.iters * 1e3

    t0 = time.time()
    for _ in range(args.iters):
        ulysses_attn(q, k, v, group)
    torch.cuda.synchronize(); dist.barrier()
    t_sp = (time.time() - t0) / args.iters * 1e3

    # 全序列单卡 attention 时间（只在 rank0 上跑，作为参照上限）
    if rank == 0:
        qg = torch.randn(B, args.seq, H, D, device=0, dtype=dt)
        kg = torch.randn(B, args.seq, H, D, device=0, dtype=dt)
        vg = torch.randn(B, args.seq, H, D, device=0, dtype=dt)
        for _ in range(2):
            F.scaled_dot_product_attention(qg.transpose(1, 2), kg.transpose(1, 2), vg.transpose(1, 2))
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.iters):
            F.scaled_dot_product_attention(qg.transpose(1, 2), kg.transpose(1, 2), vg.transpose(1, 2))
        torch.cuda.synchronize()
        t_single = (time.time() - t0) / args.iters * 1e3
        print("[BENCH] seq=%d heads=%d dim=%d  single-card full-attn=%.1fms" % (args.seq, args.heads, args.dim, t_single), flush=True)
        print("[BENCH] per-rank local-only attn=%.1fms   ulysses SP attn=%.1fms   speedup vs single=%.2fx" %
              (t_local, t_sp, t_single / t_sp), flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
