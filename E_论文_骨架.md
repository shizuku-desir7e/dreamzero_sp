# E 线论文骨架（诚实定位版，2026-10-20）

> 工作标题候选：
> - **Making Action-Conditioned Video Diffusion Trainable at 33 Frames on 4×80GB:
>   An Engineering Study of Sequence Parallelism for Wan2.1-14B**
> - （中文：把 action-conditioned 视频扩散的 33 帧配置塞进 4 张 80GB 卡：一个序列并行的工程研究）
>
> **定位（必须先读这段，全文的口径都由它决定）**：
> 本文**不提出新的序列并行算法**。Ulysses / ring-attention / DeepSpeed-Ulysses 都是已有方法。
> 本文记录的是：把一个已有方法**适配到 action-conditioned 视频扩散**时，
> 那些通用 SP 论文不会覆盖的工程障碍，以及每一项的量化代价。
> 这类"适配税"在系统论文里通常被略过，但它决定了别人能不能复现。

---

## 1. Introduction（诚实定位版，可直接用）

### 1.1 我们做了什么
在 4×A100-80G 上用 Ulysses 序列并行训练 Wan2.1-14B（DreamZero 的 action-conditioned 视频扩散主干）。
**33 帧（官方配置）在纯数据并行下放不下**；加入 SP=4 后可以训练，且同配置步时快 2.46×。

### 1.2 我们没做什么（主动写清楚）
- 不提出新的并行算法。Ulysses 的 all-to-all 换头/换 token 是 2023 年的工作。
- 不声称 SP 本身有 novelty。novelty 在**适配层**：一个 action-conditioned 视频扩散模型有三个
  通用 SP 论文不会遇到的约束（见 §1.3），每一个都会让"直接套 Ulysses"得到**静默错误**的结果。
- 不声称这是最快的方案。4 卡规模下 all-to-all 开销占 attention 的 ~28%，
  更大的集群上这个比例会变（我们只测到 P=4）。

### 1.3 三个"通用 SP 论文不覆盖"的障碍（本文真正的贡献）

| # | 障碍 | 直接套 Ulysses 会怎样 | 我们的处理 | 代价 |
|---|---|---|---|---|
| 1 | 序列里有 **400 个 action/state token** 拼在尾部，attention 用 `(s-400)//2` 切 clean/noisy | all-to-all 会把它切开，`(s-ar)//2` 在每张卡上算出不同的切点 → **静默算错** | 拆成三段（首帧 / body / action），段内连续 ⇒ 零复制，全局序天然成立 | 每层 a2a 从 16 次增到 20 次（+5.5GB 瞬时显存） |
| 2 | **block-causal mask 以整帧（880 token）为界** | `N/P` 不是 880 的倍数时切在帧中间 → mask 边界错 → **NaN 或静默错** | 按段均分，保证每个 a2a chunk 整帧对齐 | 段长必须能被 P 整除（880=4×220、400=4×100） |
| 3 | `all_to_all_single` / `all_gather_into_tensor` **没有 autograd** | q/k/v 梯度**精确为 0**，而且 **backward 不报错**；因 action 输出不走 gather，其他参数仍有非零梯度 → 极易误判"能训" | 三种通信全部包 `autograd.Function`，反向用**逆置换**（不是 reduce_scatter） | 需要显式实现，无法靠 `torch.distributed.nn` 现成函数 |

**第 3 条是最有价值的**：它是一个"看起来在工作"的失败模式。我们最初的实现 loss 正常下降、
多数参数梯度非零，但 `self_attn.q.weight` 的梯度精确为 0。只有逐参数对 DP 校验才能发现。

### 1.4 贡献清单
1. **可用性边界**：纯 DP 在 4×80GB 上最多 9 帧；SP=4 达到官方 33 帧（79.9 GB / 20.89 s/step）。
2. **三个适配障碍** + 每个的量化代价（§1.3 表）。
3. **正确性协议**：一套能抓住"静默错误"的等价性检查
   （必须断言参考输出非零；必须逐参数比梯度；复制输出的 loss 要按 1/P 缩放）。
4. **诚实的 baseline 对比**：DP+ckpt、DP+offload、以及 FSDP 的算术分析（§5）。

## 2. Setup
- 模型：Wan2.1-14B（16.48B，bf16 33.0 GB），40 层 / 40 head / dim 5120 / 每帧 880 token。
- 官方配置：`num_frames=33`、`use_gradient_checkpointing=true`（`config.json` 原文）。
- 硬件：4×A100-80G（单卡 85.9 GB，扣 CUDA context 约 82.9 GB 可用）。
- 权重：官方 step-100000 checkpoint（1317 个张量与模型逐一对上，shape mismatch = 0）。

## 3. 测量：为什么需要 SP
| 项 | 数值 |
|---|---|
| attention 占单层墙钟 | 60.6% |
| 4 卡 Ulysses attention 单算子 | 85.8ms → 30.0ms（2.86×） |
| all-to-all 占 attention 时间 | 28% |
| 无 ckpt 时单层激活 | ≈ 470 KB / token |
| 权重 + 梯度地板 | 33.0 + 33.0 = 66.0 GB |

## 4. 主结果
（表见 `E_结果_2026-10-20.md` §4）

## 5. Baselines（三个必须回答的问题，都有实测/算术答案）

| 配置 | 峰值 | step |
|---|---|---|
| DP+ckpt F=9 | 75.2 GB | 11.94s |
| DP+ckpt F=17 / F=33 | OOM @82.0 / 82.1 | — |
| **DP+ckpt+activation offload F=9** | **75.6 GB** | **18.08s** |
| DP+ckpt+activation offload F=17 / F=33 | **仍 OOM** | — |
| SP=4 F=33 | 79.9 GB | 20.89s |

1. **DP 真的不行吗？** → F=17 差约 1 GB、F=33 差 13 GB。
2. **DP + activation offload 呢？** → **打不到点上**：换出 423 个 saved tensor，
   峰值 75.2→75.6 GB（没降），代价 1.5× 时间。因为开 ckpt 后 saved 的只有 40 个 block 输入
   （≈6.6 GB），峰值由**块内 recompute 瞬时占用 + 33 GB 梯度**决定。
   这是一个反直觉但可解释的负面结果，值得单独写一段。
3. **FSDP 呢？** → 算术上**能放下**（≈50 GB），但 FSDP 切的是状态不是激活，
   激活仍是全序列 ⇒ **无法再往上加帧**；且每层要一次≈0.8 GB 权重 all-gather。
   SP 与 FSDP 正交，组合是明确的后续工作。

## 5.1 真权重短程训练（堵"语义没验"）
真权重 + Wan2.1 i2v 标准输入（16 噪声 + 4 mask + 16 首帧 = in_dim 36）+ flow-matching 目标
（z_t=(1-t)z0+tε，v*=ε−z0），SP=4、F=9、SGD lr=4e-3、**50 步**：

```
loss   1.7582 → 1.3393 → 1.1689 → 1.0819 → 1.0304 → 0.9993   (降 43.2%)
gradn  16.02  → 5.74   → 4.14   → 3.46   → 2.87            (单调降，无 NaN)
每步 4.9–5.0 s
```

DP 对照（同初始权重/同 batch/10 步）：step0 差 0.02%、step5 差 0.14% —— 数值等价。
**限制**：z0 是高斯 latent 而非 VAE 编码的真实视频（无视频数据），
所以证明的是优化稳定性与 DP/SP 一致性，不是官方 loss 复现。这句必须写进 limitations。

## 6. 限制（直接写进 limitations）
- 4 卡，只有 P=4 一个点（P=8 的拐点未测）。
- 训练曲线用的是高斯 latent 而非 VAE 编码的真实视频（无视频数据）；
  证明的是"真权重 + 真目标形式下的优化稳定性与 DP/SP 一致性"，不是官方 loss 复现。
- 未与 FSDP/ZeRO 组合。
- 未做端到端推理质量评估（FVD/成功率）。

## 7. 如果只能写一页
> 33 帧的 action-conditioned 视频扩散在 4×80GB 上训不了，原因是激活，不是参数量。
> 已有的 Ulysses 能解决，但直接套会得到**静默错误**的结果——三个具体障碍（action token 的位置、
> 帧对齐、集合通信没有 autograd）各有一个代价，我们把它们量化了。
> 做完适配后：9 帧 → 33 帧，同配置 2.46×，梯度与 DP 相差 0.27%。
