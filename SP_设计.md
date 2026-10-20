# E：Action-conditioned 世界模型的 objective-aware 序列并行

> 2026-10-19 立项。基于 DreamZero（NVIDIA GEAR，Apache-2.0）开源训练代码。
> 状态：**可行性已验证到"模型能在 4×A100 上构造"**，SP 实现未开始。

## 0. 一句话

DreamZero 训练用的是 **DeepSpeed ZeRO-2（纯数据并行）**，每个 rank 都要装下完整序列的激活；
而它的序列是 `[clean history | action | noisy target]` 的 **block-causal** 结构。
我们把序列并行（SP）加进去，并解决一个别人没处理的问题：**在 block-causal + 三流（clean / action / noisy）的约束下，token 该怎么切。**

## 1. 为什么这是空白（已验证）

代码检索（`grep -rn "ulysses|sequence_parallel|ring_attention|all_to_all"`）在 DreamZero 仓库内**零命中**。
现有的 SP 工作：
- **LongVILA MM-SP**：两阶段分片（vision tower 按帧、LLM 按 token），是 **LLM**，不是 video DiT
- **LongLive-2.0 Balanced SP**：处理 **clean/noisy 两流**（AR 视频生成），**没有 action 条件**
- **DualParal / FCP / Megatron DCP**：通用 context parallel，不感知"哪段是 clean、哪段是 noisy、action 属于哪个 block"

⇒ **action-conditioned 的三流分片无人处理。** 这是 E 的立足点。

## 2. DreamZero 的序列结构（从代码读出，非猜测）

`groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py`：

```
forward(x, timestep, timestep_action, context, seq_len,
        clean_x=..., action=..., state=..., embodiment_id=...)
```

- `x`：**noisy target** 视频 `[C_in, F, H, W]`
- `clean_x`：**clean history**（作为 KV 条件）
- `action` / `state`：动作块与本体状态
- 块粒度：`num_frame_per_block=2`、`num_action_per_block=24`、`num_state_per_block=1`
- 注意力：`CausalWanSelfAttention` 里**手写的 block-wise 循环**，按 `clean_image / noisy_image / noisy_action` 三段分别调 `flash_attention`

⇒ 每个 block 的 token 组成是 **2 帧视频 + 24 action + 1 state**，且只能看到之前的 block（causal）。

## 3. 为什么 naive SP 会不均（这是要证明的核心命题）

**(a) block-causal 本身就让计算量随位置增长。**
第 i 个 block 的 Q 要 attend 到前 i 个 block 的 KV ⇒ FLOPs ∝ i。
按 token 均分到 4 卡（前 1/4 token 给 rank0 …）⇒ **rank3 的 attention FLOPs 约为 rank0 的 7 倍**（末块 vs 首块）。

**(b) 三流让切分不能"随便切"。**
- `clean_x` 是**只读 KV**：切了之后每个 rank 只有一段，Q 想 attend 全部 clean 就得 all-gather
- `action` token 是**每 block 24 个**，与它的 2 帧视频强耦合；切在 block 中间会把一对 (action, frame) 拆散
- 序列开头是 clean、结尾是 noisy ⇒ **按 token 均分等于把"便宜的 KV 段"给 rank0、"昂贵的 Q 段"给 rank3**

**(c) 4 卡的 all-to-all 开销不能忽略。**
Ulysses 每层两次 all-to-all；33 帧 × 3 视角 × 220 token ≈ 21.8k token、dim 3072 时，
每层 all-to-all 的数据量 ≈ 21780 × 3072 × 2B ≈ 134MB，4 卡 All-to-All 在 NVLink 上约 0.1–0.2ms/层，
30 层 ⇒ 3–6ms/step，相对 step time（数百 ms）可接受，但**必须实测**。

## 4. 三种分片方案（实验臂）

| 臂 | 切法 | 预期 |
|---|---|---|
| **S1: token-level Ulysses** | 序列按 token 均分，每层 all-to-all 交换 KV | 负载最不均（(a)），通信最大 |
| **S2: block-level round-robin** | 完整 block 轮流分给各 rank，clean 段广播 | 保证 (action, frame) 不拆，但 FLOPs 仍不均 |
| **S3: balanced（本方案的候选）** | 按**每 block 的累计 attention FLOPs** 做加权切分（前面多给 token、后面少给），且约束 `clean/noisy 配对同 rank` | 理论 FLOPs 方差最小 |

**S3 的关键设计**：切点不在 token 维度等分，而是让每个 rank 的 `Σ_i (i · tokens_in_rank_i)` 相等 ——
即"前缀和均衡"而不是"计数均衡"。这是把 LongLive 的"按流配对"扩展成"按计算量配对"。

## 5. 测量什么

| 指标 | 定义 | 为什么重要 |
|---|---|---|
| **imbalance ratio** | `max_rank_FLOPs / mean_rank_FLOPs` | 直接度量"不均" |
| **step time** | 单次 fwd+bwd 墙钟 | 真实收益 |
| **scaling efficiency** | `4卡吞吐 / (1卡吞吐×4)` | SP 是否值得 |
| **loss 一致性** | SP 与 DP 在同一 batch 上的 loss 差 | 验证语义未被改坏（**必须有**） |

## 6. 预写判据（到点只对数字）

**Go**：S3 的 step time 比 DP **快 ≥1.3×**，且 imbalance ratio 从 >2.0 降到 <1.2，且 loss 相对差 <1e-3。
**Partial**：step time 1.1–1.3× ⇒ 可写"4 卡规模下 SP 的收益边界"，仍是系统报告的有效结论。
**No-Go**：<1.1× ⇒ 如实报"4 卡 + 21.8k token 下 SP 不划算"，并给出临界规模估算（这个负面结论对 ASC 报告同样有用）。

## 7. 已验证的事实（2026-10-19）

| 项 | 结果 |
|---|---|
| 代码 | `dreamzero0/dreamzero` clone 成功（9.2MB，Apache-2.0），训练入口 `scripts/train/droid_training_wan22.sh` |
| 训练栈 | HF Trainer + **DeepSpeed ZeRO-2**（`groot/vla/configs/deepspeed/zero2.json`），`torch.distributed.run` |
| 依赖 | 服务器 env 有 torch 2.8 / deepspeed 0.18.2 / **flash_attn 2.8.3** / transformers 4.52；缺 hydra、lerobot（训练框架需要，SP 探针不需要） |
| diffusers | 装 0.41 会与 pinned huggingface_hub 冲突 ⇒ **写了 40 行 shim**（`dreamzero_sp/shim/diffusers/`），只提供 ConfigMixin/ModelMixin，不污染 LightningRL env |
| 模型构造 | ✅ `CausalWanModel` 小配置（dim=512, 4 层）**0.5s 构造成功，27.6M 参数** |
| 算力账 | 33 帧 × 220 token/帧 = 7,260 token；× 3 视角 ≈ 21.8k ⇒ **attention/MLP FLOPs 比 ≈ 0.8–1.0**（注意力与 MLP 同量级，SP 有空间） |
| **权重（已找到，无需下载）** | `$Z/model/` = **DreamZero-DROID checkpoint，45.85GB，10 分片，`global_step100000`**，**自包含**：`action_head.vae.*`(194) + `action_head.text_encoder.*`(242) + `action_head.image_encoder.*`(含 patch_embedding) ⇒ VAE / T5 / CLIP 全在里面 |
| **真实模型维度** | ⚠️ **不是 5B，是 Wan2.1-14B**：`dim=5120 / num_layers=40 / num_heads=40 / ffn_dim=13824`，`frame_seqlen=880`，`in_dim=36`，`patch_size=(1,2,2)` |
| **模型构造实测** | random-init 构造 **16.48B 参数 / 33.0GB 显存 / 174s**（`probe_step1.py`） |
| **token 账（修正）** | 880 tok/帧（4 视角 × 220）× 32–33 帧 ≈ **28,160–29,040 token**；**attention/MLP FLOPs = 1.40**（注意力主导，SP 有真实空间）；单次 forward ≈ **1.18 PFLOPs** |
| **显存实测** | forward 尚未跑通（见下），但**峰值已到 49.1GB**（33GB 权重 + 16GB 中间量） |
| 数据 | 本地 `droid_lerobot` 仅 70MB / 16 episodes（完整集 131GB，不下载）；系统测量足够 |

### ⚠️ 结论因此升级：SP 从"可选优化"变成"必需"
14B 权重 **33GB/卡**（ZeRO-2 不切权重）+ 28k token 的完整激活 ⇒ 4 卡 DP 每卡都要装全量激活；
而 SP 把激活切 4 份（每卡 7k token）。**这不是"能不能更快"的问题，是"能不能跑"的问题。**

## 7.5 ⭐ 实测 Amdahl 上限（2026-10-19，`bench_amdahl.py`）

真实形状（N=29,040, heads=40, head_dim=128, dim=5120, ffn=13824），A100-80G，bf16：

| 组件 | 每层墙钟 |
|---|---|
| **attention (SDPA)** | **85.73 ms** |
| MLP | 32.10 ms |
| QKVO 投影 | 23.68 ms |
| **layer total** | **141.51 ms** |

**attention share = 60.6%** ⇒ Amdahl 上限（只切 attention）：

| P | 理想加速 |
|---|---|
| 2 | 1.43× |
| **4** | **1.83×** |
| 8 | 2.13× |

若 SP 同时切 attention + MLP + 投影（sequence parallel 常规做法）⇒ 上限接近 P（4×）。

**通信开销实测估算**：Ulysses 每层 all-to-all payload = N×dim×2B = **297 MB**，40 层 = **11.9 GB**；
A100 NVLink 100–200 GB/s ⇒ **79–149 ms**，相对 40 层 attention 总时长（3.43s）只占 **2–4%**。
⇒ **通信不是瓶颈，1.83× 是可信的目标。**（这是上限，实际实现会打折。）

**这个数字把 E 的定位钉死了**：不是"可能有用"，是"P=4 理论 1.83×，通信只吃掉 2–4%"。

---

## 7.6 ✅ 14B 单卡 forward 实测跑通（2026-10-19 23:55）

```
[build] 173.0s  params=16.48B  GPU=33.0GB
[shape] F=33 H=40 W=88 -> tok/frame=880  N=29040
[ok]    forward 15.33s  peak=50.1GB   out=(1,16,33,40,88)
```

| 指标 | 实测 |
|---|---|
| 单次 forward | **15.33 s** |
| 峰值显存（forward only） | **50.1 GB** |
| 权重 | 33.0 GB（ZeRO-2 下每卡常驻） |
| 激活 | ≈ 17 GB（29k token × 40 层） |
| 有效算力 | 1.18 PFLOPs / 15.33s ≈ **77 TFLOPS**（A100 bf16 峰值 312，利用率 25%） |

**⇒ 关键推论（这是 E 的立项依据）**：
- **forward-only**：单卡 50GB < 80GB，能跑；
- **训练 fwd+bwd**：需要在反向时保留各层激活 ≈ 17GB × 3–4 ≈ 50–70GB，叠加 33GB 权重 ⇒ **83–103GB，OOM**；
- **SP 切 4 份**：激活降到 ~4–5GB/卡 ⇒ **37–38GB/卡，余量充足**；
- 且 SP 让"更长序列 / 更大 batch"成为可能（当前 33 帧已是 29k token）。

### ⚠️ 复现时必须打的一个补丁（上游 bug）
`CausalWanAttentionBlock.forward` 返回 `(outputs, kv_cache)`，但 `_forward_train` 的
**非 gradient-checkpointing 分支**写的是 `x = block(x, **kwargs)`（未解包）⇒ 第二个 block 收到 tuple，
报 `'tuple' object has no attribute 'shape'`。
官方训练脚本因为开了 gradient checkpointing（那条分支有 `outputs, kv_cache = module(...)`）所以没暴露。
本探针在 `probe_step2.py` 里 monkey-patch 解包（等价于 checkpointing 分支的语义）。
**若我们自己实现 SP，改 attention 时不要再引入同类解包不一致。**

---

## 7.7 🔴 决定性结果：**4×A100-80G 上 DP 根本跑不动**（2026-10-20 00:10）

### 无 gradient checkpointing（`probe_step3.py`）

| F（帧） | N | 峰值 | 结果 |
|---|---|---|---|
| 9 | 7,920 | 83.6 GB | **OOM** |
| 17 | 14,960 | 83.6 GB | **OOM** |
| 25 | 22,000 | 84.3 GB | **OOM** |
| **33（官方配置）** | **29,040** | 84.1 GB | **OOM** |

**连最短的 9 帧都跑不动**（A100 可用 79.25 GB）。权重常驻 33.0 GB。

### 加 gradient checkpointing（`probe_step4.py`）

| F | N | 峰值 | fwd | bwd | 结果 |
|---|---|---|---|---|---|
| **9** | 7,920 | **76.6 GB** | 5.3 s | 13.4 s | ✅ 勉强（只剩 2.6 GB） |
| 17 | 14,960 | 79.7 GB | — | — | **OOM** |
| 25 | 22,000 | 70.4 GB | — | — | **OOM** |
| **33（官方）** | **29,040** | 70.8 GB | — | — | **OOM** |

### ⇒ 结论（这是论文的 motivation 核心）

> **在 4×A100-80G 上，DreamZero（14B / 33 帧 / 29k token）的训练无论开不开 checkpointing 都跑不起来**：
> 不开 ckpt 连 9 帧都 OOM；开了 ckpt 只能到 9 帧（76.6/79.25 GB，余量 2.6 GB），
> **而官方配置要 33 帧**。
>
> SP 把激活切 4 份是**唯一可行路径**，而不是"一个可选的速度优化"。

### 附带结论：ckpt 的代价
9 帧时 fwd 5.3s + bwd 13.4s = **18.7 s/step**（bwd 是 fwd 的 2.5 倍，因为要重算 forward）。
这给了 SP 的第二个卖点：**在同等显存下，SP 不需要重算 forward**，bwd 的开销应接近常规比例（~2×）。

---

## 7.8 ✅ Ulysses 实现 + 数值验证 + 吞吐实测（2026-10-20 00:40）

`ulysses_attn.py`：标准 Ulysses（两次 all-to-all，head 维切 / token 维拼）。

### 正确性（4 卡，fp32，`--mode check`）
```
[CHECK] world=4  max_abs_err=0.000e+00  rel=0.000e+00  PASS
```
（fp32 下与单卡全序列 attention **逐元素完全一致**，不是"误差在容差内"。）

### 吞吐（4 卡，真实形状 seq=29040 / heads=40 / dim=5120，bf16，5 iters）

| 配置 | 时间 |
|---|---|
| 单卡全序列 attention | **85.8 ms** |
| **4 卡 Ulysses SP** | **30.0 ms** |
| **加速比** | **2.86×** |

拆解：纯计算 85.8/4 = 21.5 ms ⇒ **all-to-all 通信开销 ≈ 8.5 ms（28%）**。
比 `bench_amdahl.py` 的粗估（2–4%）高，原因：粗估只算了 payload 带宽，没算 kernel 启动/同步/非对齐传输。

### 端到端预期（按 Amdahl，attention 占 60.6%）
```
1 / (0.394 + 0.606/2.86) = 1.65x
```
即 **只切 attention ⇒ 整层约 1.65×**；若 SP 同时切 MLP 与投影（常规做法），上限更高。

**这个 1.65× 与 §7.5 的 1.83× 上限吻合**（后者用理想 4× 算的），说明模型自洽。

---

## 7.9 ~~✅ 端到端 SP 正确性：逐元素 0 误差~~（2026-10-20 02:00）**⚠️ 已于 10-20 17:30 作废，见 §7.11**

> **这条结论是假阳性，别再引用。** 上游 `CausalWanModel.head` 是**零初始化**，随机权重下模型输出
> 精确等于 0；两边都是全零张量，`max_abs_err` 当然是 0.0。当时没有检查 `ref.absmax > 0`。
> 而且该测试用的是"按 W 切"的 v1 实现，顺序本身也是错的（见 §7.11）。

`test_sp_e2e.py`：4 卡，小模型（2 层 dim512），纯视频路径（action=None），按视角切分。

```
[ref] done shape=(1, 16, 33, 40, 88)            # 单卡全序列
[sp]  rank0..3 out shape=(1, 16, 33, 40, 22)    # 每卡一个视角
[sp]  gathered shape=(1, 16, 33, 40, 88)
[CHECK] max_abs_err=0.0000e+00  rel=0.0000e+00  PASS
```

**这是本项目第一次证明：Ulysses SP 与 block-causal 注意力完全兼容，输出无任何偏差。**

### 注入的 4 处改动（`wan_video_dit_action_casual_chunk.py`）
1. 文件头：`_SP_STATE` + `_sp_all_to_all_4d`
2. `CausalWanSelfAttention.forward`：`qkv_fn` 后 all-to-all（head 切 / token 拼），重设 `b,s,n,d`
3. 同函数 `return` 前：all-to-all 换回（**必须在 `flatten(2)` 与 `self.o(x)` 之前**，否则 o-proj 收到 128 维）
4. `_forward_train`：`grid_size` 保持**本地**（unpatchify 要用），`freqs` 用**全局**空间尺寸（否则 rope 每个 rank 都从 0 开始）；`assert x.shape[1]*world == seq_len`

### ⚠️ 发现的结构冲突（未解决，明天处理）
`action_register`（400 token）拼在序列尾部，而 Ulysses 的 all-to-all 会把它一起切分；
但 attention 内部 `(s - action_register_length)//2` 假设它是**全局**的 400。
⇒ **带 action 的路径不能直接套用当前的切分**，需要：
   - 方案 A：action register 只放一个 rank，attention 前 all-gather 到全量并在出口只保留该 rank 的部分（实现复杂，需处理梯度）
   - 方案 B：action register 在每卡复制全量，all-to-all 前先按 world 缩小（避免重复拼接）
   - 方案 C：把 action 从"序列尾部"改成"每 block 内固定位置"，与视频 token 一起切（改动最大，但语义最干净）
**这是 E 的真正技术难点，也是论文里可以写的"为什么三流 SP 不是简单套用 Ulysses"。**

---

## 7.10 ⚠️ 14B + SP 实测：卡在死锁（2026-10-20 04:20，未完成）

**症状**：2 卡 SP、F=33，构造成功（168s / 33.1GB 常驻），但 forward 阶段 rank0 停在 `Ssl`（睡眠）、
rank1 停在 `Rsl`（CPU 自旋），GPU0 48.7GB / GPU1 33.3GB 长期不动 ⇒ **典型 NCCL 不同步**。

**已排除**：
- 不是主机内存（480GB，可用 463GB）
- 不是显存（33GB 权重 + 激活，远未到 80GB）
- 不是 GPU 被占（GPU2/3 上是**别人的 vLLM**，我们只用 0/1）

**最可能的原因（待验证）**：rank1 走的是 `meta` device 构造 + `to_empty` + 广播这条路，
而 `_create_freqs` 里有 `if any(freq.device != device)` 的设备判断、以及依赖 buffer 是否初始化的分支；
rank1 的 buffer 是未初始化内存，可能走进不同分支 ⇒ **NCCL 调用次数不一致 ⇒ 死锁**。

**明天第一件事**：4 卡**全部真实构造**（480GB 内存够 4×66GB=264GB），不用 meta/broadcast。
之前的 exit 137 实际是 SSH 会话被掐，不是 OOM —— 这一点已由"进程仍在跑、GPU 有占用"证实。

**教训**：`meta` + `to_empty` + 广播这条省内存的路，在有 buffer/分支依赖的模型上不安全；
宁可多花 3 分钟构造，也不要引入"某 rank 状态未初始化"的隐患。

---

## 8. 下一步（按顺序）

1. **S0 完成**（今晚）：模型构造 + FLOPs 账 ✅
2. **S1**：修通 `_forward_train` 的输入契约（当前卡在 block layout：`noisy_frames=33` 与 `num_image_blocks=16` 不匹配，
   28k token 需要 `x` 用 32 帧、首帧走 concat 通道），然后测 1/2/4 卡 step time 与显存
3. **S2**：实现 token-level Ulysses（改 `CausalWanSelfAttention`），测 imbalance + step time
   —— 目标已由 §7.5 量化：**P=4 上限 1.83×（只切 attention）**，通信预算 79–149 ms/40层
4. **S3**：实现 block-level 与 balanced，三方对比
5. **S4**（若 Go）：加载真实权重重跑 100 步，验证 loss 曲线一致

## 9. 风险（诚实列出）

1. **4 卡规模太小**：Ulysses 的 all-to-all 在 4 卡上收益有限，可能达不到 1.3×。→ 判据已留 Partial 分支。
2. **flash_attn 2.8.3 + A100**：无 FA3，`flash_attn_interface` 不可用；attention.py 有 SDPA fallback，但性能不同。需实测。
3. **手写 block-wise 循环**：SP 改造要动这段手写逻辑（第 404–599 行），改错会静默出错 → **必须做 loss 一致性检查**（§5 最后一项）。
4. **权重未验证**：随机权重只验证吞吐，不验证语义；S4 才验语义。若时间不够，论文只能写"系统测量"，不能写"端到端训练加速"。

---

## 7.11 🔧 v1→v3：三次返工，两个"看起来对但错"的坑（2026-10-20 17:30）

### 7.11.1 v1 的两个错

**(a) 按 W 切输入 ⇒ gathered 顺序是"编织"的，rope 全错**

局部 token 顺序是 `(f, h, w_local)`，all-to-all 按 rank 顺序拼接后变成 `(rank, f, h, w_local)`；
而全局正确顺序是 `(f, h, w)`。两者是 stride-`W_local` 的编织关系，**不是分块**。
所以 rope 索引、block-mask 的帧边界（`frame_seqlen=880`）全部错位。这个错**不会报错**，只会静默算错。

**(b) `max_abs_err = 0.0` 是假阳性（§7.9 作废的原因）**

`CausalWanModel.head` 是**零初始化**。随机权重下模型输出精确为 0，两边都是全零张量，
`max_abs_err` 当然是 0.0。**教训：等价性检查必须先断言参考值 `absmax > 0`。**

### 7.11.2 v2 的错：token 连续切分不保证帧对齐

改成"全局构造、按 row-major 连续区间切 `q = N/P`"解决了顺序问题，但引入了新问题：
`q = N/P` 必须是 `frame_seqlen` 的整数倍。
- F=17, P=4: 3740 / 880 = **4.25 帧** ✗
- F=33, P=4: 7260 / 880 = **8.25 帧** ✗
切在帧中间 ⇒ `clean_frames = clean_image_seq_len // frame_seqlen` 取整后丢 token ⇒ 输出 NaN（实测）。

### 7.11.3 v3 的正解：分段式 sharding

```
本地序列 = [首帧(复制) | 我的 8 帧 | 首帧(复制) | 我的 8 帧 | action_register(复制)]
            \___ clean:'img' ___/  \___ noisy:'img' ___/
```
- `nrep = (F % P) * frame_seqlen` 个 token 在每卡复制；剩下 `F - F%P` 帧按帧均分 ⇒ **每个 a2a chunk 都整帧对齐**
- F=33,P=4: nrep=1 帧，32/4 = 8 帧/卡 ✓；F=17: 1 + 16/4 ✓；F=9: 1 + 8/4 ✓
- 复制段（首帧 + action_register）**不进 a2a**：入口按 head 本地切片（各卡内容相同，无需通信），
  出口 `all_gather_into_tensor` 把 head 拼回。这正好把 §7.9 里"未解决的结构冲突"一起解决了。
- 复制开销 `(nrep + ar) / S ≈ 2%`

**入口/出口段长度不同**（本地 `iq` vs 全局 `iq*P`），必须用两份 segs —— 第一版这里就 assert 出来了。

### 7.11.4 正确性实测（小模型 2 层，F=17，4 卡 vs 单卡，权重已随机化保证输出非零）

| 路径 | video max_abs_err | rel | 说明 |
|---|---|---|---|
| action=None | 4.30e-02 | **8.0e-03** | bf16 tiling 噪声（H=10 vs 40） |
| action 有 | **0.0** | **0.0** | **位级一致** |

带 action 反而 bit-exact 的原因：action 分支把 head 并进了 batch 维（`[B*H, S, D]`），
按 head 切开后每个 head 的计算完全不变；而 action=None 分支是 `[B,S,H,D]` 直接进 flash-attn，
H 变了 kernel tiling 就变，bf16 累加顺序变了才有 0.8%。

### 7.11.5 死锁的原因（§7.10 的答案）

不是 NCCL 配置问题，是**构造方式**：rank0 真实构造 + 其余 rank `meta` + `to_empty` + 扁平广播。
模型里 `self.freqs / freqs_action / freqs_state` 是**普通属性不是 buffer**，`to_empty` 不会动它们，
广播也只覆盖 `parameters()` ⇒ 各 rank 状态不一致 ⇒ NCCL 调用序列发散。
**改成四卡各自真实构造（216s / 33.1GB）后死锁消失**，a2a 计数四卡完全对称。

### 7.11.6 一个反直觉的实测：无 ckpt 时激活是"每 token 每层 ~470KB"

SP4 关 ckpt 逐层量（`diag_mem.py`，hook 每层 `memory_allocated`）：

| 配置 | 结果 |
|---|---|
| SP1 F=3 | OOM，撑到 L20 |
| SP4 F=3 | OOM，撑到 L22（F=3 < P=4，全复制，SP 无效） |
| SP1 F=9 | OOM，撑到 **L6** |
| SP4 F=9 | OOM，撑到 **L20**（3.3×，与 s 比 3.08 吻合 ✓） |

⇒ SP 的收益是真实的，但**基数太大**：F=33、P=4 本地 15840 token 时单层 ~7.4GB。
这个模型每层保存 ~30-47 个 `[S, 5120]` 量级的中间量，无 checkpointing 时 4 卡也放不下。
⇒ 论文的表格必须带 ckpt 列：**DP+ckpt 在 F=17 就 OOM，SP+ckpt 才有 F=33**。

---

## 7.12 ✅ 主结果（2026-10-20 19:20）—— E 线收口

### ⚠️ 先纠正两个我自己的错误

1. **显存预算的单位算错了**：`81920 MiB` = **85.9 GB**，不是 79.25 GB。之前所有"还差 2GB"的判断都基于错误预算。
2. **Amdahl 上限 1.83× 是错的**：那个估计只把 attention 当作可并行部分（60.6%），
   但 SP 里 MLP / o-proj / FFN 全是 token-local，也一起快 4×。真正不并行的只有 all-to-all 和 2.2% 复制段。

### ⚠️ 假 OOM

不设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 时，72–82 GB 就会 OOM——是**碎片**。
这一条浪费了一整轮实验（F=17/33 全被判死）。**以后所有 14B 实验必须带上这个环境变量。**

### 主表（4×A100-80G，Wan2.1-14B，ckpt 开，预热 1 轮取 2 轮均值）

| 配置 | 峰值 | fwd | bwd | 梯度同步 | step |
|---|---|---|---|---|---|
| DP F=9 | 75.2 GB | 2.90s | 9.04s | – | 11.94s |
| DP F=17 | **OOM** @82.0 | | | | |
| DP F=33 | **OOM** @82.1 | | | | |
| SP4 F=9 | 69.4 GB | 1.11s | 3.17s | 0.57s | **4.85s** |
| SP4 F=17 | 72.9 GB | 2.18s | 6.66s | 0.57s | 9.41s |
| **SP4 F=33** | **79.9 GB** | 4.82s | 15.50s | 0.57s | **20.89s** |

- 可训练长度 **9 → 33 帧（3.7×）**
- 同 F=9 步时 **11.94 → 4.85s（2.46×）**，不含梯度同步 2.78×
- 显存地板 = 权重 33.0 + 梯度 33.0 = 66.0 GB；F=33/P=4 的激活约 13.9 GB（79.9 − 66.0）

### 还差什么才能到论文级

1. **换真实 checkpoint** 跑 100 步，验证 loss 曲线与 DP 一致（现在只有随机权重 + `out.mean()` 代理 loss）
2. **梯度分片（ZeRO-2/FSDP）**：F=33 只剩 6 GB 余量，加之后能到 ~60 GB，可以再上一档帧数
3. **8 卡**：all-to-all 的 28% 在 8 卡上会放大，需要重测 P=8 的拐点
4. 目前 SP 的梯度只做了 all_reduce（1 个 per-param 循环，0.57s），**没有和优化器 step 一起测**

---

## 7.13 🔴 最危险的一个坑：集合通信没有 autograd（v3 反向是坏的）

### 症状
真权重、F=9、SP=4，梯度 `all_reduce(SUM)` 后：

```
blocks.0.self_attn.q.weight   |SP|=0.0000   |DP|=0.0318      <-- 梯度精确为 0
```

### 根因
`dist.all_to_all_single` 与 `dist.all_gather_into_tensor` **都不支持 autograd**，
产出的张量没有 `grad_fn`。q/k/v 必须经过 a2a ⇒ 梯度链在这里静默断开。

**最容易误判的地方**：backward 不报任何错，而且 ffn / cross_attn 等参数仍有非零梯度
（因为 action 输出没走 gather，action loss 那一项漏回来了）。
只看"loss 在降、梯度非零"会得出"能训"的错误结论。**必须逐参数对 DP 校验。**

### 修法（`sp_v4_grad.py`）
三种通信全部包 `autograd.Function`，反向用**逆置换**（不是 sum/reduce）：

| 前向 | 反向 | 为什么不是归约 |
|---|---|---|
| a2a(scatter=sd, gather=gd) | a2a(scatter=gd, gather=sd) | 纯重排 |
| all_gather(token) | 取自己那段 | 各 rank token 不重叠 |
| all_gather(head, 复制段) | 取自己那片 head | 各 rank head 不重叠 |

参数梯度跨卡 **求和**（`all_reduce SUM`），发生在 backward 之后、optimizer 之前。

### 附带的使用规则（不是实现问题，是语义）
`action_register` 的输出在每张卡上都是完整副本 ⇒ 它上面的 loss 必须 `/P` 再反传，
否则梯度求和后**正好放大 P 倍**（实测 4.00×）。上报 loss 用未缩放值。

### 验证（小模型 F=17，4 卡 vs 单卡）
```
gradnorm SP=478.0827 DP=478.4491  rel=7.66e-04
比值>2.5 的参数个数 = 0        逐参数比值全为 1.00
blocks.0.self_attn.q.weight  |SP|=0.08967 |DP|=0.08969  relL2=8.0e-03
```

### 一个自己骗自己的插曲
中途看到"所有 action 路径参数恰好 4.00×"，一度以为是模型 bug。
实际是**我的测试脚本对 DP 对照组也除了 P** —— 对照组被缩小 4 倍。
凡是"整整齐齐等于 P 的倍数"的偏差，先怀疑自己的缩放，再怀疑实现。

---

## 7.14 ✅ v5：零复制，梯度做到 0.27%

### 为什么 v4 还不够
v4 修好 autograd 后，真权重总 gradnorm 仍差 **12.6%**，`patch_embedding` 1.92×。
根因是 v3 的**复制段**：首帧 + action_register 在每张卡上重复计算。
attention 的 q/k/v/o 按 head 切片 → 求和正确；但 **MLP / o-proj 处理全 dim 的本地 token**
→ 复制段的贡献被算 P 次。**F=9 时首帧占 11%，不是 2.2%。**

### v5 做法：三段式零复制
```
段1 首帧    880       -> 每卡 220        (880 % P == 0)
段2 body   (F-1)*880  -> 每卡 (F-1)*220  ((F-1)*880 % P == 0)
段3 action  400       -> 每卡 100        (400 % P == 0)
```
每段内部是连续区间 ⇒ a2a 按 rank 序拼接 == 全局序，**零复制**。
attention 侧仍是 5 段，`(s - ar_local)//2` == clean 本地长度 ✓，gather 后 `clean_frames` == F ✓。

尾部把 action 输出也 all_gather 回全量 ⇒ 各 rank loss 完全一致，
**v3 那条「复制输出 loss 除以 P」的规则彻底不需要了**。

### 结果
| 阶段 | 真权重 gradnorm 相对误差 |
|---|---|
| v3（a2a 无 autograd） | 18.3%（且 q/k/v 梯度精确为 0） |
| v4（加 autograd） | 12.6% |
| **v5（零复制）** | **0.27%** |

小模型（F=17，带 action）：video/action 输出 **0.0**，gradnorm rel **2.02e-06**，
比值>2.5 的参数个数 **0**。

### 教益（写给以后的自己）
1. **集合通信必须包 autograd，且反向是逆置换不是归约。** 不报错，只静默把梯度清零。
2. **等价性测试必须挑对 loss。** 我第一版测试的 action 路径梯度大了两个数量级，
   把视频路径 12.6% 的偏差完全掩盖了——「测试通过」不等于「对」。
3. **凡是"整整齐齐等于 P 的倍数"的偏差，先查自己的缩放。**
