# dreamzero_sp —— Wan2.1-14B 序列并行（可复现包）

> 目标：把 action-conditioned 视频扩散的官方 33 帧配置塞进 4×A100-80G。
> 结果：纯 DP 上限 9 帧 → SP=4 + ZeRO-2 达到 33 帧（55.9 GB），同配置加速 2.46×。
>
> **主 blog 见 `blog.md`**（从零讲起，不需要前置知识）。
> 实验原始记录见 `E_结果_2026-10-20.md`；设计决策与被证伪的假设见 `SP_设计.md`。

**文档索引**

| 文件 | 体例 | 适合谁读 |
|---|---|---|
| **`blog.md`** | **主 blog（通俗、从零讲起）** | **第一次听说这个项目的人；转给别人看的那个版本** |
| `blog_序列并行.md` | 工程复盘（更技术、按因果链） | 想看踩坑细节与决策链的人 |
| `E_结果_2026-10-20.md` | 实验记录 | 要核对具体数字、配置、失败案例的人 |
| `SP_设计.md` | 设计日志 | 要看每个决策的依据与被推翻的假设的人 |
| `E_论文_骨架.md` | 论文骨架 | 要写稿的人（含诚实定位与 limitations） |

---

## 1. 目录结构

```
dreamzero_sp/
├── apply_all.sh              ★ 一键重建：从 pristine 备份依次打 v3→v4→v5→v6
├── known_good/               ★ 已验证通过的模型文件 + pristine 备份 + md5
│   ├── wan_video_dit_action_casual_chunk.py
│   ├── pristine.bak
│   └── md5.txt
│
├── sp_v3_patch.py            补丁层 1：按 frame 切分的基础骨架
├── sp_v4_grad.py             补丁层 2：给集合通信加 autograd（反向用逆置换）
├── sp_v5_zerocopy.py         补丁层 3：零复制分段（首帧/body/action 三段均分）
├── sp_v6_flat.py             补丁层 4：多段 a2a 合并成一次扁平 all_to_all_single
│
├── train_real.py             ★ 训练循环（真权重 + flow-matching + ZeRO-2 + SGD/AdamW）
├── run_f33_adamw.sh          ★ 主结果：F=33 × 50 步 × AdamW（就是产出那张曲线的脚本）
├── run_adamw_chk.sh          2 步快速自检：AdamW 接入是否正确（--no_load，约 4 分钟）
├── run_smoke.sh              1 步冒烟：改完训练循环后先跑这个（--no_load）
├── zero2_check.py            ★ ZeRO-2 vs all_reduce 的逐元素等价性
├── test_sp_grad_small.py     ★ 小模型前向/梯度等价性（4 卡 vs 单卡）
├── test_sp_ckpt.py           ★ 显存/步时主表（支持多帧数、多 rank 数）
├── test_real_weights.py      真权重前向等价性
├── grad_check.py             真权重梯度等价性
├── diag_mem.py               逐层 memory_allocated 曲线（定位显存增长）
├── offload_baseline.py       DP + activation offload 基线（结论：无效）
├── ulysses_attn.py           Ulysses attention 单算子：正确性 + 吞吐
├── bench_amdahl.py           attention 占单层墙钟的比例
│
├── dl_droid.py               真实 DROID 数据加载（mp4 解码 + parquet action + VAE 编码）
├── train_real_data.py        真实数据训练（替代 train_real.py 的合成 batch）
│
├── shim/diffusers/           40 行的 diffusers 兼容层（避免污染共享环境）
├── logs/                     各次运行日志（不进仓库）
└── results/                  结构化结果：50 步 CSV + 原始日志
```

**根目录里有两个"反面教材"，别当可用脚本**：
- `test_sp_e2e.py` —— 曾报"逐元素 0 误差"，实为**假阳性**（模型 `head` 零初始化，输出恒为 0）。
  真正的等价性检查用 `test_sp_grad_small.py`（它会先断言参考输出非零）。
- `sp_v5_split.py` —— 中途失败的尝试（`_sp_segs` 索引错位），留作对照；后来的零复制实现是 `sp_v5_zerocopy.py`。

其余历史脚本（`probe_step*.py`、`sp_patch.py`、`sp_v2_patch.py`、`test_sp_mem*.py`、`test_sp_exact.py`）
是探索阶段的产物，已被 `test_sp_grad_small.py` / `test_sp_ckpt.py` 取代，保留仅为可追溯。

---

## 2. 环境

```bash
export Z=/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengmenghan
source $Z/auto_research_lightningrl/env.sh          # py3.12 / torch2.8 / flash-attn

# 两个必须设置的东西
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 否则 72–82GB 会"假 OOM"
export PYTHONPATH="$Z/dreamzero_sp/shim:$Z/dreamzero_sp/dreamzero:$PYTHONPATH"
# torchrun 可能不在 PATH，用 python -m torch.distributed.run
```

---

## 3. 复现步骤

### 3.1 重建补丁（关键：从 pristine 备份重建，容忍重复执行）

```bash
cd $Z/dreamzero_sp
./apply_all.sh
# 期望输出：[1/4] v3 ok … [4/4] v6 ok / patch markers: 9 / syntax ok
```

补丁层之间有依赖，**顺序不能变**（v4 定义 `_SPTokenGather`，v5 依赖它；v6 依赖 v5 的段结构）。

> ⚠️ 我们在这里踩过一次：有两处 hotfix 是直接改模型文件修的，没回写补丁脚本，
> 导致一键重建把 bug 装回来（`dim=3` 和 `_sp_segs[1:4]`）。
> **规则：任何修改先改补丁脚本，再 `apply_all.sh`，最后跑 3.2 的测试。**

### 3.2 正确性（先跑这个，再跑任何测量）

```bash
# 小模型（2 层 dim512）4 卡 vs 单卡：前向 + 梯度
python -m torch.distributed.run --nproc_per_node=4 test_sp_grad_small.py --frames 17 --action 1

# 期望：
# [PASS] video max_abs_err=0.0000e+00
# [PASS] action max_abs_err=0.0000e+00
# [grad] gradnorm SP=478.03 DP=478.03 rel≈2e-06
# 比值>2.5 的参数个数=0

# ZeRO-2 vs all_reduce 的逐元素等价
python -m torch.distributed.run --nproc_per_node=4 zero2_check.py
```

### 3.3 显存/步时主表

```bash
python -m torch.distributed.run --nproc_per_node=4 test_sp_ckpt.py \
  --ckpt 1 --iters 2 --warmup 1 --gradsync 1 --spec 4:9:1,4:17:1,4:33:1
# 注意：F=33 需要梯度分片才放得下，见 3.4
```

### 3.4 33 帧训练（SP + ZeRO-2 + AdamW）★ 主结果

最省事：**直接用产出这张曲线的那个脚本**（命令逐字一致）

```bash
bash run_f33_adamw.sh          # 内部就是下面这条
```

```bash
# 50 步完整曲线（真权重，和论文里那张图对应）
python -m torch.distributed.run --nproc_per_node=4 train_real.py \
  --frames 33 --steps 50 --lr 1e-4 --zero2 1 --opt adamw_bf16 --order 4
# 实测：峰值 72.3 GB（状态 16.5 GB），21.0–21.4 s/step
#       loss 1.0693 → 0.5861（−45.2%），步 20 后收敛，无 NaN
#       注意 batch 是固定的一个 ⇒ 证明的是「能稳定优化」，不是「在数据上收敛」

# 省显存版（SGD，无优化器状态）：55.9 GB
python -m torch.distributed.run --nproc_per_node=4 train_real.py \
  --frames 33 --steps 4 --lr 4e-3 --zero2 1 --order 4
# 期望：peak ≈ 55.9 GB，loss 1.0693 → 0.8993，20.4 s/step
```

曲线图 `fig_E_f33_curve.png`，逐步骤数据 `results/f33_adamw_50steps.csv`。

### 3.5 审稿人攻击的两个对照实验

```bash
# 纯 DP + ZeRO-2（关掉 SP，梯度照样分片）→ F=33 必须 OOM
python -m torch.distributed.run --nproc_per_node=4 train_real.py \
  --no_load 1 --zero2 1 --order 1 --dp_steps 1 --frames 9 --frames_list 9,17,33 --lr 4e-3

# DP + activation offload → 峰值不降（结论：该方向无效）
python -m torch.distributed.run --nproc_per_node=4 offload_baseline.py \
  --offload 1 --spec 1:9,1:17,1:33 --iters 2 --warmup 1
```

---

## 4. 关键参数

| 项 | 值 |
|---|---|
| 模型 | Wan2.1-14B（16.48B 参数，bf16 33.0 GB），40 层 / 40 head / dim 5120 |
| 每帧 token | 880（= 40/2 × 88/2） |
| 官方配置 | `num_frames=33`、`use_gradient_checkpointing=true` |
| 权重 | `$Z/model`（step-100000，1317 张量，shape mismatch = 0） |
| 段结构（v6） | 首帧 880/P + body (F-1)×880/P + action 400/P |
| 整除约束 | 880 % P == 0、400 % P == 0（P=4 时任何 F 都成立） |

---

## 5. 已知限制

1. 只测到 P=4（本机 4 卡），P=8 的拐点未测。
2. **优化器的默认值仍是 SGD**（`--opt` 缺省 `sgd`，无优化器状态，55.9 GB）。
   **主结果那张 50 步曲线用的是 `--opt adamw_bf16`**：状态 bf16 + 1/P 分片 ⇒ 72.3 GB（实测）。
   ⚠️ 跑 `--zero2 1` 时**必须显式带 `--opt adamw_bf16`**，否则会静默用 SGD —— 两者显存差 16.4 GB，
   loss 曲线也不可比。fp32 状态放不下（88.9 GB）；Adafactor 内存最优但需大 ~100× 的 lr。见 `opt_test.py`。
3. `action=None`（纯视频）分支在 F=17 出 NaN —— **DP 单卡同样 NaN**，属上游问题。
4. **50 步曲线已补到 F=33**（真权重 + SP4 + ZeRO-2 + bf16 状态 AdamW）：1.0693 → 0.5861，
   峰值 72.3 GB，21.0–21.4 s/step，无 NaN，步 20 后收敛。但 **batch 固定为单一样本**，
   且 `z0` 是高斯 latent 而非 VAE 编码的真实视频 ⇒ 证明的是优化稳定性与梯度路径正确，
   不是数据分布上的收敛。优化器状态为 bf16，长训练精度损失未评估；only one lr。
5. 未做端到端质量评估（FVD / 任务成功率）。
