# 当前数据的分组 3D 高光谱分析

## 状态与边界

本目录是一套**环境与前向路径已验证、但尚未正式运行训练**的深度学习分析协议。项目 `.venv` 已安装 `torch 2.12.1+cu126`、`h5py 3.16.0` 及锁定的科学计算依赖；RTX 4060 CUDA 运算、真实 MAT 读取以及 HS3I/无 SelecVar 单样本前向路径均已通过。**没有产生任何新的 3D-CNN、HS3I-Net、空间打乱或 mask-only 性能结果**。只有在固定协议实际完成后，才能把输出标记为“新执行结果”。

脚本：[grouped_hs3i_current_data.py](grouped_hs3i_current_data.py)

这套分析只回答一个受限问题：使用当前 16 个来源立方体时，模型能否从一组完整立方体迁移到另一组完整立方体。它不能把当前数据变成独立农场、批次、年份或仪器验证，也不能单独证明地理产地因果效应。

## 与旧分析相比的关键修正

- 来源立方体完整隔离：只执行 `所有 *-1 训练 → 所有 *-2 测试` 和反向分析。一个来源立方体中的种子不会跨训练集和测试集。
- 测试集不参与开发：固定训练轮数，不做早停，不根据测试指标选择模型、检查点、波段、随机种子或超参数；测试 MAT patch 只在固定训练完成后迭代一次。
- 波长来自真实 CSV：读取每个种子配套 CSV 的第一列，要求 392 个波长严格递增且全数据一致；不再用首末波长线性插值。
- 同一主干显式切换 `HS3I/SelecVar` 与 `无 SelecVar`，避免两份脚本继续漂移。
- 加入空间信息的否证对照：`spatial_shuffle` 在前景内以完整像素光谱为单位打乱位置；`mask_only` 删除所有光谱与强度，只保留轮廓、面积、方向和 patch 位置。
- 固定多个训练随机种子，分别保存结果；随机种子间波动只表示训练不稳定性，不是对新批次泛化误差的置信区间。
- 不提供 Grad-CAM。现有网络末端单元的光谱感受野很宽，将其归因到单一 nm 会制造虚假精度。SelecVar 权重按真实波长导出，但仍只属于模型参数，不能当作化学机制证据。

## 固定协议

默认完整运行包含：

- 两个方向：`suffix_1_to_2`、`suffix_2_to_1`；
- 两个模型：`hs3i`、`no_selecvar`；
- 三种输入：`full`、`spatial_shuffle`、`mask_only`；
- 三个预先声明的训练种子：`42`、`2024`、`2025`。

因此默认共训练 36 个模型。固定超参数保留原研究的主体设定：360 epochs、10-epoch warm-up、AdamW、主干学习率 `3e-4`、SelecVar 学习率 `1e-3`、label smoothing、Mixup 和相同的残差 3D 主干。学习率调度被统一成单一的 warm-up + cosine schedule，避免旧脚本中两个 scheduler 先后作用造成不透明行为。batch size 固定为 16，以降低该 3D 网络的显存压力。

模型/条件 CLI 参数仅用于执行预先定义的实验单元，例如分批占用 GPU；不得在看过测试结果后据此删选“最好”的单元。正式报告应运行完整矩阵并披露全部随机种子。

## 环境与运行

当前验证环境为 Python 3.11.9、PyTorch 2.12.1+cu126、CUDA runtime 12.6、h5py 3.16.0 和 NumPy 2.4.4。项目根目录提供完整 PyPI 锁和独立的官方 CUDA wheel 通道。从仓库根目录重建：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m pip install -r requirements-torch-cu126.txt
```

从仓库根目录执行完整协议：

```powershell
python deep_models/grouped_hs3i_current_data.py `
  --data-root data `
  --output-dir deep_models/outputs/grouped_confirmatory
```

如需把实验拆到不同设备上，可显式选择预先定义的单元，例如：

```powershell
python deep_models/grouped_hs3i_current_data.py `
  --data-root data `
  --output-dir deep_models/outputs/hs3i_controls `
  --models hs3i `
  --conditions spatial_shuffle mask_only `
  --seeds 42 2024 2025 `
  --device cuda:0
```

输出目录必须不存在，脚本拒绝覆盖或混合旧结果。`--save-checkpoints` 可额外保存最终参数；这些参数是固定 epoch 的最终状态，不是测试集挑选出的检查点。

## 结果文件

- `manifest.csv`：逐样本标签、来源立方体、后缀、种子编号和相对路径；
- `wavelengths.csv`：从 CSV 读取的真实 392 波长；
- `predictions.csv`：每个方向、模型、条件、随机种子的逐样本预测与 8 类概率；
- `metrics_by_run.csv`：每次运行的 accuracy、balanced accuracy、macro precision/recall/F1；
- `metrics_seed_aggregate.csv`：相同实验单元跨随机种子的均值、标准差和范围；
- `metrics.json`：混淆矩阵、逐类指标、数据指纹、环境、固定超参数和解释限制；
- `training_history.csv`：逐 epoch 损失和学习率；
- `selecvar_gate_weights.csv`：HS3I 的门控权重及其真实波长；
- 可选 `*.pt`：最终模型状态。

CSV 使用 UTF-8 with BOM，便于中文 Windows/Excel 环境读取。JSON 中的 `status: executed` 只会在所有计划运行完成后写出；中途失败不会生成一份伪装成完整结果的 JSON 总结。

## 对照的正确解释

`full` 与 `spatial_shuffle` 的差异用于检验模型是否利用前景内部的空间排列。打乱操作始终移动一个像素的完整 392 波段向量，因此不会人为拆散该像素内部的光谱协方差；轮廓保持不变。

`mask_only` 只包含重复到全部波段的二值轮廓。如果它仍有较高准确率，说明类别可能由面积、形状、方向、居中或裁切方式区分，不能把完整模型成绩直接解释为化学或光谱差异。

即使 `full` 稳定优于这些对照，也只能支持“在当前两组采集立方体之间存在有用的光谱—空间信息”。由于每个测试类别只有一个来源立方体，而且两后缀是否代表独立商业批次尚不清楚，种子仍是同一立方体内的技术子样本。当前结果不能外推到新农场、收获年、供应商或仪器。

## 运行前后审计清单

运行前：

1. 保持上述超参数、模型单元、对照和随机种子不因任何 `*-2` 或 `*-1` 测试表现而改变。
2. 确认 `data/0-1` 至 `data/7-2` 均存在，且每个 MAT 有同名 CSV。
3. 记录 Python、PyTorch、CUDA、GPU 和 h5py 版本。
4. 预留足够时间与显存；默认 36 次、每次 360 epochs，属于高成本完整协议。

运行后：

1. 检查终端是否正常完成全部 36 次运行，并确认 `metrics.json` 存在。
2. 保留全部 seed、双向结果和失败记录，不只报告最好结果。
3. 对照 `manifest.csv` 确认训练与测试来源立方体交集为空。
4. 在论文或报告中把这些结果标为“当前数据、来源立方体后缀迁移”，不要写成独立地理产地外部验证。
5. 按仓库 `AGENTS.md` 要求，把实际执行命令、环境、生成文件、验证和限制追加到研究修订总账；不得把计划运行登记为实测结果。

## 本次环境与冒烟验证

2026-07-21 在仓库 `.venv` 中实际完成：

- `pip check`：无损坏或冲突依赖；
- CUDA：`torch.cuda.is_available() == True`，识别 NVIDIA GeForce RTX 4060 Laptop GPU，GPU 矩阵运算通过；
- 数据入口：发现 1,264 个样本和 392 个真实波长，`h5py` 成功读取真实 MAT patch；
- 模型入口：HS3I 与无 SelecVar 分支各完成一次单样本 GPU 前向计算，输出形状均为 `(1, 8)`；
- 既有分析回归：`python -m unittest discover -s current_data_study/tests -v` 为 9/9 通过。

这些检查证明当前依赖、MAT 方向、设备选择和基础前向路径可运行；它们**不证明**360-epoch 训练可稳定完成，也不证明数值收敛、泛化性能或任何空间/光谱优势。首次正式运行必须写入新的输出目录，并将运行时修正、失败和全部预声明结果另行登记。
