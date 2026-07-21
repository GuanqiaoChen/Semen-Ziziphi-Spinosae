# 当前数据的预注册来源立方体隔离分析

## 状态、入口与证据范围

本目录的正式分析已于 2026-07-21 完整执行并标记为 `executed_complete`。当前唯一正式入口是 [top_journal_current_data.py](top_journal_current_data.py)，锁定后处理入口是 [summarize_top_journal_results.py](summarize_top_journal_results.py)，结果前冻结的方案见 [当前数据顶刊方法预注册](../docs/当前数据顶刊方法预注册.md)。

本次分析按顶刊级方法学审计规范实施：预先固定研究问题和估计量、来源立方体完全隔离、测试集不参与开发、强光谱基线、共享初始化的深度消融、同一锁定模型反事实、概率校准、正确层级的条件性不确定性以及可追溯逐样本产物。

这些方法不能创造数据中不存在的生物学重复。当前数据只有 8 个存档商业样品标签、16 个来源立方体和 1,264 粒立方体内技术子样本；每个方向、每类只有 1 个测试来源立方体。因此结果是**当前配对来源立方体之间的闭集迁移诊断**，不是地理产地外部验证。

## 冻结协议

### 分析单位与拆分

- 数据：`data/0-1` 至 `data/7-2` 的全部同名 MAT/CSV 对，共 1,264 粒种子、392 个实测波长、16 个来源立方体。
- 互反测试：全部 `*-1` 立方体开发 → 全部 `*-2` 测试；以及全部 `*-2` 开发 → 全部 `*-1` 测试。
- 来源隔离：任一来源立方体只属于开发或测试一侧，测试立方体不参与模型、epoch、超参数、温度或输入变换选择。
- 开发内控制：每个开发立方体按固定规则做 80% 拟合、20% 验证，仅用于模型控制、checkpoint 选择和温度缩放。它仍是同立方体内的种子级验证，不能解释为独立分组验证。
- 数据完整性：逐文件 SHA-256 写入 manifest；MAT 前景平均与同名 CSV 光谱对 1,264 个样本全部通过 `1e-5` 容差核验，最大绝对差为 `2.8260272e-7`。

### 固定模型与训练矩阵

正式矩阵按运行清单包含 18 个训练单元（其中 `snv_lr` 单元是模型拟合）：

\[
2\ \text{个方向}\times 3\ \text{个模型}\times 3\ \text{个种子}=18。
\]

三个模型为：

- `snv_lr`：每粒种子的前景平均光谱做 SNV，再由开发集拟合标准化和类别等权多项 logistic regression；`C={0.01, 0.1, 1, 10, 100}` 仅在开发内验证选择。
- `spectral_only`：SNV 平均光谱进入一维残差编码器；参数量 79,624，是融合模型的深度光谱消融。
- `fusion_net`：同一一维残差光谱分支加高效空间分支；空间分支先逐像素学习 `392→16` 谱投影，再用二维残差编码；参数量 209,000。

`spectral_only` 与 `fusion_net` 使用相同拆分、种子、谱分支和分类器结构，并在每个方向/种子配对单元共享初始谱分支与分类器参数；正式输出的初始化哈希核验为通过。

固定优化种子为 `42`、`2024`、`2025`，反事实置乱种子为 `9173`。神经模型统一使用 batch size 32、最多 60 epochs、至少 12 epochs、patience 8、AdamW（学习率 `1e-3`、weight decay `1e-4`）、label smoothing 0.05、梯度裁剪 5.0 和 CUDA AMP。checkpoint 按开发内 validation macro-F1 最大、再按 NLL 最小、再按最早 epoch 的冻结规则选择；温度仅由开发内 validation logits 在固定网格上确定。

### 锁定模型反事实

每个神经训练单元先只用 `full` 输入完成拟合和锁定；随后同一 checkpoint 不再训练，成对测试四种输入：

1. `full`：完整前景高光谱 patch；
2. `spatial_shuffle`：在前景内移动完整 392 波段像素光谱，删除内部排列但保留轮廓与像素光谱分布；
3. `mean_broadcast`：把该种子前景平均光谱复制到轮廓内，删除像素异质性和内部排列；
4. `mask_only`：仅保留二值轮廓，删除实测光谱与强度。

因此，“18 个训练单元”不应误写为 72 次模型训练；四条件是对锁定模型的反事实测试。`spectral_only` 对 `full`、`spatial_shuffle` 与 `mean_broadcast` 的预测一致性也作为反事实实现校验。

### 主要估计量与推断

主要预测器是三个预声明种子的**温度校准概率平均集成**。先对每粒测试种子的 8 类概率求平均，再计算集成指标，不把优化种子当作生物重复。

主要性能估计量为：

\[
\theta_m=\frac{BA_{1\rightarrow2,m}+BA_{2\rightarrow1,m}}{2}，
\]

其中每个方向内 8 个类别/来源立方体等权，两个方向再等权。主要机制效应是 `fusion_net(full) - fusion_net(spatial_shuffle)` 的双向等权差。

条件性 95% 区间由 10,000 次配对 block bootstrap 生成，重采样单位是 8 个商业标签对，所有模型与条件共享种子 `20260721` 的同一重采样矩阵。主要机制检验是 8 标签对全部 `2^8=256` 个符号分配的精确 sign-flip 检验。区间和检验只条件于当前 16 个存档立方体，不是新批次总体推断。

## 正式结果

下表均为温度校准三种子概率集成的 balanced accuracy；区间是上述标签对 block bootstrap 的条件性 95% 百分位区间。

| 模型 | `*-1 → *-2` | `*-2 → *-1` | \(\theta\) | 条件性 95% 区间 |
| --- | ---: | ---: | ---: | ---: |
| `snv_lr` | 84.77% | 89.11% | 86.94% | 75.38%–96.20% |
| `spectral_only` | 52.94% | 36.56% | 44.75% | 25.31%–63.58% |
| `fusion_net` | 43.85% | 48.02% | 45.94% | 31.98%–60.70% |

关键配对效应：

| 对比 | 双向效应 | 条件性 95% 区间 | 解释 |
| --- | ---: | ---: | --- |
| `fusion(full) - fusion(spatial_shuffle)` | +13.15 pp | +1.82–+24.38 pp | 主要空间排列效应 |
| `fusion(spatial_shuffle) - fusion(mean_broadcast)` | +0.28 pp | −0.50–+0.99 pp | 未见稳定无序像素异质性增益 |
| `fusion(full) - fusion(mean_broadcast)` | +13.43 pp | +1.71–+24.76 pp | 完整空间/异质信息的合并效应 |
| `fusion(mean_broadcast) - fusion(mask_only)` | +11.99 pp | +4.10–+21.96 pp | 当前数据中的平均光谱增量 |
| `fusion(full) - spectral_only(full)` | +1.19 pp | −11.29–+15.55 pp | 没有稳定谱空融合优势 |
| `fusion(full) - snv_lr(full)` | −41.00 pp | −50.86–−31.46 pp | 融合网络显著低于强光谱基线 |
| `fusion(mask_only) - chance` | +8.02 pp | −5.70–+23.54 pp | 区间跨零，仍须保留形态/裁切捷径解释 |

主要空间效应在 `*-1 → *-2` 为 +0.78 pp、在反向为 +25.52 pp，方向均为正但高度不对称；6/6 个方向×优化种子差均为正，双向均值超过预声明的 2 pp 门槛。精确单侧 sign-flip `p=0.0429688`，双侧敏感性 `p=0.0859375`。因此结果**达到预声明的有限空间排列支持门槛**，但只能说明融合模型在当前配对立方体迁移中使用了可用的前景内部排列信息，不能解释为空间因果、地理组织结构或外部泛化。

融合网络的 \(\theta\) 比 `spectral_only` 只高 1.19 pp，区间跨零，而且两个方向差值一负一正；不能声称稳定优于深度光谱消融。它还比 SNV–LR 低 41.00 pp，区间完全低于零。当前最强性能基线是简洁的 SNV–LR，而不是深度融合模型。

完整中英摘要见 [postprocessing/summary.md](outputs/top_journal_preregistered/postprocessing/summary.md)。

## 环境与复现

正式环境为 Python 3.11.9、NumPy 2.4.4、scikit-learn 1.8.0、h5py 3.16.0、PyTorch 2.12.1+cu126、CUDA 12.6 和 NVIDIA GeForce RTX 4060 Laptop GPU。依赖已锁定在仓库根目录的 `requirements-lock.txt` 与 `requirements-torch-cu126.txt`。

从仓库根目录重建环境：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m pip install -r requirements-torch-cu126.txt
.venv\Scripts\python.exe -m pip check
```

本次正式运行记录的确切命令为：

```powershell
.venv\Scripts\python.exe deep_models\top_journal_current_data.py `
  --data-root data `
  --output-dir deep_models\outputs\top_journal_preregistered `
  --models snv_lr spectral_only fusion_net `
  --seeds 42 2024 2025 `
  --device cuda:0 `
  --num-workers 2
```

正式运行耗时 `1387.738` 秒，运行前工作树为干净的 `main`，代码提交为 `4bc191c2e9b8a809e866ccd15d96fea29378969d`。脚本要求输出目录不存在，**不要向现有正式目录重复执行**；独立复跑应使用新的空目录，并保留全部预声明单元。

锁定后处理的确切命令为：

```powershell
.venv\Scripts\python.exe deep_models\summarize_top_journal_results.py `
  --input-dir deep_models\outputs\top_journal_preregistered
```

默认后处理目录是输入目录下的 `postprocessing`，同样拒绝覆盖已有目录。若对独立复跑后处理，应把 `--input-dir` 指向该完整新输出目录。

发布检查点把波长元数据保存为普通 Python 列表，可使用 PyTorch 的安全默认模式加载：

```python
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
```

正式运行最初生成的检查点含等值 NumPy 波长数组；发布前只对这一元数据序列化格式进行了安全规范化。12 个发布检查点均已在 `weights_only=True` 下加载，并与本地保留的原始版本逐一核对 600 个 state-dict 张量完全相同；模型权重、温度、波长值和全部结果均未改变。

## 正式产物清单

所有正式产物位于 [outputs/top_journal_preregistered](outputs/top_journal_preregistered/)：

- [results.json](outputs/top_journal_preregistered/results.json) 与 [run_status.json](outputs/top_journal_preregistered/run_status.json)：执行状态、命令、环境、Git、协议、数据指纹、完整运行清单和限制；
- [manifest.csv](outputs/top_journal_preregistered/manifest.csv)（1,264 行）、[wavelengths.csv](outputs/top_journal_preregistered/wavelengths.csv)、[splits.csv](outputs/top_journal_preregistered/splits.csv)：样本、实测波长和零交叉拆分；
- [predictions.csv](outputs/top_journal_preregistered/predictions.csv)（45,504 行）与 [ensemble_predictions.csv](outputs/top_journal_preregistered/ensemble_predictions.csv)（15,168 行）：逐样本、逐条件原始/校准概率及三种子集成概率；
- [metrics.csv](outputs/top_journal_preregistered/metrics.csv)（144 行）、[ensemble_metrics.csv](outputs/top_journal_preregistered/ensemble_metrics.csv)（48 行）、[cube_metrics.csv](outputs/top_journal_preregistered/cube_metrics.csv) 与 [ensemble_cube_metrics.csv](outputs/top_journal_preregistered/ensemble_cube_metrics.csv)：运行、集成和来源立方体层级指标；
- [primary_estimands.csv](outputs/top_journal_preregistered/primary_estimands.csv) 与 [spatial_mechanism_decision.json](outputs/top_journal_preregistered/spatial_mechanism_decision.json)：主要 \(\theta\) 与冻结门槛判定；
- [model_selection.csv](outputs/top_journal_preregistered/model_selection.csv)、[development_validation_calibration.csv](outputs/top_journal_preregistered/development_validation_calibration.csv) 和 [training_history.csv](outputs/top_journal_preregistered/training_history.csv)：开发内选择、温度与训练轨迹；
- [counterfactual_pairs.csv](outputs/top_journal_preregistered/counterfactual_pairs.csv) 与 [ensemble_counterfactual_pairs.csv](outputs/top_journal_preregistered/ensemble_counterfactual_pairs.csv)：逐种子和集成的配对反事实；
- 12 个可由 `weights_only=True` 安全加载的神经 checkpoint（2 个方向 × 2 个神经模型 × 3 个种子，`*.pt`）；
- [MAT/CSV 一致性 JSON](outputs/top_journal_preregistered/mat_csv_mean_consistency.json) 与逐样本 CSV：表示完整性审计。

锁定后处理位于 [postprocessing](outputs/top_journal_preregistered/postprocessing/)：

- [summary.md](outputs/top_journal_preregistered/postprocessing/summary.md)：中英双语审计摘要；
- [bootstrap_theta_intervals.csv](outputs/top_journal_preregistered/postprocessing/bootstrap_theta_intervals.csv)、[bootstrap_effect_intervals.csv](outputs/top_journal_preregistered/postprocessing/bootstrap_effect_intervals.csv) 与 [bootstrap_label_pair_indices.csv](outputs/top_journal_preregistered/postprocessing/bootstrap_label_pair_indices.csv)：条件性区间及共享重采样矩阵；
- [primary_sign_flip_test.json](outputs/top_journal_preregistered/postprocessing/primary_sign_flip_test.json)、逐标签效应和完整 256 个符号分配：主要精确检验；
- [postprocessing_manifest.json](outputs/top_journal_preregistered/postprocessing/postprocessing_manifest.json)：输入/输出 SHA-256、锁定常数和推断范围；
- [主性能图](outputs/top_journal_preregistered/postprocessing/figure_main_performance.pdf)、[反事实效应图](outputs/top_journal_preregistered/postprocessing/figure_counterfactual_effects.pdf)、[集成混淆矩阵](outputs/top_journal_preregistered/postprocessing/figure_ensemble_confusion_matrices.pdf) 与 [校准可靠性图](outputs/top_journal_preregistered/postprocessing/figure_calibration_reliability.pdf)，均同时提供 PNG 和矢量 PDF。

## 复现审计要点

正式结果应满足以下不可拆分的完整性条件：

1. `results.json` 和后处理 manifest 均为 `executed_complete`，18 个训练单元、两个方向、三个模型和三个种子齐全；
2. 训练/测试来源立方体交集为空，测试只在开发内选择与温度锁定后访问；
3. `spectral_only` 与 `fusion_net` 的配对初始化哈希一致；
4. 所有 8 类概率有限且逐行和为 1，指标可由逐样本概率重算；
5. 四个反事实对同一神经 checkpoint 测试，不为任何条件重新训练；
6. 10,000 次区间对所有模型/条件共享同一 8 标签对重采样矩阵，精确检验保留全部 256 个符号分配；
7. 不删选较好方向、种子、模型或校准版本；原始与温度校准结果均保留；
8. 任何复跑必须使用新输出目录，不覆盖或拼接正式结果。
9. 12 个发布 checkpoint 均可安全加载，并与原始正式运行版本逐张量一致。

## 不可越过的解释边界

- 每类每方向只有一个测试来源立方体，种子是聚类技术子样本；
- 两个后缀是否代表独立商业批次、供应商或采集会话缺少可核验元数据；
- 开发内 validation 与温度校准来自同一开发立方体，不能证明跨立方体校准可迁移；
- 反事实效应诊断的是模型依赖，不是化学因果或组织机制；
- 当前没有未知类、独立农场/批次、多年份、跨仪器或跨实验室测试。

因此不得把本次结果表述为地理产地认证、真实溯源、防伪系统、独立外部验证、开放集识别、部署就绪或对新来源总体的性能估计。最严格且可辩护的贡献是：一个来源立方体隔离、结果前冻结、包含强基线、反事实证伪、校准和层级不确定性的高光谱方法学审计案例。
