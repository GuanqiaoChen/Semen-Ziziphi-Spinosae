# 当前数据分析流水线

本目录以 `original` 分支中的初始脚本为历史审计基线，但不导入、调用或修改它们；这些旧脚本不属于 `main` 当前发布树。本流水线重新分析目前已有的 1,264 粒种子、392 波段 CSV 平均光谱。目标不是制造更高的数字，而是把“随机种子级拟合能力”和“跨采集立方体的有限域迁移能力”区分开来，并保存可审核的样本层级信息。

## 一键运行

在仓库根目录执行：

```powershell
python current_data_study/analyze.py
```

若需安装依赖：

```powershell
python -m pip install -r current_data_study/requirements.txt
```

`requirements-lock.txt` 记录本次实际生成结果所用的精确版本；若需最大程度复算相同输出，使用该文件。`results.json` 同时保存实际 Python、NumPy、pandas、SciPy 和 scikit-learn 版本及平台信息。

可选参数：

```powershell
python current_data_study/analyze.py `
  --data-root data `
  --config current_data_study/config.json `
  --output-dir current_data_study/outputs
```

运行基本测试：

```powershell
python -m unittest discover -s current_data_study/tests -v
```

生成与锁定结果直接关联的稿件图（PNG 与可编辑 PDF）：

```powershell
python current_data_study/make_figures.py
```

该命令生成协议性能对比、按类别分面的两来源立方体光谱图，以及 SNV–LR 的三种来源隔离混淆矩阵。图中不绘制会把种子误当成独立批次的推断性误差条；原始绘图数值同时保存在 `figures/figure_performance_source.csv`。

## 数据约定

脚本读取 `data/<label>-<replicate>/<seed_id>.csv`。每个文件必须为 392 行、两列纯数值：第一列是真实波长（nm），第二列是平均反射率。载入时检查：

- 392 波段及两列结构；
- 全部值有限且波长严格递增；
- 所有样本使用相同实测波长网格；
- 标签从 0 连续编码；
- 每个文件的 SHA-256 及全数据组合指纹。

`dataset_manifest.csv` 保留 `label`、`source_cube`、`seed_id`、相对路径与文件哈希。分析绝不把 1,264 粒种子表述为 1,264 个独立产地批次。

## 固定模型与预处理

所有参数在运行前由 `config.json` 固定，不进行网格搜索或测试集调参：

| 标识 | 预处理 | 分类器 |
|---|---|---|
| `raw_lr` | StandardScaler | 多分类 Logistic Regression，C=1，tol=1e-10 |
| `raw_svm` | StandardScaler | RBF-SVM，C=10，gamma=scale |
| `raw_pls_da` | StandardScaler | PLS-DA，20 个固定潜变量、one-hot 响应后取 argmax |
| `raw_rf` | StandardScaler（为复现旧脚本） | Random Forest，200 棵树 |
| `snv_lr` | 单光谱 SNV + StandardScaler | Logistic Regression |
| `msc_lr` | 仅从训练集学习参考谱的 MSC + StandardScaler | Logistic Regression |
| `sg_smooth_lr` | SG 平滑（窗口 11、二阶多项式）+ StandardScaler | Logistic Regression |
| `sg_first_derivative_lr` | SG 一阶导数（窗口 11、二阶多项式）+ StandardScaler | Logistic Regression |

SNV 和 SG 是逐样本确定性变换；MSC 参考光谱及 StandardScaler 参数只从各折训练数据拟合。

全部 LR 变体统一使用 `tol=1e-10`（而非依据准确率逐模型挑选），以降低训练行顺序对 `lbfgs` 提前停止位置的影响。数值诊断中，随机留出训练索引保持原顺序或排序后，严格收敛的 raw-LR 均为 275/316、SNV-LR 均为 305/316；两次拟合的最大系数绝对差分别约为 `5.33e-6` 和 `3.96e-6`，预测均无差异。旧脚本默认较松的停止阈值下，SNV-LR 曾出现一粒种子的行序敏感差异，因此不能挑选较高的一次作为结果。

## 验证协议

1. `random_seed_holdout`：随机种子级分层 75/25 留出，固定随机种子 42，用于复现旧 LR/SVM/RF 结果；同一 `source_cube` 会跨训练和测试，因此不能称为独立外部验证。
2. `suffix_1_to_2`：全部 `*-1` 立方体训练、全部 `*-2` 立方体测试。
3. `suffix_2_to_1`：上述方向反转。
4. `leave_one_cube_out`：16 折，每次完整留出一个采集立方体，最后合并所有 out-of-fold 预测计算总体指标。

后 3 种分析保证同一 `source_cube` 不跨训练和测试，但每类的两个立方体仍可能属于同一商业批次，且训练中总有同类的另一个立方体。它们不是未知农场、独立批次、年份或仪器的外部验证。

## 指标和区间

输出 accuracy、balanced accuracy、macro-F1、混淆矩阵，以及：

- accuracy 的 Wilson 95% 区间；
- 三项指标的 2,000 次普通种子级 bootstrap 百分位区间。

种子嵌套于立方体且并非独立批次，Wilson 和普通 bootstrap 的独立性前提不成立。因此这些区间明确标记为 `descriptive_seed_level_only`，只能描述当前逐种子预测，不能作为批次级或产地级推断区间。当前只有每类两个立方体，也不足以可靠估计 cluster bootstrap。

## 结果文件

脚本确定性覆盖 `current_data_study/outputs/` 中的同名结果文件：

- `dataset_manifest.csv`：样本层级与数据校验信息；
- `wavelengths.csv`：真实 392 波段网格；
- `metrics.csv`：所有协议与模型的汇总指标；
- `predictions.csv`：可追溯到种子和立方体的逐条预测；
- `fold_metrics.csv`：留一立方体逐折准确率；
- `confusion_matrices.csv`：长表格式混淆矩阵；
- `results.json`：配置、软件版本、数据指纹、限制声明和指标；
- `report.md`：从机器可读结果自动生成的中文报告。

`current_data_study/figures/` 保存三组稿件候选图的 300 dpi PNG、矢量 PDF 及绘图源表。所有图由 `make_figures.py` 从上述锁定结果和原始 CSV 光谱确定性生成。

## 最重要的解释边界

本流水线能够回答“当前 16 个采集立方体之间有多少可重复的类别判别信号”，不能回答“模型是否能鉴别来自未知农场/年份/供应商的真实地理产地”。在得到带链路追踪的独立批次前，论文应将任务降格表述为**同一采集域中的八组商业样品闭集判别**，并把随机划分结果作为乐观上界而非部署性能。
