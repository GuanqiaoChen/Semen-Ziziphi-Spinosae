# 酸枣仁高光谱研究重构工作区

本仓库已从“以种子级随机拆分证明地理产地鉴别”重构为**来源立方体感知的探索性研究**。原始稿件和旧代码保留用于审计；新分析不会把 1,264 粒种子误写成 1,264 个独立产地重复。

`original` 只作为不可变审计基线；`main` 的算法、依赖和实现不受旧脚本约束。`main` 始终按顶刊证据标准设计方法、基线、消融、证伪、验证和复现流程；若当前数据不足，必须收窄结论，而不是降低方法标准。

## 首先阅读

1. [研究审查与修订总账](docs/研究审查与修订总账.md)：中文证据审查、顶刊差距、重构路线与只追加修订记录。
2. [现有数据条件下的研究重构方案](docs/现有数据条件下的研究重构方案.md)：当前数据可以和不可以回答的问题、验证阶梯与结论门槛。
3. [数据采集协作需求书](docs/数据采集协作需求书.md)：提交给数据采集团队的中文采样、元数据、化学参照、交付与验收要求。
4. [现有数据重写稿](paper/manuscript_current_data_reframed.md)：英文完整论文草稿；原始 `paper/manuscript_7.5.docx` 未覆盖。

## 已执行的现有数据分析

入口与说明见 [current_data_study/README.md](current_data_study/README.md)。在仓库根目录运行：

```powershell
python current_data_study/analyze.py
python -m unittest discover -s current_data_study/tests -v
python current_data_study/make_figures.py
```

锁定复跑读取 1,264 个 CSV 平均光谱、392 个真实波长和 16 个来源立方体，输出逐种子预测、混淆矩阵、分折指标、软件版本与数据 SHA-256 指纹。关键结果为：

- 随机种子留出：HS3I-Net 旧结果 96.84%；新执行 SNV–LR 96.52%，SG 一阶导数–LR 97.15%。
- 双向来源立方体隔离：PLS-DA 为 85.34% / 89.85%，SNV–LR 为 83.39% / 88.46%，原始 LR 为 79.64% / 80.77%。
- 留一来源立方体汇总：SNV–LR 79.03%；模型排序相较双向拆分发生明显变化。

这些结果只能描述当前八个商业样品标签在有限采集域中的闭集区分，不能估计未知农场、独立批次、年份、供应商或设备上的地理产地性能。

## 深度模型状态

[deep_models/grouped_hs3i_current_data.py](deep_models/grouped_hs3i_current_data.py) 提供来源立方体完全隔离的 HS3I、无 SelecVar、空间打乱和仅掩膜协议，并固定双向拆分与多个训练随机种子。项目 `.venv` 已安装并验证 CUDA 版 PyTorch 与 h5py；真实 MAT 读取、RTX 4060 GPU 张量运算以及两个模型分支的单样本前向路径均已通过。**正式 36 单元训练仍未执行，因此仍无新增深度模型性能结果。**详见 [deep_models/README.md](deep_models/README.md)。

从仓库根目录重建相同环境：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m pip install -r requirements-torch-cu126.txt
```

## 结果与图形

- 中文自动分析报告：`current_data_study/outputs/report.md`
- 机器可读汇总：`current_data_study/outputs/metrics.csv`
- 逐种子预测：`current_data_study/outputs/predictions.csv`
- 数据与环境审计：`current_data_study/outputs/results.json`
- 300 dpi PNG、矢量 PDF 与绘图源表：`current_data_study/figures/`

## 投稿边界

当前重写稿是一篇诚实的探索性/方法学警示型草稿，不是顶刊级地理溯源确认研究。达到该目标仍需可追溯独立农场或批次、多年份、混板采集、第二设备或实验室、配对化学测定、未知类/掺伪挑战和锁定外部测试。具体执行要求已经写入数据采集协作需求书。

此后凡修改论文、方法、代码、数据、统计或图表，必须按 [AGENTS.md](AGENTS.md) 在中文总账末尾追加记录，不得覆盖旧记录或把计划分析写成已执行结果。
