# 酸枣仁高光谱研究：来源立方体隔离审计

本仓库已完成一套**结果前冻结、来源立方体完全隔离、包含强基线与反事实证伪**的当前数据分析。研究对象是 8 个存档商业样品标签、16 个来源高光谱立方体和 1,264 粒立方体内技术子样本；不会把种子数误写成独立产地重复。

`original` 是不可变审计基线；初始 Python 算法和 MATLAB 提取脚本只在该分支及本地忽略副本中保留，不属于 `main` 当前发布树。`main` 的正式代码由 `current_data_study/` 经典审计和 `deep_models/` 预注册方法组成，独立于旧脚本并按预注册、泄漏安全且可审计的方法学标准设计。当前唯一正式深度分析入口是 [deep_models/source_cube_audit.py](deep_models/source_cube_audit.py)，完整协议与复现说明见 [deep_models/README.md](deep_models/README.md)。

## 正式分析状态

- 预注册协议：[来源立方体预注册分析方案](docs/来源立方体预注册分析方案.md)
- 正式运行：`executed_complete`，2 个互反方向 × 3 个模型 × 3 个固定优化种子，按运行清单共 18 个训练单元（含 SNV–LR 拟合）
- 数据隔离：所有 `*-1` 来源立方体开发、所有 `*-2` 测试，以及完全反向测试；测试立方体不参与选择、早停或温度校准
- 模型：SNV–LR 强基线、`spectral_only` 深度光谱消融、`fusion_net` 高效谱空融合网络
- 证伪：每个锁定神经模型均在不重新拟合的前提下评估 `full`、`spatial_shuffle`、`mean_broadcast` 和 `mask_only`
- 主要预测器：3 个预声明种子的温度校准概率平均集成
- 主要估计量：两个方向 balanced accuracy 等权平均的 \(\theta\)

## 正式结果

| 模型 | \(\theta\) | 条件性 95% 区间 |
| --- | ---: | ---: |
| SNV–LR | 86.94% | 75.38%–96.20% |
| `spectral_only` | 44.75% | 25.31%–63.58% |
| `fusion_net` | 45.94% | 31.98%–60.70% |

`fusion_net(full) - fusion_net(spatial_shuffle)` 为 13.15 个百分点（条件性 95% 区间 1.82–24.38）；8 标签对精确单侧 sign-flip `p=0.0429688`，双侧敏感性 `p=0.0859375`，并达到预声明的有限空间排列支持门槛。但效应高度方向不对称（0.78 与 25.52 个百分点）。

融合模型相对 `spectral_only` 仅高 1.19 个百分点（区间 −11.29–15.55），没有稳定谱空增益；相对 SNV–LR 低 41.00 个百分点（区间 −50.86–−31.46）。因此，当前数据支持的是一个严格、可证伪的来源立方体迁移审计，而不是“深度模型优于强光谱基线”。

完整双语摘要、条件性区间与图表见：

- [正式后处理摘要](deep_models/outputs/source_cube_preregistered_audit/postprocessing/summary.md)
- [机器可读正式结果](deep_models/outputs/source_cube_preregistered_audit/results.json)
- [空间机制预声明判定](deep_models/outputs/source_cube_preregistered_audit/spatial_mechanism_decision.json)
- [主性能图](deep_models/outputs/source_cube_preregistered_audit/postprocessing/figure_main_performance.pdf)
- [反事实效应图](deep_models/outputs/source_cube_preregistered_audit/postprocessing/figure_counterfactual_effects.pdf)
- [集成混淆矩阵](deep_models/outputs/source_cube_preregistered_audit/postprocessing/figure_ensemble_confusion_matrices.pdf)
- [校准可靠性图](deep_models/outputs/source_cube_preregistered_audit/postprocessing/figure_calibration_reliability.pdf)

## 论文与研究记录

- [当前数据英文重写稿](paper/manuscript_current_data_reframed.md)
- [正式执行与结果审计](docs/来源立方体预注册分析执行与结果审计.md)
- [研究审查与修订总账](docs/研究审查与修订总账.md)
- [现有数据研究重构方案](docs/现有数据条件下的研究重构方案.md)
- [数据采集协作需求书](docs/数据采集需求.md)
- [经典光谱基线与遗留结果说明](current_data_study/README.md)

## 结论边界

所有区间和检验都只条件于当前 16 个存档来源立方体。每类、每个测试方向只有 1 个测试来源立方体，两个后缀是否对应独立批次或采集会话也没有充分元数据；优化种子不是生物重复。当前结果不能证明地理产地认证、真实溯源、化学机制、开放集识别或对新农场、批次、年份、供应商、仪器和实验室的外部泛化。

若要获得上述证据，仍需按 [数据采集协作需求书](docs/数据采集需求.md) 增加可追溯独立来源、多年份/批次、混板采集、跨设备或实验室验证、配对化学测定与锁定外部测试。

此后修改论文、方法、代码、数据、结果或图表，必须按 [AGENTS.md](AGENTS.md) 在中文总账末尾追加对应记录，不得把计划分析写成已执行结果。
