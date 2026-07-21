# 当前数据顶刊方法后处理摘要 / Top-journal current-data post-processing summary

## 中文审计摘要

本后处理只读取已标记为 `executed_complete` 的完整固定矩阵。主要预测器是三个预声明优化种子的温度校准概率平均集成；主要估计量是两个来源立方体迁移方向等权的 balanced accuracy（θ）。全部模型与反事实条件使用同一个由固定随机种子 `20260721` 生成的 10,000 次八标签对block重采样矩阵；区间为固定95%百分位区间。可靠性数据严格使用预注册的10个等宽置信度箱。

| 模型 / Model | θ | 条件性95%区间 / Conditional 95% interval |
| --- | --- | --- |
| snv_lr | 86.94% | 75.38%–96.20% |
| spectral_only | 44.75% | 25.31%–63.58% |
| fusion_net | 45.94% | 31.98%–60.70% |

| 效应 / Effect | 双向效应 / Effect | 条件性95%区间 / Conditional 95% interval |
| --- | --- | --- |
| fusion_net(full) - fusion_net(spatial_shuffle) | 13.15% | 1.82%–24.38% |
| fusion_net(spatial_shuffle) - fusion_net(mean_broadcast) | 0.28% | -0.50%–0.99% |
| fusion_net(full) - fusion_net(mean_broadcast) | 13.43% | 1.71%–24.76% |
| fusion_net(mean_broadcast) - fusion_net(mask_only) | 11.99% | 4.10%–21.96% |
| fusion_net(full) - spectral_only(full) | 1.19% | -11.29%–15.55% |
| fusion_net(full) - snv_lr(full) | -41.00% | -50.86%–-31.46% |
| fusion_net(mask_only) - chance (0.125) | 8.02% | -5.70%–23.54% |

主要 `fusion_net(full) − fusion_net(spatial_shuffle)` 效应为 13.15%，条件性95%区间为 1.82%–24.38%。八标签对精确单侧sign-flip检验 `p=0.0429688`，双侧敏感性值 `p=0.0859375`。依据训练入口已冻结的方向、效应量和5/6 seed稳定性联合规则，本次结果**达到预声明的有限空间排列支持门槛**。

上述区间与检验只条件于当前16个存档来源立方体。它们不是新农场、新批次、年份、供应商、仪器或实验室的总体区间；三个优化种子也不是生物重复。即使空间门槛通过，也只能说明当前配对来源立方体迁移中模型使用了可用的前景内部排列信息，不能解释为地理产地组织结构、化学机制或外部泛化。

## English audit summary

This post-processor accepts only an `executed_complete` locked matrix. The primary predictor is the temperature-scaled probability ensemble across the three preregistered optimization seeds. The primary estimand is balanced accuracy averaged equally over the two reciprocal source-cube directions. Every model and counterfactual shares one 10,000-draw, eight-label-pair block-bootstrap index matrix generated with seed `20260721`; intervals are fixed 95% percentile intervals. Reliability data use the preregistered ten equal-width confidence bins.

| 模型 / Model | θ | 条件性95%区间 / Conditional 95% interval |
| --- | --- | --- |
| snv_lr | 86.94% | 75.38%–96.20% |
| spectral_only | 44.75% | 25.31%–63.58% |
| fusion_net | 45.94% | 31.98%–60.70% |

| 效应 / Effect | 双向效应 / Effect | 条件性95%区间 / Conditional 95% interval |
| --- | --- | --- |
| fusion_net(full) - fusion_net(spatial_shuffle) | 13.15% | 1.82%–24.38% |
| fusion_net(spatial_shuffle) - fusion_net(mean_broadcast) | 0.28% | -0.50%–0.99% |
| fusion_net(full) - fusion_net(mean_broadcast) | 13.43% | 1.71%–24.76% |
| fusion_net(mean_broadcast) - fusion_net(mask_only) | 11.99% | 4.10%–21.96% |
| fusion_net(full) - spectral_only(full) | 1.19% | -11.29%–15.55% |
| fusion_net(full) - snv_lr(full) | -41.00% | -50.86%–-31.46% |
| fusion_net(mask_only) - chance (0.125) | 8.02% | -5.70%–23.54% |

The primary `fusion_net(full) − fusion_net(spatial_shuffle)` effect was 13.15%, with a conditional 95% interval of 1.82%–24.38%. The exact eight-label-pair sign-flip result was `p=0.0429688` for the preregistered greater alternative and `p=0.0859375` for the two-sided sensitivity analysis. Under the frozen joint direction/effect-size/5-of-6-seed rule, the result **met the preregistered limited-support gate**.

All intervals and tests are conditional on the 16 archived source cubes. They are not population intervals for new farms, lots, years, suppliers, instruments, or laboratories. Optimization seeds are not biological replicates. Passing the spatial gate would support only the use of within-foreground arrangement in these reciprocal cube transfers; it would not establish geographical tissue structure, chemical causality, or external generalization.

## Audit artifacts

- `bootstrap_label_pair_indices.csv`: the single shared resampling matrix;
- `bootstrap_theta_intervals.csv` and `bootstrap_effect_intervals.csv`: conditional intervals;
- `primary_sign_flip_label_effects.csv`, `primary_sign_flip_null_distribution.csv`, and `primary_sign_flip_test.json`;
- `main_results.csv`, `counterfactual_effects.csv`, `reliability_data.csv`, and `ensemble_confusion_matrices.csv`;
- four publication figures, each in PNG and PDF;
- `postprocessing_manifest.json`: input hashes, locked constants, and generated-file hashes.
