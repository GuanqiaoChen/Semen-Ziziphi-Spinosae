# 采集域稳健的校准化产地溯源开发集报告

> 状态：仅使用构造批次 0–7 新执行的开发集探索；构造批次 8–9 的数值与字节均未读取。跨立方体（来源图像）迁移是采集域稳健性压力测试，不是外部地理认证。

## 诚实结论摘要

1. **表示选择有明显效果**：SG 一阶导数 + 收缩 LDA 的跨采集域平衡准确率显著高于 SNV/MSC/SG2 等表示（见比较矩阵），是当前数据上最稳健的产地判别表示。
2. **无监督进样立方体自标准化有中等正效应**：仅用目标立方体的未标注特征做逐立方体标准化，可提升跨采集域平衡准确率（见自适应效应表，含簇自助区间）。
3. **温度校准修复跨域校准崩塌（大且稳健）**：但**分组（shift-aware）温度与普通 iid 温度不可区分**——在单一训练立方体内，构造批次不是足够不同的采集域，分组校准并未带来额外收益（阴性结果）。
4. **保形集在跨域下欠覆盖**，分组与 iid 保形无实质差异（阴性结果）。
5. **逐种子协方差/像素群体特征在跨采集域下有害**（阴性结果）。

因此当前数据支持的是一个严格、可证伪的采集域稳健性与校准框架及适度的部署流水线，而不是具有巨大效应的全新算法。

## 无监督进样立方体自适应（准确率效应, sg1/lda + shift 校准）

| 方法 | 平衡准确率 | 准确率 | NLL | ECE |
|---|---:|---:|---:|---:|
| source_standardize | 0.9361 | 0.9348 | 0.1788 | 0.0095 |
| target_standardize | 0.9506 | 0.9496 | 0.1529 | 0.0149 |

自适应效应（target − source 标准化）：平衡准确率 +0.0145，95% 簇自助区间 [+0.0075, +0.0211]，自助为正比例 1.000。

## 跨采集域校准效应（hero = sg1 / lda）

| 比较 | 指标 | 点降幅 | 95% 簇自助区间 | 自助为正比例 |
|---|---|---:|---:|---:|
| uncalibrated_minus_shift_aware_temperature | expected_calibration_error | +0.0283 | [+0.0093, +0.0347] | 0.998 |
| uncalibrated_minus_iid_temperature | expected_calibration_error | +0.0290 | [+0.0088, +0.0355] | 0.998 |
| iid_minus_shift_aware | expected_calibration_error | -0.0007 | [-0.0015, +0.0015] | 0.425 |
| uncalibrated_minus_shift_aware_temperature | negative_log_likelihood | +0.0571 | [+0.0205, +0.0984] | 1.000 |
| uncalibrated_minus_iid_temperature | negative_log_likelihood | +0.0578 | [+0.0204, +0.0998] | 1.000 |
| iid_minus_shift_aware | negative_log_likelihood | -0.0006 | [-0.0016, +0.0001] | 0.048 |
| uncalibrated_minus_shift_aware_temperature | multiclass_brier_score | +0.0054 | [-0.0011, +0.0123] | 0.956 |
| uncalibrated_minus_iid_temperature | multiclass_brier_score | +0.0055 | [-0.0011, +0.0126] | 0.954 |
| iid_minus_shift_aware | multiclass_brier_score | -0.0001 | [-0.0003, +0.0001] | 0.122 |

## 同域参考（LOBO, sg1/lda）

| 校准 | 平衡准确率 | NLL | ECE | Brier |
|---|---:|---:|---:|---:|
| uncalibrated | 0.9758 | 0.1044 | 0.0133 | 0.0402 |
| shift_aware_temperature | 0.9758 | 0.0882 | 0.0064 | 0.0391 |

## 跨立方体保形覆盖（目标 90%）

| 方法 | 方向 | 覆盖率 | 平均集大小 | 空集比例 |
|---|---|---:|---:|---:|
| shift_aware_group_conformal | cube1_to_cube2 | 0.8049 | 0.8252 | 0.1748 |
| shift_aware_group_conformal | cube2_to_cube1 | 0.8135 | 0.8173 | 0.1827 |
| iid_split_conformal | cube1_to_cube2 | 0.8110 | 0.8354 | 0.1646 |
| iid_split_conformal | cube2_to_cube1 | 0.7981 | 0.8019 | 0.1981 |

## 跨采集域比较矩阵（pooled 两方向）

| 表示 | 分类器 | 校准 | 平衡准确率 | NLL | ECE | Brier |
|---|---|---|---:|---:|---:|---:|
| raw | lda | uncalibrated | 0.8518 | 0.4896 | 0.0240 | 0.2262 |
| raw | lda | iid_temperature | 0.8518 | 0.4966 | 0.0346 | 0.2267 |
| raw | lda | shift_aware_temperature | 0.8518 | 0.5004 | 0.0391 | 0.2273 |
| raw | lr | uncalibrated | 0.7974 | 0.6318 | 0.0752 | 0.3015 |
| raw | lr | iid_temperature | 0.7974 | 0.6222 | 0.0281 | 0.2882 |
| raw | lr | shift_aware_temperature | 0.7974 | 0.6236 | 0.0276 | 0.2874 |
| raw | svm | uncalibrated | 0.6921 | 0.9979 | 0.0502 | 0.4891 |
| raw | svm | iid_temperature | 0.6921 | 0.9976 | 0.0564 | 0.4901 |
| raw | svm | shift_aware_temperature | 0.6921 | 0.9981 | 0.0502 | 0.4893 |
| snv | lda | uncalibrated | 0.8683 | 0.4144 | 0.0671 | 0.2053 |
| snv | lda | iid_temperature | 0.8683 | 0.3625 | 0.0417 | 0.1955 |
| snv | lda | shift_aware_temperature | 0.8683 | 0.3644 | 0.0424 | 0.1960 |
| snv | lr | uncalibrated | 0.8574 | 0.4940 | 0.0334 | 0.2044 |
| snv | lr | iid_temperature | 0.8574 | 0.5024 | 0.0397 | 0.2053 |
| snv | lr | shift_aware_temperature | 0.8574 | 0.5004 | 0.0384 | 0.2048 |
| snv | svm | uncalibrated | 0.7846 | 0.7832 | 0.1300 | 0.3861 |
| snv | svm | iid_temperature | 0.7846 | 0.7629 | 0.0281 | 0.3616 |
| snv | svm | shift_aware_temperature | 0.7846 | 0.7635 | 0.0140 | 0.3610 |
| msc | lda | uncalibrated | 0.8653 | 0.4709 | 0.0686 | 0.2181 |
| msc | lda | iid_temperature | 0.8653 | 0.4203 | 0.0501 | 0.2111 |
| msc | lda | shift_aware_temperature | 0.8653 | 0.4230 | 0.0509 | 0.2115 |
| msc | lr | uncalibrated | 0.8491 | 0.5214 | 0.0305 | 0.2179 |
| msc | lr | iid_temperature | 0.8491 | 0.5324 | 0.0432 | 0.2190 |
| msc | lr | shift_aware_temperature | 0.8491 | 0.5300 | 0.0424 | 0.2183 |
| msc | svm | uncalibrated | 0.7846 | 0.7843 | 0.1290 | 0.3869 |
| msc | svm | iid_temperature | 0.7846 | 0.7642 | 0.0262 | 0.3630 |
| msc | svm | shift_aware_temperature | 0.7846 | 0.7650 | 0.0141 | 0.3622 |
| sg1 | lda | uncalibrated | 0.9361 | 0.2359 | 0.0379 | 0.0985 |
| sg1 | lda | iid_temperature | 0.9361 | 0.1781 | 0.0088 | 0.0930 |
| sg1 | lda | shift_aware_temperature | 0.9361 | 0.1788 | 0.0095 | 0.0931 |
| sg1 | lr | uncalibrated | 0.8951 | 0.3670 | 0.0258 | 0.1657 |
| sg1 | lr | iid_temperature | 0.8951 | 0.3846 | 0.0361 | 0.1681 |
| sg1 | lr | shift_aware_temperature | 0.8951 | 0.3887 | 0.0361 | 0.1686 |
| sg1 | svm | uncalibrated | 0.8654 | 0.6318 | 0.2123 | 0.3061 |
| sg1 | svm | iid_temperature | 0.8654 | 0.5544 | 0.0259 | 0.2403 |
| sg1 | svm | shift_aware_temperature | 0.8654 | 0.5675 | 0.0392 | 0.2411 |
| sg2 | lda | uncalibrated | 0.9058 | 0.5243 | 0.0770 | 0.1672 |
| sg2 | lda | iid_temperature | 0.9058 | 0.2694 | 0.0426 | 0.1432 |
| sg2 | lda | shift_aware_temperature | 0.9058 | 0.2570 | 0.0361 | 0.1411 |
| sg2 | lr | uncalibrated | 0.8551 | 0.4577 | 0.0577 | 0.2206 |
| sg2 | lr | iid_temperature | 0.8551 | 0.4677 | 0.0595 | 0.2218 |
| sg2 | lr | shift_aware_temperature | 0.8551 | 0.4744 | 0.0610 | 0.2227 |
| sg2 | svm | uncalibrated | 0.8393 | 0.6573 | 0.1850 | 0.3298 |
| sg2 | svm | iid_temperature | 0.8393 | 0.6592 | 0.0876 | 0.2910 |
| sg2 | svm | shift_aware_temperature | 0.8393 | 0.6895 | 0.1015 | 0.2944 |
| sg1 | lda_lr_ensemble | uncalibrated | 0.9370 | 0.2385 | 0.0228 | 0.1065 |
| sg1 | lda_lr_ensemble | iid_temperature | 0.9309 | 0.2158 | 0.0174 | 0.1093 |
| sg1 | lda_lr_ensemble | shift_aware_temperature | 0.9309 | 0.2161 | 0.0169 | 0.1094 |

## 负消融：逐种子协方差描述子（跨立方体）

| 特征 | 跨立方体平衡准确率 | 跨立方体 NLL |
|---|---:|---:|
| sg1_reference | 0.9365 | 0.2391 |
| sg1_plus_covariance | 0.9204 | 0.6057 |

## 解释边界

构造批次由导师授权、按确定性规则划分，是当前数据内部的分组单位，但仍共享每产地两张来源图像，不是新采集的物理批次。本报告的采集域稳健性证据不能替代跨年份、跨农场、跨仪器或未知产地的外部验证。是否将本方法升级为主预测器，须由一次性锁定评估决定。
