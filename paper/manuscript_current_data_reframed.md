# Source-cube-isolated hyperspectral classification of commercial *Ziziphi Spinosae* Semen: a preregistered counterfactual audit

> **Author-facing draft status—remove before submission.** This manuscript integrates the completed preregistered current-data analysis executed from clean `main` commit `4bc191c2e9b8a809e866ccd15d96fea29378969d`. It is a methodologically complete audit of the 16 archived source cubes, not a submission-ready claim of geographical-origin authentication. The third author affiliation, final CRediT roles, and a public repository DOI remain unavailable in the audited materials and must be supplied and verified by the authors. No placeholder is presented as an observed result.

Ziying Wang<sup>a</sup>, Guanqiao Chen<sup>b</sup>, Jiale Wei<sup>c</sup>, Xumei Wang<sup>a,*</sup>

<sup>a</sup> School of Pharmacy, Xi'an Jiaotong University, Xi'an 710049, China  
<sup>b</sup> Tandon School of Engineering, New York University, New York, NY 11201, USA  
<sup>c</sup> School of Artificial Intelligence, The Chinese University of Hong Kong, Shenzhen, Guangdong 518172, China
<sup>*</sup> Corresponding author: wangxumei@mail.xjtu.edu.cn

## Highlights

- The 1,264 seed patches are technical subsamples nested within only 16 source hyperspectral cubes.
- A preregistered 18-unit matrix used reciprocal complete-cube isolation, three optimization seeds, calibration, and probability ensembling.
- SNV–logistic regression reached 86.94% balanced accuracy, substantially exceeding both frozen neural models.
- Removing within-foreground spatial arrangement reduced fusion performance by 13.15 percentage points and met a limited-support gate.
- The spatial effect was direction-dependent and did not establish deep-model superiority, geographical causality, or external validity.

## Abstract

Randomly dividing objects extracted from the same hyperspectral scene can yield optimistic estimates of performance on new acquisition units. We audited this problem in a near-infrared hyperspectral dataset of commercial *Ziziphi Spinosae* Semen (SZR). The archive contains 1,264 segmented seed patches assigned to eight supplier-reported location labels but nested within only 16 source cubes, two per label. Cultivation provenance, independent lot identity, harvest year, processing history, and chain of custody were unavailable. We first retained the legacy seed-level split and classical spectral diagnostics as data-informed exploratory analyses. We then froze and executed a new protocol before inspecting its test results: models were developed on all suffix-1 cubes and tested on all suffix-2 cubes, and vice versa; no cube crossed either boundary. Within each development direction, seed-level validation controlled fitting and temperature scaling but was not treated as independent validation. Three fixed optimization seeds were used for standard-normal-variate logistic regression (SNV–LR), a residual spectral network, and an efficient spectral–spatial fusion network, giving 18 training units. The primary predictor averaged seed-specific temperature-scaled class probabilities. Balanced accuracy was averaged equally over the two reciprocal directions, and uncertainty was estimated by 10,000 paired bootstrap resamples of the eight commercial-label cube pairs. SNV–LR achieved 86.94% (conditional 95% interval, 75.38–96.20), compared with 44.75% (25.31–63.58) for the spectral network and 45.94% (31.98–60.70) for fusion. For the same locked fusion models, full input exceeded within-foreground spatial shuffling by 13.15 percentage points (1.82–24.38); the exact eight-pair sign-flip test gave one-sided *p*=0.0429688 and two-sided sensitivity *p*=0.0859375. The preregistered limited-support gate was met, although the directional effects were 0.78 and 25.52 points. Fusion did not outperform the spectral network (+1.19 points; −11.29 to 15.55) and was markedly inferior to SNV–LR (−41.00 points; −50.86 to −31.46). Thus, the fusion model used some foreground-internal arrangement in these particular reciprocal cube transfers, but this did not confer competitive classification. All intervals are conditional on the 16 archived cubes. The study supports an acquisition-aware methodological audit, not geographical-origin authentication, chemical mechanism, or generalization to new lots, suppliers, years, instruments, or laboratories.

**Keywords:** hyperspectral imaging; *Ziziphi Spinosae* Semen; grouped validation; source-cube shift; counterfactual intervention; spectral–spatial fusion; calibration; conditional uncertainty

## 1. Introduction

*Ziziphi Spinosae* Semen (SZR) is used as a medicinal and food-homologous seed material, and its commercial value has motivated research on identity, quality, adulteration, and provenance [1–3]. Laboratory chromatography and mass spectrometry can characterize constituents or volatile profiles, but they require specialized workflows and are not always suited to rapid screening. Near-infrared hyperspectral imaging (HSI) is attractive because it records a reflectance spectrum at each spatial location while leaving an intact seed available for subsequent analysis [4].

HSI data are intrinsically hierarchical. Multiple pixels belong to one seed, multiple seeds can be segmented from one scene, and multiple scenes may belong to the same commercial lot, supplier, harvest, acquisition session, or instrument. A classifier evaluated after randomly allocating lower-level observations can exploit scene- or batch-specific signals shared by training and test data. Such an estimate describes interpolation within the sampled acquisition domain, not necessarily transfer to an unseen cube, commercial lot, or geographical origin. Grouped validation is therefore required whenever the intended deployment unit is higher in the hierarchy than the unit used for fitting [5].

This distinction is especially important in provenance research. A reported location can be confounded with supplier, lot, cultivar, maturity, drying, storage, moisture, acquisition date, illumination, and instrument response. Botanical authentication establishes species identity but does not verify cultivation location. Evidence for origin authentication requires traceable, independently replicated lots and validation that withholds those lots or other deployment-relevant domains. Without such information, “origin” is an observational class label rather than an identified cause of spectral differences.

Model architecture does not resolve this identification problem. Zhao et al. previously combined HSI, image and spectrum convolutional neural networks, Savitzky–Golay and standard-normal-variate (SNV) preprocessing, learned wavelength weights, lipid and protein measurements, and HPLC spinosin analysis for SZR from five reported regions [6]. Zheng et al. described a first-layer attention module followed by a three-dimensional residual network for joint band weighting and spectral–spatial prediction [7]. Zhang et al. studied 30 SZR batches from three provinces using electronic-eye, gas-chromatographic electronic-nose, and headspace GC–MS measurements [8]. A 3D network with band weights is therefore not by itself a new analytical principle, and chemical attribution requires matched measurements rather than wavelength assignment alone.

The present study reframes the available archive as a preregistered, acquisition-aware methodological audit. It separates evidence generated before the new protocol was frozen from evidence generated by the formal run. We address five bounded questions:

1. What is the recoverable experimental hierarchy, and which units are shared across the legacy development and prediction subsets?
2. Does the apparent advantage of the legacy cube model remain after competitive spectral preprocessing?
3. How do fixed models perform when complete source cubes define both reciprocal train–test boundaries?
4. Does a locked fusion model depend on within-foreground spatial arrangement under prespecified counterfactual interventions?
5. Which conclusions remain justified after uncertainty is evaluated at the eight paired-label blocks rather than by treating 1,264 seeds as independent replicates?

The contribution is not definitive geographical authentication. It is a transparent demonstration that experimental-unit definition, baseline strength, preregistration, counterfactual falsification, and independence-aware uncertainty can materially change the interpretation of a high-accuracy hyperspectral study.

## 2. Materials and methods

### 2.1 Intended use, analysis units, and estimands

The eight archived labels are retained to preserve correspondence with the supplied data. Throughout this paper they denote **commercial sample groups identified by supplier-reported purchase location**; they are not treated as independently verified cultivation origins.

The highest recoverable evaluation unit is the source cube. A seed patch is a technical subsample of a cube, not an independent provenance replicate. The formal current-data target is closed-set discrimination among eight archived commercial labels when every test seed comes from a source cube excluded from fitting. This target remains narrower than performance for a new lot, supplier, harvest, laboratory, instrument, or unknown class.

For model or ensemble predictor (m), the primary performance estimand was

$$
\theta_m=\frac{1}{2}\left(BA_{1\rightarrow2,m}+BA_{2\rightarrow1,m}\right),
$$

where (BA) is balanced accuracy, suffix 1 and suffix 2 identify the two cubes available per label, and the two directions receive equal weight. Because each direction contains one test cube for each of eight classes, this is also an equal-weight summary of the 16 cube-specific class recalls. Both directions are nevertheless reported separately because their average can conceal acquisition asymmetry.

The sole primary mechanism contrast was

$$
\Delta_{\mathrm{spatial}}=
\theta_{\mathrm{fusion,full}}-
\theta_{\mathrm{fusion,spatial\ shuffle}}.
$$

The three fixed optimization seeds were combined by averaging their eight-class probability vectors for each seed patch before calculating an ensemble metric. Seed-wise metrics quantify optimization instability; optimization seeds are not experimental or biological replicates.

We use the following terms:

- **Seed patch:** one segmented $32\times32\times392$ array associated with an individual connected component.
- **Source cube:** the hyperspectral scene from which a set of seed patches was extracted, represented by folders `0-1` through `7-2`.
- **Commercial label:** one of eight supplier-reported location labels encoded by prefixes 0–7.
- **Paired source cubes:** the two cubes sharing a commercial label; their independence as commercial lots is unknown.
- **External origin validation:** testing on traceable, independently acquired farms or lots outside model development. No such dataset is available.

### 2.2 Commercial samples and data hierarchy

Commercial dried SZR material was purchased from local suppliers under eight reported location labels: Shijiazhuang City, Hebei Province (HBS); Xian County, Hebei Province (HBX); Anyang City, Henan Province (HNA); Xinxiang City, Henan Province (HNX); Ningxia Hui Autonomous Region (NX); Daning County, Shanxi Province (SXD); Qingjian County, Shaanxi Province (SXQ); and Hetian City, Xinjiang Uyghur Autonomous Region (XJH). Professor Xumei Wang, School of Pharmacy, Xi'an Jiaotong University, confirmed the botanical identity as genuine SZR.

No farm identifiers, GPS coordinates, producer records, cultivar or genotype, wild/grafted status, harvest year, maturity, drying process, storage history, moisture content, independent supplier/lot identifiers, or chain-of-custody records are available in the present repository. Expert authentication is therefore interpreted as botanical authentication only.

The processed dataset comprises 1,264 seeds nested within 16 source cubes. Each label has exactly two cubes, but seed counts are unequal (Table 1). The source cube—not the seed—is the highest recoverable acquisition unit. The number of independent biological or commercial lots may be smaller than 16 and cannot be reconstructed from the metadata.

**Table 1. Recoverable hierarchy and allocation under the legacy random seed-level split.**

| Code | Supplier-reported location label | Source cube | Total seed patches | Legacy prediction patches | Legacy development patches |
|---|---|---:|---:|---:|---:|
| 0 | HBS | 0-1 | 80 | 20 | 60 |
| 0 | HBS | 0-2 | 80 | 20 | 60 |
| 1 | HBX | 1-1 | 80 | 14 | 66 |
| 1 | HBX | 1-2 | 80 | 26 | 54 |
| 2 | HNA | 2-1 | 80 | 21 | 59 |
| 2 | HNA | 2-2 | 80 | 19 | 61 |
| 3 | HNX | 3-1 | 80 | 20 | 60 |
| 3 | HNX | 3-2 | 80 | 20 | 60 |
| 4 | NX | 4-1 | 90 | 22 | 68 |
| 4 | NX | 4-2 | 80 | 21 | 59 |
| 5 | SXD | 5-1 | 80 | 22 | 58 |
| 5 | SXD | 5-2 | 80 | 18 | 62 |
| 6 | SXQ | 6-1 | 80 | 23 | 57 |
| 6 | SXQ | 6-2 | 80 | 17 | 63 |
| 7 | XJH | 7-1 | 80 | 23 | 57 |
| 7 | XJH | 7-2 | 54 | 10 | 44 |
| **Total** |  | **16 cubes** | **1,264** | **316** | **948** |

### 2.3 Hyperspectral acquisition and reflectance calibration

Images were acquired with a push-broom line-scan hyperspectral system from Hangzhou Hyperspectral Imaging Technology Co., Ltd., China. The archived manuscript reports a 600–1700 nm camera range, 512 original bands, spectral resolution of no more than 2.5 nm, and 640 spatial pixels. Two halogen lamps and a motorized translation stage were used. Seeds were arranged on a matte black board, nominally in a $10\times10$ grid.

The reported white/dark reflectance calibration was

$$
R = \frac{I_{\mathrm{raw}}-I_{\mathrm{dark}}}
{I_{\mathrm{white}}-I_{\mathrm{dark}}},
$$

where $I_{\mathrm{raw}}$, $I_{\mathrm{white}}$, and $I_{\mathrm{dark}}$ denote the raw scene, white reference, and dark reference, respectively. Raw `.hdr` scenes and calibration-reference files are absent from the audited repository; calibration and initial acquisition quality control therefore cannot be independently rerun. The archived preprocessing script manually crops each scene before segmentation. The retained processed field was reported as 460 × 535 pixels with 392 wavelengths spanning 949.764–1650.855 nm.

### 2.4 Segmentation and seed-level representations

A false-colour image was produced from three bands. Within a manually selected rectangular scene region, grayscale Otsu thresholding followed by $2\times2$ morphological closing and opening generated a binary foreground mask. Connected-component labelling identified individual seed candidates. Each component was centred in a $32\times32$ window; pixels outside its binary mask were set to zero, with zero padding when a window crossed a crop boundary.

Two representations were retained for each object:

1. a masked hyperspectral patch $X_i\in\mathbb{R}^{32\times32\times392}$; and
2. a mean foreground spectrum $\bar{x}_i\in\mathbb{R}^{392}$, averaging only mask-positive pixels.

The processed `.mat` files contain the patch, mask, and metadata; paired `.csv` files contain wavelength–reflectance rows for the mean spectrum. The formal loader checked every MAT-derived foreground mean against its paired CSV representation. All 1,264 pairs passed an absolute tolerance of $10^{-5}$; the maximum absolute discrepancy was $2.8260\times10^{-7}$. Because zero-valued background encodes the seed silhouette, a cube model can potentially use morphology, centring, orientation, truncation, or segmentation artefacts as well as tissue-level spectral heterogeneity.

### 2.5 Wavelength vector and spectral precision

All revised analyses read the measured wavelength vector stored in the CSV files. The legacy SelecVar plotting code instead constructed 392 equally spaced values with `linspace(949.764, 1650.855, 392)`. Comparison with the recorded nonlinear grid showed absolute discrepancies up to 5.77 nm. Previously reported single-wavelength peaks mapped by that linear grid are therefore nominal and cannot support sub-band interpretation. The formal loader verified a common, strictly increasing 392-band grid across all 1,264 spectra and exported it with the run artifacts.

### 2.6 Validation designs and analysis chronology

#### 2.6.1 Legacy seed-level benchmark

The original split was reconstructed by stratifying all 1,264 seed patches by label and applying scikit-learn `train_test_split` (`test_size=0.25`, `random_state=42`), giving a 948-patch development subset and a 316-patch prediction subset. Five-fold stratified cross-validation had been used inside the development subset for legacy model development.

This prediction subset is not independent at source-cube or lot level. It contains 10–26 patches from every one of the same 16 cubes represented in development (Table 1). The archived HS3I-Net aggregate is retained as a legacy reported result; classical pipelines newly executed on the reconstructed split are data-informed exploratory evidence. Neither is relabelled as formal current-data confirmation.

#### 2.6.2 Preregistered reciprocal complete-cube isolation

The formal protocol was frozen on 2026-07-21 in `docs/来源立方体预注册分析方案.md`, before any new neural test result was produced. It defined exactly two reciprocal evaluations:

- **Direction A (`suffix_1_to_2`):** all eight `*-1` cubes for development (650 seeds), all eight `*-2` cubes for testing (614 seeds).
- **Direction B (`suffix_2_to_1`):** all eight `*-2` cubes for development (614 seeds), all eight `*-1` cubes for testing (650 seeds).

No source cube crossed a development–test boundary. Within each development cube, 80% of seeds were assigned to fitting and 20% to validation under each fixed optimization seed. This validation subset served only hyperparameter choice for the prespecified SNV–LR candidate set, neural checkpoint selection, stopping, learning-rate control, and temperature scaling. Because fitting and validation seeds still came from the same development cube, it was explicitly classified as **development-internal seed validation**, not independent grouped validation. Test cubes were not used to select architecture, hyperparameters, epoch, seed, calibration temperature, or counterfactual.

The formal matrix comprised three models × three fixed seeds (`42`, `2024`, and `2025`) × two directions, for 18 completed training units. All models, seeds, directions, and counterfactual conditions were retained regardless of performance.

#### 2.6.3 Data-informed leave-one-source-cube-out diagnostic

Before the formal deep run, each of the 16 source cubes had also been held out in turn for classical mean-spectrum models. The other 15 cubes were used for fitting, and predictions for the held-out cube were pooled. This construction prevents direct cube overlap but is not leave-one-lot-out validation: the training data include the paired cube with the same label, the paired cubes' commercial independence is unknown, and the analysis was already visible when the new protocol was frozen. It is therefore retained as a complementary exploratory stability diagnostic rather than part of the formal estimand.

#### 2.6.4 Prespecified spatial counterfactuals and decision gate

Each fitted model was locked and then evaluated on the same test seeds under four deterministic inputs. `full` retained the complete masked patch. `spatial_shuffle` moved intact 392-band pixel spectra among foreground locations using counterfactual seed `9173`, preserving the mask and the multiset of foreground spectra while removing their internal arrangement. `mean_broadcast` replicated the seed's foreground-mean spectrum throughout the same mask, preserving mean spectrum and silhouette while removing pixel heterogeneity and arrangement. `mask_only` repeated the binary mask across all 392 bands, removing measured reflectance and intensity while preserving silhouette and crop-related cues. No model was refitted on a counterfactual.

The interpretation hierarchy was fixed as follows: full minus spatial shuffle targets within-foreground arrangement; spatial shuffle minus mean broadcast targets unordered pixel heterogeneity; mean broadcast minus mask only targets mean spectral information conditional on the frozen model; and mask only minus 12.5% chance diagnoses possible shape, centring, crop, or segmentation shortcuts. The mean-spectrum models' exact invariance to spatial shuffle and mean broadcast served as an implementation check rather than a biological result.

Positive spatial evidence was allowed only if all four preregistered criteria were satisfied: the ensemble full-minus-shuffle effect was positive in both directions; its equal-direction mean was at least 2 percentage points; at least five of six direction-by-seed effects were positive; and the complete run/counterfactual matrix was reported. Passing this gate was defined in advance as **limited support that the fitted fusion model used foreground-internal arrangement in the current paired-cube transfers**. It was explicitly not evidence of geographical tissue structure, chemical causality, external generalization, or superiority over a spectral baseline.

### 2.7 Spectral baselines and preprocessing

The exploratory classical panel used multinomial logistic regression (LR), a radial-basis-function support-vector machine (SVM), 20-component partial least-squares discriminant analysis (PLS-DA), random forest (RF), and the legacy one-dimensional CNN. Standard scaling and any training-dependent transform were fitted within the corresponding training partition. Fixed sensitivity transforms included SNV, training-reference multiplicative scatter correction, Savitzky–Golay smoothing, and an 11-band, second-order Savitzky–Golay first derivative. These alternatives were all disclosed rather than selecting only the best observed test result.

For spectrum $x_i$ with $B=392$ bands, SNV was

$$
x_{i,b}^{\mathrm{SNV}}=\frac{x_{i,b}-\bar{x}_i}{s_i},
$$

where $\bar{x}_i$ and $s_i$ are the within-spectrum mean and standard deviation [9].

The formal strong baseline was class-weighted multinomial SNV–LR. A standard scaler was fitted on development-fitting seeds only. Regularization candidates $C\in\{0.01,0.1,1,10,100\}$ were ranked on development-internal validation by macro-F1, then negative log-likelihood (NLL), then smaller $C$. The fitted candidate was not selected or changed using either reciprocal test cube.

### 2.8 Legacy HS3I-Net and preregistered neural models

The legacy HS3I-Net receives a masked $1\times392\times32\times32$ tensor, applies one learned non-negative coefficient per band, and passes the weighted cube through a four-block 3D residual backbone. Its original random-split training used AdamW for 360 epochs, Mixup, label smoothing, geometric/intensity augmentation, and jointly optimized sparsity and adjacent-band smoothness penalties. That architecture and its reported random-split result are retained only as a legacy benchmark; they were not reused as a constraint on the formal `main` method.

The preregistered neural analysis used a factorized architecture designed to make spectral and spatial contributions explicit while avoiding a large 3D activation tensor. The `spectral_only` model transformed the SNV foreground-mean spectrum with a 1D encoder: stride-2 convolutions, group normalization and GELU activations, two residual blocks (the second dilated), global pooling, and a 64-dimensional embedding. The `fusion_net` used the same spectral encoder and added a spatial branch. A learned $1\times1$ convolution projected each 392-band pixel spectrum to 16 channels; the binary mask was concatenated as a seventeenth channel; three 2D residual blocks produced a second 64-dimensional embedding. The concatenated 128-dimensional vector passed through a 96-unit GELU/dropout classifier and an eight-class output layer. The spectral-only version supplied a fixed zero spatial embedding to the same classifier topology. Trainable parameter counts were 79,624 and 209,000, respectively.

For each direction and seed, the spectral-only and fusion models used identical splits, optimization seed, spectral-branch and classifier definitions, and bitwise-identical initial shared parameters verified by SHA-256. This pairing controls initialization but does not make the models statistically independent or imply identical learned spectral representations.

Both neural models used batch size 32, at most 60 epochs, at least 12 epochs before patience-based stopping, AdamW (learning rate $10^{-3}$, weight decay $10^{-4}$), class-weighted cross-entropy with label smoothing 0.05, gradient-norm clipping at 5.0, deterministic seeding, and CUDA automatic mixed precision. ReduceLROnPlateau read development-validation NLL only. The checkpoint rule was maximum validation macro-F1, with lower validation NLL and then earlier epoch as fixed tie-breakers. Horizontal and vertical flips and 0.9–1.1 intensity scaling were applied during fusion training. Architecture and training rules were not revised after reciprocal test performance became available.

For every fitted model and seed, a scalar temperature was selected from a fixed 0.25–4.0 logarithmic grid by minimizing development-validation NLL. Raw probabilities were retained as sensitivity outputs. The primary ensemble averaged the three seed-specific temperature-scaled probability vectors before calculating metrics. Calibration was not fitted or corrected on either test direction.

### 2.9 Exploratory model-behaviour analyses

SelecVar coefficients are fitted model parameters, not causal wavelength effects. The prior model weights lack independent-group and seed-stability evidence. Any later interpretation must use the measured nonlinear wavelength vector and should require perturbation or remove-and-retrain evidence plus matched chemical measurements. The present archive contains no moisture, lipid, protein, starch, spinosin, jujuboside, fatty-acid, or metabolomic measurements.

The previous manuscript also described Grad-CAM curves and spatial maps from the last 3D residual block. The repository contains neither the required implementation nor corresponding checkpoints and machine-readable activations. Moreover, the last-layer feature map has only approximately 12 spectral positions and an estimated receptive field spanning roughly 321 input bands. Upsampling cannot identify a unique wavelength at single-band precision. Grad-CAM is therefore excluded from confirmatory evidence. It should be restored only as explicitly exploratory material after code, checkpoints, target-layer definition, interpolation, and model/label-randomization and perturbation checks are deposited [10,11].

### 2.10 Outcomes, calibration, and conditional inference

The primary metric was balanced accuracy; accuracy, macro-precision, macro-recall, macro-F1, multiclass NLL, multiclass Brier score, and ten-bin expected calibration error (ECE) were also computed. Confusion matrices, per-class recall, per-cube accuracy, selected hyperparameters or epochs, temperatures, training histories, failure status, and every seed-level eight-class probability vector were exported. For the legacy 316-seed prediction subset, Wilson intervals remain descriptive binomial summaries that do not account for within-cube dependence [12].

The formal performance estimator used the temperature-scaled three-seed probability ensemble. Seed means, standard deviations, and ranges were reported only as optimization diagnostics. They were never used as confidence intervals or biological replication.

Uncertainty was computed with one shared 10,000-draw paired block-bootstrap matrix generated with seed `20260721`. Each resampling unit was one of the eight commercial-label pairs; both reciprocal source cubes/directions and all compared models or counterfactuals shared the same sampled indices. Fixed 95% percentile intervals therefore describe heterogeneity among these eight observed label pairs conditional on the 16 archived cubes. They are not population intervals for unseen lots or origins.

For the single primary mechanism contrast, an exact sign-flip test enumerated all $2^8=256$ sign assignments of the eight paired-label effects. The frozen directional alternative was fusion(full) > fusion(spatial shuffle); the one-sided value is primary and a two-sided value is reported as a sensitivity analysis. Secondary effect intervals are estimation-focused; no additional unreported significance screen was used. The unavailable aligned legacy HS3I-Net prediction vector prevented a paired McNemar comparison with classical baselines, so one- or two-seed differences on the random split are descriptive only.

### 2.11 Software, execution provenance, and reproducibility controls

The formal run was executed in the repository-local `.venv` with Python 3.11.9, NumPy 2.4.4, h5py 3.16.0, scikit-learn 1.8.0, PyTorch 2.12.1+cu126, CUDA device `cuda:0`, and an NVIDIA GeForce RTX 4060 Laptop GPU. It began at 2026-07-21 21:21:14 UTC and completed at 21:44:22 UTC (1,387.74 s). The Git worktree was clean before execution on branch `main` at commit `4bc191c2e9b8a809e866ccd15d96fea29378969d`. After repository-facing paths were renamed, the equivalent reproduction command is:

```text
.venv\Scripts\python.exe deep_models\source_cube_audit.py --data-root data --output-dir deep_models\outputs\source_cube_preregistered_audit --models snv_lr spectral_only fusion_net --seeds 42 2024 2025 --device cuda:0 --num-workers 2
```

The path migration did not rerun training or post-processing. The original execution command is retained by canonical SHA-256 `6725b38f3244c1b23125eb3f80014bb79a7e3322d52f6db56a1c7ed752308d8c` and remains recoverable from the run commit. The run completed all 18 prespecified training units and all four counterfactual evaluations. Machine-readable outputs include 45,504 per-run prediction rows, 15,168 ensemble prediction rows, 12 neural checkpoints, full metrics and confusion matrices, model-selection records, training histories, calibration records, split assignments, and per-cube results. Dataset fingerprints were SHA-256 `873eab7b...b240e2d` for CSV contents, `50b61b46...ef75a` for MAT contents, and `325d7e68...61cd7` for the manifest. Full hashes are retained in `results.json` rather than abbreviated in the archive.

The post-processor accepted only a run marked `executed_complete`, rejected an incomplete matrix, used frozen models/conditions and one shared bootstrap matrix, and emitted input/output hashes. It generated tabular estimates, exact-test artifacts, reliability data, confusion matrices, and four figures in both PDF and PNG. Visual quality control after results were available identified a direction-label rendering error and a colour-bar overlap; only those presentation elements were corrected. SHA-256 comparison showed that every CSV, inferential JSON file, and the bilingual numerical summary was identical between passes. The classical reanalysis remains separately reproducible under `current_data_study/`; its results are explicitly distinguished from the new formal run.

## 3. Results

### 3.1 Experimental-unit audit

The nominal sample count was 1,264, but the highest recoverable acquisition-level count was 16 source cubes. Each of eight labels was represented by two cubes containing 54–90 segmented seeds (Table 1). Under the legacy stratified split, all 16 cubes contributed to both development and prediction; the prediction subset contained 10–26 seeds from each cube. Thus, the “independent prediction set” was independent only at extracted-patch index level, not at scene, commercial-lot, supplier, harvest, or instrument level.

The labels also mixed geographical scales: most referred to a city or county, whereas NX referred to an autonomous region. No metadata supported a harmonized provenance definition. Results are consequently interpreted as discrimination among archived commercial labels.

### 3.2 Legacy random seed-level benchmark and preprocessing sensitivity

The archived legacy result reported 96.84% accuracy for HS3I-Net (306/316 correct). The archived mean-spectrum LR, 1D-CNN, SVM, and RF results were 87.03%, 81.96%, 72.47%, and 65.51%, respectively. Newly executed fixed preprocessing analyses changed that comparison: SNV–LR classified 305/316 seeds (96.52%; macro-F1 96.59%), and Savitzky–Golay first-derivative LR classified 307/316 (97.15%; macro-F1 97.23%). MSC–LR and 20-component PLS-DA reached 94.62% and 93.99%.

**Table 2. Performance on the legacy random seed-level prediction subset.**

| Model | Input and preprocessing | Correct / 316 | Accuracy (%) | Macro-F1 (%) | Interpretation |
|---|---|---:|---:|---:|---|
| HS3I-Net | Masked 32 × 32 × 392 cube | 306 | 96.84 | 96.87 | Archived legacy aggregate; aligned predictions unavailable |
| SG first-derivative–LR | Mean spectrum; 11-band second-order first derivative | 307 | 97.15 | 97.23 | Exploratory sensitivity result |
| SNV–LR | Mean spectrum; SNV; leakage-safe scaling | 305 | 96.52 | 96.59 | Competitive exploratory baseline |
| MSC–LR | Mean spectrum; training-reference MSC | 299 | 94.62 | 94.74 | Exploratory sensitivity result |
| PLS-DA | Mean spectrum; scaling; 20 latent components | 297 | 93.99 | 94.04 | Exploratory latent-variable baseline |
| LR | Mean spectrum; legacy scaling | 275 | 87.03 | 87.06 | Legacy baseline |
| SG smooth–LR | Mean spectrum; 11-band second-order smoothing | 274 | 86.71 | 86.73 | Exploratory sensitivity result |
| 1D-CNN | Mean spectrum | 259 | 81.96 | 81.70 | Legacy neural spectral baseline |
| SVM | Mean spectrum; legacy scaling | 229 | 72.47 | 71.91 | Legacy baseline |
| RF | Mean spectrum | 207 | 65.51 | 65.24 | Legacy baseline |

The first-derivative result was one seed above HS3I-Net and SNV–LR one seed below. These differences neither establish superiority nor isolate a spatial contribution: all methods used a source-cube-overlapping prediction subset, several disclosed preprocessing candidates were inspected, seeds within a cube are dependent, and aligned legacy HS3I-Net predictions were unavailable for a paired comparison. The valid conclusion is that a large legacy cube-model advantage disappears when competitive spectrum preprocessing is included.

### 3.3 Data-informed reciprocal classical diagnostics

Complete-cube separation reduced performance and changed classical model ranking (Table 3). PLS-DA was highest in both exploratory reciprocal directions at 85.34% and 89.85% accuracy. SNV–LR reached 83.39% and 88.46%, whereas raw SVM reached 71.66% and 68.00%. These results were already observed when the formal protocol was frozen; they motivated inclusion of a strong SNV–LR baseline but are not relabelled as preregistered evidence.

**Table 3. Data-informed reciprocal source-cube diagnostics.**

| Model | Train `*-1`, test `*-2`: accuracy / macro-F1 (%) | Train `*-2`, test `*-1`: accuracy / macro-F1 (%) | Descriptive mean accuracy (%) |
|---|---:|---:|---:|
| PLS-DA | 85.34 / 84.60 | 89.85 / 89.66 | 87.59 |
| SNV–LR | 83.39 / 82.32 | 88.46 / 88.05 | 85.93 |
| SG first-derivative–LR | 84.53 / 83.77 | 83.69 / 83.28 | 84.11 |
| MSC–LR | 81.76 / 80.60 | 86.46 / 86.20 | 84.11 |
| Raw LR | 79.64 / 79.89 | 80.77 / 80.11 | 80.20 |
| SG smooth–LR | 79.48 / 79.77 | 80.92 / 80.31 | 80.20 |
| Raw SVM | 71.66 / 71.50 | 68.00 / 67.16 | 69.83 |
| Raw RF | 63.03 / 63.10 | 63.08 / 62.18 | 63.05 |

The decline from the shared-cube random split is consistent with source-cube-dependent variation. It does not prove that residual discrimination is geographical because paired cubes may share supplier, lot, processing, storage, or acquisition factors. Conversely, failure cannot be assigned solely to architecture because a direction provides only one training cube per class.

### 3.4 Data-informed leave-one-source-cube-out stability

The exploratory 16-fold leave-one-source-cube-out analysis produced another ordering (Table 4). SNV–LR was highest at 79.03% accuracy (macro-F1 77.92%), followed by first-derivative LR at 76.98% and MSC–LR at 74.37%. PLS-DA, despite leading both reciprocal directions, fell to 62.10%. This protocol dependence reinforces why a favourable result under one grouped construction cannot by itself identify a robust model.

**Table 4. Pooled exploratory performance when each source cube is held out once.**

| Model | Correct / 1,264 | Accuracy (%) | Balanced accuracy (%) | Macro-F1 (%) |
|---|---:|---:|---:|---:|
| SNV–LR | 999 | 79.03 | 79.35 | 77.92 |
| SG first-derivative–LR | 973 | 76.98 | 77.47 | 76.06 |
| MSC–LR | 940 | 74.37 | 74.87 | 73.27 |
| SG smooth–LR | 888 | 70.25 | 70.67 | 70.07 |
| Raw LR | 887 | 70.17 | 70.59 | 70.01 |
| PLS-DA | 785 | 62.10 | 62.24 | 61.39 |
| Raw SVM | 644 | 50.95 | 51.62 | 50.78 |
| Raw RF | 599 | 47.39 | 47.86 | 47.32 |

The 1,264 out-of-fold predictions remain clustered within only 16 cubes, and every training fold includes the paired cube carrying the held-out label. The table is therefore not evidence from 1,264 independent origin trials.

### 3.5 Preregistered reciprocal complete-cube performance

All 18 formal training units completed without exclusion. Table 5 reports the primary temperature-scaled probability ensembles. SNV–LR achieved $\theta=86.94\%$ (conditional 95% interval, 75.38–96.20), with balanced accuracies of 84.77% and 89.11% in the two directions. The spectral-only network achieved $\theta=44.75\%$ (25.31–63.58), and fusion achieved $\theta=45.94\%$ (31.98–60.70). The intervals reflect resampling of eight observed commercial-label pairs and are correspondingly broad.

**Table 5. Formal temperature-scaled three-seed probability ensembles under complete source-cube isolation.**

| Model | Direction A balanced accuracy / macro-F1 (%) | Direction B balanced accuracy / macro-F1 (%) | $\theta$ (%) | Conditional 95% interval for $\theta$ (%) |
|---|---:|---:|---:|---:|
| SNV–LR | 84.77 / 83.70 | 89.11 / 88.92 | 86.94 | 75.38–96.20 |
| Spectral-only network | 52.94 / 49.42 | 36.56 / 29.90 | 44.75 | 25.31–63.58 |
| Fusion network | 43.85 / 41.31 | 48.02 / 47.61 | 45.94 | 31.98–60.70 |

The neural results were also optimization-sensitive. Seed-wise full-input balanced accuracies for the spectral-only model were 12.50%, 17.05%, and 53.10% in Direction A and 12.34%, 12.50%, and 35.80% in Direction B. Fusion was less variable but remained low: 30.19%–45.58% in Direction A and 41.86%–44.69% in Direction B. Two Direction-B spectral-only calibrations selected opposite endpoints of the frozen temperature grid (0.25 and 4.0); the grid was not expanded after test results were available. These are optimization and development-calibration diagnostics, not independent repetitions. Probability averaging improved neither neural method to the level of SNV–LR.

Calibration metrics did not justify deployment claims. For SNV–LR, ensemble NLL/ECE were 0.588/0.044 in Direction A and 0.338/0.026 in Direction B. Spectral-only values were 1.622/0.283 and 1.808/0.155; fusion values were 1.658/0.109 and 1.393/0.066. These values describe only the two archived test directions, and temperature was learned from within-cube development validation.

### 3.6 Preregistered counterfactual spatial audit

The same locked fusion ensembles were evaluated under all four inputs (Table 6). Full-minus-spatial-shuffle was +0.78 points in Direction A and +25.52 points in Direction B, giving the primary equal-direction effect of +13.15 points (conditional 95% interval, 1.82–24.38). All six direction-by-seed effects were positive, exceeding the required five of six; both ensemble direction effects were positive; and the mean effect exceeded the 2-point threshold. The preregistered limited-support gate was therefore met.

At label-pair level, seven of eight effects were positive; effects ranged from −14.38 points for SXQ to +36.88 points for HBS. The exact sign-flip test over all 256 assignments gave one-sided $p=0.0429688$ for the frozen greater alternative and two-sided sensitivity $p=0.0859375$. Together with the strong directional asymmetry, these results support a conditional dependence on spatial arrangement but also show that it was not uniform across acquisition directions or labels.

**Table 6. Paired counterfactual effects for the temperature-scaled fusion ensemble.**

| Contrast | Direction A (percentage points) | Direction B (percentage points) | Equal-direction effect (percentage points) | Conditional 95% interval |
|---|---:|---:|---:|---:|
| Full − spatial shuffle (primary) | 0.78 | 25.52 | 13.15 | 1.82–24.38 |
| Spatial shuffle − mean broadcast | 0.47 | 0.09 | 0.28 | −0.50–0.99 |
| Full − mean broadcast | 1.25 | 25.61 | 13.43 | 1.71–24.76 |
| Mean broadcast − mask only | 21.10 | 2.88 | 11.99 | 4.10–21.96 |
| Fusion full − spectral-only full | −9.09 | 11.46 | 1.19 | −11.29–15.55 |
| Fusion full − SNV–LR full | −40.91 | −41.09 | −41.00 | −50.86 to −31.46 |
| Fusion mask only − 12.5% chance | 9.00 | 7.03 | 8.02 | −5.70–23.54 |

The near-zero shuffle-minus-mean-broadcast contrast did not show an incremental contribution from unordered pixel heterogeneity. Mean broadcast exceeded mask only by 11.99 points, consistent with use of mean spectral information by the locked fusion models. Mask-only performance was $\theta=20.52\%$, 8.02 points above chance by point estimate, but its interval crossed zero. A shape/crop shortcut therefore remains a plausible alternative contribution and cannot be declared either present or absent from these 16 cubes.

Most importantly, the spatial gate did not translate into competitive prediction. Fusion exceeded the spectral-only network by only 1.19 points with an interval spanning substantial harm and benefit, and it trailed SNV–LR by 41.00 points in a direction-consistent comparison. The experiment therefore identifies model dependence on arrangement, not an advantage of the fusion architecture.

### 3.7 Exploratory wavelength weighting and saliency audit

The legacy SelecVar output assigned relatively larger normalized coefficients to portions of the longer-wavelength region and reported nominal maxima near 1097, 1177, 1265, 1425, 1452, 1498, 1549, 1581, and 1649 nm. These locations remain provisional because the old plot used an equally spaced wavelength grid instead of the measured nonlinear grid, lacked stability estimates across independent lots and seeds, and was not validated by band occlusion or remove-and-retrain experiments.

The legacy manuscript also described Grad-CAM responses near 1330–1340 and 1430–1440 nm and non-uniform spatial maps. Those claims are not treated as results here. Required code and artifacts are absent, the target layer is spectrally coarse with a broad receptive field, and the post-normalization threshold was arbitrary. No exact wavelength or tissue region can therefore be said to cause classification. Because no reference chemistry was measured in these samples, assignments to lipid, protein, starch, water, spinosin, or other constituents are excluded.

### 3.8 Claim-status synthesis

**Table 7. Claims supported and not supported by the complete current-data evidence.**

| Claim | Evidence | Status |
|---|---|---|
| The archived labels are separable when development and test share all cubes | Legacy HS3I-Net 96.84%; spectrum pipelines 93.99%–97.15% | Supported only within shared acquisition units |
| A cube model has a large advantage over a strong mean-spectrum model | Formal fusion $\theta=45.94\%$; SNV–LR $\theta=86.94\%$ | Contradicted for the frozen formal models |
| The locked fusion model uses foreground-internal arrangement in reciprocal cube transfer | +13.15 points; conditional interval 1.82–24.38; gate passed; one-sided exact $p=0.0429688$ | Limited, conditional support |
| Fusion provides a stable increment over the paired spectral network | +1.19 points; interval −11.29–15.55; opposite directional signs | Not supported |
| Mask/crop cues make no contribution | Mask-only point estimate above chance, with interval crossing zero | Unresolved |
| Performance generalizes to new lots, suppliers, years, instruments, or laboratories | No such independent groups or domains | Not tested |
| Labels are verified cultivation origins | Supplier-reported labels; no chain of custody | Not established |
| Wavelengths identify causal chemical differences | No matched chemistry; old mapping and saliency limitations | Not supported |
| The system is deployment-ready for traceability | No external, open-set, shift-monitoring, or deployment study | Not supported |

## 4. Discussion

### 4.1 The experimental unit changes the scientific conclusion

The central result is the change in what an accuracy value can mean. The legacy prediction subset contains seeds from every source cube used for development. Its 96.84% HS3I-Net result is therefore patch-level interpolation within a shared acquisition domain. Seeds from one scene share illumination, calibration, background, camera state, acquisition timing, plate layout, and preprocessing, and may share a commercial lot. Source-cube isolation is the minimum defensible boundary recoverable from the archive, although it still falls short of unseen-lot validation.

The formal analysis enforced that boundary without selectively reporting a direction, seed, model, or counterfactual. Equal weighting of the two directions and eight label pairs prevents the largest cubes from defining the principal result. This design cannot create missing biological replication, but it makes the inferential deficit visible instead of allowing the 1,264 technical subsamples to imply a much larger effective sample size.

### 4.2 A strong simple baseline dominates the frozen neural models

SNV–LR reached 86.94% conditional balanced accuracy, whereas spectral-only and fusion networks reached 44.75% and 45.94%. The approximately 41-point deficit of fusion was present in both directions and had a conditional interval far below zero. This is not evidence that deep learning can never benefit SZR HSI; it is evidence that these prespecified neural models, trained from the current one-cube-per-class development directions, did not provide competitive transfer.

The result is scientifically useful because the baseline was not an afterthought. SNV directly addresses scatter and offset variation common in diffuse-reflectance spectra [9], and regularization was selected without test-cube access. In small-group settings, architecture capacity cannot compensate for absent acquisition-level diversity. The very low and seed-sensitive spectral-network results further show why a single neural initialization would have been misleading.

The earlier random-split comparison reached the same caution from another direction: a fixed first-derivative LR was one seed above legacy HS3I-Net and SNV–LR one seed below it. Taken together, these analyses remove the basis for attributing high shared-cube accuracy to a uniquely spatial deep representation.

### 4.3 Spatial dependence is not the same as predictive superiority

The counterfactual audit adds a narrower positive finding. Shuffling intact foreground pixel spectra reduced fusion balanced accuracy by 13.15 points on average, the conditional interval excluded zero, all six seed-direction effects were positive, and the prespecified gate passed. Because the operation preserved silhouette and the multiset of foreground spectra, the contrast is more specific to their arrangement than a comparison between unrelated architectures.

That finding requires three qualifications. First, almost the entire ensemble effect came from Direction B (25.52 points versus 0.78 in Direction A), and one label-pair effect was negative. The learned dependency is therefore acquisition-direction- and label-heterogeneous. Second, fusion did not reliably exceed the paired spectral network; useful arrangement for a particular model need not yield better overall prediction. Third, the intervention diagnoses information use, not its cause. Arrangement may reflect tissue, orientation, segmentation, illumination, position, or other scene-linked processes. Calling it a geographical microstructure or biological mechanism would exceed the experiment.

The secondary interventions sharpen this interpretation. Spatial shuffle and mean broadcast were nearly indistinguishable, whereas mean broadcast exceeded mask only. The frozen fusion model thus used mean-spectrum information and some arrangement, with no detected incremental value from unordered heterogeneity. Mask-only uncertainty was too large to dismiss morphology or crop artifacts. These counterfactuals partition model dependence more credibly than a saliency heat map, but they do not identify chemistry.

### 4.4 Conditional uncertainty and calibration must remain conditional

The eight-label-pair bootstrap preserves the highest paired structure available and uses common resamples for every contrast. It provides a transparent view of label heterogeneity, but eight blocks yield unstable, conditional intervals. Likewise, the exact sign-flip test is exact for sign assignments of these eight observed effects under its randomization model; it does not turn archived commercial labels into a sample from a well-defined population of future origins. The two-sided sensitivity value of 0.0859375 is reported because the evidential strength is modest even though the frozen one-sided mechanism hypothesis crossed 0.05.

Temperature scaling and probability ensembling were completed without test fitting, yet calibration transfer remains unverified. Development-internal validation contains seeds from the same cubes used for fitting. The resulting reliability curves, NLL, Brier scores, and ECE describe only the archived reciprocal shifts. They cannot support confidence thresholds, rejection policies, or deployment risk estimates.

### 4.5 Novelty relative to prior work

Prior SZR work already evaluated image and spectrum CNNs, SNV, learned band weighting, and measured chemical properties [6], and attention-weighted 3D residual HSI architectures have been published more generally [7]. Another SZR study used independently collected batches and volatile measurements [8]. The defensible novelty is therefore not the fusion module alone.

The stronger contribution is the integrated audit: reconstruction of the analysis hierarchy, chronological separation of exploratory and preregistered evidence, reciprocal complete-cube isolation, shared-initialization spectral ablation, same-model spatial counterfactuals, calibration, probability ensembling, and paired block uncertainty. The negative performance result and the positive but bounded mechanism result are reported together. This combination demonstrates why a mechanistic contrast cannot substitute for strong baselines or external validation.

### 4.6 Limits of wavelength and saliency interpretation

Input-band coefficients indicate how one fitted model scales correlated variables under a particular objective. They do not identify a compound or causal environmental process. Weights can redistribute among neighbouring bands and initializations; a large coefficient need not produce a large performance effect. Robust interpretation would require stability across independent groups, intervention and retraining, and matched chemical or physical measurements.

A coarse last-layer activation is still less suited to exact wavelength claims. Interpolation cannot recover spectral resolution absent from the feature map, and visually plausible saliency can persist after model or label randomization [11]. With missing executable Grad-CAM artifacts and no reference chemistry, excluding those claims is necessary rather than merely conservative.

### 4.7 Strengths and limitations

Strengths include an explicit analysis hierarchy, a protocol frozen before the formal neural test results, complete reporting of all 18 training units, reciprocal zero-overlap cube boundaries, a competitive classical baseline, paired neural initialization, four same-model counterfactuals, temperature-scaled probability ensembles, sample-level probability release, and uncertainty preserving eight commercial-label pairs. Data hashes, environment metadata, checkpoints, and figure-source tables provide an auditable path from processed inputs to reported estimates.

The limitations determine the scope. There are only 16 source cubes and two per label; paired cubes may not be independent commercial lots. Each formal direction has only one development and one test cube per class. Development validation is seed-level within a cube. Reported locations lack chain-of-custody documentation and mix geographical resolution. Farm, supplier, lot, harvest, acquisition-session, and processing identifiers are missing. Raw scenes and white/dark references are unavailable, preventing end-to-end acquisition reproduction. Manual cropping and a zero background preserve operator, silhouette, and positioning cues. All observations appear to derive from one instrument domain. No matched chemistry or moisture data exist. The old wavelength mapping was imprecise, and legacy Grad-CAM artifacts are absent.

These constraints prevent claims of geographical authentication, causal chemistry, operational traceability, or new-domain calibration regardless of accuracy. The conditional bootstrap cannot estimate variance among future lots, and the exact sign-flip result cannot repair that missing sampling frame.

### 4.8 Practical interpretation and next evidence required

Under current evidence, a positive prediction means only that a seed resembles one of eight archived commercial groups under the sampled acquisition conditions. It should not certify cultivation origin or adjudicate fraud. The closed-set models force every object into one of eight known labels; unknown origin, adulteration, out-of-distribution detection, abstention, and decision costs were not evaluated.

A confirmatory study should prospectively sample independently traceable farms and lots, multiple harvest years, suppliers, processing/storage conditions, acquisition days and operators, mixed-label scenes, and at least a second instrument or laboratory. The split unit should match the deployment claim, with model development completed before a locked external-domain test. Matched moisture and chemical assays are needed for mechanistic attribution. Repeated calibration acquisitions and raw data should enable end-to-end preprocessing audit. Those requirements are a data-collection mandate, not analyses that can be simulated from the existing 16 cubes.

## 5. Conclusions

The completed preregistered audit changes both the performance and mechanism narrative of this SZR archive. A temperature-scaled three-seed SNV–LR ensemble reached 86.94% balanced accuracy under reciprocal complete-cube isolation, whereas the spectral-only and fusion networks reached 44.75% and 45.94%. Fusion trailed SNV–LR by 41.00 percentage points and did not reliably improve on its paired spectral network.

The same-model counterfactual nevertheless found a 13.15-point full-versus-spatial-shuffle effect and met the preregistered limited-support gate. That result shows that the fitted fusion models used some foreground-internal arrangement in these particular cube transfers; its strong directional heterogeneity, lack of fusion superiority, unresolved mask contribution, and absence of independent lots preclude a broader spatial, biological, or provenance claim.

Accordingly, the present contribution is a reproducible source-cube-aware methodological case study. It does not establish geographical-origin authentication, chemical causality, external generalization, or deployment readiness. Those claims require prospectively collected, independently traceable, multi-domain data rather than further optimization on the same 16 archived cubes.

## CRediT authorship contribution statement

The final CRediT statement remains **pending author verification**. The audited project materials do not establish who performed each component of the new data curation, software development, preregistration, formal analysis, validation, visualization, or manuscript revision. No revised role assignment should be inferred from the repository history alone. The authors must provide and approve the complete statement before submission.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Ethics statement

This study used commercial plant-derived seed material and involved no human participants or live animals.

## Funding

This work was supported by the TCM Research and Innovation Team in Shaanxi Administration of Traditional Chinese Medicine (TZKNCXTD-09) and the Shaanxi Academy of Sciences Talent Introduction Program (2025k-3).

## Acknowledgements

The authors thank Hangzhou Hyperspectral Imaging Technology Co., Ltd. for technical support in hyperspectral image acquisition.

## Data availability

The audited repository contains 1,264 processed seed-level MAT patches and paired mean-spectrum CSV files organized under 16 source-cube folders. The formal output directory contains the exact manifest, split assignments, wavelength vector, MAT–CSV representation check, sample-level probabilities, ensemble probabilities, cube metrics, and input hashes used in this paper. Raw hyperspectral scenes, white/dark calibration acquisitions, traceable independent-lot metadata, and matched chemical measurements are unavailable and therefore cannot be shared. Their absence prevents reproduction from raw acquisition and limits the scope of inference.

No public persistent repository or DOI had been assigned at the time of this draft. Local repository paths are not a substitute for public deposition. Before submission, the authors must deposit the shareable processed data, metadata dictionary, frozen code, lock files, formal outputs, and checksums in a versioned archive and replace this paragraph with the verified accession and DOI.

## Code availability

The formal entry point is `deep_models/source_cube_audit.py`, and the fixed post-processor is `deep_models/summarize_source_cube_audit.py`. The executed artifacts are under `deep_models/outputs/source_cube_preregistered_audit/`. The formal run is tied to clean Git commit `4bc191c2e9b8a809e866ccd15d96fea29378969d` on `main`; its command, environment, timestamps, hashes, and completion status are recorded in `run_status.json`, `results.json`, and `postprocessing/postprocessing_manifest.json`. Python, PyTorch, and h5py versions are recorded and were available during execution. The classical audit remains under `current_data_study/` with its own requirements, deterministic entry points, predictions, and metrics. Original root-level model scripts and MATLAB extraction copies are retained on the immutable `original` audit branch but deliberately excluded from the current `main` release; neither they nor superseded grouped-deep drafts are prerequisites for the formal analysis.

Before release, checkpoint metadata were normalized for safe deserialization by converting `wavelengths_nm` from a NumPy array to a plain list. This was a post-run serialization-compatibility correction, not an algorithm or result change. All 12 checkpoints subsequently loaded with `torch.load(..., weights_only=True)`; comparison with the retained local originals found all 600 state-dictionary tensors identical, while temperatures, wavelengths, and protocol metadata were unchanged.

No public code DOI or immutable software release exists yet. Grad-CAM code and reproducible legacy Grad-CAM artifacts remain absent and do not support the conclusions. The authors must create and cite a versioned public release before submission.

## Appendix A. Figure status and caption specifications

**Figure 1. Experimental hierarchy and scope of inference.** Pixels nested within seeds, 1,264 seeds within 16 source cubes, and two cubes within each commercial label. Recoverable identifiers must be visually separated from unknown lot, supplier, farm, harvest, and acquisition-session relationships.

**Figure 2. Hyperspectral acquisition and patch extraction.** (a) Imaging system; (b) manual scene crop, Otsu mask, morphology, and connected components; (c) $32\times32$ masked cube and foreground-mean spectrum. The caption must state that background zeroing retains silhouette and crop cues.

**Figure 3. Validation chronology and analysis boundaries.** (a) Legacy random seed split, with every cube on both sides; (b) exploratory reciprocal and leave-one-cube classical diagnostics; (c) preregistered reciprocal complete-cube formal matrix; (d) unavailable but deployment-relevant external lot/domain test.

**Figure 4. Paired source-cube spectra within each archived commercial label.** Available as `current_data_study/figures/figure_source_cube_spectra.pdf`. Lines are cube means on the measured wavelength grid; shading is ±1 SD across seeds and is descriptive, not a lot-level interval. No chemical peak assignment is made.

**Figure 5. Exploratory performance depends on preprocessing and validation construction.** Available as `current_data_study/figures/figure_performance_by_protocol.pdf`. Panels show the legacy random holdout, reciprocal cube directions, and pooled leave-one-cube-out accuracy. These data-informed diagnostics must remain visually distinct from formal results.

**Figure 6. Exploratory SNV–LR errors across grouped constructions.** Available as `current_data_study/figures/figure_snv_lr_grouped_confusions.pdf`. Rows are normalized within archived labels for both reciprocal directions and pooled leave-one-cube-out predictions.

**Figure 7. Preregistered complete-cube performance (generated).** `deep_models/outputs/source_cube_preregistered_audit/postprocessing/figure_main_performance.pdf` and `.png`. Shows each prespecified seed and the temperature-scaled probability ensemble for all three models in both reciprocal directions; the dashed line is 12.5% chance. The paired-label $\theta$ intervals are reported in Table 5 rather than superimposed on this optimization-stability plot. The caption must define the 16-cube conditioning scope.

**Figure 8. Same-model counterfactual effects (generated).** `postprocessing/figure_counterfactual_effects.pdf` and `.png`. Show full, spatial-shuffle, mean-broadcast, and mask-only contrasts for the locked fusion ensemble. The caption must state that the primary effect is direction-heterogeneous and does not establish spatial causality or model superiority.

**Figure 9. Formal full-input ensemble confusion matrices (generated).** `postprocessing/figure_ensemble_confusion_matrices.pdf` and `.png`. Rows are true archived labels and columns are predicted labels; each row is normalized separately. Panels show the temperature-scaled full-input probability ensemble for each reciprocal direction and each of the three models.

**Figure 10. Formal reliability audit (generated).** `postprocessing/figure_calibration_reliability.pdf` and `.png`. Ten equal-width confidence bins compare empirical accuracy and confidence. Calibration is conditional on the archived cube shifts and must not be represented as externally validated.

**Exploratory SelecVar stability figure—not available.** It may be added only after weights are plotted on the measured wavelength grid across directions and seeds and supported by perturbation or remove-and-retrain evidence. Grad-CAM should not appear in the main paper; any reproducible future version belongs in the supplement with sanity checks and an explicit spectral-resolution warning.

## Appendix B. Supplementary package status

- **Available S1—classical audit:** `current_data_study/outputs/` contains manifest, wavelengths, predictions, fold/direction metrics, confusion matrices, configuration, dependency lock, and data fingerprint; `current_data_study/figures/` contains deterministic classical figure sources.
- **Available S2—formal provenance:** `run_status.json`, `results.json`, `manifest.csv`, `wavelengths.csv`, and `mat_csv_mean_consistency.*` contain the path-normalized reproduction command, original-command fingerprint, environment, Git commit, input hashes, and representation-integrity result.
- **Available S3—formal design and fitting:** `splits.csv`, `model_selection.csv`, `training_history.csv`, and `development_validation_calibration.csv` contain every split, candidate, selected epoch, history, and temperature.
- **Available S4—complete predictions:** `predictions.csv` and `ensemble_predictions.csv` contain raw and temperature-scaled eight-class probabilities for every prespecified model, seed, direction, and counterfactual.
- **Available S5—metrics:** `metrics.csv`, `metrics_seed_aggregate.csv`, `ensemble_metrics.csv`, `cube_metrics.csv`, `ensemble_cube_metrics.csv`, and `primary_estimands.csv` contain run-, seed-, ensemble-, cube-, and estimand-level outputs.
- **Available S6—counterfactual audit:** `counterfactual_pairs.csv`, `ensemble_counterfactual_pairs.csv`, and `spatial_mechanism_decision.json` contain paired effects and the complete preregistered gate evaluation.
- **Available S7—conditional inference:** the `postprocessing/` directory contains the shared 10,000-resample block matrix, theta/effect intervals, eight label-pair effects, the 256-value exact null distribution, reliability data, confusion matrices, summary, and hashed manifest.
- **Available S8—model states and figures:** 12 neural checkpoints, normalized only for safe checkpoint metadata deserialization and verified against 600 unchanged state-dictionary tensors, and four formal figure sets in PDF and PNG are archived with the output matrix.
- **Unavailable and not represented by placeholders:** raw scenes and calibration references, independently verified lot/farm metadata, matched chemistry, aligned legacy HS3I-Net prediction vectors, reproducible Grad-CAM artifacts, an external validation set, and a public repository DOI.

## References

1. Cao JX, Zhang QY, Cui SY, et al. Hypnotic effect of jujubosides from Semen Ziziphi Spinosae. *Journal of Ethnopharmacology*. 2010;130:163–166. https://doi.org/10.1016/j.jep.2010.03.023.
2. Sun YF, Liang ZS, Shan CJ, Viernstein H, Unger F. Comprehensive evaluation of natural antioxidants and antioxidant potentials in *Ziziphus jujuba* Mill. var. *spinosa* fruits based on geographical origin by TOPSIS method. *Food Chemistry*. 2011;124:1612–1619. https://doi.org/10.1016/j.foodchem.2010.08.026.
3. Kong Y, He S, Ma D, et al. Chemical composition determination and transcriptomic analyses provide insight into the differences between wild and grafted Semen Ziziphi Spinosae. *BMC Genomics*. 2024;25:978. https://doi.org/10.1186/s12864-024-10837-7.
4. Qin J, Chao K, Kim MS, Lu R, Burks TF. Hyperspectral and multispectral imaging for evaluating food safety and quality. *Journal of Food Engineering*. 2013;118:157–171. https://doi.org/10.1016/j.jfoodeng.2013.04.001.
5. Roberts DR, Bahn V, Ciuti S, et al. Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*. 2017;40:913–929. https://doi.org/10.1111/ecog.02881.
6. Zhao X, Liu X, Xie P, et al. Identification of geographical origin of semen ziziphi spinosae based on hyperspectral imaging combined with convolutional neural networks. *Infrared Physics & Technology*. 2024;136:104982. https://doi.org/10.1016/j.infrared.2023.104982.
7. Zheng Z, Liu Y, He M, Chen D, Sun L, Zhu F. Effective band selection of hyperspectral image by an attention mechanism-based convolutional network. *RSC Advances*. 2022;12:8750–8759. https://doi.org/10.1039/D1RA07662K.
8. Zhang JB, Li MX, Zhang YF, et al. E-eye, flash GC E-nose and HS-GC-MS combined with chemometrics to identify the adulterants and geographical origins of Ziziphi Spinosae Semen. *Food Chemistry*. 2023;424:136270. https://doi.org/10.1016/j.foodchem.2023.136270.
9. Barnes RJ, Dhanoa MS, Lister SJ. Standard normal variate transformation and de-trending of near-infrared diffuse reflectance spectra. *Applied Spectroscopy*. 1989;43:772–777. https://doi.org/10.1366/0003702894202201.
10. Selvaraju RR, Cogswell M, Das A, Vedantam R, Parikh D, Batra D. Grad-CAM: Visual explanations from deep networks via gradient-based localization. In: *Proceedings of the IEEE International Conference on Computer Vision*. 2017:618–626. https://doi.org/10.1109/ICCV.2017.74.
11. Adebayo J, Gilmer M, Muelly M, Goodfellow I, Hardt M, Kim B. Sanity checks for saliency maps. In: *Advances in Neural Information Processing Systems 31*. 2018.
12. Wilson EB. Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association*. 1927;22:209–212. https://doi.org/10.1080/01621459.1927.10502953.
