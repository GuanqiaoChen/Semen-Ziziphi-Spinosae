# Acquisition-aware hyperspectral classification of commercial *Ziziphi Spinosae* Semen: a source-cube-aware exploratory pilot study

> **Author-facing draft status — remove before submission.** This is a current-data-constrained rewrite, not a submission-ready claim of geographical-origin authentication. The classical spectral reanalysis reported below was executed with the locked source-cube-aware pipeline and is traceable to machine-readable outputs. The grouped 3D falsification protocol was prepared but could not be executed in the present environment because PyTorch and h5py were unavailable; no unexecuted neural result is presented as observed. Author affiliation, contribution, and public-repository metadata still require author verification. The original DOCX remains unchanged.

Ziying Wang<sup>a</sup>, Guanqiao Chen<sup>b</sup>, Jiale Wei<sup>c</sup>, Xumei Wang<sup>a,*</sup>

<sup>a</sup> School of Pharmacy, Xi'an Jiaotong University, Xi'an 710049, China  
<sup>b</sup> Tandon School of Engineering, New York University, New York, NY 11201, USA  
<sup>c</sup> Affiliation not available in the audited project materials; the authors must supply and verify it before submission.  
<sup>*</sup> Corresponding author: wangxumei@mail.xjtu.edu.cn

## Highlights

- The 1,264 seed patches are nested within only 16 source hyperspectral cubes.
- Every source cube occurs in both subsets of the legacy random seed-level split.
- Fixed spectrum-only pipelines reach 96.52%–97.15% versus 96.84% for legacy HS3I-Net on that split.
- Reciprocal and leave-one-cube-out tests reveal model- and acquisition-domain sensitivity.
- Band weighting and saliency are retained only as unconfirmed hypothesis-generating analyses.

## Abstract

Hyperspectral classifiers for geographical provenance can appear highly accurate when multiple objects extracted from the same acquisition scene are randomly divided between training and testing. We therefore re-examined the experimental hierarchy and validation claims of a near-infrared hyperspectral dataset of commercial *Ziziphi Spinosae* Semen (SZR). The dataset contains 1,264 segmented seed patches assigned to eight supplier-reported geographical labels, but these patches originate from only 16 source hyperspectral cubes—two cubes per label. Botanical identity was expert-confirmed, whereas cultivation provenance, independent lot identity, harvest year, processing history, and chain of custody were not documented. We retained the legacy stratified seed-level split (948 development and 316 prediction patches) as a conditional within-acquisition benchmark and executed fixed preprocessing and source-cube diagnostics on the archived mean spectra. The legacy three-dimensional HS3I-Net result was 306/316 (96.84%; descriptive Wilson 95% interval, 94.27–98.27%). Under the locked reanalysis, standard-normal-variate logistic regression classified 305/316 (96.52%), and Savitzky–Golay first-derivative logistic regression classified 307/316 (97.15%); the latter was one seed above HS3I-Net but is a descriptive sensitivity analysis rather than a test-selected replacement model. The legacy raw-spectrum LR, one-dimensional CNN, SVM, and RF results were 87.03%, 81.96%, 72.47%, and 65.51%, respectively. In reciprocal diagnostics, PLS-DA achieved 85.34% and 89.85%, SNV–LR achieved 83.39% and 88.46%, raw LR achieved 79.64% and 80.77%, and SVM achieved 71.66% and 68.00%. Model ranking was not stable: in 16-fold leave-one-source-cube-out prediction, SNV–LR was highest among the executed spectrum models at 79.03%, whereas PLS-DA fell to 62.10%. These analyses eliminate direct source-cube overlap in testing but cannot estimate performance on unseen farms, suppliers, harvests, or instruments because paired cubes may not be independent commercial lots. SelecVar weights and previously reported Grad-CAM maps are therefore treated as unconfirmed model-behaviour observations rather than chemical or biological evidence. The present data support closed-set discrimination among eight commercial sample groups within a restricted acquisition domain; they do not establish geographical-origin authentication or traceability.

**Keywords:** hyperspectral imaging; *Ziziphi Spinosae* Semen; grouped validation; batch effects; source-cube shift; spectral preprocessing; pilot study

## 1. Introduction

*Ziziphi Spinosae* Semen (SZR) is used as a medicinal and food-homologous seed material, and its commercial value has motivated research on identity, quality, adulteration, and provenance [1–3]. Laboratory chromatography and mass spectrometry can characterize constituents or volatile profiles, but they require specialized workflows and are not always suited to rapid screening. Near-infrared hyperspectral imaging (HSI) is attractive because it records a reflectance spectrum at each spatial location while leaving an intact seed available for subsequent analysis [4].

HSI data, however, have a hierarchical structure. Multiple pixels belong to one seed, multiple seeds can be segmented from one hyperspectral scene, and multiple scenes may belong to the same lot, supplier, harvest, acquisition session, or instrument. A classifier evaluated after randomly allocating lower-level observations can exploit stable scene- or batch-specific signals that occur in both training and test data. The resulting accuracy estimates discrimination within the sampled acquisition domain, not necessarily generalization to an unseen commercial lot or geographical origin. Grouped or blocked validation is therefore required whenever the intended deployment unit is higher in the hierarchy than the unit used for model fitting [5].

This distinction is particularly important in origin research. A geographical label can be confounded with supplier, lot, cultivar, maturity, drying, storage, moisture, acquisition date, illumination, or instrument response. Botanical authentication establishes species identity but does not by itself verify the cultivation location. Evidence for origin authentication consequently requires traceable and independently replicated lots, as well as validation that withholds those lots or other relevant domains. Without such information, “origin” is an observational class label rather than an identified causal source of the spectral differences.

The methodological novelty also requires careful positioning. Zhao et al. previously used HSI, image and spectrum CNNs, Savitzky–Golay and standard normal variate (SNV) preprocessing, learned wavelength weights, lipid and protein measurements, and HPLC spinosin analysis to classify SZR from five reported regions [6]. More generally, Zheng et al. described an end-to-end attention module placed before a three-dimensional residual network for simultaneous band weighting and spectral–spatial prediction [7]. Zhang et al. studied 30 SZR batches from three provinces using electronic-eye, gas-chromatographic electronic-nose, and headspace GC–MS measurements, and associated origin groups with measured volatile compounds [8]. Accordingly, a 3D CNN plus an input-band weighting layer is not, by itself, a new analytical principle, and chemical interpretations require measurements rather than wavelength assignment alone.

The present study therefore reframes the available dataset as an acquisition-aware exploratory pilot. We ask four bounded questions:

1. What is the actual experimental hierarchy, and which units are shared across the legacy development and prediction subsets?
2. How much of the reported cube-model advantage remains after adding a standard, competitive spectral preprocessing baseline?
3. How strongly does performance change when the source hyperspectral cube, rather than the individual seed patch, defines the train–test boundary?
4. Which conclusions about spatial information, wavelength importance, chemistry, provenance, and deployment are justified by the current evidence?

The contribution is thus not a claim of definitive geographical authentication. It is a transparent case study of how experimental-unit definition, preprocessing, and acquisition-aware validation alter the interpretation of a high-accuracy hyperspectral classifier.

## 2. Materials and methods

### 2.1 Intended use, estimand, and terminology

The original eight labels are retained to preserve correspondence with the archived data, but throughout this paper they denote **commercial sample groups identified by supplier-reported purchase origin**. They are not treated as independently verified cultivation origins.

The primary current-data estimand is closed-set discrimination among these eight labels when evaluation patches come from a source hyperspectral cube not used for fitting. This estimand remains narrower than performance on a new lot, supplier, harvest year, laboratory, or instrument. The seed-level random split is retained only to reproduce the legacy benchmark and to show how conclusions depend on the split unit.

We use the following terms:

- **Seed patch:** one segmented 32 × 32 × 392 array associated with an individual connected component.
- **Source cube:** the hyperspectral scene from which a set of seed patches was extracted, represented in the repository by folders `0-1` through `7-2`.
- **Commercial label:** one of eight reported location labels encoded by folder-prefix integers 0–7.
- **Paired source cubes:** the two source cubes sharing a commercial label. Their independence as lots is unknown.
- **External origin validation:** testing on traceable, independently acquired lots or farms outside model development. No such dataset is available here.

### 2.2 Commercial samples and data hierarchy

Commercial dried SZR material was purchased from local suppliers under eight reported location labels: Shijiazhuang City, Hebei Province (HBS); Xian County, Hebei Province (HBX); Anyang City, Henan Province (HNA); Xinxiang City, Henan Province (HNX); Ningxia Hui Autonomous Region (NX); Daning County, Shanxi Province (SXD); Qingjian County, Shaanxi Province (SXQ); and Hetian City, Xinjiang Uyghur Autonomous Region (XJH). Professor Xumei Wang, School of Pharmacy, Xi'an Jiaotong University, confirmed the botanical identity as genuine SZR.

No farm identifiers, GPS coordinates, producer records, cultivar or genotype, wild/grafted status, harvest year, maturity, drying process, storage history, moisture content, independent supplier/lot identifiers, or chain-of-custody records are available in the present repository. Expert authentication is therefore interpreted as botanical authentication only.

The processed dataset comprises 1,264 seeds nested within 16 source cubes. Each label has exactly two source cubes, but seed counts are unequal (Table 1). The source cube—not the seed—is the highest recoverable acquisition unit. The number of independent biological or commercial lots may be smaller than 16 and cannot be recovered from the available metadata.

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

Images were acquired with a push-broom line-scan hyperspectral system from Hangzhou Hyperspectral Imaging Technology Co., Ltd., China. The archived manuscript reports a 600–1700 nm camera range, 512 original bands, spectral resolution of no more than 2.5 nm, and 640 spatial pixels. Two halogen lamps and a motorized translation stage were used. Seeds were arranged on a matte black board, nominally in a 10 × 10 grid.

The reported white/dark reflectance calibration was

$$
R = \frac{I_{\mathrm{raw}}-I_{\mathrm{dark}}}
{I_{\mathrm{white}}-I_{\mathrm{dark}}},
$$

where $I_{\mathrm{raw}}$, $I_{\mathrm{white}}$, and $I_{\mathrm{dark}}$ denote the raw scene, white reference, and dark reference, respectively. Raw `.hdr` scenes and calibration-reference files are not present in the audited repository; consequently, calibration and initial acquisition quality control cannot be independently rerun from raw data.

The archived preprocessing script manually crops each scene before segmentation. The retained processed field was reported as 460 × 535 pixels with 392 wavelengths spanning 949.764–1650.855 nm.

### 2.4 Segmentation and seed-level representations

A false-colour image was produced from three bands. Within a manually selected rectangular scene region, grayscale Otsu thresholding followed by 2 × 2 morphological closing and opening generated a binary foreground mask. Connected-component labelling identified individual seed candidates. Each component was centred in a 32 × 32 window; pixels outside its binary mask were set to zero, with zero padding when a window crossed a crop boundary.

Two representations were retained for each object:

1. a masked hyperspectral patch $X_i \in \mathbb{R}^{32\times32\times392}$; and
2. a mean foreground spectrum $\bar{x}_i \in \mathbb{R}^{392}$ obtained by averaging only pixels for which the binary mask was positive.

The processed `.mat` files contain the hyperspectral patch, mask, and metadata; paired `.csv` files contain wavelength–reflectance rows for the mean spectrum. Because the zero-valued background encodes the seed silhouette, a cube model can potentially use morphology, centring, orientation, truncation, or segmentation artefacts in addition to spectral–spatial tissue variation.

### 2.5 Wavelength vector and spectral precision

All revised analyses read the measured wavelength vector stored in the paired CSV files. The legacy SelecVar plotting code instead constructed 392 equally spaced values using `linspace(949.764, 1650.855, 392)`. Comparison with the recorded nonlinear wavelength grid showed absolute discrepancies of up to 5.77 nm. Therefore, previously reported single-wavelength peaks mapped by the linear grid are nominal and must not be interpreted at sub-band precision. The executed pipeline verified a common, strictly increasing 392-band grid across all 1,264 CSV spectra and exported it in `current_data_study/outputs/wavelengths.csv`.

### 2.6 Validation designs

#### 2.6.1 Legacy seed-level benchmark

To reproduce the original analysis, all 1,264 seed patches were stratified by label and randomly divided with scikit-learn `train_test_split` (`test_size=0.25`, `random_state=42`) into a 948-patch development subset and a 316-patch prediction subset. Five-fold stratified cross-validation was performed inside the development subset for legacy model development.

This prediction subset is **not independent at the source-cube or lot level**. It contains 10–26 patches from every one of the same 16 source cubes represented in development (Table 1). Results from this split quantify interpolation among seeds under shared source-cube conditions and are not estimates of new-cube, new-lot, or new-origin performance.

#### 2.6.2 Reciprocal source-cube diagnostic

The highest-level grouping recoverable from current metadata is the source cube. We therefore define two reciprocal evaluations:

- **Direction A:** train on source cubes `0-1, 1-1, …, 7-1`; test on `0-2, 1-2, …, 7-2`.
- **Direction B:** train on source cubes `0-2, 1-2, …, 7-2`; test on `0-1, 1-1, …, 7-1`.

All preprocessing parameters are fit using the training direction only. Test-direction labels are not used for hyperparameter selection. Accuracy, balanced accuracy, macro-precision, macro-recall, macro-F1, per-label recall, and confusion matrices are reported separately by direction and then descriptively averaged.

These reciprocal tests remove direct source-cube overlap, but they remain diagnostic. With only two cubes per label, “cube” and “split direction” are strongly constrained; no independent validation cube remains for model selection; and paired cubes may share the same supplier or commercial lot. The tests therefore assess acquisition-domain sensitivity rather than external geographical provenance.

#### 2.6.3 Leave-one-source-cube-out diagnostic

As a secondary diagnostic, each of the 16 source cubes was held out in turn. The model was fitted on the other 15 cubes and predicted every seed in the held-out cube; the 16 out-of-fold prediction vectors were then pooled for descriptive accuracy, balanced accuracy, and macro-F1. This design prevents the same source cube from crossing a fold boundary, but it is not leave-one-lot-out validation: the training data still include the paired cube with the same commercial label, and the relationship between that paired cube and an independent lot is unknown.

#### 2.6.4 Spatial falsification status

Mask-only classification, joint foreground-pixel shuffling, and removal of SelecVar are necessary to determine whether the 3D network uses internal spatial organization rather than silhouette or acquisition shortcuts. A grouped implementation covering both reciprocal directions, two architectures, three input conditions, and three fixed optimization seeds was prepared in `deep_models/grouped_hs3i_current_data.py`. It was syntax-checked but not trained in the current environment because PyTorch and h5py were unavailable. Accordingly, these controls define the evidential requirement for a future run; they are not reported as completed experiments, and this paper makes no positive claim of spatial information gain.

### 2.7 Spectral baselines and preprocessing

Mean foreground spectra were evaluated using multinomial logistic regression (LR), a radial-basis-function support-vector machine (SVM), partial least-squares discriminant analysis (PLS-DA), random forest (RF), and the legacy one-dimensional convolutional neural network (1D-CNN). Fixed classical settings were LR with `C=1`, L-BFGS, tolerance $10^{-10}$ and at most 5,000 iterations; SVM with `C=10` and `gamma="scale"`; PLS-DA as 20-component PLS regression on one-hot outcomes followed by response argmax; and RF with 200 trees. Standard scaling was fitted within each training split or fold only. No grid search, test-direction tuning, or post hoc hyperparameter selection was performed.

SNV was added as a necessary competitive baseline. For spectrum $x_i$ with $B=392$ bands,

$$
x_{i,b}^{\mathrm{SNV}} = \frac{x_{i,b}-\bar{x}_i}{s_i},
$$

where $\bar{x}_i$ and $s_i$ are the within-spectrum mean and standard deviation. SNV reduces multiplicative scatter and additive offsets that can arise from seed geometry and illumination [9]. Additional fixed candidates were multiplicative scatter correction (MSC) using only the training-set mean as reference, Savitzky–Golay smoothing, and a Savitzky–Golay first derivative (11-band window, second-order polynomial). SNV and Savitzky–Golay transforms were applied independently to each spectrum; the MSC reference and every subsequent scaler were learned only from the corresponding training partition. These alternatives form a transparent sensitivity panel, not a test-set search from which only the highest accuracy is retained.

### 2.8 HS3I-Net and neural baselines

HS3I-Net receives a masked tensor of shape 1 × 392 × 32 × 32. Its first module learns one global non-negative coefficient per input band. For learned parameter $a_b$,

$$
w_b=\operatorname{softplus}(a_b), \qquad X'_{h,w,b}=w_bX_{h,w,b}.
$$

The weights are optimized jointly with a four-block 3D residual backbone whose channels increase from 16 to 128. The first two blocks use spectral–spatial kernels of 11 × 3 × 3 and 7 × 3 × 3; later blocks use 3 × 3 × 3 kernels. Global average pooling produces a 128-dimensional representation followed by fully connected classification layers. The training objective combines classification loss with Hoyer-style sparsity and adjacent-band smoothness penalties on $w$.

Legacy training used AdamW for 360 epochs, batch size 32, a main-network learning rate of $3\times10^{-4}$, a SelecVar learning rate of $10^{-3}$, weight decay $10^{-4}$, label smoothing 0.1, Mixup $\alpha=0.3$, dropout 0.35, 10 warm-up epochs, and cosine-annealing warm restarts. SelecVar sparsity and smoothness coefficients were $5\times10^{-5}$ and $10^{-5}$, respectively. Horizontal and vertical flips and small intensity scaling were applied during training; spectral-axis reversal was not used. Test-time augmentation was disabled.

The 1D-CNN uses only the 392-element mean spectrum and comprises four one-dimensional convolutional blocks, adaptive average pooling, and fully connected layers. The no-SelecVar model retains the 3D backbone and training procedure while omitting input-band reweighting.

For any future grouped neural run, model selection must be separated from both source-cube test directions. Because the current dataset provides no third cube per label, architecture and optimization settings are frozen from the documented legacy configuration rather than selected on reciprocal test performance. This avoids direct tuning to the two diagnostic test sets but does not remove the fundamental small-group limitation. No grouped neural performance is reported in this revision because that run was not executed.

### 2.9 Exploratory model-behaviour analyses

SelecVar coefficients are model parameters, not causal wavelength effects. No revised SelecVar plot is used as evidence here because fold- and random-seed stability was not available. Any future plot must use the measured wavelength vector, and a band must not be called “chemically meaningful” without a perturbation test, remove-and-retrain evidence, and matched chemical measurements. The present dataset contains no moisture, lipid, protein, starch, spinosin, jujuboside, fatty-acid, or metabolomic measurements.

The previous manuscript reported Grad-CAM curves and spatial maps from the last 3D residual block. At audit, however, the repository did not contain the Grad-CAM implementation, corresponding checkpoints, or machine-readable activation outputs. Moreover, the final feature map has only approximately 12 spectral positions, and each activation has an estimated receptive field spanning approximately 321 input bands. Upsampling such a representation to 392 wavelengths cannot resolve a unique peak at single-band precision. Grad-CAM is therefore excluded from confirmatory evidence in this rewrite. It may be restored as a clearly labelled exploratory supplement only after code, checkpoints, target-layer definition, interpolation procedure, and randomization/perturbation sanity checks are deposited [10,11].

### 2.10 Outcomes and statistical analysis

Accuracy is reported together with the number correct. Macro-F1 is emphasized for directional source-cube tests because class sizes differ. For the legacy 316-patch prediction subset, Wilson score intervals are calculated as descriptive binomial intervals [12], with the explicit caveat that they do not account for dependence among seeds from the same cube.

The archived project did not contain the aligned HS3I-Net prediction vector. Consequently, no paired McNemar test between HS3I-Net and a spectral baseline was performed: its discordant cells cannot be recovered from aggregate correct counts. Differences of one or two correct seeds are reported descriptively only.

No population-level confidence interval or hypothesis test is used to claim origin generalization from the reciprocal source-cube experiment. Sixteen source cubes—and only two per label—are insufficient to identify between-lot or between-origin variance. Direction-specific and per-cube results are reported descriptively.

### 2.11 Software and reproducibility controls

Legacy neural scripts specify Python 3.8, PyTorch 2.0.0, CUDA 11.8, and an NVIDIA RTX 3090 GPU. The executed spectral reanalysis used Python 3.11.9, NumPy 2.4.4, pandas 3.0.2, SciPy 1.17.1, and scikit-learn 1.8.0 on Windows. It exports:

- a manifest containing sample path, label, source-cube ID, and split assignment;
- the exact measured wavelength vector;
- the fixed preprocessing and model configuration;
- package versions, operating system, and random seed;
- seed-level predictions and confusion matrices; and
- machine-readable per-fold and per-direction metrics.

The audited legacy scripts point to a `cube/` directory, whereas the supplied processed dataset is stored under `data/`. The revised spectral entry point is `python current_data_study/analyze.py`; dependencies are recorded in `current_data_study/requirements-lock.txt`, and deterministic outputs are written to `current_data_study/outputs/`. The workspace is not a valid Git repository, so no commit hash is asserted; a SHA-256 fingerprint over relative CSV paths and contents is stored in `results.json`. Missing legacy checkpoints and Grad-CAM code remain disclosed rather than treated as reproducible.

## 3. Results

### 3.1 Experimental-unit audit

The nominal sample count was 1,264, but the highest recoverable acquisition-level count was 16 source cubes. Each of the eight labels was represented by two cubes, containing 54–90 segmented seeds per cube (Table 1). Under the legacy stratified split, all 16 cubes contributed to both development and prediction. The prediction subset contained 10–26 seeds from each source cube. Thus, the “independent prediction set” was independent only at the extracted-patch index level; it was not independent at the scene, commercial-lot, supplier, harvest, or instrument level.

The label hierarchy also mixed geographical scales: most labels referred to a city or county, whereas NX referred to an autonomous region. No available metadata supported a finer or harmonized provenance definition. For this reason, all subsequent results are interpreted as discrimination of the archived commercial labels.

### 3.2 Random seed-level benchmark and preprocessing sensitivity

The legacy seed-level benchmark yielded 96.84% accuracy for HS3I-Net (306/316 correct). The original mean-spectrum LR, 1D-CNN, SVM, and RF achieved 87.03%, 81.96%, 72.47%, and 65.51%, respectively (Table 2). These values created the apparent advantage reported in the original draft when HS3I-Net was compared with unoptimized or differently preprocessed spectrum baselines.

Fixed preprocessing alternatives changed that interpretation. Under the locked pipeline, SNV–LR classified 305/316 seeds (96.52%; macro-F1 96.59%), while Savitzky–Golay first-derivative LR classified 307/316 (97.15%; macro-F1 97.23%). MSC–LR and 20-component PLS-DA also reached 94.62% and 93.99%. The first-derivative result was one correct seed above HS3I-Net, whereas SNV–LR was one below. These small descriptive differences cannot be attributed to spatial information or statistical superiority: all models used the same source-cube-overlapping prediction subset, several fixed preprocessing candidates were inspected, seeds within cubes are dependent, and the aligned HS3I-Net prediction vector required for a paired test was unavailable.

**Table 2. Performance on the legacy random seed-level prediction subset.**

| Model | Input and preprocessing | Correct / 316 | Accuracy (%) | Macro-F1 (%) | Interpretation |
|---|---|---:|---:|---:|---|
| HS3I-Net | Masked 32 × 32 × 392 cube | 306 | 96.84 | 96.87 | Conditional within-acquisition benchmark |
| SG first-derivative–LR | Mean spectrum; 11-band second-order first derivative | 307 | 97.15 | 97.23 | Highest executed spectral sensitivity result; not externally selected |
| SNV–LR | Mean spectrum; SNV; leakage-safe scaling | 305 | 96.52 | 96.59 | Competitive fixed spectral baseline |
| MSC–LR | Mean spectrum; training-reference MSC | 299 | 94.62 | 94.74 | Fixed spectral sensitivity analysis |
| PLS-DA | Mean spectrum; scaling; 20 latent components | 297 | 93.99 | 94.04 | Fixed latent-variable baseline |
| LR | Mean spectrum; legacy scaling | 275 | 87.03 | 87.06 | Sensitive to preprocessing choice |
| SG smooth–LR | Mean spectrum; 11-band second-order smoothing | 274 | 86.71 | 86.73 | Smoothing alone did not improve LR |
| 1D-CNN | Mean spectrum | 259 | 81.96 | 81.70 | Legacy neural spectral baseline |
| SVM | Mean spectrum; legacy scaling | 229 | 72.47 | 71.91 | Legacy baseline |
| RF | Mean spectrum | 207 | 65.51 | 65.24 | Legacy baseline |

The overlap of HS3I-Net with simple, fixed spectrum-only pipelines means that the legacy result does not isolate a benefit from spatial organization. It shows that several models can interpolate strongly among seed patches when training and prediction share all source cubes.

### 3.3 Reciprocal source-cube diagnostics

Separating the paired source cubes reduced performance and changed model ranking (Table 3). PLS-DA was highest in both reciprocal directions at 85.34% and 89.85%, followed by different orderings of SNV–LR and first-derivative LR. Raw LR fell to 79.64% and 80.77%, and SVM to 71.66% and 68.00%. Directional differences were material: for example, SNV–LR changed by 5.07 points and PLS-DA by 4.51 points.

**Table 3. Reciprocal source-cube diagnostics.**

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

The decrease relative to the random seed split is consistent with source-cube-dependent variation. It does not prove that the residual classification is geographical: paired cubes could share supplier, lot, processing, storage, or other label-specific factors. Conversely, any failure in these reciprocal tests should not be attributed solely to model architecture because each class provides only one training cube in a direction.

### 3.4 Leave-one-source-cube-out stability

The 16-fold leave-one-source-cube-out analysis gave a different ordering (Table 4). SNV–LR was highest at 79.03% (macro-F1 77.92%), followed by first-derivative LR at 76.98% and MSC–LR at 74.37%. PLS-DA, despite leading both reciprocal directions, fell to 62.10% (macro-F1 61.39%). Raw SVM and RF fell to 50.95% and 47.39%. This instability shows that a favourable result under one grouped construction is not sufficient to identify a robust model.

**Table 4. Pooled out-of-fold performance when each source cube is held out once.**

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

These pooled seed-level metrics are descriptive. Each fold still trains on the paired cube carrying the same label, and the 1,264 predictions are clustered within only 16 held-out cubes; the table does not represent 1,264 independent provenance trials.

### 3.5 Spatial mechanism remains untested

The legacy 96.84% result cannot establish that the network uses within-seed spatial organization. Background zeroing retains the silhouette, and the 32 × 32 centroid crop retains size, shape, orientation, position, and truncation cues. The grouped mask-only, spatial-shuffle, and no-SelecVar protocol was not executed, so there are no current-data results capable of separating internal spatial organization from these alternatives. The defensible result is full-cube classification performance under the legacy random split, not an identified spectral–spatial mechanism.

### 3.6 Exploratory wavelength weighting and saliency audit

The legacy SelecVar output assigned relatively larger normalized coefficients to parts of the longer-wavelength region and reported nominal local maxima near 1097, 1177, 1265, 1425, 1452, 1498, 1549, 1581, and 1649 nm. These locations are provisional because they were mapped using an equally spaced wavelength grid rather than the measured nonlinear grid. The coefficients also lack stability estimates across independent lots, harvests, or instruments and have not been validated by band occlusion or remove-and-retrain experiments.

The legacy manuscript further described Grad-CAM responses near 1330–1340 and 1430–1440 nm and non-uniform spatial maps. Those observations are not presented as results here. The analysis code and checkpoints are absent, the target layer has coarse spectral resolution and a broad receptive field, and the threshold of 0.5 after normalization was arbitrary. No claim can therefore be made that an exact wavelength or seed tissue region caused a classification decision.

No reference chemistry was measured in these samples. Assignments to lipid, protein, starch, water, spinosin, or other constituents have consequently been removed. At most, the current model weights nominate broad spectral intervals for future experiments with matched chemical assays.

### 3.7 Summary of claim status

**Table 5. Claims supported and not supported by the current dataset.**

| Claim | Current evidence | Status |
|---|---|---|
| Eight archived commercial labels can be discriminated under shared source-cube conditions | HS3I-Net 96.84%; executed spectral pipelines 93.99%–97.15% on the legacy split | Supported conditionally |
| HS3I-Net has a large advantage over spectrum-only models | Fixed spectral pipelines bracket the HS3I-Net point estimate | Not supported |
| Spatial organization provides independent predictive information | Required grouped falsification controls were not executed | Not established |
| Spectrum models transfer between the paired source cubes | Reciprocal accuracy is model- and direction-dependent (63.03%–89.85%); leave-one-cube ranking changes | Supported only as a restricted diagnostic |
| Performance generalizes to an unseen lot, supplier, year, or instrument | No such independent groups | Not tested |
| Labels are verified cultivation origins | Supplier-reported labels; no chain of custody | Not established |
| Selected bands identify causal chemical differences | No matched chemistry; wavelength mapping issue | Not supported |
| The method establishes practical traceability | No open-set, external, calibration, or deployment test | Not supported |

## 4. Discussion

### 4.1 Validation design changes the scientific conclusion

The principal result of this reanalysis is not another high accuracy value; it is the change in what the accuracy can mean. The legacy prediction subset included seeds from every source cube used for development. Its 96.84% result is therefore a measure of patch-level interpolation within a shared acquisition domain. Calling that subset “independent” without specifying the grouping level obscures the most important dependence in the data.

This is not merely terminological. Seeds from the same scene share illumination, reflectance calibration, background, camera state, acquisition timing, plate layout, and preprocessing decisions. They may also share a commercial lot. Deep and classical models can learn any stable combination of these properties. Source-cube-aware evaluation is consequently the minimum defensible boundary recoverable from the current repository, although it still falls short of unseen-lot validation.

### 4.2 Competitive spectral preprocessing removes the apparent large cube advantage

The original comparison suggested a 14.88-percentage-point improvement of HS3I-Net over the 1D-CNN. In the locked reanalysis, SNV–LR was only 0.32 points (one seed) below HS3I-Net, while first-derivative LR was 0.32 points (one seed) above it. These are descriptive sensitivity results, not pairwise superiority tests, but they remove the empirical basis for a large cube-model advantage. This finding has two implications.

First, baseline quality is part of causal attribution. A cube model cannot be credited with a spatial advantage merely because it outperforms an inadequately preprocessed spectrum model. SNV directly addresses scatter and offset variation common in reflectance spectra and should have been included given its established role in NIR analysis and its use in prior SZR HSI work [6,9].

Second, accuracy alone does not reveal which information a model uses. HS3I-Net may use useful local spectral heterogeneity, but it may also use shape or acquisition artefacts. The current aggregate results cannot distinguish these possibilities. The unavailable paired HS3I-Net prediction vector precludes McNemar testing, and the unexecuted spatial controls preclude a mechanistic claim. The source-cube-isolated spectral diagnostics are therefore the strongest analyses actually completed with the current data.

### 4.3 What the reciprocal experiment can and cannot establish

Raw LR fell to 79.64% and 80.77%, and SVM to 71.66% and 68.00%, after separating source cubes. Even the stronger pipelines varied by direction: PLS-DA reached 85.34% and 89.85%, while SNV–LR reached 83.39% and 88.46%. In leave-one-source-cube-out prediction, SNV–LR was highest at 79.03% and PLS-DA fell to 62.10%. This protocol-dependent ranking is valuable diagnostic evidence: acquisition grouping matters, and random patch splits overstate robustness to at least one recoverable source of shift.

Nevertheless, reciprocal source-cube performance is not external origin performance. With only two source cubes per label, each direction trains on one cube and tests on one cube for that class. The paired cubes may be repeat scenes of the same purchased material. Thus, success could still reflect supplier or lot signatures, while failure could reflect idiosyncratic cube differences. No statistical method can manufacture independent lots from this structure. The correct response is to narrow the estimand, show both directions, report each cube, and avoid population-level claims.

### 4.4 Novelty relative to prior work

The revised positioning also changes the novelty claim. Zhao et al. already compared image and spectrum CNNs for SZR geographical labels, used SNV, learned feature-wavelength weights, measured lipid and protein, and compared HSI with HPLC spinosin analysis [6]. Zheng et al. previously combined first-layer band attention with a 3D residual network [7]. Zhang et al. used independently collected SZR batches and measured volatile compounds in an origin-related analysis [8].

HS3I-Net should therefore be described as an application-specific implementation of established components, not the first interpretable cube-based SZR framework. The potentially publishable current-data contribution is methodological transparency: an empirical demonstration that experimental hierarchy, preprocessing, and validation construction reverse the original interpretation. This is suitable as a rigorous pilot or methodological cautionary case; a top-tier confirmatory origin-authentication article still requires independently traceable lots and external domains, not merely completion of more models on the same 16 cubes.

### 4.5 Limits of model interpretation

Input-band coefficients indicate how a fitted model scales bands under a particular loss and regularization scheme. They do not identify a compound or a causal environmental mechanism. Correlated neighboring bands can substitute for each other; weights can change across initializations; and a large coefficient need not imply a large performance effect. Robust interpretation would require stability across independent groups, perturbation and retraining, and measurements of candidate chemical or physical properties.

The Grad-CAM claims require even greater restraint. A coarse final-layer activation cannot be converted into precise wavelength evidence by interpolation. Visually plausible saliency can also persist after model or label randomization, motivating explicit sanity checks [11]. Without executable code and archived outputs, reproducibility fails before biological interpretation is considered. The appropriate current-data action is to omit confirmatory Grad-CAM claims and preserve the old plots, if desired, only as undocumented exploratory material that is not used to support the conclusion.

### 4.6 Strengths and limitations

This study has several strengths within its revised scope. It audits the experimental unit from the repository rather than inferring independence from the nominal seed count; reproduces the legacy split; adds a strong and simple preprocessing baseline; uses recoverable source-cube identifiers to construct a more demanding diagnostic; separates supported from unsupported claims; and specifies falsification tests before interpreting the full-cube model.

The limitations are decisive. The data contain only 16 source cubes and no verified independent lot hierarchy. Reported locations lack chain-of-custody documentation and mix geographical resolutions. Paired cubes may not be independent. All data appear to come from one acquisition system and period, although the exact session metadata are absent. Raw hyperspectral scenes and calibration references are unavailable. Manual cropping introduces operator dependence. The black-background mask preserves morphology and positioning. No matched chemistry or moisture measurements exist. The legacy code has a data-path mismatch, the measured wavelength vector was not used in one importance plot, and Grad-CAM code and checkpoints are missing. Neural-model variability and all current-data spatial ablations remain to be quantified.

These limitations prevent claims of geographical authentication, causal chemistry, or operational traceability regardless of the eventual accuracy of the revised models.

### 4.7 Practical interpretation and next evidence required

Under current evidence, a positive classification means only that a seed resembles one of eight archived commercial groups under the sampled acquisition conditions. It should not be used to certify cultivation origin or adjudicate fraud. The model also forces every seed into one of eight known labels; no unknown-origin or adulterant rejection has been evaluated.

A future confirmatory study would require independently traceable farms or lots, multiple harvest years, randomized mixed-origin acquisition scenes, replicated days and operators, a second instrument or laboratory, matched moisture and chemical assays, and a locked external test. Such data are outside the present study. The current pilot can instead inform that design by identifying likely acquisition sensitivity and by screening whether any purported spatial or band-specific signal survives falsification.

## 5. Conclusions

The present dataset supports an exploratory, closed-set classification study of eight commercial SZR sample labels, not a validated system for geographical-origin authentication. HS3I-Net achieved 96.84% on a random seed-level split, but the split shared all 16 source cubes between development and prediction. Under the same split, fixed spectrum-only pipelines ranged up to 97.15%; SNV–LR achieved 96.52%. After source-cube separation, performance and model ranking changed substantially: reciprocal accuracies ranged from 63.03% to 89.85%, and the strongest leave-one-source-cube-out spectral result was 79.03%. These diagnostics show that preprocessing and acquisition grouping affect performance while remaining insufficient to estimate unseen-lot or unseen-origin generalization.

SelecVar and Grad-CAM outputs do not provide chemical confirmation: the dataset lacks matched assays, the legacy wavelength mapping was imprecise, and the Grad-CAM implementation and artifacts are absent. Accordingly, the scientifically defensible conclusion is that preprocessing and validation hierarchy dominate the interpretation of this pilot. Spatial, chemical, provenance, and traceability claims should be reconsidered only if they survive source-cube-aware falsification and, ultimately, independent lot-level data.

## CRediT authorship contribution statement

Ziying Wang: Writing—original draft, validation, methodology, investigation, formal analysis, conceptualization. Guanqiao Chen: Investigation, methodology, formal analysis, validation, software. Jiale Wei: Investigation, methodology, formal analysis, validation. Xumei Wang: Writing—review and editing, supervision, project administration, funding acquisition, conceptualization.

Author verification is required before submission because the audited files do not establish who performed the new data curation, software, and validation work.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Ethics statement

This study used commercial plant-derived seed material and involved no human participants or live animals.

## Funding

This work was supported by the TCM Research and Innovation Team in Shaanxi Administration of Traditional Chinese Medicine (TZKNCXTD-09) and the Shaanxi Academy of Sciences Talent Introduction Program (2025k-3).

## Acknowledgements

The authors thank Hangzhou Hyperspectral Imaging Technology Co., Ltd. for technical support in hyperspectral image acquisition.

## Data availability

The audited workspace contains 1,264 processed seed-level `.mat` files and paired mean-spectrum `.csv` files organized under 16 source-cube folders. The revised analysis exports a de-identified manifest linking each seed to its commercial label and source-cube ID, plus the exact measured wavelength vector and deterministic split predictions, under `current_data_study/outputs/`. No public persistent repository or DOI had been assigned at the time of this rewrite; those local paths are not a substitute for deposition and must be replaced by a verified archive citation before submission. Raw hyperspectral scenes, white/dark calibration acquisitions, independently verified lot metadata, and matched chemical measurements are not available and therefore cannot be shared. These absences limit reproduction of acquisition calibration and segmentation from raw data.

## Code availability

The legacy workspace contains scripts for HS3I-Net, a 3D-CNN without SelecVar, a 1D-CNN, LR/RF/KNN, SVM/PLS-DA, and MATLAB patch extraction. The executed classical reanalysis is in `current_data_study/`, with locked requirements, configuration, tests, a deterministic entry point, predictions, metrics, and a data fingerprint. The unexecuted grouped neural protocol is in `deep_models/` and is explicitly labelled as syntax-checked only. The workspace is not a valid Git repository and has no release DOI; a versioned public archive must therefore be created before journal submission. Grad-CAM code, checkpoints, and machine-readable outputs remain absent and are not evidence for the conclusions.

## Appendix A. Figure status and caption specifications

**Figure 1. Experimental hierarchy and scope of inference.** Diagram showing pixels nested within seeds, seeds within 16 source cubes, two cubes within each commercial label, and unknown lot/supplier relationships. The figure must visually distinguish recoverable identifiers from unavailable provenance metadata.

**Figure 2. Hyperspectral acquisition and patch extraction.** (a) Imaging system; (b) manual scene crop, Otsu mask, morphology, and connected components; (c) 32 × 32 masked cube and foreground mean spectrum. The caption must state that background zeroing retains the silhouette.

**Figure 3. Comparison of validation designs.** (a) Legacy random seed split showing every cube in both development and prediction; (b) reciprocal `*-1`→`*-2` and `*-2`→`*-1` source-cube diagnostics; (c) the unavailable but deployment-relevant unseen-lot external test.

**Figure 4. Paired source-cube spectra within each archived commercial label.** Generated as `current_data_study/figures/figure_source_cube_spectra.pdf`. Lines are source-cube means on the measured wavelength grid; shading is ±1 SD across seeds and is descriptive rather than a batch-level interval. No chemical peak assignment is made.

**Figure 5. Performance depends on preprocessing and validation construction.** Generated as `current_data_study/figures/figure_performance_by_protocol.pdf`. The three panels show the random seed holdout, both reciprocal source-cube directions, and pooled leave-one-source-cube-out accuracy. The legacy HS3I-Net point estimate is a labelled reference line. No seed-level interval is plotted as though seeds were independent lots.

**Figure 6. SNV–LR errors across source-cube validation constructions.** Generated as `current_data_study/figures/figure_snv_lr_grouped_confusions.pdf`. Rows are normalized within archived label for the two reciprocal directions and the pooled leave-one-source-cube-out predictions.

**Figure 7. Spatial falsification results—not included.** A comparison of mask-only, shuffled-pixel, mean-spectrum, no-SelecVar, and HS3I-Net models must not be drawn until the complete grouped protocol is executed. An empty panel or expected result must not be substituted.

**Figure 8. Exploratory SelecVar stability—not included.** If generated in a later verified run, plot weights against the measured nonlinear wavelength vector for every direction and random seed, label the panel “model band weights,” and include it only with remove-and-retrain or occlusion checks.

Grad-CAM should not appear in the main paper. If reproducibly regenerated, place it in the supplement with sanity checks and a warning that the last-layer spectral resolution does not support exact wavelength localization.

## Appendix B. Supplementary package status

- **Available Table/File S1:** `current_data_study/outputs/dataset_manifest.csv`, containing the seed-level path, class code, source-cube ID, patch number, and file hash.
- **Available Table/File S2:** `current_data_study/outputs/wavelengths.csv`, containing the exact measured 392-band vector; cross-sample consistency is enforced by the loader.
- **Available Table/File S3:** `current_data_study/outputs/metrics.csv`, `fold_metrics.csv`, and `confusion_matrices.csv`, containing complete executed-model summaries.
- **Available Table/File S4:** `current_data_study/outputs/predictions.csv`, containing every executed seed-level prediction and fold identifier.
- **Available File S5:** `current_data_study/config.json`, `requirements-lock.txt`, and `outputs/results.json`, containing fixed settings, the executed environment, data fingerprint, and interpretation warning.
- **Available Figure Source S1:** `current_data_study/figures/figure_performance_source.csv` and the deterministic `make_figures.py` entry point.
- **Not available:** neural grouped results, training curves, aligned legacy HS3I-Net predictions, a McNemar table, verified Grad-CAM artifacts, and perturbation-based SelecVar evidence. These items are omitted rather than represented by placeholders.

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
11. Adebayo J, Gilmer J, Muelly M, Goodfellow I, Hardt M, Kim B. Sanity checks for saliency maps. In: *Advances in Neural Information Processing Systems 31*. 2018.
12. Wilson EB. Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association*. 1927;22:209–212. https://doi.org/10.1080/01621459.1927.10502953.
