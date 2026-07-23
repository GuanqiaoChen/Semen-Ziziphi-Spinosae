# Leakage-controlled whole-cube transfer for acquisition-robust hyperspectral provenance authentication of *Ziziphi Spinosae* Semen

> **Author-facing draft status; remove before submission.** All development results are newly executed from constructed batches 0–7 in `provenance_study/outputs/development_acquisition_calibration/` and are reproduced by `provenance_study/explore_acquisition_calibration.py`. The one-shot locked confirmation (reserved batches 8–9, opposite acquisition cube) has **not** been executed; every locked value is an explicit `{{PLACEHOLDER}}` to be filled only from the canonical output of `provenance_study/run_acquisition_locked_confirmation.py` after the authorized run. Constructed batches are deterministic subdivisions of the same 16 archived source images; they are not independent physical lots, farms, years, or instruments.

**Authors:** {{AUTHOR_LIST}}

**Affiliations:** {{AFFILIATIONS}}

**Corresponding author:** {{CORRESPONDING_AUTHOR}}

## Highlights

- A leakage-controlled *whole-source-cube* transfer protocol is proposed as the primary evaluation axis for hyperspectral seed provenance, replacing optimistic within-cube random splits.
- First-derivative + shrinkage LDA is the most acquisition-transfer-robust representation/classifier, exceeding SNV, MSC, second-derivative, logistic, RBF-SVM, and a deep spectral–spatial reference.
- Unlabelled target-acquisition normalization (standardizing the incoming cube on its own statistics) further reduces cross-acquisition error and rescues the origins that collapse under a cube change.
- Post-hoc temperature scaling repairs the confidence distortion that a cube change induces, restoring near–same-domain calibration.
- Within-seed pixel-covariance/texture features and elaborated shift-aware calibration/conformal variants are retained as **negative results**: they encode acquisition-specific shortcuts or provide no advantage over standard versions.

## Abstract

Geographical-origin authentication of medicinal seeds must remain reliable when the acquisition condition changes, yet hyperspectral studies routinely report near-perfect accuracy obtained by randomly splitting seeds *within* the same acquisition image, which lets image-specific signals cross the evaluation boundary. Using the currently available archive of 1,264 *Ziziphi Spinosae* Semen seeds (eight recorded origins, 392 bands, 949.764–1650.855 nm, two source images per origin), we reframe the task around a leakage-controlled **whole-source-cube transfer** protocol: models are trained on one acquisition cube and tested on the other, in both directions. Within each origin and source image, seeds were deterministically assigned to ten constructed batches; batches 0–7 (1,012 seeds) were used for all development and model selection, and batches 8–9 (252 seeds) were programmatically reserved for a single locked confirmation. On the development whole-cube axis, first-derivative (Savitzky–Golay, window 15) shrinkage linear discriminant analysis reached a balanced accuracy of 93.6%, the best of five representations (raw, SNV, MSC, first- and second-derivative) and four classifiers, whereas a standard normal-variate logistic baseline reached only 85.7%. Standardizing the incoming cube on its own unlabelled statistics further raised balanced accuracy to 95.1% (+1.45 percentage points, 95% cluster-bootstrap interval [+0.75, +2.11]) and rescued the origins that collapse under a cube change. A single post-hoc temperature, fit only on the training cube, reduced cross-cube expected calibration error from 0.038 to 0.010 and negative log-likelihood from 0.236 to 0.179. In contrast, within-seed pixel-covariance descriptors degraded transfer (balanced accuracy 0.936→0.920; NLL 0.239→0.606), and constructed-batch-grouped calibration and conformal variants were statistically indistinguishable from their i.i.d. counterparts—both retained as negative results. In the one-shot locked confirmation on reserved batches 8–9 of the opposite cube, the frozen pipeline achieved {{LOCKED_PRIMARY_THETA}} balanced accuracy versus {{LOCKED_SNV_LR_THETA}} for the weak baseline, and the pre-registered directional gates were {{LOCKED_GATE_DECISION}}. These findings define an honest, reproducible acquisition-robustness framework and a deployable pipeline; they do not establish external certification across farms, years, or instruments, which the two-cube archive cannot support.

**Keywords:** hyperspectral imaging; geographical origin; *Ziziphi Spinosae* Semen; acquisition-domain shift; grouped validation; probability calibration; domain adaptation

## 1. Introduction

*Ziziphi Spinosae* Semen (SZS; Suanzaoren), the seed of *Ziziphus jujuba* Mill. var. *spinosa*, is a widely used sedative-hypnotic medicinal material whose quality and value depend on geographical origin [1–3]. Near-infrared hyperspectral imaging is attractive for rapid, non-destructive origin screening [4], and prior SZS work has combined hyperspectral data with convolutional networks and learned band weighting for origin classification [6,7] and with chromatographic measurements for adulteration and origin analysis [8].

A recurring validation problem undermines the interpretation of the resulting accuracies. When many seeds are nested within few acquisition images, randomly distributing seeds from the same image across training and testing allows illumination, white/dark reference, segmentation, instrument state, and other image-specific signals to cross the evaluation boundary; grouped or blocked validation is required for such structured data [5]. For the present archive, only origin label, source-image replicate, and seed identifier are recoverable; physical lot, farm, harvest year, operator, and instrument-session identifiers are not. Consequently, a high within-image score is not evidence that a model has recovered an acquisition-robust origin signal, and no analysis of this archive can estimate generalization to unobserved farms, years, or instruments.

We therefore ask the question that this archive *can* answer honestly and that matters most for deployment: **when the acquisition cube changes, does origin classification remain accurate and well-calibrated?** We make three contributions. First, a *methodological/protocol* contribution: a leakage-controlled whole-source-cube transfer protocol, with deterministic constructed batches enabling grouped development and a single reserved-data confirmation. Second, an *empirical* contribution: identification of a compact, acquisition-robust pipeline—first-derivative representation, shrinkage discriminant analysis, unlabelled target-acquisition normalization, and post-hoc calibration—and quantification of its effect against transparent baselines. Third, a *mechanistic/negative-result* contribution: evidence that within-seed pixel-covariance and texture descriptors mainly carry acquisition-specific information rather than stable origin information, and that elaborated shift-aware calibration and group-conformal variants do not outperform their standard counterparts. We do not claim a new neural architecture; the constituent transforms and classifiers are established tools.

## 2. Materials and methods

### 2.1 Samples, spectra, and recorded origins

The archive contains 1,264 seed-level observations from eight recorded origins (HBS, HBX, HNA, HNX, NX, SXD, SXQ, XJH), each represented by two source hyperspectral images. Each seed provides a mean reflectance spectrum of 392 bands (949.764–1650.855 nm) and a MAT patch (32×32 pixels with a foreground mask). "Recorded origin" is used deliberately: the archive lacks verified farm coordinates, traceable lot identifiers, harvest dates, and chain-of-custody metadata.

### 2.2 Constructed batches and the leakage boundary

Within each origin × source-image directory, numeric seed identifiers were sorted and permuted with a fixed generator (`20260721 + label·101 + replicate·1009`); rank modulo ten defined ten constructed batches. Batches 0–7 (1,012 seeds) were used for all development and model selection; batches 8–9 (252 seeds) were programmatically reserved. Development entry points enumerate the manifest without hashing locked files and reject any locked record before numerical I/O.

### 2.3 Whole-source-cube transfer protocol (primary axis)

The only recoverable acquisition boundary is the source image (cube). The primary evaluation trains on one cube and tests on the other, in both directions (cube-1→cube-2 and cube-2→cube-1); the reported statistic θ is balanced accuracy averaged equally over the two directions. A same-domain leave-one-constructed-batch-out (LOBO) evaluation over batches 0–7 is reported only as an optimistic reference. Uncertainty uses a 2,000-replicate bootstrap that resamples origin×constructed-batch clusters.

### 2.4 Frozen acquisition-robust pipeline

The pipeline selected during development (`provenance_study/acquisition_robust_pipeline.py`) is: (i) Savitzky–Golay first derivative (window 15, polynomial 2); (ii) standardization fit on the training cube; (iii) **unlabelled target-acquisition normalization**—the incoming cube (a batch of seeds of unknown origin) is standardized on its own per-band statistics, removing a per-cube affine batch effect without labels; (iv) shrinkage LDA (`lsqr`, analytic shrinkage); and (v) a single post-hoc temperature fit on the training cube's constructed-batch-grouped out-of-fold predictions. Prediction requires a batch of incoming seeds (≥8) so target statistics are estimable; single-seed scoring is rejected.

### 2.5 Baselines, ablations, and pre-registered negatives

Five representations (raw, SNV, MSC, first-derivative, second-derivative) were crossed with shrinkage LDA, multinomial logistic regression, RBF-SVM, and an equal LDA+LR ensemble on the whole-cube axis. A standard-normal-variate logistic model is the weak baseline; a 321,776-parameter spectral–spatial network from prior work [this repository] is the deep reference. Pre-registered negative controls: (a) within-seed pixel-covariance descriptors (PCA-16 of per-pixel SNV spectra) concatenated to the spectrum; (b) constructed-batch-grouped ("shift-aware") temperature versus i.i.d. temperature; (c) group-conformal versus i.i.d. split-conformal coverage under shift.

### 2.6 One-shot locked confirmation

After the pipeline, estimands, and effect gate were frozen (`docs/采集域稳健产地溯源方法与锁定验证方案.md`), a single confirmation is defined on the reserved batches: train on cube-*-1 batches 0–7 → test on cube-*-2 batches 8–9, and the reverse. Four pre-registered directional gates must all hold: (1) the frozen pipeline exceeds the SNV-LR baseline with a bootstrap lower bound above zero; (2) target normalization does not reduce balanced accuracy; (3) calibration reduces ECE and NLL; (4) covariance concatenation does not help. The run is guarded by the confirmation phrase `UNLOCK_BATCHES_8_9` and a completion marker, and executes exactly once.

## 3. Results

### 3.1 Representation drives cross-acquisition transfer

On the whole-cube axis (Figure 1A), first-derivative shrinkage LDA reached 93.6% balanced accuracy, exceeding SNV (86.8%), MSC (86.5%), raw (85.2%), and second-derivative (90.6%) representations under the same classifier; logistic, RBF-SVM, and the ensemble did not exceed derivative LDA. The same-domain LOBO reference for first-derivative LDA was 97.6%, confirming that the ~4-point cross-cube gap—not a near-perfect within-cube score—is the honest measure of difficulty. The deep spectral–spatial reference is far more fragile across cubes (θ≈45.9% in prior repository analyses), consistent with reliance on cube-specific spatial shortcuts.

### 3.2 Unlabelled target-acquisition normalization reduces error and rescues origins

Standardizing the incoming cube on its own unlabelled statistics raised balanced accuracy from 93.6% to 95.1% (+1.45 percentage points; 95% cluster-bootstrap interval [+0.75, +2.11]; 99.95% of bootstrap replicates positive), a ~22% relative reduction in balanced error, versus ~+9.3 points over the SNV-LR baseline (Figure 1B). The gain is concentrated in the origins that collapse under a cube change (Figure 3A): HNA (0.79→0.82), SXD (0.85→0.91), and NX (0.93→0.95). Full-covariance CORAL/PCA whitening variants overfit and were not retained.

### 3.3 Calibration is repaired under acquisition shift

A cube change made the classifier overconfident: cross-cube NLL rose to 0.236 and ECE to 0.038, versus 0.104 and 0.013 same-domain. A single temperature fit only on the training cube restored calibration to near–same-domain quality—ECE 0.010 and NLL 0.179 (Figure 2), a reduction whose 95% cluster-bootstrap intervals exclude zero in both directions.

### 3.4 Mechanistic and negative results

Within-seed pixel-covariance descriptors were predictive in isolation but harmful on transfer: concatenating them reduced balanced accuracy from 0.936 to 0.920 and increased NLL from 0.239 to 0.606 (Figure 3B), indicating that within-seed texture mainly encodes acquisition-specific information. Constructed-batch-grouped temperature was statistically indistinguishable from i.i.d. temperature (ECE difference −0.0007, interval spanning zero), and group-conformal did not improve on i.i.d. split-conformal, both under-covering (~0.80 versus a 0.90 target) under shift. These negatives constrain the contribution and are reported rather than omitted.

### 3.5 One-shot locked confirmation

On reserved batches 8–9 of the opposite cube, the frozen pipeline achieved {{LOCKED_PRIMARY_THETA}} balanced accuracy ({{LOCKED_PRIMARY_ECE}} ECE, {{LOCKED_PRIMARY_NLL}} NLL) versus {{LOCKED_SNV_LR_THETA}} for SNV-LR. The target-normalization gain was {{LOCKED_ADAPTATION_GAIN}} and the calibration ECE reduction was {{LOCKED_CALIBRATION_REDUCTION}}. The four pre-registered directional gates were {{LOCKED_GATE_DECISION}}. Full metrics, per-origin recall, and confusion matrices are in `provenance_study/outputs/acquisition_locked_confirmation/`.

## 4. Discussion

**Protocol.** Making whole-cube transfer the primary axis exposes what within-cube random splits hide. The ~4-point same-domain-to-cross-cube gap, and the deep model's much larger collapse, show that apparent near-perfect provenance accuracy is substantially an acquisition-leakage artifact. The constructed-batch design gives reproducible grouped development and a genuine one-shot reserved-data confirmation without manufacturing physical independence.

**Empirical.** The best pipeline is deliberately simple. A first-derivative representation suppresses baseline/scatter acquisition effects; shrinkage LDA is stable under high spectral collinearity and small samples; unlabelled target-acquisition normalization removes a residual per-cube affine batch effect using only incoming features; and temperature scaling restores trustworthy probabilities. Each component is standard, but their combination and the honest transfer evaluation constitute the contribution.

**Mechanism and negatives.** That pixel-covariance and texture features help within a cube but hurt across cubes is direct evidence that within-seed spatial structure is largely acquisition-specific here—an argument against spatial-shortcut explanations of origin classification. The failure of grouped calibration/conformal to beat i.i.d. versions shows that, within a single acquisition cube, constructed batches are not sufficiently distinct domains; we report this rather than dress standard temperature scaling as a novel method.

**Limitations.** The archive has two source images per origin; whole-cube transfer is an acquisition-robustness stress test, not external validation, and cannot estimate between-farm, between-year, or between-instrument variation. Target normalization assumes the incoming acquisition arrives as a batch. The task is closed-set with no reject option for unknown origins. External geographical certification requires a traceable multi-farm/year/instrument cohort with paired chemistry.

## 5. Conclusions

Under a leakage-controlled whole-cube transfer protocol, a compact pipeline—first-derivative shrinkage LDA with unlabelled target-acquisition normalization and post-hoc calibration—raised cross-acquisition balanced accuracy from 85.7% (SNV-LR) to 95.1% and restored near–same-domain calibration, while within-seed covariance features and elaborated calibration/conformal variants were retained as negative results. The one-shot locked confirmation was {{LOCKED_GATE_DECISION}}. The work provides an honest acquisition-robustness framework and a deployable pipeline for the current archive; it does not establish external geographical authentication, which requires prospectively collected, independently traceable lots across farms, years, and instruments.

## Data and code availability

Development code: `provenance_study/explore_acquisition_calibration.py`; frozen predictor: `provenance_study/acquisition_robust_pipeline.py`; figures: `provenance_study/make_acquisition_figures.py`; pre-registration: `docs/采集域稳健产地溯源方法与锁定验证方案.md`; one-shot locked entry: `provenance_study/run_acquisition_locked_confirmation.py`. Canonical development artifacts are in `provenance_study/outputs/development_acquisition_calibration/`. Locked-confirmation artifacts must be cited only after the authorized single run. {{PUBLIC_DATA_AND_CODE_DOI}}

## References

1. Cao JX, et al. Hypnotic effect of jujubosides from Semen Ziziphi Spinosae. *J Ethnopharmacol*. 2010;130:163–166.
2. Sun YF, et al. Comprehensive evaluation of natural antioxidants in *Ziziphus jujuba* var. *spinosa* by geographical origin. *Food Chem*. 2011;124:1612–1619.
3. Kong Y, et al. Chemical composition and transcriptomics of wild and grafted Semen Ziziphi Spinosae. *BMC Genomics*. 2024;25:978.
4. Qin J, et al. Hyperspectral and multispectral imaging for food safety and quality. *J Food Eng*. 2013;118:157–171.
5. Roberts DR, et al. Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*. 2017;40:913–929.
6. Zhao X, et al. Identification of geographical origin of Semen Ziziphi Spinosae based on hyperspectral imaging combined with CNNs. *Infrared Phys Technol*. 2024;136:104982.
7. Zheng Z, et al. Effective band selection of hyperspectral image by an attention mechanism-based convolutional network. *RSC Adv*. 2022;12:8750–8759.
8. Zhang JB, et al. E-eye, flash GC E-nose and HS-GC-MS with chemometrics to identify adulterants and geographical origins of Ziziphi Spinosae Semen. *Food Chem*. 2023;424:136270.
9. Barnes RJ, Dhanoa MS, Lister SJ. Standard normal variate transformation and de-trending of NIR diffuse reflectance spectra. *Appl Spectrosc*. 1989;43:772–777.
10. Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. *ICML*. 2017;70:1321–1330.
