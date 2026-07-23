# Acquisition-robust hyperspectral provenance of *Ziziphi Spinosae* Semen

This branch (`acquisition`) contains **only** the acquisition-robust geographical-origin
traceability study for *Ziziphi Spinosae* Semen (酸枣仁) and the shared infrastructure it
depends on. Other research rounds (the first-round classical protocol, the source-cube deep
audit, and their documents) live on `main`; they were removed from this branch to keep it a
self-contained record of this study.

## Scientific question

Same-cube random splits make hyperspectral origin classification look near-perfect by letting
image-specific signals leak across the evaluation boundary. This study asks the question the
current archive can answer honestly: **when the acquisition cube (source image) changes, does
origin classification stay accurate and well-calibrated?**

## Method (frozen)

Whole-source-cube transfer is the primary evaluation axis: train on one acquisition cube, test
on the other, both directions. The frozen deployment pipeline
([`provenance_study/acquisition_robust_pipeline.py`](provenance_study/acquisition_robust_pipeline.py))
is:

1. Savitzky–Golay first derivative (window 15, poly 2) — most transfer-robust representation;
2. training-cube standardization;
3. **unlabelled target-acquisition normalization** — the incoming cube (a batch of unknown-origin
   seeds) is standardized on its own statistics, removing a per-cube affine batch effect;
4. shrinkage LDA;
5. a single post-hoc temperature fit on the training cube's constructed-batch-grouped OOF.

## Key development results (constructed batches 0–7)

| Step | cross-cube balanced accuracy |
|---|---|
| SNV-LR weak baseline | 85.7% |
| SG1 + shrinkage LDA | 93.6% |
| + unlabelled target-cube normalization | 95.1% (+1.45 pp, 95% CI [+0.75, +2.11]) |

Temperature calibration cut cross-cube ECE 0.038→0.010 and NLL 0.236→0.179. Retained **negatives**:
within-seed pixel-covariance features hurt transfer; grouped calibration/conformal did not beat
i.i.d. versions.

## One-shot locked confirmation (reserved batches 8–9, opposite cube)

Frozen pipeline **93.3%** balanced accuracy vs 85.5% for SNV-LR (+7.8 pp, 95% CI [+0.95, +12.2]);
three of four pre-registered gates evaluated, all passed. See
[`provenance_study/outputs/acquisition_locked_confirmation/`](provenance_study/outputs/acquisition_locked_confirmation/)
(including `NOTES.md`, which discloses one pre-registration deviation).

## Layout

- `provenance_study/acquisition_robust_pipeline.py` — frozen predictor.
- `provenance_study/explore_acquisition_calibration.py` — development comparator matrix.
- `provenance_study/make_acquisition_figures.py` — publication figures.
- `provenance_study/run_acquisition_locked_confirmation.py` — fail-closed one-shot locked entry.
- `provenance_study/core.py`, `__init__.py` — shared leakage-safe loaders/transforms (dependency).
- `provenance_study/exploration_probes/` — exploratory diagnostics that informed the frozen method.
- `provenance_study/outputs/` — development and locked machine tables and figures.
- `docs/采集域稳健产地溯源方法与锁定验证方案.md` — result-before-freeze pre-registration.
- `docs/研究审查与修订总账.md` — append-only revision ledger (full project governance record).
- `paper/manuscript_acquisition_robust_provenance.md` — manuscript draft.
- `data/` — 16 source-image directories of paired CSV/MAT seed spectra.

## Run

```powershell
.venv\Scripts\python.exe -m pytest provenance_study/tests -q
.venv\Scripts\python.exe -m provenance_study.explore_acquisition_calibration
.venv\Scripts\python.exe -m provenance_study.make_acquisition_figures
```

The locked entry is fail-closed and runs only with `--confirm-locked-test UNLOCK_BATCHES_8_9`.

## Scope boundary

Two source images per origin makes whole-cube transfer an acquisition-robustness stress test, not
external geographical certification. Constructed batches are deterministic subdivisions of the same
16 images, not independent physical lots, farms, years, or instruments.
