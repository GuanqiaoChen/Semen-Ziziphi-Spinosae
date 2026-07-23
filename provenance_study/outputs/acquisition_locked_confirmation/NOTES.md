# Locked confirmation — execution notes and disclosures

Canonical one-shot run of `provenance_study/run_acquisition_locked_confirmation.py`.

- **State:** `executed_complete` (see `run_status.json`); executed once from committed
  code `b35b6b6` on branch `acquisition`.
- **Design:** whole-cube transfer to reserved batches 8–9 of the *opposite* cube
  (direction A: train cube-*-1 batches 0–7 → test cube-*-2 batches 8–9; direction B
  reversed). 252 reserved test seeds.
- **Headline:** frozen pipeline balanced accuracy 0.9332 vs SNV-LR 0.8548; ECE 0.047,
  NLL 0.232.

## Disclosure 1 — pre-registration deviation (three of four gates evaluated)

The pre-registration (`docs/采集域稳健产地溯源方法与锁定验证方案.md` §5.1) lists four
directional gates. The one-shot locked run evaluated **three**:

- gate 1 (frozen pipeline beats SNV-LR): **passed**, +0.0784, 95% CI [+0.0095, +0.1223].
- gate 2 (target-normalization gain ≥ 0): **passed**, +0.0191, 95% CI [+0.0000, +0.0376].
- gate 3 (calibration reduces ECE): **passed** (point +0.0193; 95% CI [−0.0062, +0.0378],
  so the reduction is directional but not itself CI-significant on n=252).

The fourth gate — within-seed covariance features must not help — was **not** re-run on
the reserved data. It would require streaming reserved MAT pixel patches; the
development negative (n=1,012: balanced accuracy 0.936→0.920, NLL 0.239→0.606, in
`../development_acquisition_calibration/negative_ablation.csv`) stands as the evidence.
This is a transparent deviation and does not affect gates 1–3.

## Disclosure 2 — a metrics key is mislabelled

In `results.json`/`metrics.csv`, the pooled key `sg1_target_std_calibrated` actually
holds the target-standardized probabilities **without** temperature scaling (T=1); it is
the calibration ablation used by gate 3. The temperature-**calibrated** frozen pipeline is
the separate `frozen_pipeline` key. Both share identical predictions (temperature does not
change arg-max), so balanced accuracy is equal (0.9332); they differ only in ECE/NLL
(uncalibrated 0.066 vs calibrated 0.047). The gate-3 computation used the correct arrays;
only the display key name is misleading. The one-shot artifact was not rewritten to fix the
label, to preserve the single-execution record.

## Worktree state at execution

`run_status.json` records 50 dirty worktree entries; these are a parallel exploration
round's untracked files and a modified `requirements-lock.txt`, none authored by this run
and none affecting the frozen code (committed at `b35b6b6`) or the data fingerprint.
