# Exploration probes (research byproducts)

These four scripts are the exploratory diagnostics that steered the method choice for the
acquisition-robust provenance study. They are **byproducts**, retained for the record; they are
throwaway diagnostics with hard-coded absolute paths and are **superseded** by the formal,
leakage-safe, tested pipeline in
[`../explore_acquisition_calibration.py`](../explore_acquisition_calibration.py) and
[`../acquisition_robust_pipeline.py`](../acquisition_robust_pipeline.py). They are development-only
(constructed batches 0–7) and read no reserved data.

- `probe_crossacq.py` — quantified the cross-acquisition headroom, the low-rank between-cube shift
  structure, and a first pixel-population test.
- `probe2.py` — measured the calibration collapse under cube shift, per-origin recall, and the
  within-seed covariance descriptor (found harmful).
- `probe3.py` — showed the cross-cube calibration collapse is recoverable by a training-only
  temperature.
- `probe4.py` — compared unsupervised incoming-cube adaptation methods; `target_std` won.

To reproduce, adjust the hard-coded `ROOT` path and run with the project `.venv` interpreter.
