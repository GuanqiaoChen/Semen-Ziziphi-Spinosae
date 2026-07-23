#!/usr/bin/env python3
"""One-shot locked confirmation of the acquisition-robust provenance pipeline.

This entry point is FAIL-CLOSED.  It reads the reserved constructed batches 8-9
only after the exact confirmation phrase is supplied, and it refuses to overwrite
a prior completed run.  It executes the pre-registered design in
``docs/采集域稳健产地溯源方法与锁定验证方案.md`` §5:

    direction A: train cube *-1 batches 0-7  -> test cube *-2 batches 8-9
    direction B: train cube *-2 batches 0-7  -> test cube *-1 batches 8-9

The test seeds are reserved (batches 8-9) AND from the opposite acquisition cube,
so they were never used in development selection.  The frozen pipeline
(``AcquisitionRobustProvenanceClassifier``) and the four pre-registered
directional gates are applied exactly once.

Run (only when authorized):
    .venv\\Scripts\\python.exe -m provenance_study.run_acquisition_locked_confirmation \\
        --confirm-locked-test UNLOCK_BATCHES_8_9
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from provenance_study.core import (  # noqa: E402
    StandardNormalVariate,
    discover_manifest,
    load_csv_split,
    multiclass_metrics,
)
from provenance_study.acquisition_robust_pipeline import (  # noqa: E402
    AcquisitionRobustProvenanceClassifier,
    _self_standardize,
    _sg1,
    fit_temperature,
    softmax_temperature,
)

CONFIRMATION_PHRASE = "UNLOCK_BATCHES_8_9"
COMPLETE_STATE = "executed_complete"
BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_SEED = 20260722


class LockedConfirmationError(RuntimeError):
    pass


def require_confirmation(value: str | None) -> None:
    if value != CONFIRMATION_PHRASE:
        raise LockedConfirmationError(
            "Reserved batches 8-9 remain closed. Supply exactly "
            f"--confirm-locked-test {CONFIRMATION_PHRASE}."
        )


def _git_state(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo_root, check=True, capture_output=True, text=True, timeout=15
        ).stdout.strip()

    try:
        return {
            "available": True,
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "porcelain": run("status", "--porcelain=v1", "--untracked-files=all").splitlines(),
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def _snv_lr_baseline():
    return Pipeline(
        [
            ("snv", StandardNormalVariate()),
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, tol=1e-4)),
        ]
    )


def _lda_source_vs_target(X_tr, y_tr, X_te, target_norm: bool):
    """SG1 + shrinkage LDA with source- or target-cube standardization."""

    F_tr, mean, scale = _self_standardize(_sg1(X_tr))
    if target_norm:
        F_te, _, _ = _self_standardize(_sg1(X_te))
    else:
        F_te = (_sg1(X_te) - mean) / scale
    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(F_tr, y_tr)
    return model.predict_proba(F_te)


def _cluster_bootstrap(y, proba_a, proba_b, clusters, metric, higher_is_better=True):
    """Bootstrap metric(a) - metric(b) over clusters (a = treatment)."""

    def value(idx, proba):
        return float(multiclass_metrics(y[idx], proba[idx])[metric])

    uniq = np.unique(clusters)
    members = {c: np.where(clusters == c)[0] for c in uniq}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    point = value(np.arange(y.size), proba_a) - value(np.arange(y.size), proba_b)
    draws = np.empty(BOOTSTRAP_REPETITIONS)
    for r in range(BOOTSTRAP_REPETITIONS):
        chosen = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([members[c] for c in chosen])
        draws[r] = value(idx, proba_a) - value(idx, proba_b)
    return {
        "point": point,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "fraction_positive": float((draws > 0).mean()),
    }


def run_locked_confirmation(args: argparse.Namespace) -> Path:
    require_confirmation(args.confirm_locked_test)  # first, before any I/O
    output_dir = Path(args.output_dir)
    status_path = output_dir / "run_status.json"
    if status_path.exists():
        state = json.loads(status_path.read_text(encoding="utf-8")).get("state")
        if state == COMPLETE_STATE:
            raise LockedConfirmationError(
                f"A completed confirmation already exists at {status_path}; refusing to overwrite."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    repo_root = Path(args.repo_root).resolve()
    status_path.write_text(json.dumps({"state": "executing_running"}, indent=2), encoding="utf-8")

    # Authorized read: hash the full manifest and load both splits.
    manifest = discover_manifest(Path(args.data_root), hash_files=True)
    dev = load_csv_split(manifest, split="development", verify_hashes=True)
    locked = load_csv_split(manifest, split="locked", verify_hashes=True)

    def arrays(dataset):
        rep = np.array([r.replicate for r in dataset.records], dtype=np.int64)
        batch = np.array([r.constructed_batch for r in dataset.records], dtype=np.int64)
        return dataset.X, dataset.y, rep, batch, dataset.records

    Xd, yd, repd, batchd, _ = arrays(dev)
    Xl, yl, repl, batchl, recl = arrays(locked)

    directions = (("cube1_to_cube2", 1, 2), ("cube2_to_cube1", 2, 1))
    frozen_probas, snv_probas = [], []
    src_probas, tgt_probas, uncal_probas = [], [], []
    test_y, test_cluster, prediction_rows = [], [], []
    per_direction: list[dict[str, Any]] = []

    for name, train_cube, test_cube in directions:
        tr = repd == train_cube  # training cube, development batches 0-7
        te = repl == test_cube   # opposite cube, reserved batches 8-9
        Xtr, ytr, btr = Xd[tr], yd[tr], batchd[tr]
        Xte, yte = Xl[te], yl[te]

        frozen = AcquisitionRobustProvenanceClassifier().fit(Xtr, ytr, btr)
        p_frozen = frozen.predict_proba(Xte)
        p_snv = _snv_lr_baseline().fit(Xtr, ytr).predict_proba(Xte)
        p_src = _lda_source_vs_target(Xtr, ytr, Xte, target_norm=False)
        p_tgt = _lda_source_vs_target(Xtr, ytr, Xte, target_norm=True)
        p_uncal = p_tgt  # target-normalized, temperature = 1 (calibration ablation)

        frozen_probas.append(p_frozen); snv_probas.append(p_snv)
        src_probas.append(p_src); tgt_probas.append(p_tgt); uncal_probas.append(p_uncal)
        test_y.append(yte)
        test_cluster.append(yte.astype(np.int64) * 100 + batchl[te])

        m = multiclass_metrics(yte, p_frozen)
        per_direction.append({"direction": name, "temperature": frozen.temperature_, **m})
        for rec, prob, truth in zip(np.array(recl)[te], p_frozen, yte):
            prediction_rows.append(
                {
                    "direction": name,
                    "sample_id": rec.sample_id,
                    "true_label": int(truth),
                    "predicted_label": int(prob.argmax()),
                    **{f"proba_{k}": float(prob[k]) for k in range(prob.shape[0])},
                }
            )

    y = np.concatenate(test_y)
    cluster = np.concatenate(test_cluster)
    pooled = {
        "frozen_pipeline": np.vstack(frozen_probas),
        "snv_lr_baseline": np.vstack(snv_probas),
        "sg1_source_std": np.vstack(src_probas),
        "sg1_target_std_calibrated": np.vstack(tgt_probas),
    }
    pooled_metrics = {name: multiclass_metrics(y, p) for name, p in pooled.items()}

    # Pre-registered directional gates.
    gates = {
        "gate1_frozen_beats_snv_lr": _cluster_bootstrap(
            y, pooled["frozen_pipeline"], pooled["snv_lr_baseline"], cluster, "balanced_accuracy"
        ),
        "gate2_target_normalization_gain": _cluster_bootstrap(
            y, np.vstack(tgt_probas), np.vstack(src_probas), cluster, "balanced_accuracy"
        ),
        "gate3_calibration_reduces_ece": _cluster_bootstrap(
            y, np.vstack(uncal_probas), pooled["frozen_pipeline"], cluster,
            "expected_calibration_error",
        ),
    }
    gate_decisions = {
        "gate1_frozen_beats_snv_lr": gates["gate1_frozen_beats_snv_lr"]["ci_low"] > 0,
        "gate2_target_normalization_gain": gates["gate2_target_normalization_gain"]["point"] >= 0,
        "gate3_calibration_reduces_ece": gates["gate3_calibration_reduces_ece"]["point"] > 0,
    }

    results = {
        "design": "whole_cube_transfer_to_reserved_batches_8_9",
        "directions": per_direction,
        "pooled_metrics": pooled_metrics,
        "pre_registered_gates": gates,
        "gate_decisions": gate_decisions,
        "all_gates_passed": bool(all(gate_decisions.values())),
        "n_locked_test_seeds": int(y.size),
        "data_fingerprint_sha256": manifest.data_fingerprint_sha256,
        "manifest_sha256": manifest.manifest_sha256,
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_csv(output_dir / "metrics.csv", [
        {"model": k, **v} for k, v in pooled_metrics.items()
    ])
    _write_csv(output_dir / "predictions.csv", prediction_rows)

    run_status = {
        "state": COMPLETE_STATE,
        "confirmation_phrase_sha256": hashlib.sha256(CONFIRMATION_PHRASE.encode()).hexdigest(),
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_seconds": time.perf_counter() - started,
        "git": _git_state(repo_root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "data_fingerprint_sha256": manifest.data_fingerprint_sha256,
    }
    status_path.write_text(json.dumps(run_status, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_dir


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    import csv

    if not rows:
        raise ValueError(f"Refusing to write empty table: {path.name}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_argument_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--data-root", type=Path, default=root / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "provenance_study" / "outputs" / "acquisition_locked_confirmation",
    )
    parser.add_argument("--confirm-locked-test", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    output_dir = run_locked_confirmation(args)
    print(f"Locked confirmation complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
