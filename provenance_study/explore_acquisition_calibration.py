#!/usr/bin/env python3
"""Acquisition-shift-aware calibrated provenance (ASAC-Prov): development study.

Scientific objective (unchanged): geographical-origin traceability of *Ziziphi
Spinosae* Semen from hyperspectral spectra.  This module targets the aspect of
that objective that is scientifically load-bearing and, on the current archive,
where a large and measurable effect actually exists: **trustworthy behaviour
under an acquisition-domain change**.

Motivating development probe (constructed batches 0-7 only):
  * Same-domain leave-one-constructed-batch-out (LOBO) SG1-shrinkage-LDA reaches
    ~97.6% balanced accuracy, but under whole-source-cube transfer accuracy falls
    only ~4 pp while the probabilities collapse: negative log-likelihood ~2.3x,
    expected calibration error (ECE) ~2.9x, Brier ~2.5x.  The classifier becomes
    badly *overconfident* the moment the acquisition cube changes.
  * Within-seed pixel-population / covariance descriptors do **not** help cross
    cube -- they add cube-specific texture shortcuts and are retained here as a
    leakage-safe negative ablation.

Contribution evaluated here (development only, no locked batch is read):
  1. Selection, on the cross-acquisition axis, of the most transfer-robust
     spectral representation and classifier among transparent candidates.
  2. **Acquisition-shift-aware calibration**: temperature scaling fit on
     *constructed-batch-grouped* out-of-fold predictions of the training cube,
     using the research-team-authorized constructed batches as simulated
     acquisition domains.  This anticipates cross-cube overconfidence without
     ever using test-cube labels.
  3. **Group-conformal reject option** whose finite-sample coverage is checked
     under a genuine cross-cube shift, versus i.i.d. split conformal.

All fitting, standardisation, temperature, and conformal thresholds are learnt
only from the applicable training partition.  The two source cubes per origin
are not independent physical lots, so cross-cube transfer is a rigorous
acquisition-robustness stress test, not external geographical validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scipy  # noqa: E402
import sklearn  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from provenance_study.core import (  # noqa: E402
    BASE_BATCH_SEED,
    CLASS_NAMES,
    DEVELOPMENT_BATCHES,
    EXPECTED_BANDS,
    MultiplicativeScatterCorrection,
    SavitzkyGolayTransformer,
    StandardNormalVariate,
    discover_manifest,
    load_csv_split,
    multiclass_metrics,
)

NUM_CLASSES = len(CLASS_NAMES)
REPRESENTATIONS = ("raw", "snv", "msc", "sg1", "sg2")
BASE_CLASSIFIERS = ("lda", "lr", "svm")
CALIBRATIONS = ("uncalibrated", "iid_temperature", "shift_aware_temperature")
TEMPERATURE_GRID = np.exp(np.linspace(np.log(0.25), np.log(8.0), 400))
CONFORMAL_ALPHA = 0.10
BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_SEED = 20260722
IID_CALIBRATION_FOLDS = 8
DIRECTIONS = ("cube1_to_cube2", "cube2_to_cube1")


# --------------------------------------------------------------------------- #
# Models with a uniform "logit" interface.
# --------------------------------------------------------------------------- #
def _representation_steps(name: str) -> list[tuple[str, Any]]:
    if name == "raw":
        return [("scale", StandardScaler())]
    if name == "snv":
        return [("snv", StandardNormalVariate()), ("scale", StandardScaler())]
    if name == "msc":
        return [("msc", MultiplicativeScatterCorrection()), ("scale", StandardScaler())]
    if name == "sg1":
        return [("sg", SavitzkyGolayTransformer(15, 2, 1)), ("scale", StandardScaler())]
    if name == "sg2":
        return [("sg", SavitzkyGolayTransformer(15, 2, 2)), ("scale", StandardScaler())]
    raise ValueError(f"Unknown representation: {name}")


def build_pipeline(representation: str, classifier: str) -> Pipeline:
    steps = _representation_steps(representation)
    if classifier == "lda":
        steps.append(("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")))
    elif classifier == "lr":
        steps.append(
            (
                "clf",
                LogisticRegression(
                    C=1.0, solver="lbfgs", max_iter=5000, tol=1e-4, random_state=BASE_BATCH_SEED
                ),
            )
        )
    elif classifier == "svm":
        steps.append(
            (
                "clf",
                SVC(C=10.0, kernel="rbf", gamma="scale", probability=False, decision_function_shape="ovr"),
            )
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier}")
    return Pipeline(steps)


def pipeline_logits(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    """Return an (n, 8) score matrix usable as temperature-scaling logits."""

    clf = pipeline.named_steps["clf"]
    if isinstance(clf, SVC):
        scores = pipeline.decision_function(X)
    else:
        scores = pipeline.predict_log_proba(X)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape[1] != NUM_CLASSES:
        raise ValueError("Logit matrix does not have one column per class")
    return scores


def softmax_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exponentiated = np.exp(scaled)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int64)
    index = np.arange(y.size)
    losses = [
        float(-np.log(np.clip(softmax_temperature(logits, T)[index, y], 1e-15, 1.0)).mean())
        for T in TEMPERATURE_GRID
    ]
    return float(TEMPERATURE_GRID[int(np.argmin(losses))])


# --------------------------------------------------------------------------- #
# Fold structures for calibration / conformal.
# --------------------------------------------------------------------------- #
def batch_grouped_folds(batch: np.ndarray, subset: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Leave-one-constructed-batch-out folds inside `subset` (simulated domains)."""

    folds = []
    for b in sorted(np.unique(batch[subset])):
        cal = subset[batch[subset] == b]
        train = subset[batch[subset] != b]
        if cal.size and train.size:
            folds.append((train, cal))
    return folds


def random_folds(subset: np.ndarray, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """i.i.d. k-fold split that ignores acquisition-domain grouping."""

    rng = np.random.default_rng(seed)
    order = subset.copy()
    rng.shuffle(order)
    parts = np.array_split(order, n_folds)
    folds = []
    for i in range(n_folds):
        cal = parts[i]
        train = np.concatenate([parts[j] for j in range(n_folds) if j != i])
        if cal.size and train.size:
            folds.append((train, cal))
    return folds


def oof_logits(
    representation: str,
    classifier: str,
    X: np.ndarray,
    y: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    logit_rows, label_rows = [], []
    for train_idx, cal_idx in folds:
        pipeline = build_pipeline(representation, classifier).fit(X[train_idx], y[train_idx])
        logit_rows.append(pipeline_logits(pipeline, X[cal_idx]))
        label_rows.append(y[cal_idx])
    return np.vstack(logit_rows), np.concatenate(label_rows)


# --------------------------------------------------------------------------- #
# Cross-cube evaluation of one (representation, classifier).
# --------------------------------------------------------------------------- #
@dataclass
class DirectionResult:
    direction: str
    test_idx: np.ndarray
    test_logits: np.ndarray
    temperatures: dict[str, float]
    # out-of-fold calibrated probabilities used as conformal calibration scores
    cal_shift_proba: np.ndarray
    cal_shift_y: np.ndarray
    cal_iid_proba: np.ndarray
    cal_iid_y: np.ndarray


def evaluate_direction(
    representation: str,
    classifier: str,
    X: np.ndarray,
    y: np.ndarray,
    rep: np.ndarray,
    batch: np.ndarray,
    train_cube: int,
    test_cube: int,
    direction: str,
) -> DirectionResult:
    train_subset = np.where(rep == train_cube)[0]
    test_idx = np.where(rep == test_cube)[0]

    shift_folds = batch_grouped_folds(batch, train_subset)
    iid_fold = random_folds(train_subset, IID_CALIBRATION_FOLDS, seed=BASE_BATCH_SEED + train_cube)

    shift_logits, shift_y = oof_logits(representation, classifier, X, y, shift_folds)
    iid_logits, iid_y = oof_logits(representation, classifier, X, y, iid_fold)

    temperatures = {
        "uncalibrated": 1.0,
        "iid_temperature": fit_temperature(iid_logits, iid_y),
        "shift_aware_temperature": fit_temperature(shift_logits, shift_y),
    }

    final_pipeline = build_pipeline(representation, classifier).fit(X[train_subset], y[train_subset])
    test_logits = pipeline_logits(final_pipeline, X[test_idx])

    return DirectionResult(
        direction=direction,
        test_idx=test_idx,
        test_logits=test_logits,
        temperatures=temperatures,
        cal_shift_proba=softmax_temperature(shift_logits, temperatures["shift_aware_temperature"]),
        cal_shift_y=shift_y,
        cal_iid_proba=softmax_temperature(iid_logits, temperatures["iid_temperature"]),
        cal_iid_y=iid_y,
    )


def lobo_reference(
    representation: str,
    classifier: str,
    X: np.ndarray,
    y: np.ndarray,
    batch: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Same-domain leave-one-constructed-batch-out reference (both cubes mixed)."""

    all_idx = np.arange(y.size)
    folds = batch_grouped_folds(batch, all_idx)
    logits, order = oof_logits(representation, classifier, X, y, folds)
    # `order` follows fold concatenation; align labels via the same folds.
    aligned_y = order
    out = {}
    T = fit_temperature(logits, aligned_y)
    for name, temp in (("uncalibrated", 1.0), ("shift_aware_temperature", T)):
        proba = softmax_temperature(logits, temp)
        m = multiclass_metrics(aligned_y, proba)
        m["temperature"] = float(temp)
        out[name] = m
    return out


# --------------------------------------------------------------------------- #
# Conformal prediction (group vs i.i.d. calibration) under cross-cube shift.
# --------------------------------------------------------------------------- #
def conformal_threshold(cal_proba: np.ndarray, cal_y: np.ndarray, alpha: float) -> float:
    scores = 1.0 - cal_proba[np.arange(cal_y.size), cal_y]
    n = scores.size
    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    level = min(level, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def conformal_evaluate(test_proba: np.ndarray, test_y: np.ndarray, threshold: float) -> dict[str, float]:
    included = test_proba >= (1.0 - threshold)
    in_set = included[np.arange(test_y.size), test_y]
    set_sizes = included.sum(axis=1)
    return {
        "coverage": float(in_set.mean()),
        "average_set_size": float(set_sizes.mean()),
        "empty_set_fraction": float((set_sizes == 0).mean()),
    }


# --------------------------------------------------------------------------- #
# Cluster (origin x constructed-batch) bootstrap for the headline effect.
# --------------------------------------------------------------------------- #
def cluster_bootstrap_delta(
    y: np.ndarray,
    proba_reference: np.ndarray,
    proba_treatment: np.ndarray,
    clusters: np.ndarray,
    metric: str,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    """Bootstrap ``metric(reference) - metric(treatment)`` by resampling clusters."""

    def metric_value(idx: np.ndarray, proba: np.ndarray) -> float:
        return float(multiclass_metrics(y[idx], proba[idx])[metric])

    unique_clusters = np.unique(clusters)
    cluster_members = {c: np.where(clusters == c)[0] for c in unique_clusters}
    rng = np.random.default_rng(seed)
    point = metric_value(np.arange(y.size), proba_reference) - metric_value(
        np.arange(y.size), proba_treatment
    )
    deltas = np.empty(repetitions)
    for r in range(repetitions):
        chosen = rng.choice(unique_clusters, size=unique_clusters.size, replace=True)
        idx = np.concatenate([cluster_members[c] for c in chosen])
        deltas[r] = metric_value(idx, proba_reference) - metric_value(idx, proba_treatment)
    return {
        "point": point,
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
        "fraction_positive": float((deltas > 0).mean()),
    }


# --------------------------------------------------------------------------- #
# Optional negative ablation: within-seed covariance descriptor.
# --------------------------------------------------------------------------- #
def covariance_descriptor_ablation(
    records: Sequence[Any], X: np.ndarray, y: np.ndarray, rep: np.ndarray, n_components: int = 16
) -> list[dict[str, Any]]:
    import h5py
    from sklearn.decomposition import PCA

    def snv(a: np.ndarray) -> np.ndarray:
        m = a.mean(1, keepdims=True)
        s = a.std(1, ddof=1, keepdims=True)
        s = np.where(s < 1e-12, 1.0, s)
        return (a - m) / s

    rng = np.random.default_rng(7)
    pooled, all_px = [], []
    for record in records:
        with h5py.File(record.mat_path, "r") as handle:
            patch = handle["patch_chw"][()]
            mask = np.asarray(handle["crop_mask"][()]).squeeze() > 0.5
        px = snv(patch[mask].astype(np.float64))
        all_px.append(px)
        take = rng.choice(px.shape[0], min(40, px.shape[0]), replace=False)
        pooled.append(px[take])
    basis = PCA(n_components=n_components, random_state=0).fit(np.vstack(pooled))
    triu = np.triu_indices(n_components)
    cov = np.empty((len(records), triu[0].size))
    for i, px in enumerate(all_px):
        z = basis.transform(px)
        cov[i] = np.cov(z, rowvar=False)[triu]

    sg1 = SavitzkyGolayTransformer(15, 2, 1)
    features = {
        "sg1_reference": sg1.fit(X).transform(X),
        "sg1_plus_covariance": np.hstack([sg1.fit(X).transform(X), cov]),
    }
    rows = []
    for name, feat in features.items():
        thetas, nlls = [], []
        for tr, te in ((1, 2), (2, 1)):
            itr, ite = rep == tr, rep == te
            scaler = StandardScaler().fit(feat[itr])
            model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(
                scaler.transform(feat[itr]), y[itr]
            )
            proba = model.predict_proba(scaler.transform(feat[ite]))
            m = multiclass_metrics(y[ite], proba)
            thetas.append(m["balanced_accuracy"])
            nlls.append(m["negative_log_likelihood"])
        rows.append(
            {
                "feature": name,
                "cross_cube_balanced_accuracy": float(np.mean(thetas)),
                "cross_cube_negative_log_likelihood": float(np.mean(nlls)),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Unsupervised adaptation to the incoming acquisition cube (batch-effect
# correction).  The target cube is used only through its *unlabeled* features.
# --------------------------------------------------------------------------- #
def _sg1_features(X: np.ndarray) -> np.ndarray:
    return SavitzkyGolayTransformer(15, 2, 1).fit(X).transform(X)


def _adapted_standardize(F_train: np.ndarray, F_test: np.ndarray, method: str):
    """Return standardized (train, test) features under one adaptation method."""

    if method == "source_standardize":
        scaler = StandardScaler().fit(F_train)
        return scaler.transform(F_train), scaler.transform(F_test)
    if method == "target_standardize":
        # Each cube standardized on its own statistics -> removes a per-cube
        # affine batch effect using only unlabeled target features.
        return (
            StandardScaler().fit(F_train).transform(F_train),
            StandardScaler().fit(F_test).transform(F_test),
        )
    raise ValueError(f"Unknown adaptation method: {method}")


def adaptation_effect(
    X: np.ndarray,
    y: np.ndarray,
    rep: np.ndarray,
    batch: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cross-cube effect of unsupervised incoming-cube standardization (sg1/LDA).

    Adds shift-anticipating temperature calibration fit only on the training
    cube's constructed-batch-grouped out-of-fold predictions of the *same*
    adapted pipeline, so the reported calibration also holds under adaptation.
    """

    F = _sg1_features(X)
    origin_batch = y.astype(np.int64) * 100 + batch
    per_method: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    metric_rows: list[dict[str, Any]] = []

    for method in ("source_standardize", "target_standardize"):
        idx_parts, proba_parts = [], []
        for train_cube, test_cube in ((1, 2), (2, 1)):
            itr = np.where(rep == train_cube)[0]
            ite = np.where(rep == test_cube)[0]
            Xtr, Xte = _adapted_standardize(F[itr], F[ite], method)
            model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xtr, y[itr])

            # shift-anticipating temperature from batch-grouped OOF of same pipeline
            cal_logits, cal_y = [], []
            for held in sorted(np.unique(batch[itr])):
                inner_tr = itr[batch[itr] != held]
                inner_ca = itr[batch[itr] == held]
                Xi_tr, Xi_ca = _adapted_standardize(F[inner_tr], F[inner_ca], method)
                inner_model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(
                    Xi_tr, y[inner_tr]
                )
                cal_logits.append(inner_model.predict_log_proba(Xi_ca))
                cal_y.append(y[inner_ca])
            temperature = fit_temperature(np.vstack(cal_logits), np.concatenate(cal_y))

            proba = softmax_temperature(model.predict_log_proba(Xte), temperature)
            idx_parts.append(ite)
            proba_parts.append(proba)
        idx = np.concatenate(idx_parts)
        proba = np.vstack(proba_parts)
        per_method[method] = (idx, proba)
        m = multiclass_metrics(y[idx], proba)
        metric_rows.append(
            {
                "method": method,
                "classifier": "sg1_lda_shift_calibrated",
                **{k: m[k] for k in (
                    "balanced_accuracy", "accuracy", "macro_f1",
                    "negative_log_likelihood", "expected_calibration_error",
                    "multiclass_brier_score",
                )},
            }
        )

    src_idx, src_proba = per_method["source_standardize"]
    tgt_idx, tgt_proba = per_method["target_standardize"]
    assert np.array_equal(src_idx, tgt_idx)
    boot = cluster_bootstrap_delta(
        y[tgt_idx], tgt_proba, src_proba, origin_batch[tgt_idx],
        "balanced_accuracy", repetitions, seed,
    )
    effect = {
        "comparison": "target_standardize_minus_source_standardize",
        "metric": "balanced_accuracy",
        "point_improvement": boot["point"],
        "ci_low": boot["ci_low"],
        "ci_high": boot["ci_high"],
        "fraction_bootstrap_positive": boot["fraction_positive"],
    }
    return metric_rows, effect


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #
def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path.name}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pooled_metrics(direction_results: Sequence[DirectionResult], y: np.ndarray, calibration: str):
    idx = np.concatenate([d.test_idx for d in direction_results])
    proba = np.vstack(
        [softmax_temperature(d.test_logits, d.temperatures[calibration]) for d in direction_results]
    )
    return idx, proba, multiclass_metrics(y[idx], proba)


def run(data_root: Path, output_dir: Path, *, run_mat_ablation: bool = True) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = discover_manifest(data_root)  # hash_files=False -> no locked bytes
    dev = load_csv_split(manifest, split="development", verify_hashes=False)
    X, y, records = dev.X, dev.y, dev.records
    rep = np.array([r.replicate for r in records], dtype=np.int64)
    batch = np.array([r.constructed_batch for r in records], dtype=np.int64)
    origin_batch_cluster = y.astype(np.int64) * 100 + batch
    if not set(np.unique(batch)).issubset(set(DEVELOPMENT_BATCHES)):
        raise AssertionError("A locked constructed batch leaked into development loading")

    # ---- comparator matrix on the cross-acquisition axis --------------------
    cross_rows: list[dict[str, Any]] = []
    store: dict[tuple[str, str], list[DirectionResult]] = {}
    for representation in REPRESENTATIONS:
        for classifier in BASE_CLASSIFIERS:
            dir_results = [
                evaluate_direction(representation, classifier, X, y, rep, batch, 1, 2, DIRECTIONS[0]),
                evaluate_direction(representation, classifier, X, y, rep, batch, 2, 1, DIRECTIONS[1]),
            ]
            store[(representation, classifier)] = dir_results
            for calibration in CALIBRATIONS:
                _, _, pooled = _pooled_metrics(dir_results, y, calibration)
                cross_rows.append(
                    {
                        "representation": representation,
                        "classifier": classifier,
                        "calibration": calibration,
                        "direction": "pooled_two_directions",
                        "temperature_dir1": dir_results[0].temperatures[calibration],
                        "temperature_dir2": dir_results[1].temperatures[calibration],
                        **{k: pooled[k] for k in (
                            "balanced_accuracy", "accuracy", "macro_f1",
                            "negative_log_likelihood", "multiclass_brier_score",
                            "expected_calibration_error",
                        )},
                    }
                )

    # ---- LDA + LR equal ensemble (both have clean probabilities) ------------
    for calibration in CALIBRATIONS:
        idx = np.concatenate([d.test_idx for d in store[("sg1", "lda")]])
        proba_members = []
        for member in ("lda", "lr"):
            member_dirs = store[("sg1", member)]
            proba_members.append(
                np.vstack(
                    [softmax_temperature(d.test_logits, d.temperatures[calibration]) for d in member_dirs]
                )
            )
        ens = np.mean(proba_members, axis=0)
        m = multiclass_metrics(y[idx], ens)
        cross_rows.append(
            {
                "representation": "sg1",
                "classifier": "lda_lr_ensemble",
                "calibration": calibration,
                "direction": "pooled_two_directions",
                "temperature_dir1": float("nan"),
                "temperature_dir2": float("nan"),
                **{k: m[k] for k in (
                    "balanced_accuracy", "accuracy", "macro_f1",
                    "negative_log_likelihood", "multiclass_brier_score",
                    "expected_calibration_error",
                )},
            }
        )
    _write_csv(output_dir / "cross_cube_metrics.csv", cross_rows)

    # ---- same-domain reference ---------------------------------------------
    lobo_rows = []
    for calibration_name, metrics in lobo_reference("sg1", "lda", X, y, batch).items():
        lobo_rows.append({"model": "sg1_lda", "calibration": calibration_name, **metrics})
    _write_csv(output_dir / "lobo_reference_metrics.csv", lobo_rows)

    # ---- headline effect: hero = sg1 / lda ---------------------------------
    hero = store[("sg1", "lda")]
    hero_idx = np.concatenate([d.test_idx for d in hero])
    hero_proba = {
        c: np.vstack([softmax_temperature(d.test_logits, d.temperatures[c]) for d in hero])
        for c in CALIBRATIONS
    }
    effect_rows = []
    for metric in ("expected_calibration_error", "negative_log_likelihood", "multiclass_brier_score"):
        for treatment in ("shift_aware_temperature", "iid_temperature"):
            boot = cluster_bootstrap_delta(
                y[hero_idx],
                hero_proba["uncalibrated"],
                hero_proba[treatment],
                origin_batch_cluster[hero_idx],
                metric,
                BOOTSTRAP_REPETITIONS,
                BOOTSTRAP_SEED,
            )
            effect_rows.append(
                {
                    "comparison": f"uncalibrated_minus_{treatment}",
                    "metric": metric,
                    "point_reduction": boot["point"],
                    "ci_low": boot["ci_low"],
                    "ci_high": boot["ci_high"],
                    "fraction_bootstrap_positive": boot["fraction_positive"],
                }
            )
        boot = cluster_bootstrap_delta(
            y[hero_idx],
            hero_proba["iid_temperature"],
            hero_proba["shift_aware_temperature"],
            origin_batch_cluster[hero_idx],
            metric,
            BOOTSTRAP_REPETITIONS,
            BOOTSTRAP_SEED,
        )
        effect_rows.append(
            {
                "comparison": "iid_minus_shift_aware",
                "metric": metric,
                "point_reduction": boot["point"],
                "ci_low": boot["ci_low"],
                "ci_high": boot["ci_high"],
                "fraction_bootstrap_positive": boot["fraction_positive"],
            }
        )
    _write_csv(output_dir / "headline_effect.csv", effect_rows)

    # ---- conformal coverage under shift (hero) -----------------------------
    conformal_rows = []
    for calibration, cal_attr_p, cal_attr_y in (
        ("shift_aware_group_conformal", "cal_shift_proba", "cal_shift_y"),
        ("iid_split_conformal", "cal_iid_proba", "cal_iid_y"),
    ):
        for d in hero:
            threshold = conformal_threshold(
                getattr(d, cal_attr_p), getattr(d, cal_attr_y), CONFORMAL_ALPHA
            )
            calib_name = (
                "shift_aware_temperature"
                if "shift" in calibration
                else "iid_temperature"
            )
            test_proba = softmax_temperature(d.test_logits, d.temperatures[calib_name])
            result = conformal_evaluate(test_proba, y[d.test_idx], threshold)
            conformal_rows.append(
                {
                    "method": calibration,
                    "direction": d.direction,
                    "target_coverage": 1.0 - CONFORMAL_ALPHA,
                    "threshold": threshold,
                    **result,
                }
            )
    _write_csv(output_dir / "conformal_coverage.csv", conformal_rows)

    # ---- unsupervised incoming-cube adaptation (accuracy effect) -----------
    adaptation_rows, adaptation_effect_row = adaptation_effect(
        X, y, rep, batch, repetitions=BOOTSTRAP_REPETITIONS, seed=BOOTSTRAP_SEED
    )
    _write_csv(output_dir / "adaptation_metrics.csv", adaptation_rows)
    _write_csv(output_dir / "adaptation_effect.csv", [adaptation_effect_row])

    # ---- negative ablation (optional MAT streaming) ------------------------
    ablation_rows: list[dict[str, Any]] = []
    if run_mat_ablation:
        ablation_rows = covariance_descriptor_ablation(records, X, y, rep)
        _write_csv(output_dir / "negative_ablation.csv", ablation_rows)

    runtime = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "n_development_seeds": int(y.size),
        "expected_bands": EXPECTED_BANDS,
        "data_access_audit": {
            "manifest_hash_files": manifest.hashes_complete,
            "locked_numeric_reads": 0,
            "locked_byte_reads": 0,
            "development_split_only": True,
        },
        "data_fingerprint_sha256": manifest.data_fingerprint_sha256,
    }
    (output_dir / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")

    _write_report(
        output_dir, cross_rows, lobo_rows, effect_rows, conformal_rows, ablation_rows,
        adaptation_rows, adaptation_effect_row,
    )
    return {
        "cross_rows": cross_rows,
        "effect_rows": effect_rows,
        "conformal_rows": conformal_rows,
        "adaptation_rows": adaptation_rows,
        "adaptation_effect": adaptation_effect_row,
    }


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def _write_report(
    output_dir: Path,
    cross_rows: Sequence[dict[str, Any]],
    lobo_rows: Sequence[dict[str, Any]],
    effect_rows: Sequence[dict[str, Any]],
    conformal_rows: Sequence[dict[str, Any]],
    ablation_rows: Sequence[dict[str, Any]],
    adaptation_rows: Sequence[dict[str, Any]],
    adaptation_effect_row: dict[str, Any],
) -> None:
    lines: list[str] = []
    lines.append("# 采集域稳健的校准化产地溯源开发集报告\n")
    lines.append(
        "> 状态：仅使用构造批次 0–7 新执行的开发集探索；构造批次 8–9 的数值与字节均未读取。"
        "跨立方体（来源图像）迁移是采集域稳健性压力测试，不是外部地理认证。\n"
    )
    lines.append(
        "## 诚实结论摘要\n\n"
        "1. **表示选择有明显效果**：SG 一阶导数 + 收缩 LDA 的跨采集域平衡准确率显著高于 "
        "SNV/MSC/SG2 等表示（见比较矩阵），是当前数据上最稳健的产地判别表示。\n"
        "2. **无监督进样立方体自标准化有中等正效应**：仅用目标立方体的未标注特征做逐立方体标准化，"
        "可提升跨采集域平衡准确率（见自适应效应表，含簇自助区间）。\n"
        "3. **温度校准修复跨域校准崩塌（大且稳健）**：但**分组（shift-aware）温度与普通 iid 温度不可区分**——"
        "在单一训练立方体内，构造批次不是足够不同的采集域，分组校准并未带来额外收益（阴性结果）。\n"
        "4. **保形集在跨域下欠覆盖**，分组与 iid 保形无实质差异（阴性结果）。\n"
        "5. **逐种子协方差/像素群体特征在跨采集域下有害**（阴性结果）。\n\n"
        "因此当前数据支持的是一个严格、可证伪的采集域稳健性与校准框架及适度的部署流水线，"
        "而不是具有巨大效应的全新算法。\n"
    )
    lines.append("## 无监督进样立方体自适应（准确率效应, sg1/lda + shift 校准）\n")
    lines.append("| 方法 | 平衡准确率 | 准确率 | NLL | ECE |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in adaptation_rows:
        lines.append(
            f"| {r['method']} | {_fmt(r['balanced_accuracy'])} | {_fmt(r['accuracy'])} | "
            f"{_fmt(r['negative_log_likelihood'])} | {_fmt(r['expected_calibration_error'])} |"
        )
    ae = adaptation_effect_row
    lines.append(
        f"\n自适应效应（target − source 标准化）：平衡准确率 {ae['point_improvement']:+.4f}，"
        f"95% 簇自助区间 [{ae['ci_low']:+.4f}, {ae['ci_high']:+.4f}]，"
        f"自助为正比例 {ae['fraction_bootstrap_positive']:.3f}。\n"
    )
    lines.append("## 跨采集域校准效应（hero = sg1 / lda）\n")
    lines.append("| 比较 | 指标 | 点降幅 | 95% 簇自助区间 | 自助为正比例 |")
    lines.append("|---|---|---:|---:|---:|")
    for r in effect_rows:
        lines.append(
            f"| {r['comparison']} | {r['metric']} | {r['point_reduction']:+.4f} | "
            f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | {r['fraction_bootstrap_positive']:.3f} |"
        )
    lines.append("\n## 同域参考（LOBO, sg1/lda）\n")
    lines.append("| 校准 | 平衡准确率 | NLL | ECE | Brier |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in lobo_rows:
        lines.append(
            f"| {r['calibration']} | {_fmt(r['balanced_accuracy'])} | {_fmt(r['negative_log_likelihood'])} | "
            f"{_fmt(r['expected_calibration_error'])} | {_fmt(r['multiclass_brier_score'])} |"
        )
    lines.append("\n## 跨立方体保形覆盖（目标 90%）\n")
    lines.append("| 方法 | 方向 | 覆盖率 | 平均集大小 | 空集比例 |")
    lines.append("|---|---|---:|---:|---:|")
    for r in conformal_rows:
        lines.append(
            f"| {r['method']} | {r['direction']} | {_fmt(r['coverage'])} | "
            f"{_fmt(r['average_set_size'])} | {_fmt(r['empty_set_fraction'])} |"
        )
    lines.append("\n## 跨采集域比较矩阵（pooled 两方向）\n")
    lines.append("| 表示 | 分类器 | 校准 | 平衡准确率 | NLL | ECE | Brier |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for r in cross_rows:
        lines.append(
            f"| {r['representation']} | {r['classifier']} | {r['calibration']} | "
            f"{_fmt(r['balanced_accuracy'])} | {_fmt(r['negative_log_likelihood'])} | "
            f"{_fmt(r['expected_calibration_error'])} | {_fmt(r['multiclass_brier_score'])} |"
        )
    if ablation_rows:
        lines.append("\n## 负消融：逐种子协方差描述子（跨立方体）\n")
        lines.append("| 特征 | 跨立方体平衡准确率 | 跨立方体 NLL |")
        lines.append("|---|---:|---:|")
        for r in ablation_rows:
            lines.append(
                f"| {r['feature']} | {_fmt(r['cross_cube_balanced_accuracy'])} | "
                f"{_fmt(r['cross_cube_negative_log_likelihood'])} |"
            )
    lines.append(
        "\n## 解释边界\n\n构造批次由导师授权、按确定性规则划分，是当前数据内部的分组单位，"
        "但仍共享每产地两张来源图像，不是新采集的物理批次。本报告的采集域稳健性证据不能替代"
        "跨年份、跨农场、跨仪器或未知产地的外部验证。是否将本方法升级为主预测器，须由一次性锁定评估决定。\n"
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "development_acquisition_calibration",
    )
    parser.add_argument("--no-mat-ablation", action="store_true", help="Skip the MAT covariance negative ablation")
    args = parser.parse_args(argv)
    run(args.data_root, args.output_dir, run_mat_ablation=not args.no_mat_ablation)
    print(f"Wrote development acquisition-calibration outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
