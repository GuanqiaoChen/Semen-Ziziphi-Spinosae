from __future__ import annotations

import numpy as np
import pytest

from provenance_study.explore_acquisition_calibration import (
    NUM_CLASSES,
    batch_grouped_folds,
    build_pipeline,
    cluster_bootstrap_delta,
    conformal_evaluate,
    conformal_threshold,
    evaluate_direction,
    fit_temperature,
    pipeline_logits,
    random_folds,
    softmax_temperature,
)


def _synthetic(seed: int = 0, n_batches: int = 8, per_cell: int = 2, n_features: int = 20):
    """8 classes x 2 replicates x n_batches cells with separable, cube-shifted means."""

    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=3.0, size=(NUM_CLASSES, n_features))
    cube_shift = rng.normal(scale=1.0, size=(2, n_features))  # per-cube nuisance offset
    X, y, rep, batch = [], [], [], []
    for c in range(NUM_CLASSES):
        for r in (1, 2):
            for b in range(n_batches):
                for _ in range(per_cell):
                    X.append(centers[c] + cube_shift[r - 1] + rng.normal(scale=0.6, size=n_features))
                    y.append(c)
                    rep.append(r)
                    batch.append(b)
    return (
        np.asarray(X),
        np.asarray(y, dtype=np.int64),
        np.asarray(rep, dtype=np.int64),
        np.asarray(batch, dtype=np.int64),
    )


def test_softmax_temperature_normalised_and_softens() -> None:
    logits = np.array([[3.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    p1 = softmax_temperature(logits, 1.0)
    p2 = softmax_temperature(logits, 4.0)
    assert np.allclose(p1.sum(axis=1), 1.0)
    assert np.allclose(p2.sum(axis=1), 1.0)
    assert p2.max() < p1.max()  # higher temperature is less confident


def test_fit_temperature_recovers_softening_for_overconfident_logits() -> None:
    rng = np.random.default_rng(1)
    n = 600
    y = rng.integers(0, NUM_CLASSES, size=n)
    logits = np.zeros((n, NUM_CLASSES))
    # A sharply favoured class that is correct only 60% of the time and
    # *confidently wrong* otherwise -> overconfident, so NLL wants T > 1.
    for i, label in enumerate(y):
        favoured = label if rng.random() < 0.6 else (label + 1 + rng.integers(NUM_CLASSES - 1)) % NUM_CLASSES
        logits[i, favoured] = 6.0
    assert fit_temperature(logits, y) > 1.0


def test_batch_grouped_folds_leave_one_out() -> None:
    _, _, _, batch = _synthetic()
    subset = np.arange(batch.size)
    folds = batch_grouped_folds(batch, subset)
    assert len(folds) == len(np.unique(batch))
    seen = np.concatenate([cal for _, cal in folds])
    assert np.array_equal(np.sort(seen), subset)  # each sample held out exactly once
    for train, cal in folds:
        assert set(batch[train]).isdisjoint(set(batch[cal]))


def test_random_folds_partition_subset() -> None:
    subset = np.arange(80)
    folds = random_folds(subset, 8, seed=3)
    seen = np.concatenate([cal for _, cal in folds])
    assert np.array_equal(np.sort(seen), subset)
    for train, cal in folds:
        assert set(train).isdisjoint(set(cal))


def test_conformal_threshold_reaches_nominal_coverage_iid() -> None:
    rng = np.random.default_rng(5)
    n, alpha = 4000, 0.1
    # Well-calibrated-ish probabilities: true-class prob drawn, rest uniform.
    def make(n_):
        y_ = rng.integers(0, NUM_CLASSES, size=n_)
        p = rng.dirichlet(np.ones(NUM_CLASSES), size=n_)
        return y_, p
    y_cal, p_cal = make(n)
    y_te, p_te = make(n)
    thr = conformal_threshold(p_cal, y_cal, alpha)
    cov = conformal_evaluate(p_te, y_te, thr)["coverage"]
    assert cov >= (1 - alpha) - 0.03  # finite-sample split-conformal guarantee


def test_cluster_bootstrap_point_matches_direct_delta() -> None:
    rng = np.random.default_rng(6)
    n = 300
    y = rng.integers(0, NUM_CLASSES, size=n)
    ref = rng.dirichlet(np.ones(NUM_CLASSES), size=n)
    trt = rng.dirichlet(np.ones(NUM_CLASSES) * 4, size=n)
    clusters = rng.integers(0, 20, size=n)
    out = cluster_bootstrap_delta(
        y, ref, trt, clusters, "negative_log_likelihood", repetitions=200, seed=0
    )
    assert out["ci_low"] <= out["point"] <= out["ci_high"]
    assert 0.0 <= out["fraction_positive"] <= 1.0


@pytest.mark.parametrize("classifier", ["lda", "lr", "svm"])
def test_pipeline_logits_shape(classifier: str) -> None:
    X, y, _, _ = _synthetic()
    pipe = build_pipeline("sg1", classifier).fit(X, y)
    logits = pipeline_logits(pipe, X[:10])
    assert logits.shape == (10, NUM_CLASSES)
    assert np.all(np.isfinite(logits))


def test_evaluate_direction_uses_only_test_cube_and_yields_three_temperatures() -> None:
    X, y, rep, batch = _synthetic()
    result = evaluate_direction("sg1", "lda", X, y, rep, batch, 1, 2, "cube1_to_cube2")
    assert np.array_equal(np.sort(result.test_idx), np.sort(np.where(rep == 2)[0]))
    assert set(result.temperatures) == {
        "uncalibrated",
        "iid_temperature",
        "shift_aware_temperature",
    }
    assert result.temperatures["uncalibrated"] == 1.0
    assert result.test_logits.shape == (result.test_idx.size, NUM_CLASSES)
