from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score

from provenance_study.acquisition_robust_pipeline import (
    AcquisitionRobustProvenanceClassifier,
    MIN_INCOMING_BATCH,
    fit_temperature,
    softmax_temperature,
)
from provenance_study.core import discover_manifest, load_csv_split

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def _toy(seed: int = 0, n_bands: int = 40):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=3.0, size=(8, n_bands))
    X, y, batch = [], [], []
    for c in range(8):
        for b in range(8):
            for _ in range(3):
                X.append(centers[c] + rng.normal(scale=0.5, size=n_bands))
                y.append(c)
                batch.append(b)
    return np.asarray(X), np.asarray(y), np.asarray(batch)


def test_predict_proba_normalized_and_labels() -> None:
    X, y, batch = _toy()
    clf = AcquisitionRobustProvenanceClassifier().fit(X, y, batch)
    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], 8)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert set(np.unique(clf.predict(X))).issubset(set(range(8)))


def test_single_seed_prediction_is_rejected() -> None:
    X, y, batch = _toy()
    clf = AcquisitionRobustProvenanceClassifier().fit(X, y, batch)
    with pytest.raises(ValueError, match="batch of"):
        clf.predict_proba(X[:1])
    assert MIN_INCOMING_BATCH >= 2


def test_uncalibrated_temperature_is_one_and_needs_no_batch() -> None:
    X, y, _ = _toy()
    clf = AcquisitionRobustProvenanceClassifier(calibrate=False).fit(X, y)
    assert clf.temperature_ == 1.0


def test_calibration_requires_batch() -> None:
    X, y, _ = _toy()
    with pytest.raises(ValueError, match="grouped OOF"):
        AcquisitionRobustProvenanceClassifier(calibrate=True).fit(X, y)


def test_fit_temperature_matches_grid_argmin() -> None:
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(200, 8))
    y = rng.integers(0, 8, size=200)
    T = fit_temperature(logits, y)
    nll = lambda t: -np.log(
        np.clip(softmax_temperature(logits, t)[np.arange(200), y], 1e-15, 1.0)
    ).mean()
    assert nll(T) <= nll(1.0) + 1e-12


@pytest.mark.skipif(not DATA_ROOT.is_dir(), reason="requires the archived data directory")
def test_reproduces_development_cross_cube_effect() -> None:
    """Frozen pipeline should transfer across cubes near the development ~95%."""

    manifest = discover_manifest(DATA_ROOT)
    dev = load_csv_split(manifest, split="development", verify_hashes=False)
    X, y = dev.X, dev.y
    rep = np.array([r.replicate for r in dev.records])
    batch = np.array([r.constructed_batch for r in dev.records])

    accuracies = []
    for train_cube, test_cube in ((1, 2), (2, 1)):
        itr, ite = rep == train_cube, rep == test_cube
        clf = AcquisitionRobustProvenanceClassifier().fit(X[itr], y[itr], batch[itr])
        pred = clf.predict(X[ite])
        accuracies.append(balanced_accuracy_score(y[ite], pred))
    theta = float(np.mean(accuracies))
    # Development established ~0.95; allow a tolerance band, must clearly beat SNV-LR ~0.86.
    assert theta > 0.92, f"cross-cube balanced accuracy regressed: {theta:.4f}"
