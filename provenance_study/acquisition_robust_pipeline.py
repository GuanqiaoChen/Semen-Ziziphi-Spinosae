#!/usr/bin/env python3
"""Frozen acquisition-robust provenance predictor for *Ziziphi Spinosae* Semen.

This module packages the single pipeline selected during leakage-controlled
development (constructed batches 0-7, whole-source-cube transfer axis; see
``explore_acquisition_calibration.py`` and its outputs).  The pipeline is:

    1. Savitzky-Golay first derivative (window 15, poly 2) -- the representation
       with the best cross-acquisition stability among raw / SNV / MSC / SG1 /
       SG2 in development.
    2. Shrinkage linear discriminant analysis (lsqr, analytic shrinkage) --
       more transfer-robust than logistic regression, RBF-SVM, and the deep
       reference on the whole-cube axis.
    3. Unlabelled target-acquisition normalization -- the incoming acquisition
       (a *batch* of seeds of unknown origin) is standardized on its **own**
       pooled per-band statistics, removing a per-cube affine batch effect using
       no labels.  Training features are standardized on training statistics.
    4. Post-hoc temperature scaling -- a single scalar temperature fit on the
       training cube's constructed-batch-grouped out-of-fold predictions,
       repairing the confidence distortion that a cube change induces.

Deployment assumption: prediction consumes a *batch* of incoming seeds so that
target statistics are estimable; single-seed scoring is not supported and is
rejected.  The two source cubes per origin are not independent physical lots, so
this predictor is validated as an acquisition-robustness result, not external
geographical certification.

Development note (retained, not hidden): the constructed-batch-grouped
temperature was **not** found superior to an ordinary i.i.d. temperature, and
grouped conformal did not beat i.i.d. conformal; those are reported as negative
results.  The grouped out-of-fold temperature is used here only because the
constructed batches are the natural fitting groups, not as a claimed advantage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from provenance_study.core import SavitzkyGolayTransformer

SG_WINDOW = 15
SG_POLYORDER = 2
SG_DERIVATIVE = 1
LDA_SOLVER = "lsqr"
LDA_SHRINKAGE = "auto"
TEMPERATURE_GRID = np.exp(np.linspace(np.log(0.25), np.log(8.0), 400))
MIN_INCOMING_BATCH = 8  # target statistics need a batch, not one seed
_EPS = np.finfo(np.float64).eps


def _sg1(X: np.ndarray) -> np.ndarray:
    transformer = SavitzkyGolayTransformer(SG_WINDOW, SG_POLYORDER, SG_DERIVATIVE)
    return transformer.fit(X).transform(np.asarray(X, dtype=np.float64))


def _self_standardize(F: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = F.mean(axis=0)
    scale = F.std(axis=0, ddof=0)
    scale = np.where(scale <= _EPS, 1.0, scale)
    return (F - mean) / scale, mean, scale


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


@dataclass
class AcquisitionRobustProvenanceClassifier:
    """Frozen SG1 + shrinkage-LDA + target-normalization + temperature pipeline."""

    calibrate: bool = True
    temperature_: float = field(default=1.0, init=False)
    classes_: np.ndarray = field(default_factory=lambda: np.empty(0), init=False)
    _lda: Any = field(default=None, init=False)
    _n_bands: int = field(default=0, init=False)

    def fit(self, X: np.ndarray, y: np.ndarray, batch: np.ndarray | None = None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D seed-by-band matrix")
        self._n_bands = X.shape[1]
        F, _, _ = _self_standardize(_sg1(X))
        self._lda = LinearDiscriminantAnalysis(solver=LDA_SOLVER, shrinkage=LDA_SHRINKAGE).fit(F, y)
        self.classes_ = self._lda.classes_

        if self.calibrate:
            if batch is None:
                raise ValueError("Calibration requires a `batch` group vector for grouped OOF")
            batch = np.asarray(batch, dtype=np.int64)
            if batch.shape[0] != y.shape[0]:
                raise ValueError("batch must have one entry per training seed")
            self.temperature_ = self._grouped_oof_temperature(X, y, batch)
        else:
            self.temperature_ = 1.0
        return self

    def _grouped_oof_temperature(self, X: np.ndarray, y: np.ndarray, batch: np.ndarray) -> float:
        logit_rows, label_rows = [], []
        for held in np.unique(batch):
            train_mask = batch != held
            cal_mask = batch == held
            if train_mask.sum() == 0 or cal_mask.sum() == 0:
                continue
            F_train, _, _ = _self_standardize(_sg1(X[train_mask]))
            F_cal, _, _ = _self_standardize(_sg1(X[cal_mask]))  # held batch on own stats
            fold_lda = LinearDiscriminantAnalysis(solver=LDA_SOLVER, shrinkage=LDA_SHRINKAGE).fit(
                F_train, y[train_mask]
            )
            logit_rows.append(fold_lda.predict_log_proba(F_cal))
            label_rows.append(y[cal_mask])
        if not logit_rows:
            return 1.0
        return fit_temperature(np.vstack(logit_rows), np.concatenate(label_rows))

    def predict_proba(self, X_incoming: np.ndarray) -> np.ndarray:
        if self._lda is None:
            raise RuntimeError("Classifier must be fit before prediction")
        X_incoming = np.asarray(X_incoming, dtype=np.float64)
        if X_incoming.ndim != 2 or X_incoming.shape[1] != self._n_bands:
            raise ValueError("X_incoming must be a 2-D matrix with the trained band count")
        if X_incoming.shape[0] < MIN_INCOMING_BATCH:
            raise ValueError(
                f"Target normalization needs a batch of >= {MIN_INCOMING_BATCH} incoming seeds; "
                f"got {X_incoming.shape[0]}"
            )
        F, _, _ = _self_standardize(_sg1(X_incoming))  # incoming standardized on its own stats
        logits = self._lda.predict_log_proba(F)
        return softmax_temperature(logits, self.temperature_)

    def predict(self, X_incoming: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X_incoming).argmax(axis=1)]
