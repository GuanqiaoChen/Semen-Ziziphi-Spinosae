#!/usr/bin/env python3
"""Explore batch-stable wavelength subsets without opening locked spectra.

This module is a development-only, negative-result audit for geographical-origin
traceability.  It reproduces the disclosed 132-candidate wavelength-selection
panel on constructed batches 0--7.  Batches 8--9 are enumerated structurally by
``discover_manifest(hash_files=False)`` but are rejected before numerical I/O.

Every outer fold holds out one complete constructed batch.  Within the remaining
seven batches, seven inner training subsets are formed by omitting one additional
batch.  ANOVA/Fisher scores or standardized shrinkage-LDA coefficient magnitudes
are ranked on those six-batch subsets.  The seven rankings alone determine the
outer-fold wavelength subset; the outer fold cannot affect feature selection.

All comparisons among the 132 candidates use the same development OOF predictions
and are explicitly exploratory.  The maximum over this panel is selection-biased
and is not eligible to replace the frozen primary method or justify a locked-test
analysis change.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import sklearn
from scipy.signal import savgol_filter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from provenance_study.core import (  # noqa: E402
    BASE_BATCH_SEED,
    DEVELOPMENT_BATCHES,
    EXPECTED_BANDS,
    EXPECTED_SOURCE_COUNTS,
    LOCKED_BATCHES,
    LockedDataAccessError,
    SpectralDataset,
    discover_manifest,
    load_development_csv,
    multiclass_metrics,
)


DEFAULT_OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parent / "outputs" / "development_bands"
)
SCORERS = ("ANOVA", "LDAcoef")
CLASSIFIERS = ("LDA", "LR")
FIXED_BAND_COUNTS = (8, 12, 16, 24, 32, 48, 64, 96, 128)
CONSENSUS_TOP_COUNTS = (16, 24, 32, 48, 64, 96)
CONSENSUS_THRESHOLDS = (0.50, 0.67, 0.80, 1.00)
SG_WINDOW = 15
SG_POLYORDER = 2
SG_DERIVATIVE = 1
EXPECTED_SELECTOR_COUNT = 66
EXPECTED_CANDIDATE_COUNT = 132
OUTPUT_FILENAMES = (
    "candidates.csv",
    "fold_selections.csv",
    "intervals.csv",
    "selection_frequency.csv",
    "summary.json",
    "report.md",
)


@dataclass(frozen=True)
class SelectionDefinition:
    """One deterministic wavelength-selection rule, before classifier choice."""

    selector_id: str
    scorer: str
    mode: str
    amount: int
    consensus_threshold: float | None


@dataclass(frozen=True)
class StableBandEvaluation:
    """In-memory OOF results and rows used to write the audit artifacts."""

    classes: np.ndarray
    transformed_X: np.ndarray
    baseline_probabilities: np.ndarray
    baseline_metrics: Mapping[str, Any]
    candidate_rows: tuple[dict[str, Any], ...]
    fold_selection_rows: tuple[dict[str, Any], ...]
    interval_rows: tuple[dict[str, Any], ...]
    frequency_rows: tuple[dict[str, Any], ...]


def selection_definitions() -> tuple[SelectionDefinition, ...]:
    """Return the frozen 66 selectors; two classifiers produce 132 candidates."""

    definitions: list[SelectionDefinition] = []
    for scorer in SCORERS:
        for count in FIXED_BAND_COUNTS:
            definitions.append(
                SelectionDefinition(
                    selector_id=f"{scorer}:fixed:{count}",
                    scorer=scorer,
                    mode="fixed",
                    amount=count,
                    consensus_threshold=None,
                )
            )
        for top_count in CONSENSUS_TOP_COUNTS:
            for threshold in CONSENSUS_THRESHOLDS:
                definitions.append(
                    SelectionDefinition(
                        selector_id=(
                            f"{scorer}:consensus:{top_count}:{threshold:.2f}"
                        ),
                        scorer=scorer,
                        mode="consensus",
                        amount=top_count,
                        consensus_threshold=threshold,
                    )
                )
    if len(definitions) != EXPECTED_SELECTOR_COUNT:
        raise AssertionError("Frozen selector panel no longer contains 66 rules")
    return tuple(definitions)


def _validate_development_inputs(
    X: np.ndarray,
    y: Sequence[int],
    groups: Sequence[int],
    wavelengths: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(X, dtype=np.float64)
    labels = np.asarray(y, dtype=np.int64)
    batch_ids = np.asarray(groups, dtype=np.int64)
    wavelength_values = np.asarray(wavelengths, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("X must be a non-empty sample-by-band matrix")
    if labels.ndim != 1 or batch_ids.ndim != 1:
        raise ValueError("y and groups must be one-dimensional")
    if labels.size != features.shape[0] or batch_ids.size != features.shape[0]:
        raise ValueError("X, y, and groups must contain the same number of samples")
    if wavelength_values.shape != (features.shape[1],):
        raise ValueError("wavelengths must contain one value for every input band")
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(wavelength_values)):
        raise ValueError("X and wavelengths must be finite")
    if np.any(np.diff(wavelength_values) <= 0.0):
        raise ValueError("wavelengths must be strictly increasing")
    observed_groups = set(batch_ids.tolist())
    expected_groups = set(DEVELOPMENT_BATCHES)
    if observed_groups & set(LOCKED_BATCHES):
        raise LockedDataAccessError(
            "Stable-band development exploration received a locked batch ID"
        )
    if observed_groups != expected_groups:
        raise ValueError(
            "Stable-band exploration requires exactly constructed batches 0--7; "
            f"observed {sorted(observed_groups)}"
        )
    classes = np.unique(labels)
    if classes.size < 2:
        raise ValueError("At least two origin classes are required")
    for outer_batch in DEVELOPMENT_BATCHES:
        train_classes = set(labels[batch_ids != outer_batch].tolist())
        test_classes = set(labels[batch_ids == outer_batch].tolist())
        if train_classes != set(classes.tolist()) or test_classes != set(classes.tolist()):
            raise ValueError(
                f"Outer batch {outer_batch} must contain every origin class in both sides"
            )
    if features.shape[1] < SG_WINDOW:
        raise ValueError(f"At least {SG_WINDOW} bands are required for SG preprocessing")
    return features, labels, batch_ids, wavelength_values


def sg_first_derivative(X: np.ndarray) -> np.ndarray:
    """Apply the frozen sample-wise SG(15, 2, first derivative) transform."""

    features = np.asarray(X, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] < SG_WINDOW:
        raise ValueError(f"SG preprocessing requires at least {SG_WINDOW} bands")
    return savgol_filter(
        features,
        window_length=SG_WINDOW,
        polyorder=SG_POLYORDER,
        deriv=SG_DERIVATIVE,
        axis=1,
        mode="interp",
    )


def _make_classifier(kind: str) -> Pipeline:
    if kind == "LDA":
        classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    elif kind == "LR":
        classifier = LogisticRegression(
            C=1.0,
            max_iter=4_000,
            solver="lbfgs",
            random_state=BASE_BATCH_SEED,
        )
    else:
        raise ValueError(f"Unknown classifier: {kind}")
    return Pipeline([("standardize", StandardScaler()), ("classifier", classifier)])


def _aligned_probabilities(
    model: Pipeline, X: np.ndarray, classes: np.ndarray
) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(X), dtype=np.float64)
    model_classes = np.asarray(model.classes_, dtype=np.int64)
    if set(model_classes.tolist()) != set(classes.tolist()):
        raise ValueError("Fitted classifier does not contain every development class")
    positions = {int(label): index for index, label in enumerate(model_classes)}
    aligned = probabilities[:, [positions[int(label)] for label in classes]]
    if not np.all(np.isfinite(aligned)) or not np.allclose(
        aligned.sum(axis=1), 1.0, rtol=0.0, atol=1e-8
    ):
        raise ValueError("Classifier returned invalid probabilities")
    return aligned


def _importance_ranks(X: np.ndarray, y: np.ndarray, scorer: str) -> np.ndarray:
    """Rank bands on one six-batch training subset with stable index tie-breaking."""

    if scorer == "ANOVA":
        scores = np.asarray(f_classif(X, y)[0], dtype=np.float64)
        scores = np.nan_to_num(
            scores,
            nan=-np.inf,
            posinf=np.finfo(np.float64).max,
            neginf=-np.inf,
        )
    elif scorer == "LDAcoef":
        scaler = StandardScaler().fit(X)
        standardized = scaler.transform(X)
        fitted = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(
            standardized, y
        )
        scores = np.sqrt(np.mean(np.square(fitted.coef_), axis=0))
        scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.finfo(np.float64).max)
    else:
        raise ValueError(f"Unknown band scorer: {scorer}")
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(order.size)
    return ranks


def inner_rank_matrix(
    transformed_X: np.ndarray,
    y: Sequence[int],
    groups: Sequence[int],
    *,
    outer_batch: int,
    scorer: str,
) -> np.ndarray:
    """Return seven rankings computed without the outer batch or one inner batch.

    This function is public specifically so the outer-fold isolation invariant can
    be tested on synthetic arrays.  It expects already transformed spectra because
    SG is a fixed, sample-wise operation with no learned state.
    """

    features = np.asarray(transformed_X, dtype=np.float64)
    labels = np.asarray(y, dtype=np.int64)
    batch_ids = np.asarray(groups, dtype=np.int64)
    if features.ndim != 2 or labels.shape != (features.shape[0],):
        raise ValueError("transformed_X and y have incompatible shapes")
    if batch_ids.shape != (features.shape[0],):
        raise ValueError("transformed_X and groups have incompatible shapes")
    observed_groups = set(batch_ids.tolist())
    if observed_groups & set(LOCKED_BATCHES):
        raise LockedDataAccessError("Inner selection received a locked batch ID")
    if observed_groups != set(DEVELOPMENT_BATCHES):
        raise ValueError("Inner selection requires exactly development batches 0--7")
    if int(outer_batch) not in DEVELOPMENT_BATCHES:
        raise ValueError("outer_batch must be one of the development batches 0--7")
    rankings: list[np.ndarray] = []
    for inner_batch in DEVELOPMENT_BATCHES:
        if inner_batch == outer_batch:
            continue
        training_mask = (batch_ids != outer_batch) & (batch_ids != inner_batch)
        rankings.append(_importance_ranks(features[training_mask], labels[training_mask], scorer))
    matrix = np.asarray(rankings, dtype=np.int64)
    expected_shape = (len(DEVELOPMENT_BATCHES) - 1, features.shape[1])
    if matrix.shape != expected_shape:
        raise AssertionError(f"Expected inner rank matrix {expected_shape}; observed {matrix.shape}")
    return matrix


def select_bands(
    rank_matrix: np.ndarray,
    definition: SelectionDefinition,
) -> np.ndarray:
    """Convert seven leakage-safe rankings into one outer-fold band subset."""

    ranks = np.asarray(rank_matrix, dtype=np.int64)
    if ranks.ndim != 2 or ranks.shape[0] != len(DEVELOPMENT_BATCHES) - 1:
        raise ValueError("rank_matrix must contain seven inner rankings")
    n_bands = ranks.shape[1]
    if definition.scorer not in SCORERS:
        raise ValueError(f"Unknown scorer in definition: {definition.scorer}")
    if definition.amount <= 0 or definition.amount > n_bands:
        raise ValueError(
            f"Selection amount {definition.amount} is incompatible with {n_bands} bands"
        )
    aggregate_rank = ranks.mean(axis=0)
    if definition.mode == "fixed":
        selected = np.argsort(aggregate_rank, kind="mergesort")[: definition.amount]
    elif definition.mode == "consensus":
        threshold = definition.consensus_threshold
        if threshold is None or not 0.0 < threshold <= 1.0:
            raise ValueError("Consensus selection requires a threshold in (0, 1]")
        required = int(math.ceil(threshold * ranks.shape[0] - 1e-12))
        frequency = (ranks < definition.amount).sum(axis=0)
        selected = np.flatnonzero(frequency >= required)
        if selected.size == 0:
            # Frozen deterministic fallback disclosed in the original exploration.
            selected = np.argsort(aggregate_rank, kind="mergesort")[: min(4, n_bands)]
    else:
        raise ValueError(f"Unknown selection mode: {definition.mode}")
    return np.sort(np.asarray(selected, dtype=np.int64))


def contiguous_intervals(
    selected_indices: Sequence[int], wavelengths: Sequence[float]
) -> tuple[dict[str, Any], ...]:
    """Merge directly adjacent selected indices into wavelength intervals."""

    selected = np.unique(np.asarray(selected_indices, dtype=np.int64))
    wavelength_values = np.asarray(wavelengths, dtype=np.float64)
    if selected.size == 0:
        return ()
    if selected[0] < 0 or selected[-1] >= wavelength_values.size:
        raise ValueError("Selected band index lies outside the wavelength vector")
    runs: list[tuple[int, int]] = []
    start = previous = int(selected[0])
    for value in selected[1:]:
        current = int(value)
        if current == previous + 1:
            previous = current
            continue
        runs.append((start, previous))
        start = previous = current
    runs.append((start, previous))
    return tuple(
        {
            "first_band_index": first,
            "last_band_index": last,
            "start_nm": float(wavelength_values[first]),
            "end_nm": float(wavelength_values[last]),
            "n_bands": last - first + 1,
        }
        for first, last in runs
    )


def _probability_metrics(
    y: np.ndarray,
    groups: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> dict[str, Any]:
    predicted = classes[np.argmax(probabilities, axis=1)]
    shared_metrics = multiclass_metrics(y, probabilities, classes=classes)
    fold_scores = [
        float(
            balanced_accuracy_score(
                y[groups == batch], predicted[groups == batch]
            )
        )
        for batch in DEVELOPMENT_BATCHES
    ]
    return {
        "accuracy": float(shared_metrics["accuracy"]),
        "balanced_accuracy": float(shared_metrics["balanced_accuracy"]),
        "macro_f1": float(shared_metrics["macro_f1"]),
        # Keep the artifact column name for compatibility, but compute the value
        # with the single project-wide NLL definition in ``core.multiclass_metrics``.
        # In particular, its 1e-15 true-class probability floor must not drift to
        # sklearn's version-dependent epsilon or the former local 1e-12 floor.
        "log_loss": float(shared_metrics["negative_log_likelihood"]),
        "fold_balanced_accuracy_min": float(min(fold_scores)),
        "fold_balanced_accuracy_max": float(max(fold_scores)),
        "fold_balanced_accuracy_mean": float(np.mean(fold_scores)),
        "fold_balanced_accuracy_sd": float(np.std(fold_scores, ddof=1)),
        "fold_balanced_accuracies": fold_scores,
    }


def _pairwise_jaccard(selections: Sequence[Sequence[int]]) -> tuple[float, float]:
    values: list[float] = []
    sets = [set(int(index) for index in selected) for selected in selections]
    for left, right in combinations(sets, 2):
        union = left | right
        values.append(1.0 if not union else len(left & right) / len(union))
    if not values:
        return 1.0, 1.0
    return float(np.mean(values)), float(min(values))


def _baseline_oof(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, classes: np.ndarray
) -> np.ndarray:
    probabilities = np.full((y.size, classes.size), np.nan, dtype=np.float64)
    for outer_batch in DEVELOPMENT_BATCHES:
        train_mask = groups != outer_batch
        test_mask = groups == outer_batch
        fitted = _make_classifier("LDA").fit(X[train_mask], y[train_mask])
        probabilities[test_mask] = _aligned_probabilities(
            fitted, X[test_mask], classes
        )
    if not np.all(np.isfinite(probabilities)):
        raise AssertionError("Full-spectrum LDA OOF did not cover every sample")
    return probabilities


def evaluate_stable_band_candidates(
    X: np.ndarray,
    y: Sequence[int],
    groups: Sequence[int],
    wavelengths: Sequence[float],
    *,
    definitions: Sequence[SelectionDefinition] | None = None,
) -> StableBandEvaluation:
    """Evaluate the development-only candidate panel with nested feature selection."""

    features, labels, batch_ids, wavelength_values = _validate_development_inputs(
        X, y, groups, wavelengths
    )
    transformed = sg_first_derivative(features)
    classes = np.unique(labels)
    requested_definitions = (
        selection_definitions() if definitions is None else tuple(definitions)
    )
    if not requested_definitions:
        raise ValueError("At least one selection definition is required")
    baseline_probabilities = _baseline_oof(transformed, labels, batch_ids, classes)
    baseline_metrics = _probability_metrics(
        labels, batch_ids, baseline_probabilities, classes
    )

    ranks_by_outer_and_scorer: dict[tuple[int, str], np.ndarray] = {}
    for outer_batch in DEVELOPMENT_BATCHES:
        for scorer in SCORERS:
            ranks_by_outer_and_scorer[(outer_batch, scorer)] = inner_rank_matrix(
                transformed,
                labels,
                batch_ids,
                outer_batch=outer_batch,
                scorer=scorer,
            )

    candidate_rows: list[dict[str, Any]] = []
    fold_selection_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []

    for definition in requested_definitions:
        fold_selections = [
            select_bands(
                ranks_by_outer_and_scorer[(outer_batch, definition.scorer)], definition
            )
            for outer_batch in DEVELOPMENT_BATCHES
        ]
        mean_jaccard, minimum_jaccard = _pairwise_jaccard(fold_selections)
        counts = np.asarray([selected.size for selected in fold_selections], dtype=np.int64)
        frequency = np.zeros(features.shape[1], dtype=np.int64)
        for selected in fold_selections:
            frequency[selected] += 1

        for classifier_kind in CLASSIFIERS:
            candidate_id = f"{definition.selector_id}:{classifier_kind}"
            probabilities = np.full(
                (labels.size, classes.size), np.nan, dtype=np.float64
            )
            for position, outer_batch in enumerate(DEVELOPMENT_BATCHES):
                train_mask = batch_ids != outer_batch
                test_mask = batch_ids == outer_batch
                selected = fold_selections[position]
                fitted = _make_classifier(classifier_kind).fit(
                    transformed[train_mask][:, selected], labels[train_mask]
                )
                probabilities[test_mask] = _aligned_probabilities(
                    fitted, transformed[test_mask][:, selected], classes
                )
                merged = contiguous_intervals(selected, wavelength_values)
                fold_selection_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "selector_id": definition.selector_id,
                        "classifier": classifier_kind,
                        "outer_held_batch": int(outer_batch),
                        "selected_band_count": int(selected.size),
                        "interval_count": len(merged),
                        "selected_band_indices": ";".join(map(str, selected.tolist())),
                    }
                )
                for interval_number, interval in enumerate(merged, start=1):
                    interval_rows.append(
                        {
                            "candidate_id": candidate_id,
                            "outer_held_batch": int(outer_batch),
                            "interval_number": interval_number,
                            **interval,
                        }
                    )
            if not np.all(np.isfinite(probabilities)):
                raise AssertionError(f"Candidate OOF incomplete: {candidate_id}")
            metrics = _probability_metrics(
                labels, batch_ids, probabilities, classes
            )
            ensemble_probabilities = 0.5 * probabilities + 0.5 * baseline_probabilities
            ensemble_metrics = _probability_metrics(
                labels, batch_ids, ensemble_probabilities, classes
            )
            candidate_rows.append(
                {
                    "candidate_id": candidate_id,
                    "selector_id": definition.selector_id,
                    "scorer": definition.scorer,
                    "selection_mode": definition.mode,
                    "selection_amount": definition.amount,
                    "consensus_threshold": (
                        ""
                        if definition.consensus_threshold is None
                        else definition.consensus_threshold
                    ),
                    "classifier": classifier_kind,
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "log_loss": metrics["log_loss"],
                    "fold_balanced_accuracy_min": metrics[
                        "fold_balanced_accuracy_min"
                    ],
                    "fold_balanced_accuracy_max": metrics[
                        "fold_balanced_accuracy_max"
                    ],
                    "fold_balanced_accuracy_sd": metrics[
                        "fold_balanced_accuracy_sd"
                    ],
                    "selected_band_count_min": int(counts.min()),
                    "selected_band_count_median": float(np.median(counts)),
                    "selected_band_count_max": int(counts.max()),
                    "mean_pairwise_selection_jaccard": mean_jaccard,
                    "minimum_pairwise_selection_jaccard": minimum_jaccard,
                    "delta_vs_full_spectrum_lda_pp": 100.0
                    * (
                        float(metrics["balanced_accuracy"])
                        - float(baseline_metrics["balanced_accuracy"])
                    ),
                    "ensemble_with_full_lda_balanced_accuracy": ensemble_metrics[
                        "balanced_accuracy"
                    ],
                    "ensemble_with_full_lda_macro_f1": ensemble_metrics["macro_f1"],
                    "ensemble_with_full_lda_log_loss": ensemble_metrics["log_loss"],
                    "ensemble_fold_balanced_accuracy_min": ensemble_metrics[
                        "fold_balanced_accuracy_min"
                    ],
                    "ensemble_fold_balanced_accuracy_max": ensemble_metrics[
                        "fold_balanced_accuracy_max"
                    ],
                    "ensemble_delta_vs_full_spectrum_lda_pp": 100.0
                    * (
                        float(ensemble_metrics["balanced_accuracy"])
                        - float(baseline_metrics["balanced_accuracy"])
                    ),
                    "comparison_scope": "same_development_oof_exploratory",
                    "promotion_eligible": False,
                }
            )
            for band_index, times_selected in enumerate(frequency):
                if times_selected == 0:
                    continue
                frequency_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "band_index": band_index,
                        "wavelength_nm": float(wavelength_values[band_index]),
                        "outer_folds_selected": int(times_selected),
                        "selection_frequency": float(
                            times_selected / len(DEVELOPMENT_BATCHES)
                        ),
                    }
                )

    expected_candidates = len(requested_definitions) * len(CLASSIFIERS)
    if len(candidate_rows) != expected_candidates:
        raise AssertionError(
            f"Expected {expected_candidates} candidate rows; observed {len(candidate_rows)}"
        )
    return StableBandEvaluation(
        classes=classes,
        transformed_X=transformed,
        baseline_probabilities=baseline_probabilities,
        baseline_metrics=baseline_metrics,
        candidate_rows=tuple(candidate_rows),
        fold_selection_rows=tuple(fold_selection_rows),
        interval_rows=tuple(interval_rows),
        frequency_rows=tuple(frequency_rows),
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty artifact: {path.name}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary_payload(
    evaluation: StableBandEvaluation,
    *,
    dataset: SpectralDataset,
    manifest_fingerprint: str,
) -> dict[str, Any]:
    rows = list(evaluation.candidate_rows)
    best_selected = max(
        rows, key=lambda row: (float(row["balanced_accuracy"]), -float(row["log_loss"]))
    )
    best_ensemble = max(
        rows,
        key=lambda row: (
            float(row["ensemble_with_full_lda_balanced_accuracy"]),
            -float(row["ensemble_with_full_lda_log_loss"]),
        ),
    )
    loaded_batches = sorted(
        {int(record.constructed_batch) for record in dataset.records}
    )
    if loaded_batches != list(DEVELOPMENT_BATCHES):
        raise LockedDataAccessError(
            f"Artifact summary received non-development batches: {loaded_batches}"
        )
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_objective": "hyperspectral geographical-origin traceability",
        "analysis_scope": "constructed_batches_0_to_7_development_only",
        "status": "exploratory_negative_result_not_in_primary_method",
        "selection_bias_warning": (
            "The largest value among 132 candidates was selected on the same development "
            "OOF comparison and is optimistic; it is not confirmatory evidence."
        ),
        "protocol": {
            "analysis_seed": BASE_BATCH_SEED,
            "sg": {
                "window_length": SG_WINDOW,
                "polyorder": SG_POLYORDER,
                "derivative": SG_DERIVATIVE,
                "mode": "interp",
            },
            "outer_folds": list(DEVELOPMENT_BATCHES),
            "inner_rankings_per_outer_fold": len(DEVELOPMENT_BATCHES) - 1,
            "scorers": list(SCORERS),
            "classifiers": list(CLASSIFIERS),
            "selector_count": len(selection_definitions()),
            "candidate_count": len(rows),
            "late_ensemble": "0.5*candidate_probability + 0.5*full_spectrum_LDA_probability",
        },
        "data_access": {
            "manifest_discovery_hash_files": False,
            "loaded_constructed_batches": loaded_batches,
            "locked_constructed_batches": list(LOCKED_BATCHES),
            "locked_numeric_reads": 0,
            "locked_byte_reads": 0,
            "locked_hash_reads": 0,
            "mat_numeric_reads": 0,
        },
        "data": {
            "n_samples": int(dataset.X.shape[0]),
            "n_bands": int(dataset.X.shape[1]),
            "class_counts": {
                str(int(label)): int(np.sum(dataset.y == label))
                for label in evaluation.classes
            },
            "batch_counts": {
                str(batch): sum(
                    int(record.constructed_batch) == batch for record in dataset.records
                )
                for batch in DEVELOPMENT_BATCHES
            },
            "manifest_structural_fingerprint_sha256": manifest_fingerprint,
            "development_csv_fingerprint_sha256": dataset.loaded_csv_fingerprint_sha256,
        },
        "full_spectrum_sg1_lda": dict(evaluation.baseline_metrics),
        "best_selected_candidate": dict(best_selected),
        "best_equal_ensemble_with_full_lda": dict(best_ensemble),
        "disposition": {
            "enter_primary_method": False,
            "use": "reported development ablation and feature-stability audit only",
            "reason": (
                "The route was screened over 132 same-OOF candidates and did not provide "
                "independent evidence sufficient to change the frozen primary method."
            ),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.4f}%"


def _report_text(summary: Mapping[str, Any]) -> str:
    baseline = summary["full_spectrum_sg1_lda"]
    best = summary["best_selected_candidate"]
    ensemble = summary["best_equal_ensemble_with_full_lda"]
    access = summary["data_access"]
    return "\n".join(
        [
            "# 构造批次稳定选带开发探索",
            "",
            "本分析只服务于酸枣仁高光谱产地溯源。它比较稳定选带是否能在当前开发数据内改进判别，同时保留完整的防泄漏边界。",
            "",
            "## 数据边界",
            "",
            f"- 仅载入构造批次：{access['loaded_constructed_batches']}。",
            f"- 锁定批次数值读取：{access['locked_numeric_reads']}；锁定字节读取：{access['locked_byte_reads']}；MAT数值读取：{access['mat_numeric_reads']}。",
            "- 来源清单使用 `discover_manifest(hash_files=False)`；批次8--9没有参与选带、拟合或OOF评价。",
            "",
            "## 方法",
            "",
            "逐粒平均光谱先执行SG一阶导数（窗口15、二阶多项式）。每个外折完整留出一个构造批次；外折训练部分再逐一去掉其余七个批次之一，只在六批次子集上生成ANOVA或收缩LDA系数排名。七份排名通过固定数量或共识阈值得到该外折的选带结果。随后比较收缩LDA、LR及其与全谱收缩LDA的1:1概率组合。",
            "",
            "## 开发结果",
            "",
            f"- 全谱SG1-收缩LDA：balanced accuracy {_format_percent(float(baseline['balanced_accuracy']))}，折范围 {_format_percent(float(baseline['fold_balanced_accuracy_min']))}--{_format_percent(float(baseline['fold_balanced_accuracy_max']))}。",
            f"- 最佳独立选带候选 `{best['candidate_id']}`：balanced accuracy {_format_percent(float(best['balanced_accuracy']))}，相对全谱LDA {float(best['delta_vs_full_spectrum_lda_pp']):+.4f} 个百分点。",
            f"- 最佳1:1晚期组合来源候选 `{ensemble['candidate_id']}`：balanced accuracy {_format_percent(float(ensemble['ensemble_with_full_lda_balanced_accuracy']))}，相对全谱LDA {float(ensemble['ensemble_delta_vs_full_spectrum_lda_pp']):+.4f} 个百分点。",
            "- `candidates.csv` 给出132个候选的全部指标；`fold_selections.csv`、`intervals.csv`和`selection_frequency.csv`记录外折选带、连续波长区间与选择频率。",
            "",
            "## 证据性质与处置",
            "",
            "132个候选在同一开发OOF上并列比较，报告的最大值存在选择乐观偏差，不是独立确认结果。该路线作为开发消融、稳定性与可解释性负结果保存，不进入冻结主方法，也不能作为修改锁定评估方案的依据。",
            "",
        ]
    )


def write_artifacts(
    output_dir: Path,
    evaluation: StableBandEvaluation,
    *,
    dataset: SpectralDataset,
    manifest_fingerprint: str,
) -> Path:
    """Write the complete, machine-readable development-only audit."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _write_csv(output_path / "candidates.csv", evaluation.candidate_rows)
    _write_csv(output_path / "fold_selections.csv", evaluation.fold_selection_rows)
    _write_csv(output_path / "intervals.csv", evaluation.interval_rows)
    _write_csv(output_path / "selection_frequency.csv", evaluation.frequency_rows)
    summary = _summary_payload(
        evaluation,
        dataset=dataset,
        manifest_fingerprint=manifest_fingerprint,
    )
    summary_path = output_path / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    (output_path / "report.md").write_text(_report_text(summary), encoding="utf-8")
    return summary_path


def run_exploration(data_root: Path, output_dir: Path) -> Path:
    """Discover structure blindly, load batches 0--7, evaluate, and serialize."""

    manifest = discover_manifest(
        Path(data_root),
        expected_source_counts=EXPECTED_SOURCE_COUNTS,
        base_seed=BASE_BATCH_SEED,
        hash_files=False,
    )
    if manifest.hashes_complete:
        raise AssertionError("Development manifest unexpectedly read file bytes")
    development_records = manifest.records_for_split("development")
    if any(
        record.constructed_batch not in DEVELOPMENT_BATCHES
        or record.analysis_split != "development"
        for record in development_records
    ):
        raise LockedDataAccessError("Development record selection contains a locked sample")
    dataset = load_development_csv(
        development_records,
        expected_bands=EXPECTED_BANDS,
        verify_hashes=False,
    )
    groups = np.asarray(
        [record.constructed_batch for record in dataset.records], dtype=np.int64
    )
    evaluation = evaluate_stable_band_candidates(
        dataset.X, dataset.y, groups, dataset.wavelengths
    )
    if len(evaluation.candidate_rows) != EXPECTED_CANDIDATE_COUNT:
        raise AssertionError("Full stable-band run must contain exactly 132 candidates")
    return write_artifacts(
        output_dir,
        evaluation,
        dataset=dataset,
        manifest_fingerprint=manifest.data_fingerprint_sha256,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the development-only stable-band exploration for origin traceability"
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    summary_path = run_exploration(arguments.data_root, arguments.output_dir)
    print(f"Stable-band development exploration completed: {summary_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
