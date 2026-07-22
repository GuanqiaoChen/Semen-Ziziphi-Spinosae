#!/usr/bin/env python3
"""Development-only MAT multi-view exploration for geographical-origin traceability.

The entry point discovers the complete structural manifest with ``hash_files=False``
but numerically opens only constructed batches 0--7.  Both the collection loader
and the single-record MAT reader reject a locked record before the first HDF5 I/O
call.  Batches 8--9 are therefore kept blind.

This is an exploratory companion to :mod:`provenance_study.explore_development`.
It evaluates whether within-seed dispersion, radial contrast, or mask morphology
adds development evidence beyond the fixed SG1 spectral probability ensemble.
Late-fusion weights are compared on the same OOF predictions and are explicitly
non-confirmatory; they cannot determine the primary predictor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from provenance_study.core import (  # noqa: E402
    BASE_BATCH_SEED,
    CLASS_NAMES,
    DEVELOPMENT_BATCHES,
    EXPECTED_BANDS,
    LOCKED_BATCHES,
    LockedDataAccessError,
    Manifest,
    SampleRecord,
    SavitzkyGolayTransformer,
    SpectralDataset,
    build_sg15_logistic_regression,
    build_sg15_rbf_svm,
    build_sg15_shrinkage_lda,
    discover_manifest,
    equal_weight_probability_average,
    grouped_oof_probabilities,
    load_development_csv,
    multiclass_metrics,
    nested_grouped_oof_temperature_probabilities,
)


VIEW_ORDER = (
    "morphology",
    "foreground_std",
    "foreground_iqr",
    "radial_inner_minus_outer",
    "mat_all",
    "sg1_plus_mat_all_early",
)
MAT_ONLY_VIEWS = VIEW_ORDER[:-1]
CLASSIFIER_ORDER = ("shrinkage_lda", "logistic_regression", "rbf_svm", "extra_trees")
SPECTRAL_REFERENCE_IDS = (
    "sg1_spectral_shrinkage_lda",
    "sg1_spectral_logistic_regression",
    "sg1_spectral_rbf_svm_group_temperature",
    "sg1_spectral_probability_ensemble",
)
LATE_SPECTRAL_WEIGHTS = (0.25, 0.50, 0.75)
MORPHOLOGY_FEATURE_NAMES = (
    "area_fraction",
    "boundary_pixels_over_sqrt_area",
    "minor_major_axis_ratio",
    "bounding_box_extent",
    "radial_distance_cv",
)
OUTPUT_FILENAMES = (
    "metrics.csv",
    "fold_metrics.csv",
    "feature_manifest.csv",
    "summary.json",
    "report.md",
)

EXPLORATION_CONFIGURATION: Mapping[str, Any] = {
    "analysis_seed": BASE_BATCH_SEED,
    "outer_validation": "eight_fold_leave_one_constructed_development_batch_out",
    "development_batches": list(DEVELOPMENT_BATCHES),
    "locked_batches": list(LOCKED_BATCHES),
    "sg1": {
        "window_length": 15,
        "polyorder": 2,
        "derivative": 1,
        "axis": 1,
        "mode": "interp",
    },
    "scaling": {
        "lda_lr_svm": "StandardScaler fitted on each outer-training partition only",
        "extra_trees": "none",
    },
    "view_models": {
        "shrinkage_lda": {
            "solver": "lsqr",
            "shrinkage": "auto",
        },
        "logistic_regression": {
            "C": 1.0,
            "solver": "lbfgs",
            "penalty": "l2",
            "max_iter": 4000,
            "tol": 1e-4,
            "class_weight": None,
            "random_state": BASE_BATCH_SEED,
        },
        "rbf_svm": {
            "C": 10.0,
            "kernel": "rbf",
            "gamma": "scale",
            "probability": True,
            "probability_calibration": "libsvm internal Platt scaling inside each outer-training partition",
            "random_state": BASE_BATCH_SEED,
        },
        "extra_trees": {
            "n_estimators": 500,
            "max_features": "sqrt",
            "min_samples_split": 2,
            "min_samples_leaf": 2,
            "max_depth": None,
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": BASE_BATCH_SEED,
        },
    },
    "fixed_spectral_reference": {
        "members": list(SPECTRAL_REFERENCE_IDS[:3]),
        "weights": [1.0 / 3.0] * 3,
        "svm_probability_calibration": "strictly nested grouped temperature scaling",
        "logistic_regression_max_iter": 5000,
    },
    "late_fusion": {
        "spectral_weights": list(LATE_SPECTRAL_WEIGHTS),
        "mat_weight": "one_minus_spectral_weight",
        "selection_data": "same development OOF predictions",
        "role": "exploratory_only_not_primary_model_selection",
    },
}


@dataclass(frozen=True)
class MatViews:
    """Sample-aligned MAT-derived feature matrices and extraction metadata."""

    foreground_std: np.ndarray
    foreground_iqr: np.ndarray
    radial_inner_minus_outer: np.ndarray
    morphology: np.ndarray
    feature_manifest_rows: tuple[dict[str, Any], ...]
    development_mat_numeric_reads: int


@dataclass(frozen=True)
class ProbabilityEvaluation:
    """One grouped OOF probability result plus machine-readable metric rows."""

    model_id: str
    probabilities: np.ndarray
    metric_row: dict[str, Any]
    fold_rows: tuple[dict[str, Any], ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _assert_development_record(record: SampleRecord) -> None:
    """Reject a locked or inconsistent record before any file operation."""

    if (
        record.analysis_split != "development"
        or int(record.constructed_batch) not in DEVELOPMENT_BATCHES
    ):
        raise LockedDataAccessError(
            "MAT development reader received a locked/non-development record "
            f"before I/O: {record.sample_id}, batch={record.constructed_batch}, "
            f"split={record.analysis_split}"
        )


def _preflight_development_records(records: Sequence[SampleRecord]) -> tuple[SampleRecord, ...]:
    checked = tuple(records)
    if not checked:
        raise ValueError("No development MAT records supplied")
    # Validate the complete collection before the first MAT file can be opened.
    for record in checked:
        _assert_development_record(record)
    return checked


def _orient_hwb(raw: np.ndarray, mask: np.ndarray, expected_bands: int, path: Path) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float32).squeeze()
    if values.ndim != 3:
        raise ValueError(f"patch_chw is not three-dimensional in {path}: {values.shape}")
    band_axes = [axis for axis, size in enumerate(values.shape) if size == expected_bands]
    if len(band_axes) != 1:
        raise ValueError(
            f"Cannot identify exactly one {expected_bands}-band axis in {path}: {values.shape}"
        )
    cube = np.moveaxis(values, band_axes[0], -1)
    if cube.shape[:2] != mask.shape and cube.shape[:2][::-1] == mask.shape:
        cube = np.transpose(cube, (1, 0, 2))
    if cube.shape != (*mask.shape, expected_bands):
        raise ValueError(f"Patch/mask mismatch in {path}: cube={cube.shape}, mask={mask.shape}")
    if not np.all(np.isfinite(cube)):
        raise ValueError(f"Non-finite MAT cube value in {path}")
    return cube


def read_development_mat(
    record: SampleRecord,
    *,
    expected_bands: int = EXPECTED_BANDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Read one development MAT patch, with the split guard before HDF5 I/O."""

    _assert_development_record(record)
    if expected_bands <= 0:
        raise ValueError("expected_bands must be positive")
    # Do not move this context manager above the record guard.
    with h5py.File(record.mat_path, "r") as handle:
        if "patch_chw" not in handle or "crop_mask" not in handle:
            raise KeyError(f"Missing patch_chw or crop_mask in {record.mat_path}")
        raw = handle["patch_chw"][()]
        mask = np.asarray(handle["crop_mask"][()]).squeeze() > 0.5
    if mask.ndim != 2 or not np.any(mask):
        raise ValueError(f"Invalid or empty crop_mask in {record.mat_path}")
    return _orient_hwb(raw, mask, expected_bands, record.mat_path), mask


def _mask_morphology(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not np.any(mask):
        raise ValueError("Morphology requires a non-empty two-dimensional mask")
    coordinates = np.argwhere(mask).astype(np.float64)
    area = float(mask.sum())
    boundary = mask & ~binary_erosion(mask)
    centered = coordinates - coordinates.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1.0, area - 1.0)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    axis_ratio = float(
        np.sqrt(max(float(eigenvalues[1]), 0.0) / max(float(eigenvalues[0]), 1e-12))
    )
    height = float(coordinates[:, 0].max() - coordinates[:, 0].min() + 1.0)
    width = float(coordinates[:, 1].max() - coordinates[:, 1].min() + 1.0)
    radii = np.sqrt(np.sum(centered**2, axis=1))
    radial_cv = float(radii.std(ddof=1) / max(float(radii.mean()), 1e-12))
    features = np.asarray(
        [
            area / float(mask.size),
            float(boundary.sum()) / math.sqrt(area),
            axis_ratio,
            area / (height * width),
            radial_cv,
        ],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(features)):
        raise ValueError("Mask morphology produced non-finite features")
    return features


def extract_mat_feature_views(
    cube: np.ndarray,
    mask: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Extract the frozen dispersion, radial-contrast, and morphology views."""

    cube = np.asarray(cube, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if cube.ndim != 3 or mask.ndim != 2 or cube.shape[:2] != mask.shape:
        raise ValueError("cube must be H x W x B and align with a two-dimensional mask")
    if not np.all(np.isfinite(cube)) or not np.any(mask):
        raise ValueError("cube must be finite and mask must be non-empty")
    foreground = np.asarray(cube[mask], dtype=np.float64)
    if foreground.shape[0] < 8:
        raise ValueError("At least eight foreground pixels are required")

    foreground_std = foreground.std(axis=0, ddof=1).astype(np.float32)
    q25, q75 = np.percentile(foreground, (25.0, 75.0), axis=0)
    foreground_iqr = (q75 - q25).astype(np.float32)

    distance = distance_transform_edt(mask)
    median_distance = float(np.median(distance[mask]))
    inner = mask & (distance > median_distance)
    outer = mask & ~inner
    if int(inner.sum()) < 4 or int(outer.sum()) < 4:
        eroded = binary_erosion(mask, iterations=2)
        erosion_outer = mask & ~eroded
        if int(eroded.sum()) < 4 or int(erosion_outer.sum()) < 4:
            raise ValueError("Cannot define stable inner and outer foreground regions")
        inner = eroded
        outer = erosion_outer
    radial = (cube[inner].mean(axis=0) - cube[outer].mean(axis=0)).astype(np.float32)
    morphology = _mask_morphology(mask)
    views = {
        "foreground_std": foreground_std,
        "foreground_iqr": foreground_iqr,
        "radial_inner_minus_outer": radial,
        "morphology": morphology,
    }
    if any(not np.all(np.isfinite(values)) for values in views.values()):
        raise ValueError("MAT feature extraction produced non-finite values")
    counts = {
        "foreground_pixels": int(mask.sum()),
        "inner_pixels": int(inner.sum()),
        "outer_pixels": int(outer.sum()),
    }
    return views, counts


def _feature_sha256(views: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in ("foreground_std", "foreground_iqr", "radial_inner_minus_outer", "morphology"):
        values = np.ascontiguousarray(views[name], dtype=np.float32)
        digest.update(name.encode("ascii"))
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def load_development_mat_views(
    records: Sequence[SampleRecord],
    *,
    expected_bands: int = EXPECTED_BANDS,
) -> MatViews:
    """Load every development MAT once after collection-wide split preflight."""

    checked = _preflight_development_records(records)
    std_rows: list[np.ndarray] = []
    iqr_rows: list[np.ndarray] = []
    radial_rows: list[np.ndarray] = []
    morphology_rows: list[np.ndarray] = []
    manifest_rows: list[dict[str, Any]] = []
    for record in checked:
        cube, mask = read_development_mat(record, expected_bands=expected_bands)
        views, counts = extract_mat_feature_views(cube, mask)
        std_rows.append(views["foreground_std"])
        iqr_rows.append(views["foreground_iqr"])
        radial_rows.append(views["radial_inner_minus_outer"])
        morphology_rows.append(views["morphology"])
        manifest_rows.append(
            {
                "sample_index": record.sample_index,
                "sample_id": record.sample_id,
                "label": record.label,
                "class_code": record.class_name,
                "source_cube": record.source_cube,
                "source_replicate": record.replicate,
                "seed_id": record.seed_id,
                "constructed_batch": record.constructed_batch,
                "analysis_split": record.analysis_split,
                "relative_mat_path": record.relative_mat_path,
                "mat_path_sha256": record.mat_path_sha256,
                "mat_size_bytes": record.mat_size_bytes,
                **counts,
                "foreground_std_features": expected_bands,
                "foreground_iqr_features": expected_bands,
                "radial_features": expected_bands,
                "morphology_features": len(MORPHOLOGY_FEATURE_NAMES),
                "extracted_mat_feature_sha256": _feature_sha256(views),
                "mat_content_sha256": "not_computed_in_blind_development",
            }
        )
    return MatViews(
        foreground_std=np.asarray(std_rows, dtype=np.float32),
        foreground_iqr=np.asarray(iqr_rows, dtype=np.float32),
        radial_inner_minus_outer=np.asarray(radial_rows, dtype=np.float32),
        morphology=np.asarray(morphology_rows, dtype=np.float32),
        feature_manifest_rows=tuple(manifest_rows),
        development_mat_numeric_reads=len(checked),
    )


def build_view_matrices(mean_spectra: np.ndarray, mat_views: MatViews) -> dict[str, np.ndarray]:
    """Construct the six frozen single/combined views in sample-aligned order."""

    mean_spectra = np.asarray(mean_spectra, dtype=np.float64)
    if mean_spectra.ndim != 2:
        raise ValueError("mean_spectra must be a sample-by-band matrix")
    n_samples, n_bands = mean_spectra.shape
    expected_shapes = {
        "foreground_std": (n_samples, n_bands),
        "foreground_iqr": (n_samples, n_bands),
        "radial_inner_minus_outer": (n_samples, n_bands),
        "morphology": (n_samples, len(MORPHOLOGY_FEATURE_NAMES)),
    }
    observed = {
        "foreground_std": mat_views.foreground_std.shape,
        "foreground_iqr": mat_views.foreground_iqr.shape,
        "radial_inner_minus_outer": mat_views.radial_inner_minus_outer.shape,
        "morphology": mat_views.morphology.shape,
    }
    if observed != expected_shapes:
        raise ValueError(f"MAT/CSV feature alignment mismatch: {observed} != {expected_shapes}")
    sg1 = SavitzkyGolayTransformer(15, 2, 1).fit_transform(mean_spectra)
    mat_all = np.column_stack(
        [
            mat_views.foreground_std,
            mat_views.foreground_iqr,
            mat_views.radial_inner_minus_outer,
            mat_views.morphology,
        ]
    )
    matrices = {
        "morphology": mat_views.morphology,
        "foreground_std": mat_views.foreground_std,
        "foreground_iqr": mat_views.foreground_iqr,
        "radial_inner_minus_outer": mat_views.radial_inner_minus_outer,
        "mat_all": mat_all,
        "sg1_plus_mat_all_early": np.column_stack([sg1, mat_all]),
    }
    if tuple(matrices) != VIEW_ORDER:
        raise AssertionError("View ordering drifted")
    return {name: np.asarray(values, dtype=np.float64) for name, values in matrices.items()}


def build_view_estimators(
    *,
    analysis_seed: int = BASE_BATCH_SEED,
) -> dict[str, Any]:
    """Build the exact four classifiers used in the executed development screen."""

    return {
        "shrinkage_lda": Pipeline(
            [
                ("standardize", StandardScaler()),
                ("classifier", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("standardize", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        solver="lbfgs",
                        max_iter=4000,
                        tol=1e-4,
                        random_state=int(analysis_seed),
                    ),
                ),
            ]
        ),
        "rbf_svm": Pipeline(
            [
                ("standardize", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        C=10.0,
                        kernel="rbf",
                        gamma="scale",
                        probability=True,
                        random_state=int(analysis_seed),
                    ),
                ),
            ]
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=int(analysis_seed),
        ),
    }


def _batch_ids(dataset: SpectralDataset) -> np.ndarray:
    if dataset.analysis_split != "development":
        raise LockedDataAccessError("MAT-view evaluation received a non-development dataset")
    if any(
        record.analysis_split != "development"
        or record.constructed_batch not in DEVELOPMENT_BATCHES
        for record in dataset.records
    ):
        raise LockedDataAccessError("MAT-view evaluation contains a locked record")
    groups = np.asarray(
        [record.constructed_batch for record in dataset.records], dtype=np.int64
    )
    if set(groups.tolist()) != set(DEVELOPMENT_BATCHES):
        raise ValueError("Development data must cover constructed batches 0--7 exactly")
    return groups


def _metric_and_fold_rows(
    *,
    model_id: str,
    role: str,
    view: str,
    classifier: str,
    probabilities: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    classes: np.ndarray,
    ece_bins: int,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    fold_rows: list[dict[str, Any]] = []
    for batch in DEVELOPMENT_BATCHES:
        selected = groups == batch
        values = multiclass_metrics(
            y[selected], probabilities[selected], classes=classes, ece_bins=ece_bins
        )
        fold_rows.append(
            {
                "model_id": model_id,
                "role": role,
                "view": view,
                "classifier": classifier,
                "held_out_batch": batch,
                **values,
                "errors": int(selected.sum() - round(float(values["accuracy"]) * selected.sum())),
            }
        )
    overall = multiclass_metrics(y, probabilities, classes=classes, ece_bins=ece_bins)
    predictions = classes[np.asarray(probabilities).argmax(axis=1)]
    row = {
        "model_id": model_id,
        "role": role,
        "view": view,
        "classifier": classifier,
        **overall,
        "equal_constructed_batch_accuracy": float(
            np.mean([float(fold["accuracy"]) for fold in fold_rows])
        ),
        "fold_balanced_accuracy_min": float(
            min(float(fold["balanced_accuracy"]) for fold in fold_rows)
        ),
        "fold_balanced_accuracy_max": float(
            max(float(fold["balanced_accuracy"]) for fold in fold_rows)
        ),
        "fold_balanced_accuracy_mean": float(
            np.mean([float(fold["balanced_accuracy"]) for fold in fold_rows])
        ),
        "fold_balanced_accuracy_sd": float(
            np.std(
                [float(fold["balanced_accuracy"]) for fold in fold_rows], ddof=1
            )
        ),
        "errors": int(np.sum(predictions != y)),
        "same_oof_selected": 0,
    }
    return row, tuple(fold_rows)


def _evaluate_probability_matrix(
    *,
    model_id: str,
    role: str,
    view: str,
    classifier: str,
    probabilities: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    classes: np.ndarray,
    ece_bins: int,
) -> ProbabilityEvaluation:
    metric, folds = _metric_and_fold_rows(
        model_id=model_id,
        role=role,
        view=view,
        classifier=classifier,
        probabilities=np.asarray(probabilities, dtype=np.float64),
        y=y,
        groups=groups,
        classes=classes,
        ece_bins=ece_bins,
    )
    return ProbabilityEvaluation(model_id, np.asarray(probabilities), metric, folds)


def evaluate_mat_view_models(
    matrices: Mapping[str, np.ndarray],
    dataset: SpectralDataset,
    *,
    analysis_seed: int = BASE_BATCH_SEED,
    ece_bins: int = 10,
) -> dict[str, ProbabilityEvaluation]:
    """Evaluate all 6 x 4 frozen view/model combinations by grouped OOF."""

    groups = _batch_ids(dataset)
    y = np.asarray(dataset.y, dtype=np.int64)
    classes = np.arange(len(CLASS_NAMES), dtype=np.int64)
    if set(y.tolist()) != set(classes.tolist()):
        raise ValueError("Development data must contain all eight origin labels")
    estimators = build_view_estimators(analysis_seed=analysis_seed)
    evaluations: dict[str, ProbabilityEvaluation] = {}
    for view_name in VIEW_ORDER:
        if view_name not in matrices:
            raise ValueError(f"Missing view matrix: {view_name}")
        for classifier_name in CLASSIFIER_ORDER:
            model_id = f"{view_name}__{classifier_name}"
            oof = grouped_oof_probabilities(
                estimators[classifier_name], matrices[view_name], y, groups
            )
            evaluations[model_id] = _evaluate_probability_matrix(
                model_id=model_id,
                role="development_mat_view_candidate",
                view=view_name,
                classifier=classifier_name,
                probabilities=oof.probabilities,
                y=y,
                groups=groups,
                classes=classes,
                ece_bins=ece_bins,
            )
    return evaluations


def evaluate_fixed_spectral_references(
    dataset: SpectralDataset,
    *,
    analysis_seed: int = BASE_BATCH_SEED,
    ece_bins: int = 10,
) -> tuple[dict[str, ProbabilityEvaluation], dict[str, Any]]:
    """Reproduce the fixed SG1 LDA/LR/group-temperature SVM ensemble."""

    groups = _batch_ids(dataset)
    X = np.asarray(dataset.X, dtype=np.float64)
    y = np.asarray(dataset.y, dtype=np.int64)
    classes = np.arange(len(CLASS_NAMES), dtype=np.int64)
    lda = grouped_oof_probabilities(build_sg15_shrinkage_lda(), X, y, groups)
    lr = grouped_oof_probabilities(
        build_sg15_logistic_regression(
            C=1.0,
            max_iter=5000,
            tol=1e-4,
            random_state=int(analysis_seed),
        ),
        X,
        y,
        groups,
    )
    svm = nested_grouped_oof_temperature_probabilities(
        build_sg15_rbf_svm(), X, y, groups
    )
    matrices = {
        SPECTRAL_REFERENCE_IDS[0]: lda.probabilities,
        SPECTRAL_REFERENCE_IDS[1]: lr.probabilities,
        SPECTRAL_REFERENCE_IDS[2]: svm.probabilities,
    }
    matrices[SPECTRAL_REFERENCE_IDS[3]] = equal_weight_probability_average(
        [matrices[name] for name in SPECTRAL_REFERENCE_IDS[:3]]
    )
    evaluations: dict[str, ProbabilityEvaluation] = {}
    classifier_names = (
        "shrinkage_lda",
        "logistic_regression",
        "rbf_svm_group_temperature",
        "equal_probability_ensemble",
    )
    for model_id, classifier_name in zip(SPECTRAL_REFERENCE_IDS, classifier_names):
        evaluations[model_id] = _evaluate_probability_matrix(
            model_id=model_id,
            role="fixed_spectral_reference",
            view="sg1_mean_spectrum",
            classifier=classifier_name,
            probabilities=matrices[model_id],
            y=y,
            groups=groups,
            classes=classes,
            ece_bins=ece_bins,
        )
    calibration = {
        "method": "strictly_nested_grouped_temperature_scaling",
        "outer_fold_temperatures": [
            {"held_out_batch": int(batch), "temperature": float(value)}
            for batch, value in svm.fold_temperatures
        ],
        "deployment_temperature": float(svm.final_temperature),
    }
    return evaluations, calibration


def evaluate_same_oof_late_fusions(
    mat_evaluations: Mapping[str, ProbabilityEvaluation],
    spectral_ensemble: ProbabilityEvaluation,
    dataset: SpectralDataset,
    *,
    ece_bins: int = 10,
) -> tuple[dict[str, ProbabilityEvaluation], str]:
    """Enumerate late weights on the same OOF rows; return exploratory winner ID."""

    groups = _batch_ids(dataset)
    y = np.asarray(dataset.y, dtype=np.int64)
    classes = np.arange(len(CLASS_NAMES), dtype=np.int64)
    candidates: dict[str, ProbabilityEvaluation] = {}
    for model_id, evaluation in mat_evaluations.items():
        if evaluation.metric_row["view"] not in MAT_ONLY_VIEWS:
            continue
        for spectral_weight in LATE_SPECTRAL_WEIGHTS:
            probability = (
                spectral_weight * spectral_ensemble.probabilities
                + (1.0 - spectral_weight) * evaluation.probabilities
            )
            late_id = f"late_w{spectral_weight:.2f}__{model_id}"
            result = _evaluate_probability_matrix(
                model_id=late_id,
                role="same_oof_late_fusion_exploratory_only",
                view=str(evaluation.metric_row["view"]),
                classifier=f"spectral_ensemble_plus_{evaluation.metric_row['classifier']}",
                probabilities=probability,
                y=y,
                groups=groups,
                classes=classes,
                ece_bins=ece_bins,
            )
            result.metric_row["late_spectral_weight"] = spectral_weight
            result.metric_row["late_mat_weight"] = 1.0 - spectral_weight
            candidates[late_id] = result
    if not candidates:
        raise ValueError("No MAT-only evaluations supplied for late fusion")
    winner = max(
        candidates,
        key=lambda name: (
            float(candidates[name].metric_row["balanced_accuracy"]),
            float(candidates[name].metric_row["macro_f1"]),
            -float(candidates[name].metric_row["negative_log_likelihood"]),
        ),
    )
    candidates[winner].metric_row["same_oof_selected"] = 1
    return candidates, winner


def _best_evaluation(
    evaluations: Mapping[str, ProbabilityEvaluation],
    *,
    allowed_views: Sequence[str],
) -> ProbabilityEvaluation:
    selected = [
        evaluation
        for evaluation in evaluations.values()
        if str(evaluation.metric_row["view"]) in set(allowed_views)
    ]
    if not selected:
        raise ValueError("No evaluation matches the requested views")
    return max(
        selected,
        key=lambda evaluation: (
            float(evaluation.metric_row["balanced_accuracy"]),
            float(evaluation.metric_row["macro_f1"]),
            -float(evaluation.metric_row["negative_log_likelihood"]),
        ),
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_artifacts(
    output_dir: Path,
    *,
    metric_rows: Sequence[Mapping[str, Any]],
    fold_rows: Sequence[Mapping[str, Any]],
    feature_manifest_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    report: str,
    overwrite: bool = False,
) -> None:
    """Write exactly the five declared artifacts, refusing accidental overwrite."""

    output_dir = Path(output_dir)
    existing = [output_dir / name for name in OUTPUT_FILENAMES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "MAT-view output already exists; pass --overwrite to replace declared files: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "metrics.csv", metric_rows)
    _write_csv(output_dir / "fold_metrics.csv", fold_rows)
    _write_csv(output_dir / "feature_manifest.csv", feature_manifest_rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def build_chinese_report(summary: Mapping[str, Any]) -> str:
    """Render the development-only result and its non-confirmatory boundary."""

    spectral = summary["fixed_spectral_ensemble"]
    mat_only = summary["best_mat_only"]
    early = summary["best_early_fusion"]
    late = summary["best_same_oof_late_fusion"]

    def percentage(value: Any) -> str:
        return f"{100.0 * float(value):.2f}%"

    conclusion = summary["primary_conclusion"]
    return f"""# MAT 多视图开发期探索报告

## 研究问题与边界

本分析只服务于酸枣仁高光谱地理产地溯源，检验逐粒 MAT 中的前景离散度、径向谱差和掩膜形态是否为固定 SG1 均值光谱组合提供开发期增益。构造批次 8–9 保持锁定，未读取其 CSV/MAT 数值或文件字节。

## 开发期结果

| 比较对象 | 模型 | balanced accuracy | 八折范围 | macro-F1 | log-loss |
|---|---|---:|---:|---:|---:|
| 固定光谱组合 | {spectral['model_id']} | {percentage(spectral['balanced_accuracy'])} | {percentage(spectral['fold_balanced_accuracy_min'])}–{percentage(spectral['fold_balanced_accuracy_max'])} | {percentage(spectral['macro_f1'])} | {float(spectral['negative_log_likelihood']):.4f} |
| 最佳 MAT 单独视图 | {mat_only['model_id']} | {percentage(mat_only['balanced_accuracy'])} | {percentage(mat_only['fold_balanced_accuracy_min'])}–{percentage(mat_only['fold_balanced_accuracy_max'])} | {percentage(mat_only['macro_f1'])} | {float(mat_only['negative_log_likelihood']):.4f} |
| 最佳早期融合 | {early['model_id']} | {percentage(early['balanced_accuracy'])} | {percentage(early['fold_balanced_accuracy_min'])}–{percentage(early['fold_balanced_accuracy_max'])} | {percentage(early['macro_f1'])} | {float(early['negative_log_likelihood']):.4f} |
| 同 OOF 权重筛选的晚期融合 | {late['model_id']} | {percentage(late['balanced_accuracy'])} | {percentage(late['fold_balanced_accuracy_min'])}–{percentage(late['fold_balanced_accuracy_max'])} | {percentage(late['macro_f1'])} | {float(late['negative_log_likelihood']):.4f} |

主结论：{conclusion}

晚期融合在相同 OOF 预测上同时比较了多个 MAT 候选和三个权重，只属于探索性筛选；即使其点估计更高，也不能据此替换或评价固定主模型。

## 捷径风险

- 构造批次仍共享两张来源图像，来源图像的照明和采集指纹可能跨折存在。
- 前景标准差、IQR、径向谱差和掩膜形态可能编码分割质量、姿态、裁剪流程或照明，而不一定是地理来源机制。
- 构造批次不是新增农场、收获年份、物理批次、设备或外部采集数据。
- 固定光谱组合的 SVM 使用严格嵌套分组温度校准；MAT 候选的 SVM 保留开发筛选时的训练折内 libsvm Platt 概率，仅用于探索。
"""


def run_mat_view_exploration(
    *,
    data_root: Path,
    config_path: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Execute the full locked-blind MAT development exploration."""

    started = time.perf_counter()
    started_utc = _utc_now()
    config = _load_json(config_path)
    analysis_seed = int(config["analysis_seed"])
    expected_bands = int(config["expected_bands"])
    ece_bins = int(config["evaluation"]["ece_bins"])
    if analysis_seed != BASE_BATCH_SEED:
        raise ValueError("MAT exploration requires the frozen constructed-batch seed")
    if expected_bands != EXPECTED_BANDS:
        raise ValueError("MAT exploration requires the frozen 392-band grid")

    manifest: Manifest = discover_manifest(
        data_root,
        base_seed=analysis_seed,
        hash_files=False,
    )
    if manifest.hashes_complete or manifest.csv_content_sha256 or manifest.mat_content_sha256:
        raise AssertionError("Blind discovery unexpectedly read file content")
    development_records = manifest.records_for_split("development")
    locked_records = manifest.records_for_split("locked")
    dataset = load_development_csv(
        development_records,
        expected_bands=expected_bands,
        verify_hashes=False,
    )
    mat_started = time.perf_counter()
    mat_views = load_development_mat_views(
        development_records,
        expected_bands=expected_bands,
    )
    mat_read_seconds = time.perf_counter() - mat_started
    if tuple(record.sample_id for record in dataset.records) != tuple(
        row["sample_id"] for row in mat_views.feature_manifest_rows
    ):
        raise AssertionError("CSV and MAT development sample ordering differs")

    matrices = build_view_matrices(dataset.X, mat_views)
    mat_evaluations = evaluate_mat_view_models(
        matrices,
        dataset,
        analysis_seed=analysis_seed,
        ece_bins=ece_bins,
    )
    spectral_references, spectral_calibration = evaluate_fixed_spectral_references(
        dataset,
        analysis_seed=analysis_seed,
        ece_bins=ece_bins,
    )
    spectral_ensemble = spectral_references[SPECTRAL_REFERENCE_IDS[3]]
    late_evaluations, late_winner_id = evaluate_same_oof_late_fusions(
        mat_evaluations,
        spectral_ensemble,
        dataset,
        ece_bins=ece_bins,
    )
    best_mat = _best_evaluation(mat_evaluations, allowed_views=MAT_ONLY_VIEWS)
    best_early = _best_evaluation(
        mat_evaluations, allowed_views=("sg1_plus_mat_all_early",)
    )
    best_late = late_evaluations[late_winner_id]

    all_evaluations = [
        *mat_evaluations.values(),
        *spectral_references.values(),
        *late_evaluations.values(),
    ]
    metric_rows = [evaluation.metric_row for evaluation in all_evaluations]
    fold_rows = [row for evaluation in all_evaluations for row in evaluation.fold_rows]
    access_audit = {
        "manifest_discovery": "discover_manifest(hash_files=False)",
        "development_samples": len(development_records),
        "development_csv_numeric_reads": len(development_records),
        "development_mat_numeric_reads": mat_views.development_mat_numeric_reads,
        "locked_paths_enumerated": len(locked_records),
        "locked_csv_numeric_reads": 0,
        "locked_mat_numeric_reads": 0,
        "locked_byte_reads": 0,
        "locked_csv_hashes": 0,
        "locked_mat_hashes": 0,
        "collection_preflight_before_mat_io": True,
        "single_record_guard_before_hdf5_io": True,
    }
    spectral_row = dict(spectral_ensemble.metric_row)
    mat_row = dict(best_mat.metric_row)
    early_row = dict(best_early.metric_row)
    late_row = dict(best_late.metric_row)
    early_exceeds_spectral = float(early_row["balanced_accuracy"]) > float(
        spectral_row["balanced_accuracy"]
    )
    conclusion = (
        "预先定义的 MAT 早期融合超过了固定光谱组合；该差异仍只属于开发期证据。"
        if early_exceeds_spectral
        else "预先定义的 MAT 早期融合未超过固定光谱组合，当前没有可靠的 MAT 增益。"
    )
    summary: dict[str, Any] = {
        "status": "newly_executed_development_only",
        "scientific_objective": "hyperspectral_geographical_origin_traceability_of_semen_ziziphi_spinosae",
        "started_utc": started_utc,
        "completed_utc": _utc_now(),
        "duration_seconds": float(time.perf_counter() - started),
        "mat_feature_read_seconds": float(mat_read_seconds),
        "analysis_units": "deterministically_constructed_batches",
        "development_batches": list(DEVELOPMENT_BATCHES),
        "locked_batches": list(LOCKED_BATCHES),
        "structural_manifest_sha256": manifest.manifest_sha256,
        "structural_assignment_fingerprint_sha256": manifest.data_fingerprint_sha256,
        "full_manifest_content_hashing_performed": False,
        "feature_dimensions": {name: int(values.shape[1]) for name, values in matrices.items()},
        "feature_definitions": {
            "foreground_std": "per-band sample SD across mask foreground pixels (ddof=1)",
            "foreground_iqr": "per-band foreground P75 minus P25",
            "radial_inner_minus_outer": "per-band inner mean minus outer mean; inner distance-transform values exceed the within-mask median",
            "morphology": list(MORPHOLOGY_FEATURE_NAMES),
            "mat_all": "foreground_std + foreground_iqr + radial_inner_minus_outer + morphology",
            "sg1_plus_mat_all_early": "SG(15,2,1) CSV mean spectrum + mat_all",
        },
        "configuration": EXPLORATION_CONFIGURATION,
        "access_audit": access_audit,
        "svm_spectral_temperature_calibration": spectral_calibration,
        "fixed_spectral_ensemble": spectral_row,
        "best_mat_only": mat_row,
        "best_early_fusion": early_row,
        "best_same_oof_late_fusion": late_row,
        "best_same_oof_late_fusion_warning": (
            "Selected across all MAT-only candidates and weights on the same development OOF rows; "
            "not an unbiased model comparison and not eligible to replace the fixed primary predictor."
        ),
        "balanced_accuracy_delta_early_minus_spectral": float(
            early_row["balanced_accuracy"] - spectral_row["balanced_accuracy"]
        ),
        "balanced_accuracy_delta_late_minus_spectral_descriptive_only": float(
            late_row["balanced_accuracy"] - spectral_row["balanced_accuracy"]
        ),
        "mat_early_exceeded_fixed_spectral_ensemble": early_exceeds_spectral,
        "primary_conclusion": conclusion,
        "limitations": [
            "constructed batches share the same two source images per origin",
            "MAT views may encode segmentation, pose, crop, or illumination shortcuts",
            "same-OOF late-fusion selection is optimistically selected",
            "no unseen physical lot, year, farm, device, or external site is available",
        ],
    }
    report = build_chinese_report(summary)
    write_artifacts(
        output_dir,
        metric_rows=metric_rows,
        fold_rows=fold_rows,
        feature_manifest_rows=mat_views.feature_manifest_rows,
        summary=summary,
        report=report,
        overwrite=overwrite,
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    package_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Explore development-only MAT views for hyperspectral origin traceability"
    )
    parser.add_argument("--data-root", type=Path, default=repository_root / "data")
    parser.add_argument("--config", type=Path, default=package_root / "config.json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=package_root / "outputs" / "development_mat",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_mat_view_exploration(
        data_root=args.data_root,
        config_path=args.config,
        output_dir=args.output_dir,
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
