"""One-shot locked evaluation for the current-data origin-traceability study.

The module is deliberately inert when imported.  The CLI checks an exact
confirmation phrase, a canonical completion marker, and the tracked Git state
before manifest discovery or any data-file access.  Once authorized, batches
8--9 are loaded once, their probability predictions are generated once per
frozen model, and every reported table is derived from those stored
probabilities.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import sklearn
import torch
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from provenance_study.core import (
    BASE_BATCH_SEED,
    CLASS_NAMES,
    DEVELOPMENT_BATCHES,
    EXPECTED_BANDS,
    LOCKED_BATCHES,
    StandardNormalVariate,
    build_sg15_logistic_regression,
    build_sg15_rbf_svm,
    build_sg15_shrinkage_lda,
    decision_scores_to_probabilities,
    discover_manifest,
    equal_weight_probability_average,
    fit_decision_temperature,
    grouped_oof_decision_scores,
    load_csv_split,
    multiclass_metrics,
    sha256_file,
)
from provenance_study.cnn_baseline import (
    CNN_PARAMETER_COUNT,
    FittedCNN,
    fit_full_development_cnn,
)


CONFIRMATION_PHRASE = "UNLOCK_BATCHES_8_9"
STATUS_FILENAME = "execution_status.json"
COMPLETE_STATE = "executed_complete"
MODEL_ORDER = (
    "raw_pls_da",
    "snv_logistic_regression",
    "sg1_shrinkage_lda",
    "sg1_logistic_regression",
    "sg1_rbf_svm_group_temperature",
    "residual_1d_cnn_reference",
    "batch_constrained_sg1_probability_ensemble",
)
PRIMARY_MODEL = "batch_constrained_sg1_probability_ensemble"
PRIMARY_BASELINE = "sg1_shrinkage_lda"
MODEL_ROLES = {
    "raw_pls_da": "strong_baseline",
    "snv_logistic_regression": "strong_baseline",
    "sg1_shrinkage_lda": "strong_single_model_baseline",
    "sg1_logistic_regression": "ensemble_member",
    "sg1_rbf_svm_group_temperature": "ensemble_member",
    "residual_1d_cnn_reference": "secondary_modern_spectral_reference",
    "batch_constrained_sg1_probability_ensemble": "primary_frozen_predictor",
}
TABLE_CONTEXT_FIELDS = (
    "run_id",
    "git_head",
    "config_sha256",
    "manifest_sha256",
    "data_fingerprint_sha256",
)


class LockedTestConfirmationError(PermissionError):
    """Raised before any data access when the exact unlock phrase is absent."""


class CompletedEvaluationError(RuntimeError):
    """Raised when the canonical output already records a completed execution."""


@dataclass(frozen=True)
class EvaluationTables:
    """Recomputable result tables derived only from stored prediction rows."""

    metrics: tuple[dict[str, Any], ...]
    batch_metrics: tuple[dict[str, Any], ...]
    class_metrics: tuple[dict[str, Any], ...]
    confusion: tuple[dict[str, Any], ...]
    predictions_by_model: Mapping[str, np.ndarray]
    probabilities_by_model: Mapping[str, np.ndarray]
    y_true: np.ndarray
    constructed_batches: np.ndarray


class PLSDAProbabilityClassifier(BaseEstimator):
    """Frozen raw-spectrum PLS-DA with softmax-normalized response scores.

    One-hot PLS regression followed by argmax is the fixed discrimination rule.
    Applying softmax preserves that argmax while supplying finite normalized
    probabilities needed for the pre-specified calibration metrics.
    """

    def __init__(self, n_components: int = 20, max_iter: int = 1000) -> None:
        self.n_components = n_components
        self.max_iter = max_iter

    def fit(self, X: np.ndarray, y: Sequence[int]) -> "PLSDAProbabilityClassifier":
        features = np.asarray(X, dtype=np.float64)
        labels = np.asarray(y, dtype=np.int64)
        if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.size:
            raise ValueError("X and y have incompatible shapes")
        self.classes_ = np.unique(labels)
        if self.classes_.size < 2:
            raise ValueError("PLS-DA requires at least two classes")
        if self.n_components <= 0 or self.n_components > min(features.shape):
            raise ValueError("n_components is incompatible with the training matrix")
        class_positions = {int(label): index for index, label in enumerate(self.classes_)}
        responses = np.zeros((labels.size, self.classes_.size), dtype=np.float64)
        responses[
            np.arange(labels.size),
            np.asarray([class_positions[int(label)] for label in labels]),
        ] = 1.0
        self.pipeline_ = Pipeline(
            [
                ("standardize", StandardScaler()),
                (
                    "pls_regression",
                    PLSRegression(
                        n_components=int(self.n_components),
                        max_iter=int(self.max_iter),
                    ),
                ),
            ]
        )
        self.pipeline_.fit(features, responses)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "pipeline_"):
            raise RuntimeError("PLSDAProbabilityClassifier must be fitted first")
        scores = np.asarray(
            self.pipeline_.predict(np.asarray(X, dtype=np.float64)), dtype=np.float64
        )
        if scores.ndim != 2 or scores.shape[1] != self.classes_.size:
            raise ValueError("PLS regression returned an invalid response matrix")
        return scores

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        shifted = scores - scores.max(axis=1, keepdims=True)
        exponentiated = np.exp(shifted)
        return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def require_confirmation(value: str | None) -> None:
    """Reject without touching configuration, output, Git, or data paths."""

    if value != CONFIRMATION_PHRASE:
        raise LockedTestConfirmationError(
            "Locked batches remain closed. Supply exactly "
            f"--confirm-locked-test {CONFIRMATION_PHRASE}."
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(
        _json_ready(dict(payload)), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    )
    _atomic_write_text(path, text + "\n")


def _read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def refuse_completed_output(output_dir: Path) -> None:
    """Protect a canonical completed result from accidental second execution."""

    status = _read_json_object(Path(output_dir) / STATUS_FILENAME)
    results = _read_json_object(Path(output_dir) / "results.json")
    if (status and status.get("state") == COMPLETE_STATE) or (
        results and results.get("execution_state") == COMPLETE_STATE
    ):
        raise CompletedEvaluationError(
            f"Canonical locked evaluation is already complete: {Path(output_dir).resolve()}"
        )


def _run_git(repo_root: Path, arguments: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Git command failed ({' '.join(arguments)}): {detail}")
    return result.stdout.strip()


def assert_tracked_worktree_clean(repo_root: Path) -> str:
    """Return HEAD only when tracked/index state is clean; ignore untracked files."""

    repo_root = Path(repo_root).resolve()
    tracked_status = _run_git(repo_root, ("status", "--porcelain", "--untracked-files=no"))
    if tracked_status:
        raise RuntimeError(
            "Locked evaluation requires a clean tracked worktree and index; "
            f"observed:\n{tracked_status}"
        )
    head = _run_git(repo_root, ("rev-parse", "HEAD"))
    if len(head) != 40:
        raise RuntimeError(f"Unexpected Git HEAD identifier: {head!r}")
    return head


def environment_snapshot() -> dict[str, Any]:
    """Capture interpreter, platform, numerical-library, and installed-package state."""

    distributions: list[str] = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name") or "unknown"
        distributions.append(f"{name}=={distribution.version}")
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "installed_distributions": sorted(set(distributions), key=str.casefold),
    }


def _load_config(config_path: Path) -> dict[str, Any]:
    value = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Configuration root must be a JSON object")
    return value


def validate_frozen_config(config: Mapping[str, Any]) -> None:
    """Fail closed if code-facing frozen protocol values have drifted."""

    expected_values = {
        "analysis_seed": BASE_BATCH_SEED,
        "expected_bands": EXPECTED_BANDS,
        "class_codes": list(CLASS_NAMES),
    }
    for key, expected in expected_values.items():
        if config.get(key) != expected:
            raise ValueError(f"Frozen configuration mismatch for {key}: {config.get(key)!r}")
    batches = config["constructed_batches"]
    if batches["development_indices"] != list(DEVELOPMENT_BATCHES):
        raise ValueError("Frozen development batch indices have changed")
    if batches["locked_test_indices"] != list(LOCKED_BATCHES):
        raise ValueError("Frozen locked batch indices have changed")
    preprocessing = config["preprocessing"]
    if (
        preprocessing["savgol_window_length"],
        preprocessing["savgol_polyorder"],
        preprocessing["savgol_derivative"],
        preprocessing["standard_scaler"],
    ) != (15, 2, 1, True):
        raise ValueError("Frozen SG1 preprocessing has changed")
    primary = config["primary_predictor"]
    if primary["members"] != [
        "sg1_shrinkage_lda",
        "sg1_logistic_regression",
        "sg1_rbf_svm_group_temperature",
    ]:
        raise ValueError("Frozen ensemble members have changed")
    if not np.allclose(
        np.asarray(primary["probability_weights"], dtype=float),
        np.repeat(1.0 / 3.0, 3),
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("Frozen ensemble weights are not exactly equal")
    evaluation = config["evaluation"]
    if evaluation["primary_metric"] != "equal_constructed_batch_accuracy":
        raise ValueError("Frozen primary metric has changed")
    if int(evaluation["cluster_bootstrap_repetitions"]) != 10_000:
        raise ValueError("Frozen bootstrap repetition count has changed")
    gate = config["effect_gate"]
    required_gate_fields = (
        "paired_cluster_interval_lower_bound_must_exceed_zero",
        "minimum_relative_balanced_error_reduction",
        "maximum_allowed_log_loss_increase",
        "maximum_allowed_brier_increase",
        "minimum_strictly_improved_classes",
        "minimum_strictly_improved_constructed_batches",
        "leave_one_constructed_batch_out_effect_must_remain_positive",
    )
    missing = [field for field in required_gate_fields if field not in gate]
    if missing:
        raise ValueError(f"Frozen effect-gate fields are missing: {missing}")
    cnn = config["cnn_reference"]
    if (
        cnn["architecture"],
        int(cnn["parameter_count"]),
        int(cnn["full_development_epochs"]),
        int(cnn["training_seed"]),
        cnn["role"],
    ) != (
        "three_stage_residual_1d_cnn",
        CNN_PARAMETER_COUNT,
        88,
        BASE_BATCH_SEED,
        "secondary_modern_spectral_reference",
    ):
        raise ValueError("Frozen CNN reference configuration has changed")


def validate_probability_matrix(probabilities: np.ndarray, n_classes: int) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != n_classes:
        raise ValueError("Probability matrix has an invalid shape")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("Probabilities must be finite and lie in [0, 1]")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("Every probability row must sum to one")
    return values


def _align_columns(
    values: np.ndarray, estimator_classes: Sequence[int], classes: np.ndarray
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    observed_classes = np.asarray(estimator_classes, dtype=np.int64)
    positions = {int(label): index for index, label in enumerate(observed_classes)}
    if set(positions) != set(classes.tolist()):
        raise ValueError("Fitted model classes do not match the frozen class order")
    if matrix.ndim != 2 or matrix.shape[1] != observed_classes.size:
        raise ValueError("Model output has an invalid class dimension")
    return matrix[:, [positions[int(label)] for label in classes]]


def build_snv_logistic_regression(config: Mapping[str, Any]) -> Pipeline:
    values = config["models"]["snv_logistic_regression"]
    return Pipeline(
        [
            ("snv", StandardNormalVariate()),
            ("standardize", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(values["C"]),
                    solver=str(values["solver"]),
                    max_iter=int(values["max_iter"]),
                    tol=float(values["tol"]),
                    random_state=int(config["analysis_seed"]),
                ),
            ),
        ]
    )


def fit_frozen_models_and_predict_once(
    X_development: np.ndarray,
    y_development: np.ndarray,
    development_batches: np.ndarray,
    X_locked: np.ndarray,
    config: Mapping[str, Any],
    *,
    cnn_fit_function: Any = fit_full_development_cnn,
) -> tuple[dict[str, np.ndarray], float, FittedCNN]:
    """Fit on development and invoke each locked-set output method exactly once."""

    classes = np.asarray(range(len(config["class_codes"])), dtype=np.int64)
    if not np.array_equal(np.unique(y_development), classes):
        raise ValueError("Development split does not contain exactly the frozen classes")

    svm = build_sg15_rbf_svm()
    svm_oof = grouped_oof_decision_scores(
        svm,
        X_development,
        y_development,
        development_batches,
        group_order=tuple(config["constructed_batches"]["development_indices"]),
    )
    fitted_temperature = fit_decision_temperature(
        svm_oof.decision_scores,
        y_development,
        classes=classes,
        log_temperature_bounds=tuple(
            config["models"]["sg1_rbf_svm"]["temperature_log_bounds"]
        ),
    )
    frozen_temperature = float(
        config["primary_predictor"]["svm_temperature_frozen_development_value"]
    )
    if not math.isclose(fitted_temperature, frozen_temperature, rel_tol=0.0, abs_tol=5e-10):
        raise AssertionError(
            "Re-fitted development SVM temperature does not reproduce the frozen value: "
            f"observed={fitted_temperature:.12g}, frozen={frozen_temperature:.12g}"
        )

    pls_values = config["models"]["raw_pls_da"]
    estimators: dict[str, BaseEstimator] = {
        "raw_pls_da": PLSDAProbabilityClassifier(
            n_components=int(pls_values["n_components"]),
            max_iter=int(pls_values["max_iter"]),
        ),
        "snv_logistic_regression": build_snv_logistic_regression(config),
        "sg1_shrinkage_lda": build_sg15_shrinkage_lda(),
        "sg1_logistic_regression": build_sg15_logistic_regression(),
        "sg1_rbf_svm_group_temperature": svm,
    }
    probabilities: dict[str, np.ndarray] = {}
    for model_name, estimator in estimators.items():
        fitted = estimator.fit(X_development, y_development)
        if model_name == "sg1_rbf_svm_group_temperature":
            # This is the sole locked-set SVM output call.
            locked_scores = _align_columns(
                fitted.decision_function(X_locked), fitted.classes_, classes
            )
            model_probabilities = decision_scores_to_probabilities(
                locked_scores, fitted_temperature
            )
        else:
            # This is the sole locked-set probability call for this model.
            model_probabilities = _align_columns(
                fitted.predict_proba(X_locked), fitted.classes_, classes
            )
        probabilities[model_name] = validate_probability_matrix(
            model_probabilities, classes.size
        )

    cnn_config = config["cnn_reference"]
    fitted_cnn = cnn_fit_function(
        X_development,
        y_development,
        epochs=int(cnn_config["full_development_epochs"]),
        optimization_seed=int(cnn_config["training_seed"]),
    )
    if int(getattr(fitted_cnn, "epochs")) != int(cnn_config["full_development_epochs"]):
        raise AssertionError("Fitted CNN did not use the frozen epoch count")
    if int(getattr(fitted_cnn, "optimization_seed")) != int(cnn_config["training_seed"]):
        raise AssertionError("Fitted CNN did not use the frozen optimization seed")
    if int(sum(parameter.numel() for parameter in fitted_cnn.model.parameters())) != int(
        cnn_config["parameter_count"]
    ):
        raise AssertionError("Fitted CNN parameter count differs from the frozen architecture")
    # This is the sole locked-set CNN output call.
    cnn_probabilities = fitted_cnn.predict_proba(X_locked)
    probabilities["residual_1d_cnn_reference"] = validate_probability_matrix(
        _align_columns(cnn_probabilities, fitted_cnn.classes, classes), classes.size
    )

    probabilities[PRIMARY_MODEL] = validate_probability_matrix(
        equal_weight_probability_average(
            [
                probabilities["sg1_shrinkage_lda"],
                probabilities["sg1_logistic_regression"],
                probabilities["sg1_rbf_svm_group_temperature"],
            ]
        ),
        classes.size,
    )
    if tuple(probabilities) != MODEL_ORDER:
        raise AssertionError("Frozen model output order drifted")
    return probabilities, fitted_temperature, fitted_cnn


def save_cnn_reference_checkpoint(
    fitted: FittedCNN,
    path: Path,
    *,
    run_context: Mapping[str, Any],
) -> None:
    """Save a weights-only-compatible CNN state dictionary and primitive metadata."""

    payload = {
        "format_version": 1,
        "model_class": "ResidualSpectralCNN",
        "model_role": MODEL_ROLES["residual_1d_cnn_reference"],
        "state_dict": {
            name: tensor.detach().cpu() for name, tensor in fitted.model.state_dict().items()
        },
        "standardizer_mean": torch.as_tensor(fitted.standardizer.mean).detach().cpu(),
        "standardizer_scale": torch.as_tensor(fitted.standardizer.scale).detach().cpu(),
        "standardizer_n_samples_seen": int(fitted.standardizer.n_samples_seen),
        "classes": torch.as_tensor(fitted.classes, dtype=torch.int64).detach().cpu(),
        "optimization_seed": int(fitted.optimization_seed),
        "epochs": int(fitted.epochs),
        "raw_band_count": int(fitted.raw_band_count),
        "training_config": asdict(fitted.training_config),
        "run_context": {str(key): str(value) for key, value in run_context.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def make_prediction_rows(
    probabilities_by_model: Mapping[str, np.ndarray],
    y_true: Sequence[int],
    sample_metadata: Sequence[Mapping[str, Any]],
    class_codes: Sequence[str],
) -> list[dict[str, Any]]:
    """Create the lossless seed-level table from which every metric is rebuilt."""

    labels = np.asarray(y_true, dtype=np.int64)
    codes = tuple(str(code) for code in class_codes)
    if labels.ndim != 1 or len(sample_metadata) != labels.size:
        raise ValueError("Labels and sample metadata have incompatible lengths")
    rows: list[dict[str, Any]] = []
    for model_name, probabilities in probabilities_by_model.items():
        values = validate_probability_matrix(probabilities, len(codes))
        if values.shape[0] != labels.size:
            raise ValueError("Probability and label row counts differ")
        predictions = values.argmax(axis=1).astype(np.int64)
        for row_index, metadata in enumerate(sample_metadata):
            true_label = int(labels[row_index])
            predicted_label = int(predictions[row_index])
            row: dict[str, Any] = {
                "model": model_name,
                "model_role": MODEL_ROLES.get(model_name, "evaluated_predictor"),
                "sample_order": row_index,
                "sample_index": int(metadata["sample_index"]),
                "sample_id": str(metadata["sample_id"]),
                "source_cube": str(metadata["source_cube"]),
                "source_replicate": int(metadata["source_replicate"]),
                "seed_id": str(metadata["seed_id"]),
                "relative_csv_path": str(metadata["relative_csv_path"]),
                "constructed_batch": int(metadata["constructed_batch"]),
                "true_label": true_label,
                "true_class": codes[true_label],
                "predicted_label": predicted_label,
                "predicted_class": codes[predicted_label],
                "correct": int(predicted_label == true_label),
            }
            for class_index, code in enumerate(codes):
                row[f"probability_{code}"] = float(values[row_index, class_index])
            rows.append(row)
    return rows


def recompute_evaluation_tables(
    prediction_rows: Sequence[Mapping[str, Any]],
    class_codes: Sequence[str],
    *,
    ece_bins: int = 10,
) -> EvaluationTables:
    """Recompute all descriptive tables directly from seed-level probability rows."""

    if not prediction_rows:
        raise ValueError("Prediction rows are empty")
    codes = tuple(str(code) for code in class_codes)
    classes = np.arange(len(codes), dtype=np.int64)
    rows_by_model: dict[str, list[Mapping[str, Any]]] = {}
    for row in prediction_rows:
        rows_by_model.setdefault(str(row["model"]), []).append(row)

    metrics_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    predictions_by_model: dict[str, np.ndarray] = {}
    probabilities_by_model: dict[str, np.ndarray] = {}
    canonical_identity: list[tuple[int, int, int]] | None = None
    canonical_labels: np.ndarray | None = None
    canonical_batches: np.ndarray | None = None

    for model_name, unsorted_rows in rows_by_model.items():
        rows = sorted(unsorted_rows, key=lambda item: int(item["sample_order"]))
        identities = [
            (
                int(row["sample_index"]),
                int(row["true_label"]),
                int(row["constructed_batch"]),
            )
            for row in rows
        ]
        if canonical_identity is None:
            canonical_identity = identities
        elif identities != canonical_identity:
            raise ValueError("Models do not share an identical locked sample ordering")
        y_true = np.asarray([int(row["true_label"]) for row in rows], dtype=np.int64)
        batches = np.asarray([int(row["constructed_batch"]) for row in rows], dtype=np.int64)
        probabilities = np.asarray(
            [[float(row[f"probability_{code}"]) for code in codes] for row in rows],
            dtype=np.float64,
        )
        probabilities = validate_probability_matrix(probabilities, len(codes))
        predictions = probabilities.argmax(axis=1).astype(np.int64)
        stored_predictions = np.asarray(
            [int(row["predicted_label"]) for row in rows], dtype=np.int64
        )
        if not np.array_equal(predictions, stored_predictions):
            raise ValueError(f"Stored predictions disagree with probabilities for {model_name}")
        observed_metrics = multiclass_metrics(
            y_true, probabilities, classes=classes, ece_bins=ece_bins
        )

        cluster_accuracies: list[float] = []
        for label in classes:
            for batch in sorted(np.unique(batches).tolist()):
                mask = (y_true == label) & (batches == batch)
                if not np.any(mask):
                    raise ValueError(f"Missing label-by-batch cluster: label={label}, batch={batch}")
                accuracy = float(np.mean(predictions[mask] == y_true[mask]))
                cluster_accuracies.append(accuracy)
                batch_rows.append(
                    {
                        "model": model_name,
                        "model_role": MODEL_ROLES.get(model_name, "evaluated_predictor"),
                        "label": int(label),
                        "class_code": codes[int(label)],
                        "constructed_batch": int(batch),
                        "n": int(mask.sum()),
                        "correct": int(np.sum(predictions[mask] == y_true[mask])),
                        "accuracy": accuracy,
                    }
                )
        metrics_rows.append(
            {
                "model": model_name,
                "model_role": MODEL_ROLES.get(model_name, "evaluated_predictor"),
                "equal_constructed_batch_accuracy": float(np.mean(cluster_accuracies)),
                **observed_metrics,
            }
        )
        for label in classes:
            mask = y_true == label
            class_rows.append(
                {
                    "model": model_name,
                    "model_role": MODEL_ROLES.get(model_name, "evaluated_predictor"),
                    "label": int(label),
                    "class_code": codes[int(label)],
                    "n": int(mask.sum()),
                    "correct": int(np.sum(predictions[mask] == label)),
                    "recall": float(np.mean(predictions[mask] == label)),
                }
            )
        matrix = confusion_matrix(y_true, predictions, labels=classes)
        for true_label in classes:
            for predicted_label in classes:
                confusion_rows.append(
                    {
                        "model": model_name,
                        "model_role": MODEL_ROLES.get(model_name, "evaluated_predictor"),
                        "true_label": int(true_label),
                        "true_class": codes[int(true_label)],
                        "predicted_label": int(predicted_label),
                        "predicted_class": codes[int(predicted_label)],
                        "count": int(matrix[int(true_label), int(predicted_label)]),
                    }
                )
        predictions_by_model[model_name] = predictions
        probabilities_by_model[model_name] = probabilities
        canonical_labels = y_true
        canonical_batches = batches

    assert canonical_labels is not None and canonical_batches is not None
    return EvaluationTables(
        metrics=tuple(metrics_rows),
        batch_metrics=tuple(batch_rows),
        class_metrics=tuple(class_rows),
        confusion=tuple(confusion_rows),
        predictions_by_model=predictions_by_model,
        probabilities_by_model=probabilities_by_model,
        y_true=canonical_labels,
        constructed_batches=canonical_batches,
    )


def cluster_accuracy_matrices(
    batch_metric_rows: Sequence[Mapping[str, Any]],
    model_order: Sequence[str],
) -> tuple[dict[str, np.ndarray], tuple[int, ...], tuple[int, ...]]:
    """Return model-by-label-by-batch accuracy arrays with strict completeness checks."""

    labels = tuple(sorted({int(row["label"]) for row in batch_metric_rows}))
    batches = tuple(sorted({int(row["constructed_batch"]) for row in batch_metric_rows}))
    matrices: dict[str, np.ndarray] = {}
    for model in model_order:
        matrix = np.full((len(labels), len(batches)), np.nan, dtype=np.float64)
        for row in batch_metric_rows:
            if str(row["model"]) != model:
                continue
            label_position = labels.index(int(row["label"]))
            batch_position = batches.index(int(row["constructed_batch"]))
            if np.isfinite(matrix[label_position, batch_position]):
                raise ValueError(f"Duplicate cluster metric for {model}")
            matrix[label_position, batch_position] = float(row["accuracy"])
        if np.any(~np.isfinite(matrix)):
            raise ValueError(f"Incomplete label-by-batch matrix for {model}")
        matrices[model] = matrix
    return matrices, labels, batches


def stratified_cluster_bootstrap(
    cluster_matrices: Mapping[str, np.ndarray],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Resample the locked batches within every origin using common paired draws."""

    if repetitions <= 0:
        raise ValueError("Bootstrap repetitions must be positive")
    if not cluster_matrices:
        raise ValueError("At least one cluster matrix is required")
    shapes = {np.asarray(matrix).shape for matrix in cluster_matrices.values()}
    if len(shapes) != 1:
        raise ValueError("All models must share one cluster-matrix shape")
    n_labels, n_batches = next(iter(shapes))
    if n_labels <= 0 or n_batches < 2:
        raise ValueError("Stratified bootstrap requires labels and at least two batches")
    rng = np.random.default_rng(int(seed))
    sampled_batch_positions = rng.integers(
        0, n_batches, size=(int(repetitions), n_labels, n_batches)
    )
    estimates: dict[str, np.ndarray] = {}
    for model, matrix_value in cluster_matrices.items():
        matrix = np.asarray(matrix_value, dtype=np.float64)
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Cluster accuracies must be finite")
        expanded = np.broadcast_to(matrix, (int(repetitions), n_labels, n_batches))
        sampled = np.take_along_axis(expanded, sampled_batch_positions, axis=2)
        estimates[model] = sampled.mean(axis=(1, 2))
    return estimates


def exact_paired_sign_flip(differences: Sequence[float]) -> dict[str, Any]:
    """Enumerate the exact paired sign-flip reference distribution."""

    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("differences must be a non-empty finite vector")
    if values.size > 22:
        raise ValueError("Exact sign-flip enumeration is limited to 22 paired units")
    pattern_ids = np.arange(1 << values.size, dtype=np.uint64)[:, None]
    bit_positions = np.arange(values.size, dtype=np.uint64)[None, :]
    signs = np.where((pattern_ids >> bit_positions) & 1, 1.0, -1.0)
    null_effects = (signs * values[None, :]).mean(axis=1)
    observed = float(values.mean())
    tolerance = 1e-15
    return {
        "n_clusters": int(values.size),
        "n_sign_patterns": int(null_effects.size),
        "observed_mean_difference": observed,
        "two_sided_p_value": float(
            np.mean(np.abs(null_effects) >= abs(observed) - tolerance)
        ),
        "one_sided_greater_p_value": float(
            np.mean(null_effects >= observed - tolerance)
        ),
        "strictly_positive_clusters": int(np.sum(values > 0.0)),
        "strictly_negative_clusters": int(np.sum(values < 0.0)),
        "zero_difference_clusters": int(np.sum(values == 0.0)),
    }


def evaluate_effect_gate(
    ensemble_metrics: Mapping[str, float],
    baseline_metrics: Mapping[str, float],
    ensemble_class_recalls: Sequence[float],
    baseline_class_recalls: Sequence[float],
    paired_cluster_differences: Sequence[float],
    bootstrap_difference_interval: tuple[float, float],
    gate_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the four frozen gate conditions without data-dependent substitution."""

    required = (
        "paired_cluster_interval_lower_bound_must_exceed_zero",
        "minimum_relative_balanced_error_reduction",
        "maximum_allowed_log_loss_increase",
        "maximum_allowed_brier_increase",
        "minimum_strictly_improved_classes",
        "minimum_strictly_improved_constructed_batches",
        "leave_one_constructed_batch_out_effect_must_remain_positive",
    )
    missing = [field for field in required if field not in gate_config]
    if missing:
        raise KeyError(f"Effect-gate configuration is incomplete: {missing}")

    ensemble_ba = float(ensemble_metrics["balanced_accuracy"])
    baseline_ba = float(baseline_metrics["balanced_accuracy"])
    baseline_error = 1.0 - baseline_ba
    relative_error_reduction = (
        (ensemble_ba - baseline_ba) / baseline_error
        if baseline_error > 0.0
        else (0.0 if ensemble_ba == baseline_ba else float("-inf"))
    )
    nll_increase = float(ensemble_metrics["negative_log_likelihood"]) - float(
        baseline_metrics["negative_log_likelihood"]
    )
    brier_increase = float(ensemble_metrics["multiclass_brier_score"]) - float(
        baseline_metrics["multiclass_brier_score"]
    )
    class_differences = np.asarray(ensemble_class_recalls, dtype=float) - np.asarray(
        baseline_class_recalls, dtype=float
    )
    cluster_differences = np.asarray(paired_cluster_differences, dtype=float)
    if class_differences.ndim != 1 or cluster_differences.ndim != 1:
        raise ValueError("Gate recall and cluster differences must be vectors")
    if cluster_differences.size < 2:
        raise ValueError("Leave-one-cluster analysis requires at least two clusters")
    leave_one_effects = (
        cluster_differences.sum() - cluster_differences
    ) / (cluster_differences.size - 1)
    improved_classes = int(np.sum(class_differences > 0.0))
    improved_clusters = int(np.sum(cluster_differences > 0.0))

    require_positive_lower = bool(
        gate_config["paired_cluster_interval_lower_bound_must_exceed_zero"]
    )
    require_positive_leave_one = bool(
        gate_config["leave_one_constructed_batch_out_effect_must_remain_positive"]
    )
    conditions = [
        {
            "condition": 1,
            "criterion": "paired_cluster_bootstrap_interval_lower_bound",
            "observed": float(bootstrap_difference_interval[0]),
            "threshold": "> 0" if require_positive_lower else "not required",
            "passed": bool(
                (not require_positive_lower) or bootstrap_difference_interval[0] > 0.0
            ),
        },
        {
            "condition": 2,
            "criterion": "relative_balanced_error_reduction",
            "observed": relative_error_reduction,
            "threshold": float(
                gate_config["minimum_relative_balanced_error_reduction"]
            ),
            "passed": bool(
                relative_error_reduction
                >= float(gate_config["minimum_relative_balanced_error_reduction"])
            ),
        },
        {
            "condition": 3,
            "criterion": "probability_quality_noninferiority",
            "observed": json.dumps(
                {"nll_increase": nll_increase, "brier_increase": brier_increase},
                sort_keys=True,
            ),
            "threshold": json.dumps(
                {
                    "maximum_nll_increase": float(
                        gate_config["maximum_allowed_log_loss_increase"]
                    ),
                    "maximum_brier_increase": float(
                        gate_config["maximum_allowed_brier_increase"]
                    ),
                },
                sort_keys=True,
            ),
            "passed": bool(
                nll_increase <= float(gate_config["maximum_allowed_log_loss_increase"])
                and brier_increase
                <= float(gate_config["maximum_allowed_brier_increase"])
            ),
        },
        {
            "condition": 4,
            "criterion": "breadth_and_leave_one_cluster_robustness",
            "observed": json.dumps(
                {
                    "strictly_improved_classes": improved_classes,
                    "strictly_improved_label_batch_clusters": improved_clusters,
                    "minimum_leave_one_cluster_out_effect": float(leave_one_effects.min()),
                },
                sort_keys=True,
            ),
            "threshold": json.dumps(
                {
                    "minimum_strictly_improved_classes": int(
                        gate_config["minimum_strictly_improved_classes"]
                    ),
                    "minimum_strictly_improved_label_batch_clusters": int(
                        gate_config["minimum_strictly_improved_constructed_batches"]
                    ),
                    "all_leave_one_cluster_out_effects_positive": require_positive_leave_one,
                },
                sort_keys=True,
            ),
            "passed": bool(
                improved_classes
                >= int(gate_config["minimum_strictly_improved_classes"])
                and improved_clusters
                >= int(gate_config["minimum_strictly_improved_constructed_batches"])
                and (
                    (not require_positive_leave_one)
                    or bool(np.all(leave_one_effects > 0.0))
                )
            ),
        },
    ]
    summary = {
        "all_four_conditions_passed": bool(all(row["passed"] for row in conditions)),
        "relative_balanced_error_reduction": relative_error_reduction,
        "nll_increase": nll_increase,
        "brier_increase": brier_increase,
        "strictly_improved_classes": improved_classes,
        "strictly_improved_label_batch_clusters": improved_clusters,
        "leave_one_cluster_out_effects": leave_one_effects.tolist(),
        "minimum_leave_one_cluster_out_effect": float(leave_one_effects.min()),
    }
    return conditions, summary


def error_transition_counts(
    y_true: Sequence[int], baseline_predictions: Sequence[int], ensemble_predictions: Sequence[int]
) -> dict[str, int]:
    labels = np.asarray(y_true, dtype=np.int64)
    baseline = np.asarray(baseline_predictions, dtype=np.int64)
    ensemble = np.asarray(ensemble_predictions, dtype=np.int64)
    if labels.shape != baseline.shape or labels.shape != ensemble.shape:
        raise ValueError("Error-transition vectors have incompatible shapes")
    baseline_correct = baseline == labels
    ensemble_correct = ensemble == labels
    return {
        "corrected_errors": int(np.sum(~baseline_correct & ensemble_correct)),
        "new_errors": int(np.sum(baseline_correct & ~ensemble_correct)),
        "both_correct": int(np.sum(baseline_correct & ensemble_correct)),
        "both_incorrect": int(np.sum(~baseline_correct & ~ensemble_correct)),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV table: {path.name}")
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _decorate_rows(
    rows: Sequence[Mapping[str, Any]], context: Mapping[str, Any]
) -> list[dict[str, Any]]:
    return [{**{field: context[field] for field in TABLE_CONTEXT_FIELDS}, **dict(row)} for row in rows]


def _metrics_by_model(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["model"]): dict(row) for row in rows}


def _class_recalls(
    rows: Sequence[Mapping[str, Any]], model: str
) -> np.ndarray:
    selected = sorted(
        (row for row in rows if str(row["model"]) == model),
        key=lambda row: int(row["label"]),
    )
    return np.asarray([float(row["recall"]) for row in selected], dtype=np.float64)


def _build_report(results: Mapping[str, Any]) -> str:
    primary = results["metrics"][PRIMARY_MODEL]
    baseline = results["metrics"][PRIMARY_BASELINE]
    interval = results["paired_difference_bootstrap"]
    gate = results["effect_gate"]
    conclusion = (
        "四项结果前门槛全部满足，可在当前数据内部表述为相对强单模型基线的稳定且实质性改进。"
        if gate["all_four_conditions_passed"]
        else "四项结果前门槛未全部满足；等权集成仅作为候选组合如实报告，不宣称已证明稳定且实质性改进。"
    )
    return (
        "# 构造批次锁定评估执行报告\n\n"
        f"- 执行状态：`{COMPLETE_STATE}`\n"
        f"- Git 提交：`{results['git_head']}`\n"
        f"- 数据指纹：`{results['data_fingerprint_sha256']}`\n"
        f"- 锁定样本数：{results['locked_n']}\n"
        f"- 复现的开发集 SVM 温度：{results['svm_temperature']:.10f}\n\n"
        "## 主要结果\n\n"
        f"主方法 16 个“产地×锁定构造批次”正确率等权平均为 "
        f"{primary['equal_constructed_batch_accuracy']:.4%}，SG1-收缩 LDA 为 "
        f"{baseline['equal_constructed_batch_accuracy']:.4%}，差值为 "
        f"{interval['observed_difference']:.4%}（分层 cluster bootstrap 95% 区间 "
        f"{interval['lower']:.4%} 至 {interval['upper']:.4%}）。\n\n"
        f"{conclusion}\n\n"
        "## 解释边界\n\n"
        "该评估只支持现有数据及其导师认可的构造批次内部的八产地判别。构造批次并非新增农场、年份、收获批次或设备，结果不能替代外部独立验证。\n"
    )


def _resolve_from_repo(repo_root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (repo_root / value).resolve()


def run_locked_evaluation(args: argparse.Namespace) -> Path:
    """Run the authorized one-shot workflow; the confirmation check is first."""

    require_confirmation(args.confirm_locked_test)
    repo_root = Path(args.repo_root).resolve()
    output_dir = _resolve_from_repo(repo_root, Path(args.output_dir))
    refuse_completed_output(output_dir)
    git_head = assert_tracked_worktree_clean(repo_root)

    config_path = _resolve_from_repo(repo_root, Path(args.config))
    data_root = _resolve_from_repo(repo_root, Path(args.data_root))
    config_sha256 = sha256_file(config_path)
    config = _load_config(config_path)
    validate_frozen_config(config)
    started_at = _utc_now()
    run_id = f"{started_at.replace(':', '').replace('+00:00', 'Z')}_{git_head[:12]}"
    status_path = output_dir / STATUS_FILENAME
    status: dict[str, Any] = {
        "state": "executing_running",
        "run_id": run_id,
        "started_at_utc": started_at,
        "completed_at_utc": None,
        "git_head": git_head,
        "command": shlex.join([sys.executable, *sys.argv]),
        "confirmation_phrase_sha256": hashlib.sha256(
            CONFIRMATION_PHRASE.encode("utf-8")
        ).hexdigest(),
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "output_dir": str(output_dir),
        "environment": environment_snapshot(),
    }
    _atomic_write_json(status_path, status)

    try:
        # This is intentionally the first discovery/hash/read of anything below data_root.
        manifest = discover_manifest(
            data_root,
            base_seed=int(config["analysis_seed"]),
            hash_files=True,
        )
        if not manifest.hashes_complete:
            raise AssertionError("Authorized manifest hashing was not completed")
        development = load_csv_split(
            manifest,
            split="development",
            expected_bands=int(config["expected_bands"]),
            verify_hashes=True,
        )
        locked = load_csv_split(
            manifest,
            split="locked",
            expected_bands=int(config["expected_bands"]),
            verify_hashes=True,
        )
        if not np.allclose(
            development.wavelengths, locked.wavelengths, rtol=0.0, atol=1e-6
        ):
            raise ValueError("Development and locked wavelength grids differ")
        development_groups = np.asarray(
            [record.constructed_batch for record in development.records], dtype=np.int64
        )
        (
            probabilities_by_model,
            fitted_temperature,
            fitted_cnn,
        ) = fit_frozen_models_and_predict_once(
            development.X,
            development.y,
            development_groups,
            locked.X,
            config,
        )
        sample_metadata = [
            {
                "sample_index": record.sample_index,
                "sample_id": record.sample_id,
                "source_cube": record.source_cube,
                "source_replicate": record.replicate,
                "seed_id": record.seed_id,
                "relative_csv_path": record.relative_csv_path,
                "constructed_batch": record.constructed_batch,
            }
            for record in locked.records
        ]
        prediction_rows = make_prediction_rows(
            probabilities_by_model,
            locked.y,
            sample_metadata,
            config["class_codes"],
        )
        tables = recompute_evaluation_tables(
            prediction_rows,
            config["class_codes"],
            ece_bins=int(config["evaluation"]["ece_bins"]),
        )
        matrices, labels, batches = cluster_accuracy_matrices(
            tables.batch_metrics, MODEL_ORDER
        )
        if batches != tuple(config["constructed_batches"]["locked_test_indices"]):
            raise ValueError(f"Observed locked batches differ from the frozen pair: {batches}")
        bootstrap_repetitions = int(
            config["evaluation"]["cluster_bootstrap_repetitions"]
        )
        bootstrap_seed = int(config["analysis_seed"]) + 31_337
        bootstrap_values = stratified_cluster_bootstrap(
            matrices,
            repetitions=bootstrap_repetitions,
            seed=bootstrap_seed,
        )
        contrast_values = bootstrap_values[PRIMARY_MODEL] - bootstrap_values[PRIMARY_BASELINE]
        confidence_level = float(config["evaluation"]["confidence_level"])
        alpha = 1.0 - confidence_level
        lower, upper = np.quantile(contrast_values, [alpha / 2.0, 1.0 - alpha / 2.0])
        cluster_differences = (
            matrices[PRIMARY_MODEL] - matrices[PRIMARY_BASELINE]
        ).reshape(-1)
        sign_flip = exact_paired_sign_flip(cluster_differences)
        metrics_lookup = _metrics_by_model(tables.metrics)
        gate_rows, gate_summary = evaluate_effect_gate(
            metrics_lookup[PRIMARY_MODEL],
            metrics_lookup[PRIMARY_BASELINE],
            _class_recalls(tables.class_metrics, PRIMARY_MODEL),
            _class_recalls(tables.class_metrics, PRIMARY_BASELINE),
            cluster_differences,
            (float(lower), float(upper)),
            config["effect_gate"],
        )
        transitions = error_transition_counts(
            tables.y_true,
            tables.predictions_by_model[PRIMARY_BASELINE],
            tables.predictions_by_model[PRIMARY_MODEL],
        )

        context = {
            "run_id": run_id,
            "git_head": git_head,
            "config_sha256": config_sha256,
            "manifest_sha256": manifest.manifest_sha256,
            "data_fingerprint_sha256": manifest.data_fingerprint_sha256,
        }
        checkpoint_filename = "cnn_reference_state.pt"
        save_cnn_reference_checkpoint(
            fitted_cnn,
            output_dir / checkpoint_filename,
            run_context=context,
        )
        manifest_rows = [
            {
                "sample_index": record.sample_index,
                "sample_id": record.sample_id,
                "label": record.label,
                "class_code": record.class_name,
                "source_replicate": record.replicate,
                "source_cube": record.source_cube,
                "seed_id": record.seed_id,
                "constructed_batch": record.constructed_batch,
                "analysis_split": record.analysis_split,
                "relative_csv_path": record.relative_csv_path,
                "relative_mat_path": record.relative_mat_path,
                "csv_size_bytes": record.csv_size_bytes,
                "mat_size_bytes": record.mat_size_bytes,
                "csv_sha256": record.csv_sha256,
                "mat_sha256": record.mat_sha256,
                "record_sha256": record.record_sha256,
            }
            for record in manifest.records
        ]
        wavelength_rows = [
            {"band_index": index, "wavelength_nm": float(wavelength)}
            for index, wavelength in enumerate(locked.wavelengths)
        ]
        bootstrap_rows: list[dict[str, Any]] = []
        for repetition in range(bootstrap_repetitions):
            for model in MODEL_ORDER:
                bootstrap_rows.append(
                    {
                        "repetition": repetition,
                        "statistic": "model_equal_constructed_batch_accuracy",
                        "model": model,
                        "baseline": "",
                        "bootstrap_seed": bootstrap_seed,
                        "value": float(bootstrap_values[model][repetition]),
                    }
                )
            bootstrap_rows.append(
                {
                    "repetition": repetition,
                    "statistic": "paired_primary_minus_lda_difference",
                    "model": PRIMARY_MODEL,
                    "baseline": PRIMARY_BASELINE,
                    "bootstrap_seed": bootstrap_seed,
                    "value": float(contrast_values[repetition]),
                }
            )
        effect_test_rows = [
            {
                "test": "exact_paired_sign_flip",
                "contrast": f"{PRIMARY_MODEL}-minus-{PRIMARY_BASELINE}",
                **sign_flip,
                **transitions,
            }
        ]
        artifact_rows: dict[str, list[dict[str, Any]]] = {
            "manifest.csv": _decorate_rows(manifest_rows, context),
            "predictions.csv": _decorate_rows(prediction_rows, context),
            "metrics.csv": _decorate_rows(tables.metrics, context),
            "batch_metrics.csv": _decorate_rows(tables.batch_metrics, context),
            "class_metrics.csv": _decorate_rows(tables.class_metrics, context),
            "confusion.csv": _decorate_rows(tables.confusion, context),
            "bootstrap.csv": _decorate_rows(bootstrap_rows, context),
            "effect_test.csv": _decorate_rows(effect_test_rows, context),
            "effect_gate.csv": _decorate_rows(gate_rows, context),
            "wavelengths.csv": _decorate_rows(wavelength_rows, context),
        }
        for filename, rows in artifact_rows.items():
            _write_csv(output_dir / filename, rows)

        observed_difference = float(cluster_differences.mean())
        results: dict[str, Any] = {
            "execution_state": COMPLETE_STATE,
            "run_id": run_id,
            "started_at_utc": started_at,
            "completed_at_utc": _utc_now(),
            "git_head": git_head,
            "config_sha256": config_sha256,
            "manifest_sha256": manifest.manifest_sha256,
            "csv_content_sha256": manifest.csv_content_sha256,
            "mat_content_sha256": manifest.mat_content_sha256,
            "data_fingerprint_sha256": manifest.data_fingerprint_sha256,
            "development_n": int(development.y.size),
            "locked_n": int(locked.y.size),
            "labels": labels,
            "locked_batches": batches,
            "svm_temperature": fitted_temperature,
            "model_roles": MODEL_ROLES,
            "cnn_reference_checkpoint": {
                "filename": checkpoint_filename,
                "role": MODEL_ROLES["residual_1d_cnn_reference"],
                "epochs": int(fitted_cnn.epochs),
                "optimization_seed": int(fitted_cnn.optimization_seed),
                "parameter_count": int(
                    sum(parameter.numel() for parameter in fitted_cnn.model.parameters())
                ),
                "safe_payload": "state_dict tensors and primitive metadata only",
            },
            "metrics": metrics_lookup,
            "paired_difference_bootstrap": {
                "contrast": f"{PRIMARY_MODEL}-minus-{PRIMARY_BASELINE}",
                "observed_difference": observed_difference,
                "repetitions": bootstrap_repetitions,
                "bootstrap_seed": bootstrap_seed,
                "confidence_level": confidence_level,
                "lower": float(lower),
                "upper": float(upper),
                "resampling": "two locked constructed batches resampled within each origin",
            },
            "exact_sign_flip": sign_flip,
            "error_transitions": transitions,
            "effect_gate": gate_summary,
            "effect_gate_conditions": gate_rows,
            "claim_boundary": (
                "current-data internal eight-origin traceability under constructed batches; "
                "not external farm/year/device validation"
            ),
            "environment": status["environment"],
        }
        report_path = output_dir / "report.md"
        _atomic_write_text(report_path, _build_report(results))
        hashed_artifacts = [*artifact_rows, "report.md", checkpoint_filename]
        results["artifact_sha256"] = {
            filename: sha256_file(output_dir / filename) for filename in hashed_artifacts
        }
        results_path = output_dir / "results.json"
        _atomic_write_json(results_path, results)
        status.update(
            {
                "state": COMPLETE_STATE,
                "completed_at_utc": results["completed_at_utc"],
                "manifest_sha256": manifest.manifest_sha256,
                "data_fingerprint_sha256": manifest.data_fingerprint_sha256,
                "results_json_sha256": sha256_file(results_path),
            }
        )
        _atomic_write_json(status_path, status)
        return results_path
    except BaseException as exc:
        status.update(
            {
                "state": "executed_failure",
                "completed_at_utc": _utc_now(),
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        _atomic_write_json(status_path, status)
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    repository_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Execute the frozen batches-8--9 geographical-origin evaluation once."
    )
    parser.add_argument("--repo-root", type=Path, default=repository_default)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=Path("provenance_study/config.json"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("provenance_study/outputs/locked_evaluation"),
    )
    parser.add_argument("--confirm-locked-test", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    results_path = run_locked_evaluation(args)
    print(f"Locked evaluation completed: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
