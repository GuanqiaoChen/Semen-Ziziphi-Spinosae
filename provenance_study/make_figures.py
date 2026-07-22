#!/usr/bin/env python3
"""Create the fixed six-figure origin-traceability result panel.

This module is a strict consumer of completed CSV/JSON artifacts.  It has no
raw-data argument and no code path that discovers, hashes, or reads ``data/``.
Locked evaluation and reciprocal source-image robustness must both report
``executed_complete`` before any figure output directory is created.  Missing
models, origins, batches, directions, probability columns, or matrix cells are
fatal; no result-dependent panel suppression is permitted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


CLASS_CODES = ("HBS", "HBX", "HNA", "HNX", "NX", "SXD", "SXQ", "XJH")
DEVELOPMENT_SPECTRAL_MODELS = (
    "snv_logistic_regression",
    "sg1_shrinkage_lda",
    "sg1_logistic_regression",
    "sg1_rbf_svm_group_temperature",
    "batch_constrained_sg1_probability_ensemble",
)
LOCKED_MODELS = (
    "raw_pls_da",
    "snv_logistic_regression",
    "sg1_shrinkage_lda",
    "sg1_logistic_regression",
    "sg1_rbf_svm_group_temperature",
    "residual_1d_cnn_reference",
    "batch_constrained_sg1_probability_ensemble",
)
ROBUSTNESS_MODELS = (
    "snv_logistic_regression",
    "sg1_shrinkage_lda",
    "sg1_logistic_regression",
    "sg1_rbf_svm_group_temperature",
    "batch_constrained_sg1_probability_ensemble",
)
SOURCE_DIRECTIONS = ("source_1_to_2", "source_2_to_1")
LOCKED_BATCHES = (8, 9)
PRIMARY_MODEL = "batch_constrained_sg1_probability_ensemble"
PRIMARY_BASELINE = "sg1_shrinkage_lda"
CNN_MODEL = "residual_1d_cnn_reference"
COMPLETE_STATE = "executed_complete"
PNG_DPI = 300
RELIABILITY_BINS = 10

MODEL_LABELS = {
    "raw_pls_da": "Raw PLS-DA",
    "snv_logistic_regression": "SNV-LR",
    "sg1_shrinkage_lda": "SG1-LDA",
    "sg1_logistic_regression": "SG1-LR",
    "sg1_rbf_svm_group_temperature": "SG1-RBF-SVM",
    "residual_1d_cnn_reference": "Residual 1D CNN",
    "batch_constrained_sg1_probability_ensemble": "SG1 probability ensemble",
}

FIGURE_SPECS = (
    ("figure_01_development_balanced_accuracy", "figure_01_source.csv"),
    ("figure_02_locked_cluster_bootstrap", "figure_02_source.csv"),
    ("figure_03_locked_confusion", "figure_03_source.csv"),
    ("figure_04_origin_recall_transitions", "figure_04_source.csv"),
    ("figure_05_locked_reliability", "figure_05_source.csv"),
    ("figure_06_source_image_stress", "figure_06_source.csv"),
)


class IncompleteFigureInputsError(RuntimeError):
    """Raised before plotting when a required completed artifact is absent."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise IncompleteFigureInputsError(f"Required JSON artifact is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IncompleteFigureInputsError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise IncompleteFigureInputsError(f"JSON root must be an object: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        raise IncompleteFigureInputsError(f"Required CSV artifact is missing: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except UnicodeDecodeError as exc:
        raise IncompleteFigureInputsError(f"Invalid CSV encoding: {path}") from exc
    if not rows:
        raise IncompleteFigureInputsError(f"Required CSV artifact is empty: {path}")
    return rows


def _require_fields(rows: Sequence[Mapping[str, Any]], fields: Iterable[str], name: str) -> None:
    required = set(fields)
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise IncompleteFigureInputsError(
                f"{name} row {index} lacks required fields: {sorted(missing)}"
            )


def _as_float(value: Any, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise IncompleteFigureInputsError(f"Non-numeric value for {context}: {value!r}") from exc
    if not math.isfinite(result):
        raise IncompleteFigureInputsError(f"Non-finite value for {context}: {value!r}")
    return result


def _as_int(value: Any, context: str) -> int:
    try:
        numeric = float(value)
        result = int(numeric)
    except (TypeError, ValueError) as exc:
        raise IncompleteFigureInputsError(f"Non-integer value for {context}: {value!r}") from exc
    if not math.isfinite(numeric) or numeric != result:
        raise IncompleteFigureInputsError(f"Non-integer value for {context}: {value!r}")
    return result


def _require_exact_set(observed: Iterable[Any], expected: Iterable[Any], context: str) -> None:
    observed_set = set(observed)
    expected_set = set(expected)
    if observed_set != expected_set:
        raise IncompleteFigureInputsError(
            f"{context} is incomplete or unexpected; "
            f"missing={sorted(expected_set - observed_set)}, "
            f"extra={sorted(observed_set - expected_set)}"
        )


def _verify_completed_result(directory: Path, *, result_state_field: str) -> dict[str, Any]:
    directory = Path(directory)
    status_path = directory / "execution_status.json"
    results_path = directory / "results.json"
    status = _read_json(status_path)
    results = _read_json(results_path)
    if status.get("state") != COMPLETE_STATE:
        raise IncompleteFigureInputsError(
            f"Result status is not {COMPLETE_STATE}: {status_path}"
        )
    if results.get(result_state_field) != COMPLETE_STATE:
        raise IncompleteFigureInputsError(
            f"Result payload is not {COMPLETE_STATE}: {results_path}"
        )
    expected_hash = str(status.get("results_json_sha256", ""))
    if not expected_hash or expected_hash != _sha256_file(results_path):
        raise IncompleteFigureInputsError(
            f"results.json hash verification failed: {results_path}"
        )
    for field in ("run_id", "git_head", "config_sha256", "manifest_sha256", "data_fingerprint_sha256"):
        if status.get(field) != results.get(field):
            raise IncompleteFigureInputsError(
                f"Status/results mismatch for {field}: {directory}"
            )
    return results


def _validate_development_inputs(
    development_dir: Path, cnn_dir: Path
) -> list[dict[str, Any]]:
    summary = _read_json(Path(development_dir) / "selection_summary.json")
    if summary.get("status") != "newly_executed_development_only":
        raise IncompleteFigureInputsError("Development spectral analysis is not complete")
    access = summary.get("access_audit", {})
    if not isinstance(access, dict) or int(access.get("locked_numeric_reads", -1)) != 0:
        raise IncompleteFigureInputsError("Development spectral artifact lacks zero locked-read audit")
    rows = _read_csv(Path(development_dir) / "metrics.csv")
    _require_fields(rows, ("model", "balanced_accuracy", "evaluation_scope"), "development metrics")
    _require_exact_set((row["model"] for row in rows), DEVELOPMENT_SPECTRAL_MODELS, "development models")
    if any(row["evaluation_scope"] != "development_grouped_oof" for row in rows):
        raise IncompleteFigureInputsError("Development metrics contain a non-OOF scope")

    cnn = _read_json(Path(cnn_dir) / "results.json")
    if cnn.get("execution_status") != "executed_complete_development_only":
        raise IncompleteFigureInputsError("Development CNN analysis is not complete")
    cnn_access = cnn.get("data_access", {})
    if not isinstance(cnn_access, dict) or int(cnn_access.get("locked_numeric_reads", -1)) != 0:
        raise IncompleteFigureInputsError("Development CNN artifact lacks zero locked-read audit")
    cnn_metrics = cnn.get("development_oof_metrics")
    if not isinstance(cnn_metrics, dict) or "balanced_accuracy" not in cnn_metrics:
        raise IncompleteFigureInputsError("Development CNN metrics are incomplete")

    source_rows: list[dict[str, Any]] = []
    for order, model in enumerate(DEVELOPMENT_SPECTRAL_MODELS):
        row = next(row for row in rows if row["model"] == model)
        source_rows.append(
            {
                "order": order,
                "model": model,
                "model_label": MODEL_LABELS[model],
                "model_family": "spectral_statistical",
                "evaluation_scope": "development_grouped_oof",
                "balanced_accuracy": _as_float(
                    row["balanced_accuracy"], f"development {model} balanced_accuracy"
                ),
            }
        )
    source_rows.append(
        {
            "order": len(source_rows),
            "model": CNN_MODEL,
            "model_label": MODEL_LABELS[CNN_MODEL],
            "model_family": "spectral_neural_reference",
            "evaluation_scope": "development_grouped_oof",
            "balanced_accuracy": _as_float(
                cnn_metrics["balanced_accuracy"], "development CNN balanced_accuracy"
            ),
        }
    )
    return source_rows


def _prediction_tables(
    prediction_rows: Sequence[Mapping[str, str]],
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    _require_fields(
        prediction_rows,
        (
            "model",
            "sample_id",
            "constructed_batch",
            "true_label",
            "predicted_label",
            *[f"probability_{code}" for code in CLASS_CODES],
        ),
        "locked predictions",
    )
    by_model: dict[str, list[dict[str, Any]]] = {model: [] for model in LOCKED_MODELS}
    for row in prediction_rows:
        model = row["model"]
        if model not in by_model:
            raise IncompleteFigureInputsError(f"Unexpected locked prediction model: {model}")
        probabilities = np.asarray(
            [_as_float(row[f"probability_{code}"], f"{model} probability") for code in CLASS_CODES]
        )
        if np.any(probabilities < 0.0) or not np.isclose(probabilities.sum(), 1.0, atol=1e-8):
            raise IncompleteFigureInputsError(f"Invalid probability row for {model}")
        true_label = _as_int(row["true_label"], f"{model} true_label")
        predicted_label = _as_int(row["predicted_label"], f"{model} predicted_label")
        batch = _as_int(row["constructed_batch"], f"{model} constructed_batch")
        if true_label not in range(len(CLASS_CODES)) or predicted_label not in range(len(CLASS_CODES)):
            raise IncompleteFigureInputsError(f"Out-of-range label for {model}")
        if predicted_label != int(probabilities.argmax()):
            raise IncompleteFigureInputsError(f"Stored prediction disagrees with probabilities for {model}")
        by_model[model].append(
            {
                "sample_id": row["sample_id"],
                "batch": batch,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "probabilities": probabilities,
            }
        )
    identities: list[str] | None = None
    for model in LOCKED_MODELS:
        rows = sorted(by_model[model], key=lambda item: item["sample_id"])
        by_model[model] = rows
        model_ids = [str(row["sample_id"]) for row in rows]
        if len(model_ids) != len(set(model_ids)):
            raise IncompleteFigureInputsError(f"Duplicate locked sample IDs for {model}")
        if identities is None:
            identities = model_ids
        elif model_ids != identities:
            raise IncompleteFigureInputsError("Locked models do not share identical samples")
        _require_exact_set((row["batch"] for row in rows), LOCKED_BATCHES, f"{model} locked batches")
        _require_exact_set((row["true_label"] for row in rows), range(8), f"{model} locked classes")
    assert identities is not None
    return by_model, identities


def _validate_locked_inputs(locked_dir: Path) -> dict[str, Any]:
    locked_dir = Path(locked_dir)
    results = _verify_completed_result(locked_dir, result_state_field="execution_state")
    _require_exact_set(results.get("labels", []), range(8), "locked result labels")
    _require_exact_set(results.get("locked_batches", []), LOCKED_BATCHES, "locked result batches")

    metrics = _read_csv(locked_dir / "metrics.csv")
    batches = _read_csv(locked_dir / "batch_metrics.csv")
    classes = _read_csv(locked_dir / "class_metrics.csv")
    confusion = _read_csv(locked_dir / "confusion.csv")
    predictions_raw = _read_csv(locked_dir / "predictions.csv")
    bootstrap = _read_csv(locked_dir / "bootstrap.csv")
    effect = _read_csv(locked_dir / "effect_test.csv")
    predictions, _ = _prediction_tables(predictions_raw)

    _require_fields(metrics, ("model", "equal_constructed_batch_accuracy"), "locked metrics")
    _require_exact_set((row["model"] for row in metrics), LOCKED_MODELS, "locked metric models")
    metric_lookup = {row["model"]: row for row in metrics}

    _require_fields(
        batches,
        ("model", "label", "class_code", "constructed_batch", "accuracy"),
        "locked batch metrics",
    )
    _require_exact_set((row["model"] for row in batches), LOCKED_MODELS, "batch metric models")
    for model in LOCKED_MODELS:
        selected = [row for row in batches if row["model"] == model]
        keys = [
            (_as_int(row["label"], "batch label"), _as_int(row["constructed_batch"], "batch ID"))
            for row in selected
        ]
        _require_exact_set(keys, ((label, batch) for label in range(8) for batch in LOCKED_BATCHES), f"{model} 16 clusters")
        observed = float(np.mean([_as_float(row["accuracy"], "cluster accuracy") for row in selected]))
        stored = _as_float(
            metric_lookup[model]["equal_constructed_batch_accuracy"],
            f"{model} equal-cluster accuracy",
        )
        if not np.isclose(observed, stored, atol=1e-12):
            raise IncompleteFigureInputsError(f"Cluster mean disagrees with locked metric for {model}")

    _require_fields(classes, ("model", "label", "class_code", "recall"), "class metrics")
    _require_exact_set((row["model"] for row in classes), LOCKED_MODELS, "class metric models")
    for model in LOCKED_MODELS:
        selected = [row for row in classes if row["model"] == model]
        _require_exact_set((_as_int(row["label"], "class label") for row in selected), range(8), f"{model} class recalls")
        prediction_rows = predictions[model]
        for row in selected:
            label = _as_int(row["label"], "class label")
            relevant = [item for item in prediction_rows if item["true_label"] == label]
            observed = float(np.mean([item["predicted_label"] == label for item in relevant]))
            if not np.isclose(observed, _as_float(row["recall"], "class recall"), atol=1e-12):
                raise IncompleteFigureInputsError(f"Recall disagrees with predictions for {model}/{label}")

    _require_fields(
        confusion,
        ("model", "true_label", "true_class", "predicted_label", "predicted_class", "count"),
        "confusion",
    )
    _require_exact_set((row["model"] for row in confusion), LOCKED_MODELS, "confusion models")
    matrices: dict[str, np.ndarray] = {}
    for model in LOCKED_MODELS:
        matrix = np.full((8, 8), -1, dtype=int)
        for row in confusion:
            if row["model"] != model:
                continue
            true_label = _as_int(row["true_label"], "confusion true label")
            predicted_label = _as_int(row["predicted_label"], "confusion predicted label")
            if not (0 <= true_label < 8 and 0 <= predicted_label < 8):
                raise IncompleteFigureInputsError("Confusion label is out of range")
            if matrix[true_label, predicted_label] >= 0:
                raise IncompleteFigureInputsError(f"Duplicate confusion cell for {model}")
            matrix[true_label, predicted_label] = _as_int(row["count"], "confusion count")
        if np.any(matrix < 0):
            raise IncompleteFigureInputsError(f"Incomplete 8x8 confusion matrix for {model}")
        recomputed = np.zeros((8, 8), dtype=int)
        for item in predictions[model]:
            recomputed[item["true_label"], item["predicted_label"]] += 1
        if not np.array_equal(matrix, recomputed):
            raise IncompleteFigureInputsError(f"Confusion matrix disagrees with predictions for {model}")
        matrices[model] = matrix

    _require_fields(bootstrap, ("repetition", "statistic", "model", "value"), "bootstrap")
    model_bootstrap = [
        row for row in bootstrap if row["statistic"] == "model_equal_constructed_batch_accuracy"
    ]
    _require_exact_set((row["model"] for row in model_bootstrap), LOCKED_MODELS, "bootstrap models")
    expected_repetitions = _as_int(
        results.get("paired_difference_bootstrap", {}).get("repetitions"),
        "bootstrap repetitions",
    )
    if expected_repetitions < 2:
        raise IncompleteFigureInputsError("At least two cluster-bootstrap repetitions are required")
    bootstrap_by_model: dict[str, np.ndarray] = {}
    for model in LOCKED_MODELS:
        selected = [row for row in model_bootstrap if row["model"] == model]
        repetitions = [_as_int(row["repetition"], "bootstrap repetition") for row in selected]
        _require_exact_set(repetitions, range(expected_repetitions), f"{model} bootstrap repetitions")
        bootstrap_by_model[model] = np.asarray(
            [_as_float(row["value"], "bootstrap value") for row in sorted(selected, key=lambda row: int(row["repetition"]))]
        )

    _require_fields(
        effect,
        ("corrected_errors", "new_errors", "both_correct", "both_incorrect"),
        "effect test",
    )
    if len(effect) != 1:
        raise IncompleteFigureInputsError("Exactly one primary-vs-LDA effect-test row is required")
    baseline_rows = predictions[PRIMARY_BASELINE]
    ensemble_rows = predictions[PRIMARY_MODEL]
    transitions = {
        "corrected_errors": 0,
        "new_errors": 0,
        "both_correct": 0,
        "both_incorrect": 0,
    }
    for baseline_item, ensemble_item in zip(baseline_rows, ensemble_rows):
        if baseline_item["sample_id"] != ensemble_item["sample_id"]:
            raise IncompleteFigureInputsError("Transition samples are misaligned")
        true_label = baseline_item["true_label"]
        baseline_correct = baseline_item["predicted_label"] == true_label
        ensemble_correct = ensemble_item["predicted_label"] == true_label
        if not baseline_correct and ensemble_correct:
            transitions["corrected_errors"] += 1
        elif baseline_correct and not ensemble_correct:
            transitions["new_errors"] += 1
        elif baseline_correct and ensemble_correct:
            transitions["both_correct"] += 1
        else:
            transitions["both_incorrect"] += 1
    for field, value in transitions.items():
        if _as_int(effect[0][field], field) != value:
            raise IncompleteFigureInputsError(f"Effect transition count disagrees for {field}")

    return {
        "results": results,
        "metric_lookup": metric_lookup,
        "batch_rows": batches,
        "class_rows": classes,
        "confusion_matrices": matrices,
        "predictions": predictions,
        "bootstrap_by_model": bootstrap_by_model,
        "bootstrap_repetitions": expected_repetitions,
        "transitions": transitions,
    }


def _validate_robustness_inputs(robustness_dir: Path) -> list[dict[str, Any]]:
    robustness_dir = Path(robustness_dir)
    results = _verify_completed_result(robustness_dir, result_state_field="execution_state")
    source = results.get("source_transfer", {})
    if not isinstance(source, dict):
        raise IncompleteFigureInputsError("Robustness source-transfer result is missing")
    _require_exact_set(source.get("directions", []), SOURCE_DIRECTIONS, "robustness directions")
    rows = _read_csv(robustness_dir / "source_transfer_metrics.csv")
    _require_fields(
        rows,
        ("analysis_role", "scope", "direction", "model", "balanced_accuracy"),
        "source transfer metrics",
    )
    selected = [row for row in rows if row["scope"] == "direction_overall"]
    observed_keys = [(row["direction"], row["model"]) for row in selected]
    expected_keys = [
        (direction, model) for direction in SOURCE_DIRECTIONS for model in ROBUSTNESS_MODELS
    ]
    _require_exact_set(observed_keys, expected_keys, "source-transfer direction/model panel")
    if any(
        row["analysis_role"] != "secondary_whole_source_image_stress_test"
        for row in selected
    ):
        raise IncompleteFigureInputsError("Source-transfer rows have an unexpected analysis role")
    return [
        {
            "direction": row["direction"],
            "model": row["model"],
            "model_label": MODEL_LABELS[row["model"]],
            "balanced_accuracy": _as_float(
                row["balanced_accuracy"], "source-transfer balanced_accuracy"
            ),
        }
        for row in selected
    ]


def _locked_bootstrap_source(locked: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for order, model in enumerate(LOCKED_MODELS):
        values = locked["bootstrap_by_model"][model]
        observed = _as_float(
            locked["metric_lookup"][model]["equal_constructed_batch_accuracy"],
            f"{model} locked primary metric",
        )
        rows.append(
            {
                "order": order,
                "model": model,
                "model_label": MODEL_LABELS[model],
                "n_label_batch_clusters": 16,
                "bootstrap_repetitions": locked["bootstrap_repetitions"],
                "equal_constructed_batch_accuracy": observed,
                "bootstrap_ci_lower_95": float(np.quantile(values, 0.025)),
                "bootstrap_ci_upper_95": float(np.quantile(values, 0.975)),
            }
        )
    return rows


def _confusion_source(locked: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in (PRIMARY_BASELINE, PRIMARY_MODEL):
        matrix = np.asarray(locked["confusion_matrices"][model], dtype=int)
        row_sums = matrix.sum(axis=1)
        if np.any(row_sums <= 0):
            raise IncompleteFigureInputsError(f"Confusion matrix has an empty true class: {model}")
        normalized = matrix / row_sums[:, None]
        for true_label in range(8):
            for predicted_label in range(8):
                rows.append(
                    {
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "true_label": true_label,
                        "true_class": CLASS_CODES[true_label],
                        "predicted_label": predicted_label,
                        "predicted_class": CLASS_CODES[predicted_label],
                        "count": int(matrix[true_label, predicted_label]),
                        "row_normalized_fraction": float(
                            normalized[true_label, predicted_label]
                        ),
                    }
                )
    return rows


def _recall_transition_source(locked: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in (PRIMARY_BASELINE, PRIMARY_MODEL):
        lookup = {
            _as_int(row["label"], "class label"): row
            for row in locked["class_rows"]
            if row["model"] == model
        }
        for label in range(8):
            rows.append(
                {
                    "panel": "origin_recall",
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "category": CLASS_CODES[label],
                    "label": label,
                    "value": _as_float(lookup[label]["recall"], "class recall"),
                    "unit": "fraction",
                }
            )
    transition_labels = {
        "corrected_errors": "Corrected errors",
        "new_errors": "New errors",
        "both_correct": "Both correct",
        "both_incorrect": "Both incorrect",
    }
    for category in ("corrected_errors", "new_errors", "both_correct", "both_incorrect"):
        rows.append(
            {
                "panel": "error_transition",
                "model": f"{PRIMARY_MODEL}_versus_{PRIMARY_BASELINE}",
                "model_label": "Ensemble versus SG1-LDA",
                "category": transition_labels[category],
                "label": "",
                "value": int(locked["transitions"][category]),
                "unit": "seeds",
            }
        )
    return rows


def _reliability_source(locked: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in (PRIMARY_BASELINE, PRIMARY_MODEL):
        model_rows = locked["predictions"][model]
        confidences = np.asarray([item["probabilities"].max() for item in model_rows])
        correct = np.asarray(
            [item["predicted_label"] == item["true_label"] for item in model_rows], dtype=float
        )
        bin_indices = np.minimum((confidences * RELIABILITY_BINS).astype(int), RELIABILITY_BINS - 1)
        for bin_index in range(RELIABILITY_BINS):
            selected = bin_indices == bin_index
            rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "bin_index": bin_index,
                    "bin_lower": bin_index / RELIABILITY_BINS,
                    "bin_upper": (bin_index + 1) / RELIABILITY_BINS,
                    "bin_midpoint": (bin_index + 0.5) / RELIABILITY_BINS,
                    "n": int(selected.sum()),
                    "mean_confidence": (
                        float(confidences[selected].mean()) if np.any(selected) else ""
                    ),
                    "empirical_accuracy": (
                        float(correct[selected].mean()) if np.any(selected) else ""
                    ),
                }
            )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty figure source: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=PNG_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_development(rows: Sequence[Mapping[str, Any]], output_stem: Path) -> None:
    values = np.asarray([float(row["balanced_accuracy"]) for row in rows])
    labels = [str(row["model_label"]) for row in rows]
    colors = ["#4C78A8"] * (len(rows) - 1) + ["#F58518"]
    fig, axis = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    bars = axis.bar(np.arange(len(rows)), values, color=colors, edgecolor="white", linewidth=0.8)
    axis.set_xticks(np.arange(len(rows)), labels, rotation=24, ha="right")
    axis.set_ylabel("Balanced accuracy")
    axis.set_title("Development grouped OOF performance")
    axis.set_ylim(max(0.0, float(values.min()) - 0.08), min(1.0, float(values.max()) + 0.04))
    axis.grid(axis="y", alpha=0.25)
    axis.bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=3, fontsize=8)
    _save_figure(fig, output_stem)


def _plot_locked_bootstrap(rows: Sequence[Mapping[str, Any]], output_stem: Path) -> None:
    observed = np.asarray([float(row["equal_constructed_batch_accuracy"]) for row in rows])
    lower = np.asarray([float(row["bootstrap_ci_lower_95"]) for row in rows])
    upper = np.asarray([float(row["bootstrap_ci_upper_95"]) for row in rows])
    positions = np.arange(len(rows))
    fig, axis = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)
    axis.errorbar(
        positions,
        observed,
        yerr=np.vstack([observed - lower, upper - observed]),
        fmt="o",
        color="#2F4B7C",
        ecolor="#7A8DA8",
        capsize=4,
        markersize=7,
    )
    axis.set_xticks(positions, [str(row["model_label"]) for row in rows], rotation=24, ha="right")
    axis.set_ylabel("Equal label–batch cluster accuracy")
    axis.set_title("Locked evaluation: 16 clusters with bootstrap 95% CI")
    axis.set_ylim(0.0, 1.03)
    axis.grid(axis="y", alpha=0.25)
    _save_figure(fig, output_stem)


def _plot_confusion(rows: Sequence[Mapping[str, Any]], output_stem: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True, sharex=True, sharey=True)
    image = None
    for axis, model in zip(axes, (PRIMARY_BASELINE, PRIMARY_MODEL)):
        selected = [row for row in rows if row["model"] == model]
        matrix = np.zeros((8, 8), dtype=float)
        for row in selected:
            matrix[int(row["true_label"]), int(row["predicted_label"])] = float(
                row["row_normalized_fraction"]
            )
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues", aspect="equal")
        axis.set_title(MODEL_LABELS[model])
        axis.set_xticks(range(8), CLASS_CODES, rotation=45, ha="right")
        axis.set_yticks(range(8), CLASS_CODES)
        axis.set_xlabel("Predicted origin")
        for row_index in range(8):
            for column_index in range(8):
                value = matrix[row_index, column_index]
                if value >= 0.20:
                    axis.text(
                        column_index,
                        row_index,
                        f"{value:.0%}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if value >= 0.55 else "black",
                    )
    axes[0].set_ylabel("True origin")
    assert image is not None
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02, label="Row-normalized fraction")
    fig.suptitle("Locked origin confusion")
    _save_figure(fig, output_stem)


def _plot_recall_transitions(rows: Sequence[Mapping[str, Any]], output_stem: Path) -> None:
    recall_rows = [row for row in rows if row["panel"] == "origin_recall"]
    transition_rows = [row for row in rows if row["panel"] == "error_transition"]
    fig, axes = plt.subplots(
        1, 2, figsize=(12.0, 4.8), constrained_layout=True, gridspec_kw={"width_ratios": [2.1, 1.0]}
    )
    x = np.arange(8)
    width = 0.36
    for offset, model, color in (
        (-width / 2, PRIMARY_BASELINE, "#4C78A8"),
        (width / 2, PRIMARY_MODEL, "#F58518"),
    ):
        selected = sorted(
            (row for row in recall_rows if row["model"] == model), key=lambda row: int(row["label"])
        )
        axes[0].bar(
            x + offset,
            [float(row["value"]) for row in selected],
            width,
            label=MODEL_LABELS[model],
            color=color,
        )
    axes[0].set_xticks(x, CLASS_CODES)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Recall")
    axes[0].set_title("Locked recall by origin")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    categories = [str(row["category"]) for row in transition_rows]
    counts = [int(row["value"]) for row in transition_rows]
    bars = axes[1].barh(
        np.arange(len(categories)), counts, color=["#54A24B", "#E45756", "#72B7B2", "#B279A2"]
    )
    axes[1].set_yticks(np.arange(len(categories)), categories)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Seeds")
    axes[1].set_title("Prediction transitions")
    axes[1].bar_label(bars, padding=3, fontsize=8)
    _save_figure(fig, output_stem)


def _plot_reliability(rows: Sequence[Mapping[str, Any]], output_stem: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.0, 5.4), constrained_layout=True)
    axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1.2, label="Ideal")
    for model, color, marker in (
        (PRIMARY_BASELINE, "#4C78A8", "o"),
        (PRIMARY_MODEL, "#F58518", "s"),
    ):
        selected = [row for row in rows if row["model"] == model and int(row["n"]) > 0]
        axis.plot(
            [float(row["mean_confidence"]) for row in selected],
            [float(row["empirical_accuracy"]) for row in selected],
            marker=marker,
            color=color,
            linewidth=1.8,
            label=MODEL_LABELS[model],
        )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Mean predicted confidence")
    axis.set_ylabel("Empirical accuracy")
    axis.set_title("Locked reliability (10 fixed bins)")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    _save_figure(fig, output_stem)


def _plot_source_stress(rows: Sequence[Mapping[str, Any]], output_stem: Path) -> None:
    x = np.arange(len(ROBUSTNESS_MODELS))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9.5, 5.0), constrained_layout=True)
    for offset, direction, color, label in (
        (-width / 2, SOURCE_DIRECTIONS[0], "#4C78A8", "Source image 1 → 2"),
        (width / 2, SOURCE_DIRECTIONS[1], "#F58518", "Source image 2 → 1"),
    ):
        lookup = {
            row["model"]: float(row["balanced_accuracy"])
            for row in rows
            if row["direction"] == direction
        }
        axis.bar(
            x + offset,
            [lookup[model] for model in ROBUSTNESS_MODELS],
            width,
            color=color,
            label=label,
        )
    axis.set_xticks(x, [MODEL_LABELS[model] for model in ROBUSTNESS_MODELS], rotation=24, ha="right")
    axis.set_ylim(0.0, 1.03)
    axis.set_ylabel("Balanced accuracy")
    axis.set_title("Reciprocal whole-source-image stress test")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    _save_figure(fig, output_stem)


def generate_figures(
    *,
    development_dir: Path,
    cnn_dir: Path,
    locked_dir: Path,
    robustness_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> Path:
    """Validate all completed inputs, then generate all six fixed panels."""

    development_dir = Path(development_dir).resolve()
    cnn_dir = Path(cnn_dir).resolve()
    locked_dir = Path(locked_dir).resolve()
    robustness_dir = Path(robustness_dir).resolve()
    output_dir = Path(output_dir).resolve()

    # Every required artifact is validated before output_dir is created.
    development_source = _validate_development_inputs(development_dir, cnn_dir)
    locked = _validate_locked_inputs(locked_dir)
    robustness_source = _validate_robustness_inputs(robustness_dir)
    locked_bootstrap_source = _locked_bootstrap_source(locked)
    confusion_source = _confusion_source(locked)
    recall_transition_source = _recall_transition_source(locked)
    reliability_source = _reliability_source(locked)

    expected_outputs = [
        *[f"{stem}.png" for stem, _ in FIGURE_SPECS],
        *[f"{stem}.pdf" for stem, _ in FIGURE_SPECS],
        *[source for _, source in FIGURE_SPECS],
        "manifest.json",
    ]
    existing = [output_dir / name for name in expected_outputs if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Figure outputs already exist; pass --overwrite to replace declared artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = (
        development_source,
        locked_bootstrap_source,
        confusion_source,
        recall_transition_source,
        reliability_source,
        robustness_source,
    )
    for (_, source_name), rows in zip(FIGURE_SPECS, sources):
        _write_csv(output_dir / source_name, rows)

    plotters = (
        _plot_development,
        _plot_locked_bootstrap,
        _plot_confusion,
        _plot_recall_transitions,
        _plot_reliability,
        _plot_source_stress,
    )
    for (stem, _), rows, plotter in zip(FIGURE_SPECS, sources, plotters):
        plotter(rows, output_dir / stem)

    input_paths = (
        development_dir / "selection_summary.json",
        development_dir / "metrics.csv",
        cnn_dir / "results.json",
        locked_dir / "execution_status.json",
        locked_dir / "results.json",
        locked_dir / "metrics.csv",
        locked_dir / "batch_metrics.csv",
        locked_dir / "class_metrics.csv",
        locked_dir / "confusion.csv",
        locked_dir / "predictions.csv",
        locked_dir / "bootstrap.csv",
        locked_dir / "effect_test.csv",
        robustness_dir / "execution_status.json",
        robustness_dir / "results.json",
        robustness_dir / "source_transfer_metrics.csv",
    )
    output_paths = [
        output_dir / name
        for name in expected_outputs
        if name != "manifest.json"
    ]
    manifest = {
        "status": "executed_complete",
        "generated_at_utc": _utc_now(),
        "scientific_objective": "hyperspectral_geographical_origin_traceability_of_semen_ziziphi_spinosae",
        "input_scope": "completed_result_csv_and_json_only",
        "raw_data_reads": 0,
        "figure_count": len(FIGURE_SPECS),
        "panels_hidden_by_results": 0,
        "png_dpi": PNG_DPI,
        "reliability_bins": RELIABILITY_BINS,
        "locked_state_required": COMPLETE_STATE,
        "robustness_state_required": COMPLETE_STATE,
        "locked_run_id": locked["results"].get("run_id"),
        "input_sha256": {str(path): _sha256_file(path) for path in input_paths},
        "output_sha256": {path.name: _sha256_file(path) for path in output_paths},
        "figures": [
            {
                "index": index,
                "stem": stem,
                "png": f"{stem}.png",
                "pdf": f"{stem}.pdf",
                "source_csv": source,
            }
            for index, (stem, source) in enumerate(FIGURE_SPECS, start=1)
        ],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def build_argument_parser() -> argparse.ArgumentParser:
    package_root = Path(__file__).resolve().parent
    outputs_root = package_root / "outputs"
    parser = argparse.ArgumentParser(
        description="Generate the fixed origin-traceability figures from completed outputs only"
    )
    parser.add_argument("--development-dir", type=Path, default=outputs_root / "development")
    parser.add_argument("--cnn-dir", type=Path, default=outputs_root / "development_cnn")
    parser.add_argument("--locked-dir", type=Path, default=outputs_root / "locked_evaluation")
    parser.add_argument("--robustness-dir", type=Path, default=outputs_root / "robustness")
    parser.add_argument("--output-dir", type=Path, default=package_root / "figures")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    manifest_path = generate_figures(
        development_dir=args.development_dir,
        cnn_dir=args.cnn_dir,
        locked_dir=args.locked_dir,
        robustness_dir=args.robustness_dir,
        output_dir=args.output_dir,
        overwrite=bool(args.overwrite),
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
