from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from provenance_study.make_figures import (
    CLASS_CODES,
    COMPLETE_STATE,
    DEVELOPMENT_SPECTRAL_MODELS,
    FIGURE_SPECS,
    LOCKED_MODELS,
    PRIMARY_BASELINE,
    PRIMARY_MODEL,
    ROBUSTNESS_MODELS,
    SOURCE_DIRECTIONS,
    IncompleteFigureInputsError,
    generate_figures,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _completed_pair(directory: Path, results: dict[str, object]) -> None:
    results_path = directory / "results.json"
    _write_json(results_path, results)
    status = {
        "state": COMPLETE_STATE,
        "run_id": results["run_id"],
        "git_head": results["git_head"],
        "config_sha256": results["config_sha256"],
        "manifest_sha256": results["manifest_sha256"],
        "data_fingerprint_sha256": results["data_fingerprint_sha256"],
        "results_json_sha256": _sha256(results_path),
    }
    _write_json(directory / "execution_status.json", status)


def _prediction(model: str, label: int, batch: int) -> int:
    if model == PRIMARY_BASELINE:
        errors = {(0, 8): 1, (1, 8): 2, (2, 9): 3}
    elif model == PRIMARY_MODEL:
        errors = {(2, 9): 3, (3, 8): 4}
    elif model == "raw_pls_da":
        errors = {(label, 8): (label + 1) % 8 for label in range(4)}
    elif model == "snv_logistic_regression":
        errors = {(0, 8): 1, (4, 9): 5, (6, 8): 7}
    elif model == "sg1_logistic_regression":
        errors = {(1, 8): 2, (5, 9): 6}
    elif model == "sg1_rbf_svm_group_temperature":
        errors = {(0, 9): 1, (3, 8): 4, (7, 9): 0}
    else:
        errors = {(6, 9): 7}
    return errors.get((label, batch), label)


def _build_fixture(root: Path) -> dict[str, Path]:
    development = root / "development"
    cnn = root / "development_cnn"
    locked = root / "locked"
    robustness = root / "robustness"
    development.mkdir(parents=True)
    cnn.mkdir(parents=True)
    locked.mkdir(parents=True)
    robustness.mkdir(parents=True)

    _write_json(
        development / "selection_summary.json",
        {
            "status": "newly_executed_development_only",
            "access_audit": {"locked_numeric_reads": 0},
        },
    )
    _write_csv(
        development / "metrics.csv",
        [
            {
                "model": model,
                "evaluation_scope": "development_grouped_oof",
                "balanced_accuracy": 0.84 + 0.025 * index,
            }
            for index, model in enumerate(DEVELOPMENT_SPECTRAL_MODELS)
        ],
    )
    _write_json(
        cnn / "results.json",
        {
            "execution_status": "executed_complete_development_only",
            "data_access": {"locked_numeric_reads": 0},
            "development_oof_metrics": {"balanced_accuracy": 0.91},
        },
    )

    prediction_rows: list[dict[str, object]] = []
    batch_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    predictions_by_model: dict[str, list[tuple[int, int, int]]] = {}
    for model in LOCKED_MODELS:
        model_predictions: list[tuple[int, int, int]] = []
        for label in range(8):
            for batch in (8, 9):
                predicted = _prediction(model, label, batch)
                model_predictions.append((label, batch, predicted))
                probabilities = np.full(8, 0.15 / 7.0)
                probabilities[predicted] = 0.85
                row: dict[str, object] = {
                    "model": model,
                    "sample_id": f"seed_{label}_{batch}",
                    "constructed_batch": batch,
                    "true_label": label,
                    "predicted_label": predicted,
                }
                for class_index, code in enumerate(CLASS_CODES):
                    row[f"probability_{code}"] = probabilities[class_index]
                prediction_rows.append(row)
                batch_rows.append(
                    {
                        "model": model,
                        "label": label,
                        "class_code": CLASS_CODES[label],
                        "constructed_batch": batch,
                        "accuracy": int(predicted == label),
                    }
                )
        predictions_by_model[model] = model_predictions
        matrix = np.zeros((8, 8), dtype=int)
        for true_label, _, predicted_label in model_predictions:
            matrix[true_label, predicted_label] += 1
        for true_label in range(8):
            class_rows.append(
                {
                    "model": model,
                    "label": true_label,
                    "class_code": CLASS_CODES[true_label],
                    "recall": matrix[true_label, true_label] / matrix[true_label].sum(),
                }
            )
            for predicted_label in range(8):
                confusion_rows.append(
                    {
                        "model": model,
                        "true_label": true_label,
                        "true_class": CLASS_CODES[true_label],
                        "predicted_label": predicted_label,
                        "predicted_class": CLASS_CODES[predicted_label],
                        "count": int(matrix[true_label, predicted_label]),
                    }
                )
        observed = float(np.mean([predicted == label for label, _, predicted in model_predictions]))
        metric_rows.append(
            {"model": model, "equal_constructed_batch_accuracy": observed}
        )

    bootstrap_repetitions = 40
    bootstrap_rows: list[dict[str, object]] = []
    for model_index, model in enumerate(LOCKED_MODELS):
        observed = float(metric_rows[model_index]["equal_constructed_batch_accuracy"])
        for repetition in range(bootstrap_repetitions):
            value = float(np.clip(observed + 0.035 * np.sin(repetition + model_index), 0, 1))
            bootstrap_rows.append(
                {
                    "repetition": repetition,
                    "statistic": "model_equal_constructed_batch_accuracy",
                    "model": model,
                    "value": value,
                }
            )

    baseline = predictions_by_model[PRIMARY_BASELINE]
    ensemble = predictions_by_model[PRIMARY_MODEL]
    transitions = {
        "corrected_errors": 0,
        "new_errors": 0,
        "both_correct": 0,
        "both_incorrect": 0,
    }
    for baseline_row, ensemble_row in zip(baseline, ensemble):
        true_label = baseline_row[0]
        baseline_correct = baseline_row[2] == true_label
        ensemble_correct = ensemble_row[2] == true_label
        if not baseline_correct and ensemble_correct:
            transitions["corrected_errors"] += 1
        elif baseline_correct and not ensemble_correct:
            transitions["new_errors"] += 1
        elif baseline_correct and ensemble_correct:
            transitions["both_correct"] += 1
        else:
            transitions["both_incorrect"] += 1

    _write_csv(locked / "predictions.csv", prediction_rows)
    _write_csv(locked / "batch_metrics.csv", batch_rows)
    _write_csv(locked / "class_metrics.csv", class_rows)
    _write_csv(locked / "confusion.csv", confusion_rows)
    _write_csv(locked / "metrics.csv", metric_rows)
    _write_csv(locked / "bootstrap.csv", bootstrap_rows)
    _write_csv(locked / "effect_test.csv", [transitions])
    common = {
        "run_id": "locked-synthetic",
        "git_head": "a" * 40,
        "config_sha256": "b" * 64,
        "manifest_sha256": "c" * 64,
        "data_fingerprint_sha256": "d" * 64,
    }
    _completed_pair(
        locked,
        {
            "execution_state": COMPLETE_STATE,
            **common,
            "labels": list(range(8)),
            "locked_batches": [8, 9],
            "paired_difference_bootstrap": {"repetitions": bootstrap_repetitions},
        },
    )

    source_rows = [
        {
            "analysis_role": "secondary_whole_source_image_stress_test",
            "scope": "direction_overall",
            "direction": direction,
            "model": model,
            "balanced_accuracy": 0.70 + 0.03 * model_index + 0.01 * direction_index,
        }
        for direction_index, direction in enumerate(SOURCE_DIRECTIONS)
        for model_index, model in enumerate(ROBUSTNESS_MODELS)
    ]
    _write_csv(robustness / "source_transfer_metrics.csv", source_rows)
    _completed_pair(
        robustness,
        {
            "execution_state": COMPLETE_STATE,
            **{**common, "run_id": "robustness-synthetic"},
            "source_transfer": {"directions": list(SOURCE_DIRECTIONS)},
        },
    )
    return {
        "development": development,
        "cnn": cnn,
        "locked": locked,
        "robustness": robustness,
    }


class FixedFigureGenerationTests(unittest.TestCase):
    def test_complete_fixture_generates_six_png_pdf_source_sets_and_hash_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = _build_fixture(root)
            output = root / "figures"
            manifest_path = generate_figures(
                development_dir=paths["development"],
                cnn_dir=paths["cnn"],
                locked_dir=paths["locked"],
                robustness_dir=paths["robustness"],
                output_dir=output,
            )
            expected = {
                *[f"{stem}.png" for stem, _ in FIGURE_SPECS],
                *[f"{stem}.pdf" for stem, _ in FIGURE_SPECS],
                *[source for _, source in FIGURE_SPECS],
                "manifest.json",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            self.assertTrue(all((output / name).stat().st_size > 100 for name in expected))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], COMPLETE_STATE)
            self.assertEqual(manifest["figure_count"], 6)
            self.assertEqual(manifest["raw_data_reads"], 0)
            self.assertEqual(manifest["panels_hidden_by_results"], 0)
            self.assertEqual(len(manifest["output_sha256"]), 18)
            self.assertEqual(len(manifest["input_sha256"]), 15)

    def test_incomplete_robustness_fails_before_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = _build_fixture(root)
            status_path = paths["robustness"] / "execution_status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status["state"] = "executed_failure"
            _write_json(status_path, status)
            output = root / "figures"
            with self.assertRaises(IncompleteFigureInputsError):
                generate_figures(
                    development_dir=paths["development"],
                    cnn_dir=paths["cnn"],
                    locked_dir=paths["locked"],
                    robustness_dir=paths["robustness"],
                    output_dir=output,
                )
            self.assertFalse(output.exists())

    def test_missing_origin_recall_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = _build_fixture(root)
            class_path = paths["locked"] / "class_metrics.csv"
            with class_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows = [
                row
                for row in rows
                if not (row["model"] == PRIMARY_MODEL and row["label"] == "7")
            ]
            _write_csv(class_path, rows)
            output = root / "figures"
            with self.assertRaises(IncompleteFigureInputsError):
                generate_figures(
                    development_dir=paths["development"],
                    cnn_dir=paths["cnn"],
                    locked_dir=paths["locked"],
                    robustness_dir=paths["robustness"],
                    output_dir=output,
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
