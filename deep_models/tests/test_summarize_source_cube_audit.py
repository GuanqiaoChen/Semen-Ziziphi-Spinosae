from __future__ import annotations

import importlib.util
import tempfile
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "summarize_source_cube_audit.py"
SPEC = importlib.util.spec_from_file_location("summarize_source_cube_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
summary = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = summary
SPEC.loader.exec_module(summary)


def make_ensemble_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for direction_index, direction in enumerate(summary.DIRECTIONS):
        suffix = 2 if direction_index == 0 else 1
        for model in summary.MODELS:
            for condition in summary.CONDITIONS:
                for label in range(summary.NUM_CLASSES):
                    # Fixed synthetic performance: fusion/full is perfect;
                    # fusion/shuffle misses labels 0 and 1; other cells are perfect.
                    predicted = label
                    if model == "fusion_net" and condition == "spatial_shuffle" and label < 2:
                        predicted = (label + 1) % summary.NUM_CLASSES
                    probabilities = np.full(summary.NUM_CLASSES, 0.01)
                    probabilities[predicted] = 0.93
                    probabilities /= probabilities.sum()
                    row: dict[str, object] = {
                        "run_id": f"{direction}__{model}__probability_ensemble",
                        "direction": direction,
                        "model": model,
                        "seed": "probability_ensemble",
                        "condition": condition,
                        "sample_index": label,
                        "sample_id": f"{label}-{suffix}/1",
                        "source_cube": f"{label}-{suffix}",
                        "cube_suffix": suffix,
                        "seed_id": "1",
                        "true_label": label,
                    }
                    for class_index, probability in enumerate(probabilities):
                        row[f"raw_probability_{class_index}"] = float(probability)
                        row[f"calibrated_probability_{class_index}"] = float(probability)
                    rows.append(row)
    return rows


def make_complete_fixture(root: Path) -> None:
    ensemble_rows = make_ensemble_rows()
    prediction_rows: list[dict[str, object]] = []
    for row in ensemble_rows:
        for seed in summary.SEEDS:
            prediction_rows.append({**row, "run_id": f"{row['direction']}__{row['model']}__seed_{seed}", "seed": seed})

    metric_rows: list[dict[str, object]] = []
    ensemble_metric_rows: list[dict[str, object]] = []
    for direction in summary.DIRECTIONS:
        for model in summary.MODELS:
            for condition in summary.CONDITIONS:
                selected = [
                    row
                    for row in ensemble_rows
                    if row["direction"] == direction
                    and row["model"] == model
                    and row["condition"] == condition
                ]
                labels = np.asarray([int(row["true_label"]) for row in selected])
                for calibration in summary.CALIBRATIONS:
                    probabilities = np.asarray(
                        [summary._probability_vector(row, calibration) for row in selected]
                    )
                    predicted = probabilities.argmax(axis=1)
                    recalls = [
                        float(np.mean(predicted[labels == label] == label))
                        for label in range(summary.NUM_CLASSES)
                    ]
                    accuracy = float(np.mean(predicted == labels))
                    common = {
                        "direction": direction,
                        "model": model,
                        "condition": condition,
                        "calibration": calibration,
                        "accuracy": accuracy,
                        "balanced_accuracy": float(np.mean(recalls)),
                        "macro_f1": accuracy,
                        "nll": float(-np.log(probabilities[np.arange(len(labels)), labels]).mean()),
                        "brier": float(
                            np.mean(
                                np.sum(
                                    (probabilities - np.eye(summary.NUM_CLASSES)[labels]) ** 2,
                                    axis=1,
                                )
                            )
                        ),
                        "ece_10": 0.05,
                    }
                    ensemble_metric_rows.append(common)
                    for seed in summary.SEEDS:
                        metric_rows.append({**common, "seed": seed})

    summary.write_json(
        root / "run_status.json", {"status": "executed_complete", "completed_runs": []}
    )
    summary.write_json(
        root / "results.json",
        {
            "status": "executed_complete",
            "protocol": {
                "directions": list(summary.DIRECTIONS),
                "models": list(summary.MODELS),
                "seeds": list(summary.SEEDS),
                "counterfactuals": list(summary.CONDITIONS),
            },
        },
    )
    summary.write_csv_rows(
        root / "manifest.csv",
        [{"sample_id": "0-1/1", "label": 0, "source_cube": "0-1"}],
    )
    summary.write_csv_rows(root / "predictions.csv", prediction_rows)
    summary.write_csv_rows(root / "metrics.csv", metric_rows)
    summary.write_csv_rows(root / "ensemble_predictions.csv", ensemble_rows)
    summary.write_csv_rows(root / "ensemble_metrics.csv", ensemble_metric_rows)
    summary.write_csv_rows(
        root / "primary_estimands.csv",
        [{"estimand": "fixture", "model": "fusion_net", "theta": 1.0}],
    )
    summary.write_json(
        root / "spatial_mechanism_decision.json",
        {"limited_support_for_spatial_arrangement": False},
    )


class BootstrapTests(unittest.TestCase):
    def test_block_draws_are_deterministic_and_shared_shape(self):
        first = summary.generate_block_draws(25, seed=99)
        second = summary.generate_block_draws(25, seed=99)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (25, 8))
        self.assertTrue(np.all((first >= 0) & (first < 8)))

    def test_theta_and_primary_effect_use_label_pair_blocks(self):
        rows = make_ensemble_rows()
        draws = np.tile(np.arange(8), (50, 1))
        theta = summary.bootstrap_theta_intervals(rows, draws)
        fusion_full = next(
            row
            for row in theta
            if row["model"] == "fusion_net"
            and row["condition"] == "full"
            and row["calibration"] == "temperature_scaled"
        )
        self.assertEqual(fusion_full["observed_theta"], 1.0)
        effects, _ = summary.bootstrap_effect_intervals(rows, draws)
        primary = next(
            row
            for row in effects
            if row["effect_id"] == "primary_fusion_full_minus_spatial_shuffle"
            and row["calibration"] == "temperature_scaled"
        )
        self.assertAlmostEqual(primary["observed_two_direction_effect"], 0.25)
        self.assertAlmostEqual(primary["conditional_ci_low"], 0.25)
        self.assertAlmostEqual(primary["conditional_ci_high"], 0.25)

    def test_exact_sign_flip_enumerates_all_eight_pair_assignments(self):
        result = summary.exact_sign_flip_test(np.full(8, 0.1))
        self.assertEqual(result["n_exact_sign_assignments"], 256)
        self.assertAlmostEqual(result["observed_effect"], 0.1)
        self.assertAlmostEqual(result["p_value_greater"], 1 / 256)
        self.assertAlmostEqual(result["p_value_two_sided_sensitivity"], 2 / 256)


class CalibrationTests(unittest.TestCase):
    def test_reliability_bins_cover_every_prediction_once(self):
        rows = [
            row
            for row in make_ensemble_rows()
            if row["direction"] == "suffix_1_to_2"
            and row["model"] == "fusion_net"
            and row["condition"] == "full"
        ]
        bins = summary.reliability_rows_for_group(rows, "temperature_scaled")
        self.assertEqual(len(bins), 10)
        self.assertEqual(sum(int(row["n"]) for row in bins), len(rows))
        self.assertAlmostEqual(sum(float(row["fraction"]) for row in bins), 1.0)

    def test_confusion_rows_are_row_normalized(self):
        rows = summary.confusion_table(make_ensemble_rows())
        selected = [
            row
            for row in rows
            if row["direction"] == "suffix_1_to_2" and row["model"] == "fusion_net"
        ]
        for label in range(8):
            total = sum(float(row["row_fraction"]) for row in selected if row["true_label"] == label)
            self.assertAlmostEqual(total, 1.0)


class DisplayTests(unittest.TestCase):
    def test_locked_direction_labels_do_not_expose_code_tokens(self):
        self.assertEqual(summary._direction_label("suffix_1_to_2"), "1 → 2")
        self.assertEqual(summary._direction_label("suffix_2_to_1"), "2 → 1")
        with self.assertRaises(ValueError):
            summary._direction_label("suffix_1_and_2")


class EndToEndTests(unittest.TestCase):
    def test_synthetic_complete_run_generates_auditable_tables_and_figures(self):
        with tempfile.TemporaryDirectory() as temporary:
            input_dir = Path(temporary) / "input"
            output_dir = Path(temporary) / "postprocessing"
            input_dir.mkdir()
            make_complete_fixture(input_dir)
            summary.summarize(input_dir, output_dir)
            expected = {
                "bootstrap_label_pair_indices.csv",
                "bootstrap_theta_intervals.csv",
                "bootstrap_effect_intervals.csv",
                "primary_sign_flip_test.json",
                "main_results.csv",
                "counterfactual_effects.csv",
                "reliability_data.csv",
                "ensemble_confusion_matrices.csv",
                "summary.md",
                "postprocessing_manifest.json",
                "figure_main_performance.png",
                "figure_main_performance.pdf",
                "figure_counterfactual_effects.png",
                "figure_counterfactual_effects.pdf",
                "figure_calibration_reliability.png",
                "figure_calibration_reliability.pdf",
                "figure_ensemble_confusion_matrices.png",
                "figure_ensemble_confusion_matrices.pdf",
            }
            self.assertTrue(expected.issubset({path.name for path in output_dir.iterdir()}))
            manifest = summary.read_json(output_dir / "postprocessing_manifest.json")
            self.assertEqual(manifest["status"], "executed_complete")
            draws = summary.read_csv_rows(output_dir / "bootstrap_label_pair_indices.csv")
            self.assertEqual(len(draws), summary.BOOTSTRAP_REPETITIONS)
            text = (output_dir / "summary.md").read_text(encoding="utf-8")
            self.assertIn("中文审计摘要", text)
            self.assertIn("English audit summary", text)


if __name__ == "__main__":
    unittest.main()
