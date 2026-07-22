from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from provenance_study.cnn_baseline import CNNTrainingConfig
from provenance_study.run_locked_evaluation import (
    COMPLETE_STATE,
    CONFIRMATION_PHRASE,
    MODEL_ORDER,
    PRIMARY_MODEL,
    CompletedEvaluationError,
    LockedTestConfirmationError,
    _decorate_rows,
    _write_csv,
    cluster_accuracy_matrices,
    evaluate_effect_gate,
    exact_paired_sign_flip,
    fit_frozen_models_and_predict_once,
    make_prediction_rows,
    recompute_evaluation_tables,
    refuse_completed_output,
    run_locked_evaluation,
    save_cnn_reference_checkpoint,
    sha256_file,
    stratified_cluster_bootstrap,
)


def _effect_gate_config() -> dict[str, object]:
    return {
        "paired_cluster_interval_lower_bound_must_exceed_zero": True,
        "minimum_relative_balanced_error_reduction": 0.2,
        "maximum_allowed_log_loss_increase": 0.01,
        "maximum_allowed_brier_increase": 0.01,
        "minimum_strictly_improved_classes": 2,
        "minimum_strictly_improved_constructed_batches": 3,
        "leave_one_constructed_batch_out_effect_must_remain_positive": True,
    }


def _model_config() -> dict[str, object]:
    return {
        "analysis_seed": 20260721,
        "class_codes": [f"C{label}" for label in range(8)],
        "constructed_batches": {"development_indices": list(range(8))},
        "models": {
            "sg1_rbf_svm": {"temperature_log_bounds": [-4.0, 4.0]},
            "raw_pls_da": {"n_components": 20, "max_iter": 1000},
            "snv_logistic_regression": {
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 5000,
                "tol": 0.0001,
            },
        },
        "primary_predictor": {
            "svm_temperature_frozen_development_value": 0.4,
        },
        "cnn_reference": {
            "full_development_epochs": 88,
            "training_seed": 20260721,
            "parameter_count": 321776,
        },
    }


class _FakeCNNModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frozen_parameter_vector = torch.nn.Parameter(torch.zeros(321_776))


class _FakeFittedCNN:
    def __init__(self) -> None:
        self.model = _FakeCNNModel()
        self.classes = np.arange(8, dtype=np.int64)
        self.optimization_seed = 20260721
        self.epochs = 88
        self.raw_band_count = 21
        self.training_config = CNNTrainingConfig()
        self.standardizer = SimpleNamespace(
            mean=np.zeros(21, dtype=np.float32),
            scale=np.ones(21, dtype=np.float32),
            n_samples_seen=64,
        )
        self.predict_calls = 0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.predict_calls += 1
        probabilities = np.full((len(X), 8), 0.05, dtype=np.float64)
        probabilities[:, 0] = 0.65
        return probabilities


class ConfirmationAndCompletionGuardTests(unittest.TestCase):
    def test_wrong_confirmation_stops_before_discovery_or_path_access(self) -> None:
        arguments = argparse.Namespace(
            confirm_locked_test="unlock_batches_8_9",
            repo_root=object(),
            output_dir=object(),
            config=object(),
            data_root=object(),
        )
        with patch(
            "provenance_study.run_locked_evaluation.discover_manifest",
            side_effect=AssertionError("data discovery was reached"),
        ) as discovery:
            with self.assertRaises(LockedTestConfirmationError):
                run_locked_evaluation(arguments)
        discovery.assert_not_called()

    def test_completed_canonical_output_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "execution_status.json").write_text(
                json.dumps({"state": COMPLETE_STATE}), encoding="utf-8"
            )
            with self.assertRaises(CompletedEvaluationError):
                refuse_completed_output(output_dir)


class ClusterStatisticsTests(unittest.TestCase):
    def test_stratified_bootstrap_uses_common_paired_draws(self) -> None:
        baseline = np.asarray([[0.70, 0.80], [0.60, 0.90]], dtype=np.float64)
        improved = baseline + 0.10
        first = stratified_cluster_bootstrap(
            {"baseline": baseline, "improved": improved}, repetitions=500, seed=19
        )
        second = stratified_cluster_bootstrap(
            {"baseline": baseline, "improved": improved}, repetitions=500, seed=19
        )
        np.testing.assert_array_equal(first["baseline"], second["baseline"])
        np.testing.assert_allclose(first["improved"] - first["baseline"], 0.10)

    def test_exact_sign_flip_enumerates_all_sixteen_clusters(self) -> None:
        result = exact_paired_sign_flip(np.repeat(0.05, 16))
        self.assertEqual(result["n_sign_patterns"], 65_536)
        self.assertAlmostEqual(result["observed_mean_difference"], 0.05)
        self.assertAlmostEqual(result["two_sided_p_value"], 2 / 65_536)
        self.assertAlmostEqual(result["one_sided_greater_p_value"], 1 / 65_536)

    def test_four_condition_gate_reads_breadth_thresholds(self) -> None:
        baseline = {
            "balanced_accuracy": 0.90,
            "negative_log_likelihood": 0.20,
            "multiclass_brier_score": 0.10,
        }
        ensemble = {
            "balanced_accuracy": 0.93,
            "negative_log_likelihood": 0.205,
            "multiclass_brier_score": 0.105,
        }
        conditions, summary = evaluate_effect_gate(
            ensemble,
            baseline,
            [0.95, 0.92, 0.90, 0.91],
            [0.90, 0.90, 0.90, 0.91],
            np.repeat(0.02, 16),
            (0.005, 0.04),
            _effect_gate_config(),
        )
        self.assertEqual(len(conditions), 4)
        self.assertTrue(summary["all_four_conditions_passed"])
        self.assertEqual(summary["strictly_improved_classes"], 2)
        self.assertEqual(summary["strictly_improved_label_batch_clusters"], 16)

        strict_config = _effect_gate_config()
        strict_config["minimum_strictly_improved_classes"] = 3
        failed_conditions, failed_summary = evaluate_effect_gate(
            ensemble,
            baseline,
            [0.95, 0.92, 0.90, 0.91],
            [0.90, 0.90, 0.90, 0.91],
            np.repeat(0.02, 16),
            (0.005, 0.04),
            strict_config,
        )
        self.assertFalse(failed_conditions[3]["passed"])
        self.assertFalse(failed_summary["all_four_conditions_passed"])


class RecomputableOutputTests(unittest.TestCase):
    def test_csv_probabilities_recompute_all_primary_cluster_metrics(self) -> None:
        class_codes = ("A", "B")
        y_true = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        batches = np.asarray([8, 8, 9, 9, 8, 8, 9, 9], dtype=np.int64)
        baseline = np.asarray(
            [
                [0.9, 0.1],
                [0.6, 0.4],
                [0.4, 0.6],
                [0.8, 0.2],
                [0.2, 0.8],
                [0.7, 0.3],
                [0.1, 0.9],
                [0.3, 0.7],
            ]
        )
        improved = baseline.copy()
        improved[2] = [0.7, 0.3]
        improved[5] = [0.2, 0.8]
        metadata = [
            {
                "sample_index": index,
                "sample_id": f"sample-{index}",
                "source_cube": f"{int(y_true[index])}-1",
                "source_replicate": 1,
                "seed_id": str(index),
                "relative_csv_path": f"{int(y_true[index])}-1/{index}.csv",
                "constructed_batch": int(batches[index]),
            }
            for index in range(len(y_true))
        ]
        prediction_rows = make_prediction_rows(
            {"baseline": baseline, "improved": improved},
            y_true,
            metadata,
            class_codes,
        )
        context = {
            "run_id": "synthetic",
            "git_head": "0" * 40,
            "config_sha256": "1" * 64,
            "manifest_sha256": "2" * 64,
            "data_fingerprint_sha256": "3" * 64,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "predictions.csv"
            _write_csv(path, _decorate_rows(prediction_rows, context))
            first_hash = sha256_file(path)
            with path.open("r", encoding="utf-8", newline="") as handle:
                reloaded = list(csv.DictReader(handle))
            tables = recompute_evaluation_tables(reloaded, class_codes, ece_bins=4)
            second_hash = sha256_file(path)

        self.assertEqual(first_hash, second_hash)
        metrics = {row["model"]: row for row in tables.metrics}
        self.assertAlmostEqual(metrics["baseline"]["equal_constructed_batch_accuracy"], 0.75)
        self.assertAlmostEqual(metrics["improved"]["equal_constructed_batch_accuracy"], 1.0)
        matrices, labels, observed_batches = cluster_accuracy_matrices(
            tables.batch_metrics, ("baseline", "improved")
        )
        self.assertEqual(labels, (0, 1))
        self.assertEqual(observed_batches, (8, 9))
        np.testing.assert_allclose(matrices["improved"], 1.0)
        self.assertEqual(len(tables.confusion), 8)


class FrozenCNNIntegrationTests(unittest.TestCase):
    def test_cnn_is_predicted_once_and_excluded_from_primary_average(self) -> None:
        rng = np.random.default_rng(29)
        X_development: list[np.ndarray] = []
        y_development: list[int] = []
        batches: list[int] = []
        for batch in range(8):
            for label in range(8):
                spectrum = rng.normal(0.0, 0.1, size=21)
                spectrum[label * 2 : label * 2 + 2] += 2.0
                X_development.append(spectrum)
                y_development.append(label)
                batches.append(batch)
        X_locked = rng.normal(size=(16, 21))
        fake_cnn = _FakeFittedCNN()
        fit_calls: list[tuple[int, int]] = []

        def fake_fit(
            X: np.ndarray,
            y: np.ndarray,
            *,
            epochs: int,
            optimization_seed: int,
        ) -> _FakeFittedCNN:
            fit_calls.append((epochs, optimization_seed))
            return fake_cnn

        fake_oof = SimpleNamespace(
            decision_scores=np.zeros((len(y_development), 8), dtype=np.float64)
        )
        with patch(
            "provenance_study.run_locked_evaluation.grouped_oof_decision_scores",
            return_value=fake_oof,
        ), patch(
            "provenance_study.run_locked_evaluation.fit_decision_temperature",
            return_value=0.4,
        ):
            probabilities, temperature, returned_cnn = fit_frozen_models_and_predict_once(
                np.asarray(X_development),
                np.asarray(y_development),
                np.asarray(batches),
                X_locked,
                _model_config(),
                cnn_fit_function=fake_fit,
            )

        self.assertEqual(fit_calls, [(88, 20260721)])
        self.assertEqual(fake_cnn.predict_calls, 1)
        self.assertIs(returned_cnn, fake_cnn)
        self.assertEqual(temperature, 0.4)
        self.assertEqual(tuple(probabilities), MODEL_ORDER)
        expected_ensemble = np.mean(
            np.stack(
                [
                    probabilities["sg1_shrinkage_lda"],
                    probabilities["sg1_logistic_regression"],
                    probabilities["sg1_rbf_svm_group_temperature"],
                ]
            ),
            axis=0,
        )
        np.testing.assert_allclose(probabilities[PRIMARY_MODEL], expected_ensemble)

    def test_checkpoint_is_loadable_with_weights_only(self) -> None:
        fake_cnn = _FakeFittedCNN()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cnn_reference_state.pt"
            save_cnn_reference_checkpoint(
                fake_cnn,
                path,
                run_context={"run_id": "synthetic", "git_head": "0" * 40},
            )
            payload = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(payload["format_version"], 1)
        self.assertEqual(payload["epochs"], 88)
        self.assertIn("frozen_parameter_vector", payload["state_dict"])


if __name__ == "__main__":
    unittest.main()
