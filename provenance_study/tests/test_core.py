from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from provenance_study.core import (
    CLASS_NAMES,
    DEVELOPMENT_BATCHES,
    LockedDataAccessError,
    MultiplicativeScatterCorrection,
    SavitzkyGolayTransformer,
    StandardNormalVariate,
    build_sg15_logistic_regression,
    build_sg15_rbf_svm,
    build_sg15_shrinkage_lda,
    crossfit_oof_decision_temperature,
    decision_scores_to_probabilities,
    discover_manifest,
    equal_weight_probability_average,
    fit_decision_temperature,
    grouped_oof_decision_scores,
    grouped_oof_probabilities,
    load_csv_split,
    load_development_csv,
    multiclass_metrics,
    nested_grouped_oof_temperature_probabilities,
)


TEST_BANDS = 21
TEST_COUNTS = {(label, replicate): 10 for label in range(8) for replicate in (1, 2)}


def _write_synthetic_repository(root: Path) -> None:
    wavelengths = np.linspace(949.0, 1651.0, TEST_BANDS)
    for label in range(8):
        for replicate in (1, 2):
            cube = root / f"{label}-{replicate}"
            cube.mkdir(parents=True)
            for seed_id in range(1, 11):
                reflectance = (
                    0.1 * label
                    + 0.01 * replicate
                    + 0.001 * seed_id
                    + np.linspace(0.0, 0.2, TEST_BANDS)
                )
                np.savetxt(
                    cube / f"{seed_id}.csv",
                    np.column_stack([wavelengths, reflectance]),
                    delimiter=",",
                )
                (cube / f"{seed_id}.mat").write_bytes(
                    f"opaque-mat-{label}-{replicate}-{seed_id}".encode("ascii")
                )


def _classification_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(19)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[int] = []
    for group in DEVELOPMENT_BATCHES:
        for label in range(3):
            for repeat in range(3):
                vector = rng.normal(0.0, 0.15, size=21)
                vector[label * 5 : label * 5 + 5] += 2.0
                vector += repeat * 0.005
                rows.append(vector)
                labels.append(label)
                groups.append(group)
    return np.asarray(rows), np.asarray(labels), np.asarray(groups)


class RecordingDecisionEstimator(BaseEstimator):
    """Fast test estimator that records group IDs seen by every cloned fit."""

    fit_group_sets: list[frozenset[int]] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> RecordingDecisionEstimator:
        values = np.asarray(X)
        type(self).fit_group_sets.append(
            frozenset(int(value) for value in values[:, -1])
        )
        self.classes_ = np.unique(y)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X)
        return values[:, : self.classes_.size]


class ManifestAndLoaderTests(unittest.TestCase):
    def test_manifest_assignment_is_deterministic_and_balanced(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data"
            _write_synthetic_repository(root)
            first = discover_manifest(
                root, expected_source_counts=TEST_COUNTS, hash_files=True
            )
            second = discover_manifest(
                root, expected_source_counts=TEST_COUNTS, hash_files=True
            )

            first_assignments = {
                record.sample_id: (record.constructed_batch, record.analysis_split)
                for record in first.records
            }
            second_assignments = {
                record.sample_id: (record.constructed_batch, record.analysis_split)
                for record in second.records
            }
            self.assertEqual(first_assignments, second_assignments)
            self.assertEqual(first.data_fingerprint_sha256, second.data_fingerprint_sha256)
            self.assertEqual(len(first.records), 160)
            self.assertEqual({record.class_name for record in first.records}, set(CLASS_NAMES))
            randomized_order = np.random.default_rng(20260721 + 1 * 1009).permutation(10)
            expected_batches = np.empty(10, dtype=int)
            for rank, source_index in enumerate(randomized_order):
                expected_batches[source_index] = rank % 10
            observed_batches = [
                record.constructed_batch
                for record in first.records
                if record.label == 0 and record.replicate == 1
            ]
            self.assertEqual(observed_batches, expected_batches.tolist())
            self.assertEqual(observed_batches, [9, 3, 8, 4, 0, 1, 6, 5, 7, 2])
            for label in range(8):
                for replicate in (1, 2):
                    batches = {
                        record.constructed_batch
                        for record in first.records
                        if record.label == label and record.replicate == replicate
                    }
                    self.assertEqual(batches, set(range(10)))
            self.assertTrue(all(record.csv_sha256 for record in first.records))
            self.assertTrue(all(record.mat_sha256 for record in first.records))

    def test_manifest_discovery_does_not_parse_numeric_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data"
            _write_synthetic_repository(root)
            # Seed 2 is development batch 3 under the frozen inverse permutation.
            (root / "0-1" / "2.csv").write_text("not,numeric\n", encoding="utf-8")
            with patch(
                "provenance_study.core.sha256_file",
                side_effect=AssertionError("default discovery opened file content"),
            ):
                manifest = discover_manifest(root, expected_source_counts=TEST_COUNTS)
            self.assertEqual(len(manifest.records), 160)
            self.assertFalse(manifest.hashes_complete)
            self.assertFalse(manifest.csv_content_sha256)
            self.assertFalse(manifest.mat_content_sha256)
            with self.assertRaisesRegex(ValueError, "two-column spectrum|Expected"):
                load_csv_split(manifest, split="development", expected_bands=TEST_BANDS)

    def test_development_loader_hard_rejects_locked_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data"
            _write_synthetic_repository(root)
            manifest = discover_manifest(root, expected_source_counts=TEST_COUNTS)
            locked = manifest.records_for_split("locked")
            self.assertTrue(locked)
            with self.assertRaises(LockedDataAccessError):
                load_development_csv(manifest.records, expected_bands=TEST_BANDS)

            development = load_csv_split(
                manifest, split="development", expected_bands=TEST_BANDS
            )
            locked_dataset = load_csv_split(
                manifest, split="locked", expected_bands=TEST_BANDS
            )
            self.assertEqual(development.X.shape, (128, TEST_BANDS))
            self.assertEqual(locked_dataset.X.shape, (32, TEST_BANDS))
            self.assertEqual(set(development.y), set(range(8)))
            self.assertEqual(set(locked_dataset.y), set(range(8)))
            self.assertTrue(
                {record.sample_id for record in development.records}.isdisjoint(
                    {record.sample_id for record in locked_dataset.records}
                )
            )

    def test_manifest_rejects_a_missing_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data"
            _write_synthetic_repository(root)
            (root / "3-2" / "4.mat").unlink()
            with self.assertRaisesRegex(ValueError, "CSV/MAT stem mismatch"):
                discover_manifest(root, expected_source_counts=TEST_COUNTS)


class PreprocessingAndModelTests(unittest.TestCase):
    def test_snv_has_zero_mean_and_unit_sample_sd(self) -> None:
        X = np.asarray([[1.0, 2.0, 4.0, 8.0], [2.0, 5.0, 6.0, 9.0]])
        transformed = StandardNormalVariate().fit_transform(X)
        np.testing.assert_allclose(transformed.mean(axis=1), 0.0, atol=1e-12)
        np.testing.assert_allclose(transformed.std(axis=1, ddof=1), 1.0, atol=1e-12)

    def test_msc_removes_affine_scatter_using_training_reference(self) -> None:
        reference = np.linspace(1.0, 3.0, TEST_BANDS)
        transformer = MultiplicativeScatterCorrection().fit(np.vstack([reference, reference]))
        scattered = np.vstack([2.0 + 3.0 * reference, -1.0 + 0.5 * reference])
        corrected = transformer.transform(scattered)
        np.testing.assert_allclose(corrected, np.vstack([reference, reference]), atol=1e-10)

    def test_savgol_and_all_frozen_model_builders_are_sklearn_compatible(self) -> None:
        X, y, _ = _classification_fixture()
        transformed = SavitzkyGolayTransformer(15, 2, 1).fit_transform(X)
        self.assertEqual(transformed.shape, X.shape)

        lda = build_sg15_shrinkage_lda().fit(X, y)
        lr = build_sg15_logistic_regression().fit(X, y)
        svm = build_sg15_rbf_svm().fit(X, y)
        self.assertEqual(lda.predict_proba(X[:4]).shape, (4, 3))
        self.assertEqual(lr.predict_proba(X[:4]).shape, (4, 3))
        self.assertEqual(svm.decision_function(X[:4]).shape, (4, 3))
        svm_classifier = svm.named_steps["classifier"]
        self.assertEqual(svm_classifier.C, 10.0)
        self.assertEqual(svm_classifier.kernel, "rbf")
        self.assertEqual(svm_classifier.gamma, "scale")
        self.assertEqual(svm_classifier.decision_function_shape, "ovr")
        self.assertFalse(svm_classifier.probability)
        lr_classifier = lr.named_steps["classifier"]
        self.assertEqual(lr_classifier.max_iter, 5000)
        self.assertEqual(lr_classifier.tol, 1e-4)
        self.assertEqual(lr_classifier.random_state, 20260721)


class OOFMetricsAndTemperatureTests(unittest.TestCase):
    def test_metrics_match_manual_multiclass_values(self) -> None:
        y = np.asarray([0, 1, 2, 1])
        probabilities = np.asarray(
            [
                [0.8, 0.1, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.3, 0.6],
                [0.1, 0.6, 0.3],
            ]
        )
        metrics = multiclass_metrics(y, probabilities, classes=[0, 1, 2], ece_bins=5)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        expected_brier = np.mean(
            np.sum((probabilities - np.eye(3)[y]) ** 2, axis=1)
        )
        self.assertAlmostEqual(metrics["multiclass_brier_score"], expected_brier)
        self.assertGreater(metrics["negative_log_likelihood"], 0.0)

    def test_fixed_grouped_oof_probability_and_decision_paths(self) -> None:
        X, y, groups = _classification_fixture()
        lr = Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2000, tol=1e-10)),
            ]
        )
        probability_result = grouped_oof_probabilities(lr, X, y, groups)
        decision_result = grouped_oof_decision_scores(build_sg15_rbf_svm(), X, y, groups)
        self.assertEqual(probability_result.probabilities.shape, (len(y), 3))
        self.assertEqual(decision_result.decision_scores.shape, (len(y), 3))
        np.testing.assert_allclose(probability_result.probabilities.sum(axis=1), 1.0)
        self.assertEqual(set(probability_result.held_out_batch), set(DEVELOPMENT_BATCHES))
        self.assertEqual(len(probability_result.fold_metrics), 8)

    def test_temperature_uses_decision_scores_and_equal_average_is_normalized(self) -> None:
        y = np.tile(np.arange(3), 8)
        groups = np.repeat(np.arange(8), 3)
        scores = np.full((y.size, 3), -4.0)
        scores[np.arange(y.size), y] = 12.0
        # Inject systematic overconfident mistakes so T > 1 improves NLL.
        for row in (0, 7, 14, 21):
            wrong = (int(y[row]) + 1) % 3
            scores[row, wrong] = 16.0
        temperature = fit_decision_temperature(scores, y, classes=[0, 1, 2])
        self.assertGreater(temperature, 1.0)
        raw = decision_scores_to_probabilities(scores, 1.0)
        calibrated = decision_scores_to_probabilities(scores, temperature)
        self.assertLessEqual(
            multiclass_metrics(y, calibrated)["negative_log_likelihood"],
            multiclass_metrics(y, raw)["negative_log_likelihood"],
        )

        crossfit = crossfit_oof_decision_temperature(
            scores, y, groups, classes=[0, 1, 2]
        )
        self.assertEqual(len(crossfit.fold_temperatures), 8)
        self.assertGreater(crossfit.final_temperature, 0.0)
        np.testing.assert_allclose(crossfit.probabilities.sum(axis=1), 1.0)
        averaged = equal_weight_probability_average([raw, calibrated, crossfit.probabilities])
        np.testing.assert_allclose(averaged.sum(axis=1), 1.0)

    def test_nested_temperature_never_fits_on_the_outer_batch(self) -> None:
        rows: list[np.ndarray] = []
        labels: list[int] = []
        groups: list[int] = []
        for group in DEVELOPMENT_BATCHES:
            for label in range(3):
                scores = np.full(3, -1.0)
                scores[label] = 2.0
                rows.append(np.concatenate([scores, [float(group)]]))
                labels.append(label)
                groups.append(group)
        X = np.asarray(rows)
        y = np.asarray(labels)
        batch_ids = np.asarray(groups)
        RecordingDecisionEstimator.fit_group_sets = []
        result = nested_grouped_oof_temperature_probabilities(
            RecordingDecisionEstimator(), X, y, batch_ids
        )

        # Each outer block has seven inner fits plus one outer refit.  Every one
        # must exclude that outer group.  The final eight fits after these blocks
        # are the separate all-development OOF calculation for deployable T.
        self.assertEqual(len(RecordingDecisionEstimator.fit_group_sets), 72)
        for outer_group in DEVELOPMENT_BATCHES:
            start = outer_group * 8
            for fitted_groups in RecordingDecisionEstimator.fit_group_sets[start : start + 8]:
                self.assertNotIn(outer_group, fitted_groups)
        np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0)
        self.assertEqual(len(result.fold_temperatures), 8)
        self.assertGreater(result.final_temperature, 0.0)


if __name__ == "__main__":
    unittest.main()
