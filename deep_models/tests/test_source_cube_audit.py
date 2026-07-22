from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "source_cube_audit.py"
SPEC = importlib.util.spec_from_file_location("source_cube_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
study = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = study
SPEC.loader.exec_module(study)


def make_samples(n_per_cube: int = 5):
    samples = []
    index = 0
    for label in range(study.NUM_CLASSES):
        for suffix in (1, 2):
            for seed_id in range(n_per_cube):
                cube = f"{label}-{suffix}"
                samples.append(
                    study.Sample(
                        sample_index=index,
                        label=label,
                        source_cube=cube,
                        cube_suffix=suffix,
                        seed_id=str(seed_id),
                        mat_path=Path(cube) / f"{seed_id}.mat",
                        csv_path=Path(cube) / f"{seed_id}.csv",
                        relative_mat_path=f"{cube}/{seed_id}.mat",
                        relative_csv_path=f"{cube}/{seed_id}.csv",
                    )
                )
                index += 1
    return samples


class ProtocolTests(unittest.TestCase):
    def test_reciprocal_split_has_complete_cube_isolation(self):
        samples = make_samples()
        for train_suffix in (1, 2):
            development, test = study.reciprocal_source_cube_split(samples, train_suffix)
            self.assertFalse(
                {sample.source_cube for sample in development}
                & {sample.source_cube for sample in test}
            )
            self.assertEqual({sample.cube_suffix for sample in development}, {train_suffix})
            self.assertEqual({sample.cube_suffix for sample in test}, {3 - train_suffix})

    def test_internal_validation_is_deterministic_complete_and_disjoint(self):
        development, test = study.reciprocal_source_cube_split(make_samples(10), 1)
        train_a, validation_a = study.internal_validation_split(development, 42)
        train_b, validation_b = study.internal_validation_split(development, 42)
        self.assertEqual([sample.sample_id for sample in train_a], [sample.sample_id for sample in train_b])
        self.assertEqual(
            [sample.sample_id for sample in validation_a],
            [sample.sample_id for sample in validation_b],
        )
        self.assertFalse({sample.sample_id for sample in train_a} & {sample.sample_id for sample in validation_a})
        self.assertEqual(
            {sample.sample_id for sample in train_a + validation_a},
            {sample.sample_id for sample in development},
        )
        self.assertFalse(
            {sample.source_cube for sample in train_a + validation_a}
            & {sample.source_cube for sample in test}
        )

    def test_counterfactuals_preserve_declared_information(self):
        rng = np.random.default_rng(3)
        cube = rng.uniform(0.1, 0.9, size=(4, 5, 7)).astype(np.float32)
        mask = np.zeros((4, 5), dtype=bool)
        mask[1:4, 1:4] = True
        full = study.apply_counterfactual(cube, mask, "full", "0-1/1")
        shuffled_a = study.apply_counterfactual(cube, mask, "spatial_shuffle", "0-1/1")
        shuffled_b = study.apply_counterfactual(cube, mask, "spatial_shuffle", "0-1/1")
        broadcast = study.apply_counterfactual(cube, mask, "mean_broadcast", "0-1/1")
        mask_only = study.apply_counterfactual(cube, mask, "mask_only", "0-1/1")
        np.testing.assert_array_equal(shuffled_a, shuffled_b)
        np.testing.assert_allclose(np.sort(full[mask], axis=0), np.sort(shuffled_a[mask], axis=0))
        np.testing.assert_allclose(full[mask].mean(axis=0), shuffled_a[mask].mean(axis=0))
        np.testing.assert_allclose(broadcast[mask], np.repeat(full[mask].mean(axis=0)[None, :], mask.sum(), axis=0))
        np.testing.assert_array_equal(mask_only[mask], np.ones((mask.sum(), cube.shape[-1])))
        np.testing.assert_array_equal(mask_only[~mask], 0.0)

    def test_snv_and_temperature_metrics_are_finite(self):
        spectra = np.asarray([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]])
        transformed = study.snv(spectra)
        self.assertAlmostEqual(float(transformed[0].mean()), 0.0)
        np.testing.assert_array_equal(transformed[1], 0.0)
        logits = np.eye(study.NUM_CLASSES) * 2.0
        labels = np.arange(study.NUM_CLASSES)
        temperature = study.select_temperature(logits, labels)
        metrics = study.multiclass_metrics(labels, study.softmax_numpy(logits, temperature))
        self.assertTrue(0.25 <= temperature <= 4.0)
        self.assertEqual(metrics["accuracy"], 1.0)
        for key in ("nll", "brier", "ece_10"):
            self.assertTrue(np.isfinite(metrics[key]))

    def test_checkpoint_rule_uses_f1_then_nll_then_earliest(self):
        self.assertTrue(study.checkpoint_is_better(0.81, 1.2, 0.80, 0.5))
        self.assertTrue(study.checkpoint_is_better(0.80, 0.4, 0.80, 0.5))
        self.assertFalse(study.checkpoint_is_better(0.80, 0.5, 0.80, 0.5))
        self.assertFalse(study.checkpoint_is_better(0.79, 0.1, 0.80, 0.5))

    def test_probability_ensemble_and_primary_mechanism_estimand(self):
        rows = []
        for direction, suffix in (("suffix_1_to_2", 2), ("suffix_2_to_1", 1)):
            for condition in ("full", "spatial_shuffle"):
                for label in range(study.NUM_CLASSES):
                    for seed in study.DEFAULT_SEEDS:
                        probabilities = np.full(study.NUM_CLASSES, 0.01)
                        predicted = label
                        if condition == "spatial_shuffle" and label == 0:
                            predicted = 1
                        probabilities[predicted] = 0.93
                        probabilities /= probabilities.sum()
                        row = {
                            "run_id": f"{direction}__fusion_net__seed_{seed}",
                            "direction": direction,
                            "model": "fusion_net",
                            "seed": seed,
                            "condition": condition,
                            "sample_index": label,
                            "sample_id": f"{label}-{suffix}/1",
                            "source_cube": f"{label}-{suffix}",
                            "cube_suffix": suffix,
                            "seed_id": "1",
                            "true_label": label,
                            "raw_predicted_label": predicted,
                            "calibrated_predicted_label": predicted,
                        }
                        for class_index, probability in enumerate(probabilities):
                            row[f"raw_probability_{class_index}"] = float(probability)
                            row[f"calibrated_probability_{class_index}"] = float(probability)
                        rows.append(row)
        ensembles = study.probability_ensemble_predictions(rows, study.DEFAULT_SEEDS)
        ensemble_metrics, _ = study.ensemble_metric_records(ensembles)
        seed_metrics = []
        for direction in ("suffix_1_to_2", "suffix_2_to_1"):
            for condition in ("full", "spatial_shuffle"):
                for seed in study.DEFAULT_SEEDS:
                    selected = [
                        row for row in rows
                        if row["direction"] == direction
                        and row["condition"] == condition
                        and row["seed"] == seed
                    ]
                    probabilities = np.asarray(
                        [
                            [row[f"calibrated_probability_{label}"] for label in range(study.NUM_CLASSES)]
                            for row in selected
                        ]
                    )
                    calculated = study.multiclass_metrics(
                        [row["true_label"] for row in selected], probabilities
                    )
                    seed_metrics.append(
                        {
                            "direction": direction,
                            "model": "fusion_net",
                            "condition": condition,
                            "seed": seed,
                            "calibration": "temperature_scaled",
                            **calculated,
                        }
                    )
        theta, mechanism = study.compute_primary_estimands(ensemble_metrics, seed_metrics)
        calibrated_theta = next(
            row for row in theta if row["calibration"] == "temperature_scaled"
        )
        self.assertEqual(calibrated_theta["theta"], 1.0)
        self.assertAlmostEqual(mechanism["mean_delta"], 0.125)
        self.assertEqual(mechanism["positive_direction_by_seed_deltas"], 6)
        self.assertTrue(mechanism["limited_support_for_spatial_arrangement"])

    def test_mat_csv_mean_comparison(self):
        rng = np.random.default_rng(19)
        cube = rng.normal(size=(3, 4, 5))
        mask = np.zeros((3, 4), dtype=bool)
        mask[1:, 1:3] = True
        exported = cube[mask].mean(axis=0)
        comparison = study.compare_patch_mean_to_csv(cube, mask, exported)
        self.assertLessEqual(comparison["max_absolute_difference"], 1e-12)

    def test_discovery_uses_exact_csv_wavelengths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wavelengths = np.linspace(949.1, 1651.7, study.EXPECTED_BANDS)
            for label in range(study.NUM_CLASSES):
                for suffix in (1, 2):
                    cube = root / f"{label}-{suffix}"
                    cube.mkdir()
                    (cube / "1.mat").write_bytes(b"fixture")
                    np.savetxt(
                        cube / "1.csv",
                        np.column_stack((wavelengths, np.linspace(0.2, 0.8, study.EXPECTED_BANDS))),
                        delimiter=",",
                    )
            samples, observed, fingerprints = study.discover_samples(root)
            self.assertEqual(len(samples), 16)
            np.testing.assert_allclose(observed, wavelengths, rtol=0.0, atol=1e-6)
            self.assertEqual(len(fingerprints["csv_content_sha256"]), 64)
            self.assertEqual(len(fingerprints["mat_content_sha256"]), 64)
            self.assertTrue(all(len(sample.mat_sha256) == 64 for sample in samples))
            self.assertTrue(all(len(sample.csv_sha256) == 64 for sample in samples))


if __name__ == "__main__":
    unittest.main()
