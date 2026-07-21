from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


@unittest.skipUnless((OUTPUT_DIR / "metrics.csv").exists(), "analysis outputs not generated")
class GeneratedOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metrics = pd.read_csv(OUTPUT_DIR / "metrics.csv")
        cls.predictions = pd.read_csv(OUTPUT_DIR / "predictions.csv")
        cls.manifest = pd.read_csv(OUTPUT_DIR / "dataset_manifest.csv")

    def test_every_summary_metric_recomputes_from_predictions(self):
        for row in self.metrics.itertuples(index=False):
            selected = self.predictions[
                (self.predictions["protocol"] == row.protocol)
                & (self.predictions["model"] == row.model)
            ]
            y_true = selected["label"].to_numpy()
            y_pred = selected["predicted_label"].to_numpy()
            self.assertEqual(len(selected), row.n_test_seeds)
            self.assertAlmostEqual(accuracy_score(y_true, y_pred), row.accuracy)
            self.assertAlmostEqual(
                balanced_accuracy_score(y_true, y_pred), row.balanced_accuracy
            )
            self.assertAlmostEqual(
                f1_score(y_true, y_pred, average="macro", zero_division=0), row.macro_f1
            )

    def test_grouped_predictions_cover_expected_cubes(self):
        for model in self.metrics["model"].unique():
            forward = self.predictions[
                (self.predictions["protocol"] == "suffix_1_to_2")
                & (self.predictions["model"] == model)
            ]
            reverse = self.predictions[
                (self.predictions["protocol"] == "suffix_2_to_1")
                & (self.predictions["model"] == model)
            ]
            loco = self.predictions[
                (self.predictions["protocol"] == "leave_one_cube_out")
                & (self.predictions["model"] == model)
            ]
            self.assertTrue(forward["source_cube"].str.endswith("-2").all())
            self.assertTrue(reverse["source_cube"].str.endswith("-1").all())
            self.assertEqual(loco["sample_index"].nunique(), len(self.manifest))
            self.assertTrue((loco["fold"] == loco["source_cube"]).all())

    def test_legacy_random_baseline_counts(self):
        # With the locked package versions and strict LR tolerance, these counts
        # make silent changes in sample ordering, preprocessing, or solver
        # convergence visible during reproducibility checks.
        expected_correct = {
            "raw_lr": 275,
            "raw_svm": 229,
            "raw_pls_da": 297,
            "raw_rf": 207,
            "snv_lr": 305,
            "msc_lr": 299,
            "sg_smooth_lr": 274,
            "sg_first_derivative_lr": 307,
        }
        selected = self.metrics[self.metrics["protocol"] == "random_seed_holdout"].set_index(
            "model"
        )
        for model, expected in expected_correct.items():
            self.assertEqual(int(selected.loc[model, "n_test_seeds"]), 316)
            self.assertEqual(int(selected.loc[model, "n_correct"]), expected)


if __name__ == "__main__":
    unittest.main()
