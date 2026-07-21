from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from analyze import (  # noqa: E402
    MultiplicativeScatterCorrection,
    PLSDAClassifier,
    SavitzkyGolayTransformer,
    load_dataset,
    snv_transform,
    wilson_interval,
)


class PreprocessingTests(unittest.TestCase):
    def test_snv_has_zero_mean_and_unit_sample_std(self):
        X = np.array([[1.0, 2.0, 4.0, 8.0], [2.0, 5.0, 6.0, 9.0]])
        transformed = snv_transform(X)
        np.testing.assert_allclose(transformed.mean(axis=1), 0.0, atol=1e-12)
        np.testing.assert_allclose(transformed.std(axis=1, ddof=1), 1.0, atol=1e-12)

    def test_msc_corrects_affine_scatter_against_training_reference(self):
        reference = np.linspace(1.0, 3.0, 20)
        training = np.vstack([reference, reference])
        transformer = MultiplicativeScatterCorrection().fit(training)
        test = np.vstack([2.0 + 3.0 * reference, -1.0 + 0.5 * reference])
        corrected = transformer.transform(test)
        np.testing.assert_allclose(corrected, np.vstack([reference, reference]), atol=1e-10)

    def test_savgol_preserves_shape(self):
        X = np.vstack([np.linspace(0, 1, 21), np.linspace(1, 0, 21)])
        transformed = SavitzkyGolayTransformer(7, 2, 1).fit(X).transform(X)
        self.assertEqual(transformed.shape, X.shape)

    def test_known_wilson_interval(self):
        low, high = wilson_interval(50, 100)
        self.assertAlmostEqual(low, 0.4038315, places=6)
        self.assertAlmostEqual(high, 0.5961685, places=6)

    def test_pls_da_returns_declared_class_indices(self):
        rng = np.random.default_rng(7)
        X = np.vstack([rng.normal(-2, 0.1, size=(12, 8)), rng.normal(2, 0.1, size=(12, 8))])
        y = np.repeat([0, 1], 12)
        classifier = PLSDAClassifier(n_components=2, n_classes=2).fit(X, y)
        prediction = classifier.predict(X)
        self.assertEqual(prediction.shape, y.shape)
        self.assertTrue(set(prediction).issubset({0, 1}))


class LoaderTests(unittest.TestCase):
    def test_loader_retains_hierarchy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data"
            wavelengths = np.arange(5, dtype=float) + 900
            for cube in ("0-1", "0-2"):
                folder = root / cube
                folder.mkdir(parents=True)
                np.savetxt(
                    folder / "1.csv",
                    np.column_stack([wavelengths, np.linspace(0.1, 0.5, 5)]),
                    delimiter=",",
                )
            dataset = load_dataset(root, expected_bands=5)
            self.assertEqual(dataset.X.shape, (2, 5))
            self.assertEqual(dataset.manifest["source_cube"].tolist(), ["0-1", "0-2"])
            self.assertEqual(dataset.manifest["seed_id"].tolist(), ["1", "1"])
            self.assertTrue(dataset.fingerprint_sha256)


if __name__ == "__main__":
    unittest.main()
