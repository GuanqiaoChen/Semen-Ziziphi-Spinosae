from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
import torch

from provenance_study.cnn_baseline import (
    CNN_PARAMETER_COUNT,
    BandStandardizer,
    DEFAULT_CNN_CONFIG,
    ResidualSpectralCNN,
    count_trainable_parameters,
    evaluate_development_batches,
    fit_full_development_cnn,
    predict_standardized_probabilities,
)


def _synthetic_grouped_spectra() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One synthetic observation per class in each of eight groups."""

    generator = np.random.default_rng(17)
    wavelengths = np.linspace(-1.0, 1.0, 15, dtype=np.float32)
    spectra: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[int] = []
    for group in range(8):
        for label in range(8):
            signal = (
                0.15 * label * wavelengths
                + 0.05 * np.sin((label + 1) * np.pi * wavelengths)
                + 0.01 * group * wavelengths**2
            )
            spectra.append(
                signal + generator.normal(0.0, 0.005, wavelengths.size)
            )
            labels.append(label)
            groups.append(group)
    return (
        np.asarray(spectra, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(groups, dtype=np.int64),
    )


class ArchitectureAndPredictionTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA regression requires a CUDA device")
    def test_unindexed_cuda_request_matches_concrete_model_device(self):
        model = ResidualSpectralCNN().to("cuda")
        values = np.zeros((2, 392), dtype=np.float32)
        probabilities = predict_standardized_probabilities(
            model, values, batch_size=2, device="cuda"
        )
        self.assertEqual(probabilities.shape, (2, 8))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-7)

    def test_parameter_count_forward_and_probabilities(self) -> None:
        model = ResidualSpectralCNN()
        self.assertEqual(count_trainable_parameters(model), CNN_PARAMETER_COUNT)
        self.assertEqual(CNN_PARAMETER_COUNT, 321_776)

        inputs = np.random.default_rng(3).normal(size=(4, 31)).astype(np.float32)
        with torch.inference_mode():
            logits = model(torch.as_tensor(inputs).unsqueeze(1))
        self.assertEqual(tuple(logits.shape), (4, 8))

        probabilities = predict_standardized_probabilities(
            model, inputs, batch_size=2, device="cpu"
        )
        self.assertEqual(probabilities.shape, (4, 8))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
        self.assertTrue(np.all(probabilities >= 0.0))

    def test_standardizer_fits_training_values_only(self) -> None:
        training = np.asarray([[0.0, 2.0], [2.0, 6.0]], dtype=np.float32)
        external = np.asarray([[101.0, 204.0]], dtype=np.float32)
        standardizer = BandStandardizer.fit(training)

        np.testing.assert_allclose(standardizer.mean, [1.0, 4.0])
        np.testing.assert_allclose(standardizer.scale, [1.0, 2.0])
        self.assertEqual(standardizer.n_samples_seen, 2)
        np.testing.assert_allclose(
            standardizer.transform(training), [[-1.0, -1.0], [1.0, 1.0]]
        )
        np.testing.assert_allclose(
            standardizer.transform(external), [[100.0, 100.0]]
        )


class SyntheticTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.short_config = replace(
            DEFAULT_CNN_CONFIG,
            max_epochs=1,
            min_epochs=1,
            patience=1,
            batch_size=64,
        )

    def test_one_epoch_full_development_fit_and_external_prediction(self) -> None:
        X, y, _ = _synthetic_grouped_spectra()
        fitted = fit_full_development_cnn(
            X,
            y,
            epochs=1,
            optimization_seed=19,
            config=self.short_config,
            device="cpu",
        )
        self.assertEqual(fitted.epochs, 1)
        self.assertEqual(fitted.standardizer.n_samples_seen, len(X))
        probabilities = fitted.predict_proba(X[:5], batch_size=2)
        self.assertEqual(probabilities.shape, (5, 8))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)

    def test_one_epoch_nested_evaluation_preserves_group_roles(self) -> None:
        X, y, groups = _synthetic_grouped_spectra()
        result = evaluate_development_batches(
            X,
            y,
            groups,
            optimization_seed=23,
            config=self.short_config,
            device="cpu",
        )
        self.assertEqual(result.probabilities.shape, (64, 8))
        self.assertEqual(result.parameter_count, CNN_PARAMETER_COUNT)
        np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0, atol=1e-6)
        self.assertEqual(set(result.held_out_batch.tolist()), set(range(8)))
        self.assertEqual(len(result.folds), 8)
        for fold in result.folds:
            self.assertEqual(fold.inner_validation_batch, (fold.outer_batch + 1) % 8)
            self.assertEqual(fold.selected_epoch, 1)
            self.assertEqual(fold.early_stopping_epochs_run, 1)

    def test_development_evaluation_rejects_a_locked_batch_id(self) -> None:
        X, y, groups = _synthetic_grouped_spectra()
        groups[groups == 7] = 8
        with self.assertRaisesRegex(ValueError, "exactly constructed batches 0--7"):
            evaluate_development_batches(
                X,
                y,
                groups,
                config=self.short_config,
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
