from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from provenance_study.core import LockedDataAccessError, SampleRecord, SpectralDataset
from provenance_study.explore_development import (
    MODEL_ORDER,
    build_manifest_rows,
    build_prediction_rows,
    hash_development_csv_records,
    summarize_probability_models,
)


def _record(root: Path, index: int, label: int, split: str = "development") -> SampleRecord:
    csv_path = root / f"{index}.csv"
    mat_path = root / f"{index}.mat"
    csv_path.write_text(f"{index}\n", encoding="ascii")
    mat_path.write_bytes(b"opaque")
    batch = index % 8 if split == "development" else 8
    return SampleRecord(
        sample_index=index,
        label=label,
        class_name=("HBS", "HBX", "HNA", "HNX", "NX", "SXD", "SXQ", "XJH")[label],
        replicate=1,
        source_cube=f"{label}-1",
        seed_id=str(index),
        constructed_batch=batch,
        analysis_split=split,  # type: ignore[arg-type]
        csv_path=csv_path,
        mat_path=mat_path,
        relative_csv_path=f"{label}-1/{index}.csv",
        relative_mat_path=f"{label}-1/{index}.mat",
        csv_path_sha256="path-csv",
        mat_path_sha256="path-mat",
        csv_size_bytes=csv_path.stat().st_size,
        mat_size_bytes=mat_path.stat().st_size,
        csv_sha256="",
        mat_sha256="",
        record_sha256=f"record-{index}",
    )


class DevelopmentAccessTests(unittest.TestCase):
    def test_hashing_rejects_locked_record_before_any_file_open(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            development = _record(root, 0, 0)
            locked = _record(root, 1, 1, split="locked")
            with patch(
                "provenance_study.explore_development.sha256_file",
                side_effect=AssertionError("hash function must not be reached"),
            ) as mocked_hash:
                with self.assertRaises(LockedDataAccessError):
                    hash_development_csv_records([development, locked])
            mocked_hash.assert_not_called()

    def test_hashing_only_development_returns_per_file_and_aggregate_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records = [_record(root, 0, 0), _record(root, 1, 1)]
            hashes, aggregate = hash_development_csv_records(records)
            self.assertEqual(set(hashes), {record.relative_csv_path for record in records})
            self.assertEqual(len(aggregate), 64)
            self.assertTrue(all(len(value) == 64 for value in hashes.values()))


class DevelopmentArtifactConstructionTests(unittest.TestCase):
    def test_manifest_and_prediction_rows_are_development_only_and_have_eight_probabilities(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records = tuple(_record(root, index, index, split="development") for index in range(8))
            wavelengths = np.linspace(949.0, 1651.0, 21)
            dataset = SpectralDataset(
                X=np.zeros((8, 21)),
                y=np.arange(8),
                wavelengths=wavelengths,
                records=records,
                analysis_split="development",
                loaded_csv_fingerprint_sha256="synthetic",
            )
            probabilities = np.full((8, 8), 0.02)
            probabilities[np.arange(8), np.arange(8)] = 0.86
            probability_map = {model: probabilities.copy() for model in MODEL_ORDER}
            origins = [f"产地{index}" for index in range(8)]
            hashes = {record.relative_csv_path: "a" * 64 for record in records}

            manifest_rows = build_manifest_rows(records, hashes, origins)
            prediction_rows = build_prediction_rows(dataset, probability_map, origins)
            self.assertEqual(len(manifest_rows), 8)
            self.assertEqual(len(prediction_rows), 8 * len(MODEL_ORDER))
            probability_columns = [
                key for key in prediction_rows[0] if key.startswith("probability_")
            ]
            self.assertEqual(len(probability_columns), 8)
            self.assertTrue(all(row["analysis_split"] == "development" for row in manifest_rows))

    def test_metric_tables_have_five_models_and_eight_folds_each(self) -> None:
        y = np.tile(np.arange(8), 8)
        groups = np.repeat(np.arange(8), 8)
        probabilities = np.full((64, 8), 0.01)
        probabilities[np.arange(64), y] = 0.93
        probability_map = {model: probabilities.copy() for model in MODEL_ORDER}
        metric_rows, fold_rows = summarize_probability_models(
            y,
            groups,
            probability_map,
            classes=np.arange(8),
            ece_bins=10,
        )
        self.assertEqual(len(metric_rows), len(MODEL_ORDER))
        self.assertEqual(len(fold_rows), len(MODEL_ORDER) * 8)
        self.assertTrue(all(row["errors"] == 0 for row in metric_rows))
        self.assertTrue(
            all(row["equal_constructed_batch_accuracy"] == 1.0 for row in metric_rows)
        )


if __name__ == "__main__":
    unittest.main()
