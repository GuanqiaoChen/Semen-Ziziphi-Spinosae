from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from provenance_study.cnn_baseline import CNNFoldResult, CNNGroupedOOFResult
from provenance_study.run_cnn_development import (
    run_development_cnn,
    validate_development_batches,
)


TEST_BANDS = 15
TEST_COUNTS = {(label, source): 10 for label in range(8) for source in (1, 2)}


def _write_split_sentinel_repository(root: Path) -> None:
    wavelengths = np.linspace(950.0, 1650.0, TEST_BANDS)
    for label in range(8):
        for source in (1, 2):
            folder = root / f"{label}-{source}"
            folder.mkdir(parents=True)
            assignment_order = np.random.default_rng(
                20260721 + label * 101 + source * 1009
            ).permutation(10)
            assigned = np.empty(10, dtype=int)
            assigned[assignment_order] = np.arange(10) % 10
            for position, batch in enumerate(assigned):
                seed_id = position + 1
                csv_path = folder / f"{seed_id}.csv"
                if batch in (8, 9):
                    # Discovery may inspect this path and size, but numeric loading
                    # would fail immediately if the entry point crossed the boundary.
                    csv_path.write_text("LOCKED_NUMERIC_SENTINEL", encoding="ascii")
                else:
                    spectrum = (
                        0.02 * label * np.linspace(-1.0, 1.0, TEST_BANDS)
                        + 0.001 * source
                        + 0.0001 * seed_id
                    )
                    np.savetxt(
                        csv_path,
                        np.column_stack([wavelengths, spectrum]),
                        delimiter=",",
                    )
                (folder / f"{seed_id}.mat").write_bytes(b"opaque")


def _fake_evaluator(X, y, groups, **kwargs) -> CNNGroupedOOFResult:
    del X, kwargs
    groups = np.asarray(groups, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    if set(groups.tolist()) != set(range(8)):
        raise AssertionError("Non-development batch reached the evaluator")
    probabilities = np.full((len(y), 8), 0.01 / 7.0, dtype=np.float64)
    probabilities[np.arange(len(y)), y] = 0.99
    folds = tuple(
        CNNFoldResult(
            outer_batch=batch,
            inner_validation_batch=(batch + 1) % 8,
            selected_epoch=1,
            early_stopping_epochs_run=1,
            inner_validation_balanced_accuracy=1.0,
            outer_balanced_accuracy=1.0,
            outer_macro_f1=1.0,
        )
        for batch in range(8)
    )
    return CNNGroupedOOFResult(
        probabilities=probabilities,
        classes=np.arange(8, dtype=np.int64),
        held_out_batch=groups.copy(),
        folds=folds,
        optimization_seed=20260721,
        parameter_count=321_776,
        elapsed_seconds=0.125,
    )


class DevelopmentCNNSerializationTests(unittest.TestCase):
    def test_batch_validator_rejects_locked_or_incomplete_requests(self) -> None:
        self.assertEqual(validate_development_batches(range(8)), tuple(range(8)))
        with self.assertRaisesRegex(ValueError, "exactly batches 0--7"):
            validate_development_batches([0, 1, 2, 3, 4, 5, 6, 8])
        with self.assertRaisesRegex(ValueError, "exactly batches 0--7"):
            validate_development_batches([0, 1, 2])

    def test_serializes_four_artifacts_without_reading_locked_sentinels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_root = root / "data"
            output = root / "outputs"
            _write_split_sentinel_repository(data_root)

            with patch(
                "provenance_study.run_cnn_development.evaluate_development_batches",
                side_effect=_fake_evaluator,
            ) as evaluator:
                results = run_development_cnn(
                    data_root,
                    output,
                    expected_source_counts=TEST_COUNTS,
                    expected_bands=TEST_BANDS,
                    device="cpu",
                )

            self.assertEqual(evaluator.call_count, 1)
            self.assertEqual(results["data_access"]["locked_numeric_reads"], 0)
            self.assertEqual(results["data_access"]["loaded_constructed_batches"], list(range(8)))
            self.assertFalse(results["manifest"]["hashes_complete"])
            self.assertEqual(results["manifest"]["n_all_path_records"], 160)
            self.assertEqual(results["manifest"]["n_development_records_loaded"], 128)
            self.assertEqual(results["cnn"]["architecture"]["parameter_count"], 321_776)

            expected_files = {"predictions.csv", "folds.csv", "results.json", "report.md"}
            self.assertEqual({path.name for path in output.iterdir()}, expected_files)
            with (output / "predictions.csv").open(encoding="utf-8", newline="") as handle:
                prediction_rows = list(csv.DictReader(handle))
            self.assertEqual(len(prediction_rows), 128)
            self.assertEqual(
                {int(row["constructed_batch"]) for row in prediction_rows}, set(range(8))
            )
            self.assertEqual(
                len([name for name in prediction_rows[0] if name.startswith("probability_")]),
                8,
            )
            with (output / "folds.csv").open(encoding="utf-8", newline="") as handle:
                fold_rows = list(csv.DictReader(handle))
            self.assertEqual(len(fold_rows), 8)
            persisted = json.loads((output / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["data_access"]["locked_records_loaded"], 0)
            report = (output / "report.md").read_text(encoding="utf-8")
            self.assertIn("锁定批次数值读取次数：**0**", report)
            self.assertIn("不是锁定测试结果", report)


if __name__ == "__main__":
    unittest.main()
