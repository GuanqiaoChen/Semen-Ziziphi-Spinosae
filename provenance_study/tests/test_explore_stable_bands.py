from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from provenance_study.core import (
    CLASS_NAMES,
    DEVELOPMENT_BATCHES,
    LockedDataAccessError,
    SampleRecord,
    SpectralDataset,
    build_sg15_shrinkage_lda,
    grouped_oof_probabilities,
    multiclass_metrics,
)
from provenance_study.explore_stable_bands import (
    EXPECTED_CANDIDATE_COUNT,
    EXPECTED_SELECTOR_COUNT,
    SelectionDefinition,
    _baseline_oof,
    _probability_metrics,
    evaluate_stable_band_candidates,
    inner_rank_matrix,
    run_exploration,
    selection_definitions,
    sg_first_derivative,
    write_artifacts,
)


def _synthetic_arrays(
    *, n_bands: int = 24, repetitions: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260721)
    wavelengths = np.linspace(950.0, 1650.0, n_bands)
    labels = []
    groups = []
    rows = []
    band_axis = np.linspace(-1.0, 1.0, n_bands)
    for batch in DEVELOPMENT_BATCHES:
        for label in range(8):
            for repetition in range(repetitions):
                class_signal = 0.08 * label * np.sin((label + 1) * band_axis)
                batch_signal = 0.01 * batch * band_axis
                rows.append(
                    0.4
                    + class_signal
                    + batch_signal
                    + rng.normal(0.0, 0.004, size=n_bands)
                )
                labels.append(label)
                groups.append(batch)
    return (
        np.asarray(rows, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(groups, dtype=np.int64),
        wavelengths,
    )


def _record(root: Path, index: int, label: int, batch: int) -> SampleRecord:
    csv_path = root / f"{index}.csv"
    mat_path = root / f"{index}.mat"
    return SampleRecord(
        sample_index=index,
        label=label,
        class_name=CLASS_NAMES[label],
        replicate=1,
        source_cube=f"{label}-1",
        seed_id=str(index),
        constructed_batch=batch,
        analysis_split="development",
        csv_path=csv_path,
        mat_path=mat_path,
        relative_csv_path=f"{label}-1/{index}.csv",
        relative_mat_path=f"{label}-1/{index}.mat",
        csv_path_sha256="path-csv",
        mat_path_sha256="path-mat",
        csv_size_bytes=0,
        mat_size_bytes=0,
        csv_sha256="",
        mat_sha256="",
        record_sha256=f"record-{index}",
    )


class StableBandProtocolTests(unittest.TestCase):
    def test_frozen_panel_has_66_selectors_and_132_model_candidates(self) -> None:
        definitions = selection_definitions()
        self.assertEqual(len(definitions), EXPECTED_SELECTOR_COUNT)
        self.assertEqual(2 * len(definitions), EXPECTED_CANDIDATE_COUNT)
        self.assertEqual(
            {definition.scorer for definition in definitions}, {"ANOVA", "LDAcoef"}
        )
        self.assertEqual(
            {definition.mode for definition in definitions}, {"fixed", "consensus"}
        )

    def test_inner_rankings_are_invariant_to_outer_fold_values(self) -> None:
        X, y, groups, _ = _synthetic_arrays()
        transformed = X.copy()
        outer = 3
        for scorer in ("ANOVA", "LDAcoef"):
            original = inner_rank_matrix(
                transformed, y, groups, outer_batch=outer, scorer=scorer
            )
            changed_X = transformed.copy()
            changed_y = y.copy()
            mask = groups == outer
            changed_X[mask] = 1_000_000.0 * np.arange(1, X.shape[1] + 1)
            changed_y[mask] = changed_y[mask][::-1]
            changed = inner_rank_matrix(
                changed_X, changed_y, groups, outer_batch=outer, scorer=scorer
            )
            np.testing.assert_array_equal(original, changed)

    def test_nll_uses_the_shared_project_probability_floor(self) -> None:
        y = np.tile(np.arange(8), 8)
        groups = np.repeat(np.arange(8), 8)
        probabilities = np.full((64, 8), 0.01 / 7.0)
        probabilities[np.arange(64), y] = 0.99
        probabilities[0] = np.asarray(
            [1e-18, 1.0 - 1e-18, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        shared = multiclass_metrics(y, probabilities, classes=np.arange(8))
        stable = _probability_metrics(
            y, groups, probabilities, classes=np.arange(8)
        )
        self.assertEqual(stable["log_loss"], shared["negative_log_likelihood"])
        nll_with_incorrect_floor = float(
            -np.log(
                np.clip(probabilities[np.arange(y.size), y], 1e-12, 1.0)
            ).mean()
        )
        self.assertGreater(stable["log_loss"], nll_with_incorrect_floor)

    def test_full_spectrum_baseline_probabilities_match_the_formal_pipeline(self) -> None:
        X, y, groups, _ = _synthetic_arrays()
        classes = np.arange(8)
        stable_probabilities = _baseline_oof(
            sg_first_derivative(X), y, groups, classes
        )
        formal = grouped_oof_probabilities(
            build_sg15_shrinkage_lda(), X, y, groups
        )
        np.testing.assert_array_equal(formal.held_out_batch, groups)
        np.testing.assert_allclose(
            stable_probabilities, formal.probabilities, rtol=1e-12, atol=1e-14
        )

    def test_locked_batch_guard_runs_before_sg_or_model_fitting(self) -> None:
        X, y, groups, wavelengths = _synthetic_arrays()
        groups[0] = 8
        with patch(
            "provenance_study.explore_stable_bands.sg_first_derivative",
            side_effect=AssertionError("SG must not run after locked input"),
        ) as mocked_sg:
            with self.assertRaises(LockedDataAccessError):
                evaluate_stable_band_candidates(X, y, groups, wavelengths)
        mocked_sg.assert_not_called()

    def test_repository_entry_uses_blind_manifest_and_passes_only_development_records(self) -> None:
        X, y, groups, wavelengths = _synthetic_arrays()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records = tuple(
                _record(root, index, int(label), int(batch))
                for index, (label, batch) in enumerate(zip(y, groups, strict=True))
            )
            dataset = SpectralDataset(
                X=X,
                y=y,
                wavelengths=wavelengths,
                records=records,
                analysis_split="development",
                loaded_csv_fingerprint_sha256="synthetic-development",
            )
            manifest = SimpleNamespace(
                hashes_complete=False,
                data_fingerprint_sha256="synthetic-structure",
                records_for_split=lambda split: records if split == "development" else (),
            )
            evaluation = SimpleNamespace(
                candidate_rows=tuple({"candidate": index} for index in range(132))
            )
            expected_summary = root / "out" / "summary.json"
            with (
                patch(
                    "provenance_study.explore_stable_bands.discover_manifest",
                    return_value=manifest,
                ) as mocked_discover,
                patch(
                    "provenance_study.explore_stable_bands.load_development_csv",
                    return_value=dataset,
                ) as mocked_load,
                patch(
                    "provenance_study.explore_stable_bands.evaluate_stable_band_candidates",
                    return_value=evaluation,
                ),
                patch(
                    "provenance_study.explore_stable_bands.write_artifacts",
                    return_value=expected_summary,
                ),
            ):
                observed = run_exploration(root / "data", root / "out")
            self.assertEqual(observed, expected_summary)
            self.assertFalse(mocked_discover.call_args.kwargs["hash_files"])
            loaded_records = mocked_load.call_args.args[0]
            self.assertEqual(loaded_records, records)
            self.assertTrue(
                all(record.constructed_batch in DEVELOPMENT_BATCHES for record in loaded_records)
            )
            self.assertFalse(mocked_load.call_args.kwargs["verify_hashes"])

    def test_synthetic_subset_writes_all_artifacts_and_zero_locked_reads(self) -> None:
        X, y, groups, wavelengths = _synthetic_arrays()
        definitions = (
            SelectionDefinition("ANOVA:fixed:8", "ANOVA", "fixed", 8, None),
            SelectionDefinition(
                "LDAcoef:consensus:8:0.67", "LDAcoef", "consensus", 8, 0.67
            ),
        )
        evaluation = evaluate_stable_band_candidates(
            X, y, groups, wavelengths, definitions=definitions
        )
        self.assertEqual(len(evaluation.candidate_rows), 4)
        self.assertEqual(len(evaluation.fold_selection_rows), 4 * 8)
        self.assertTrue(
            all(row["promotion_eligible"] is False for row in evaluation.candidate_rows)
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records = tuple(
                _record(root, index, int(label), int(batch))
                for index, (label, batch) in enumerate(zip(y, groups, strict=True))
            )
            dataset = SpectralDataset(
                X=X,
                y=y,
                wavelengths=wavelengths,
                records=records,
                analysis_split="development",
                loaded_csv_fingerprint_sha256="synthetic-development",
            )
            summary_path = write_artifacts(
                root / "outputs",
                evaluation,
                dataset=dataset,
                manifest_fingerprint="synthetic-structure",
            )
            expected = {
                "candidates.csv",
                "fold_selections.csv",
                "intervals.csv",
                "selection_frequency.csv",
                "summary.json",
                "report.md",
            }
            self.assertEqual(
                {path.name for path in summary_path.parent.iterdir()}, expected
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["data_access"]["locked_numeric_reads"], 0)
            self.assertEqual(summary["data_access"]["locked_byte_reads"], 0)
            self.assertEqual(summary["data_access"]["mat_numeric_reads"], 0)
            self.assertFalse(summary["disposition"]["enter_primary_method"])
            report = (summary_path.parent / "report.md").read_text(encoding="utf-8")
            self.assertIn("不进入冻结主方法", report)


if __name__ == "__main__":
    unittest.main()
