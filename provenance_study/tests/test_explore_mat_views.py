from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from provenance_study.core import LockedDataAccessError, Manifest, SampleRecord
from provenance_study.explore_mat_views import (
    CLASSIFIER_ORDER,
    MORPHOLOGY_FEATURE_NAMES,
    OUTPUT_FILENAMES,
    MatViews,
    build_view_estimators,
    build_view_matrices,
    extract_mat_feature_views,
    load_development_mat_views,
    read_development_mat,
    run_mat_view_exploration,
    write_artifacts,
)


TEST_BANDS = 21


def _record(
    root: Path,
    *,
    index: int = 0,
    batch: int = 0,
    split: str = "development",
) -> SampleRecord:
    mat_path = root / f"{index}.mat"
    csv_path = root / f"{index}.csv"
    return SampleRecord(
        sample_index=index,
        label=index % 8,
        class_name=("HBS", "HBX", "HNA", "HNX", "NX", "SXD", "SXQ", "XJH")[
            index % 8
        ],
        replicate=1,
        source_cube=f"{index % 8}-1",
        seed_id=str(index),
        constructed_batch=batch,
        analysis_split=split,  # type: ignore[arg-type]
        csv_path=csv_path,
        mat_path=mat_path,
        relative_csv_path=f"{index % 8}-1/{index}.csv",
        relative_mat_path=f"{index % 8}-1/{index}.mat",
        csv_path_sha256="c" * 64,
        mat_path_sha256="m" * 64,
        csv_size_bytes=0,
        mat_size_bytes=0,
        csv_sha256="",
        mat_sha256="",
        record_sha256="r" * 64,
    )


def _radial_fixture() -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros((7, 7), dtype=bool)
    mask[1:6, 1:6] = True
    from scipy.ndimage import distance_transform_edt

    distance = distance_transform_edt(mask)
    inner = mask & (distance > np.median(distance[mask]))
    outer = mask & ~inner
    cube = np.zeros((7, 7, TEST_BANDS), dtype=np.float32)
    cube[outer] = 1.0
    cube[inner] = 3.0
    return cube, mask


class MatFeatureTests(unittest.TestCase):
    def test_frozen_feature_views_have_expected_values_and_dimensions(self) -> None:
        cube, mask = _radial_fixture()
        views, counts = extract_mat_feature_views(cube, mask)
        self.assertEqual(views["foreground_std"].shape, (TEST_BANDS,))
        self.assertEqual(views["foreground_iqr"].shape, (TEST_BANDS,))
        self.assertEqual(views["radial_inner_minus_outer"].shape, (TEST_BANDS,))
        self.assertEqual(views["morphology"].shape, (len(MORPHOLOGY_FEATURE_NAMES),))
        np.testing.assert_allclose(views["radial_inner_minus_outer"], 2.0)
        self.assertEqual(counts["foreground_pixels"], 25)
        self.assertEqual(counts["inner_pixels"] + counts["outer_pixels"], 25)
        self.assertTrue(all(np.all(np.isfinite(values)) for values in views.values()))

    def test_hdf5_reader_orients_chw_and_extracts_one_manifest_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            record = _record(root)
            cube, mask = _radial_fixture()
            with h5py.File(record.mat_path, "w") as handle:
                handle.create_dataset("patch_chw", data=np.transpose(cube, (2, 0, 1)))
                handle.create_dataset("crop_mask", data=mask.astype(np.uint8))
            loaded_cube, loaded_mask = read_development_mat(
                record, expected_bands=TEST_BANDS
            )
            np.testing.assert_allclose(loaded_cube, cube)
            np.testing.assert_array_equal(loaded_mask, mask)
            result = load_development_mat_views([record], expected_bands=TEST_BANDS)
            self.assertEqual(result.development_mat_numeric_reads, 1)
            self.assertEqual(result.foreground_std.shape, (1, TEST_BANDS))
            self.assertEqual(len(result.feature_manifest_rows), 1)
            self.assertEqual(
                len(result.feature_manifest_rows[0]["extracted_mat_feature_sha256"]), 64
            )
            self.assertEqual(
                result.feature_manifest_rows[0]["mat_content_sha256"],
                "not_computed_in_blind_development",
            )

    def test_locked_record_is_rejected_before_hdf5_io(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            locked = _record(root, batch=8, split="locked")
            with patch(
                "provenance_study.explore_mat_views.h5py.File",
                side_effect=AssertionError("HDF5 I/O must not be reached"),
            ) as mocked_file:
                with self.assertRaises(LockedDataAccessError):
                    read_development_mat(locked, expected_bands=TEST_BANDS)
            mocked_file.assert_not_called()

    def test_collection_preflight_rejects_late_locked_record_before_any_io(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            development = _record(root, index=0, batch=0, split="development")
            locked = _record(root, index=1, batch=9, split="locked")
            with patch(
                "provenance_study.explore_mat_views.h5py.File",
                side_effect=AssertionError("preflight must run before any HDF5 I/O"),
            ) as mocked_file:
                with self.assertRaises(LockedDataAccessError):
                    load_development_mat_views(
                        [development, locked], expected_bands=TEST_BANDS
                    )
            mocked_file.assert_not_called()


class ViewConstructionAndConfigurationTests(unittest.TestCase):
    def test_view_dimensions_match_frozen_early_fusion(self) -> None:
        n_samples = 4
        rng = np.random.default_rng(17)
        mean = rng.normal(size=(n_samples, TEST_BANDS))
        mat = MatViews(
            foreground_std=rng.random((n_samples, TEST_BANDS)),
            foreground_iqr=rng.random((n_samples, TEST_BANDS)),
            radial_inner_minus_outer=rng.normal(size=(n_samples, TEST_BANDS)),
            morphology=rng.random((n_samples, len(MORPHOLOGY_FEATURE_NAMES))),
            feature_manifest_rows=tuple(),
            development_mat_numeric_reads=n_samples,
        )
        matrices = build_view_matrices(mean, mat)
        self.assertEqual(matrices["mat_all"].shape, (n_samples, 3 * TEST_BANDS + 5))
        self.assertEqual(
            matrices["sg1_plus_mat_all_early"].shape,
            (n_samples, 4 * TEST_BANDS + 5),
        )

    def test_model_builders_preserve_executed_hyperparameters(self) -> None:
        estimators = build_view_estimators(analysis_seed=20260721)
        self.assertEqual(tuple(estimators), CLASSIFIER_ORDER)
        lda = estimators["shrinkage_lda"].named_steps["classifier"]
        self.assertEqual(lda.solver, "lsqr")
        self.assertEqual(lda.shrinkage, "auto")
        lr = estimators["logistic_regression"].named_steps["classifier"]
        self.assertEqual(lr.C, 1.0)
        self.assertEqual(lr.solver, "lbfgs")
        self.assertEqual(lr.max_iter, 4000)
        self.assertEqual(lr.random_state, 20260721)
        svm = estimators["rbf_svm"].named_steps["classifier"]
        self.assertEqual(svm.C, 10.0)
        self.assertEqual(svm.gamma, "scale")
        self.assertTrue(svm.probability)
        self.assertEqual(svm.random_state, 20260721)
        trees = estimators["extra_trees"]
        self.assertEqual(trees.n_estimators, 500)
        self.assertEqual(trees.max_features, "sqrt")
        self.assertEqual(trees.min_samples_leaf, 2)
        self.assertEqual(trees.class_weight, "balanced")


class ArtifactContractTests(unittest.TestCase):
    def test_repository_entry_uses_blind_manifest_and_passes_only_development_records(self) -> None:
        class StopAfterDevelopmentSelection(RuntimeError):
            pass

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "analysis_seed": 20260721,
                        "expected_bands": 392,
                        "evaluation": {"ece_bins": 10},
                    }
                ),
                encoding="utf-8",
            )
            development = _record(root, index=0, batch=0, split="development")
            locked = _record(root, index=1, batch=8, split="locked")
            manifest = Manifest(
                data_root=root,
                records=(development, locked),
                manifest_sha256="manifest",
                csv_content_sha256="",
                mat_content_sha256="",
                data_fingerprint_sha256="assignment",
                hashes_complete=False,
            )
            with (
                patch(
                    "provenance_study.explore_mat_views.discover_manifest",
                    return_value=manifest,
                ) as discover,
                patch(
                    "provenance_study.explore_mat_views.load_development_csv",
                    side_effect=StopAfterDevelopmentSelection,
                ) as load_csv,
            ):
                with self.assertRaises(StopAfterDevelopmentSelection):
                    run_mat_view_exploration(
                        data_root=root,
                        config_path=config_path,
                        output_dir=root / "outputs",
                    )
            discover.assert_called_once_with(
                root, base_seed=20260721, hash_files=False
            )
            supplied_records = load_csv.call_args.args[0]
            self.assertEqual(supplied_records, (development,))
            self.assertNotIn(locked, supplied_records)

    def test_writer_emits_exact_declared_contract_and_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "development_mat"
            write_artifacts(
                output,
                metric_rows=[{"model_id": "synthetic", "balanced_accuracy": 0.5}],
                fold_rows=[{"model_id": "synthetic", "held_out_batch": 0}],
                feature_manifest_rows=[
                    {"sample_id": "0-1/1", "analysis_split": "development"}
                ],
                summary={
                    "status": "synthetic_test_only",
                    "access_audit": {"locked_mat_numeric_reads": 0},
                },
                report="# Synthetic test\n",
            )
            self.assertEqual(
                {path.name for path in output.iterdir()}, set(OUTPUT_FILENAMES)
            )
            payload = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["access_audit"]["locked_mat_numeric_reads"], 0)
            with self.assertRaises(FileExistsError):
                write_artifacts(
                    output,
                    metric_rows=[{"model_id": "synthetic"}],
                    fold_rows=[{"model_id": "synthetic"}],
                    feature_manifest_rows=[{"sample_id": "0-1/1"}],
                    summary={"status": "synthetic_test_only"},
                    report="# Synthetic test\n",
                )


if __name__ == "__main__":
    unittest.main()
