from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from provenance_study.run_locked_evaluation import COMPLETE_STATE
from provenance_study.run_robustness import (
    LockedCompletionRequiredError,
    apply_cluster_label_mapping,
    build_source_transfer_masks,
    generate_cluster_label_mapping,
    monte_carlo_upper_p_value,
    run_robustness,
    sha256_file,
    verify_locked_completion,
    verify_current_state_against_locked,
)


def _locked_results_payload() -> dict[str, object]:
    return {
        "execution_state": COMPLETE_STATE,
        "run_id": "locked-synthetic",
        "git_head": "a" * 40,
        "config_sha256": "b" * 64,
        "manifest_sha256": "c" * 64,
        "data_fingerprint_sha256": "d" * 64,
        "svm_temperature": 0.4,
    }


def _write_completed_locked_output(output_dir: Path) -> dict[str, object]:
    results = _locked_results_payload()
    results_path = output_dir / "results.json"
    results_path.write_text(
        json.dumps(results, sort_keys=True), encoding="utf-8"
    )
    status = {
        "state": COMPLETE_STATE,
        "run_id": results["run_id"],
        "git_head": results["git_head"],
        "config_sha256": results["config_sha256"],
        "manifest_sha256": results["manifest_sha256"],
        "data_fingerprint_sha256": results["data_fingerprint_sha256"],
        "results_json_sha256": sha256_file(results_path),
    }
    (output_dir / "execution_status.json").write_text(
        json.dumps(status, sort_keys=True), encoding="utf-8"
    )
    return results


class PostLockGuardTests(unittest.TestCase):
    def test_incomplete_locked_state_stops_before_data_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            locked_output = root / "locked"
            locked_output.mkdir()
            (locked_output / "execution_status.json").write_text(
                json.dumps({"state": "executing_running"}), encoding="utf-8"
            )
            arguments = argparse.Namespace(
                repo_root=root,
                locked_output_dir=locked_output,
                output_dir=root / "robustness",
                config=root / "config.json",
                data_root=root / "must-not-open",
                jobs=1,
            )
            with patch(
                "provenance_study.run_robustness.discover_manifest",
                side_effect=AssertionError("data discovery was reached"),
            ) as discovery:
                with self.assertRaises(LockedCompletionRequiredError):
                    run_robustness(arguments)
            discovery.assert_not_called()

    def test_completed_locked_hash_and_current_state_are_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            expected = _write_completed_locked_output(output_dir)
            observed = verify_locked_completion(output_dir)
            self.assertEqual(observed["run_id"], expected["run_id"])
            verify_current_state_against_locked(
                observed,
                current_git_head=str(expected["git_head"]),
                current_config_sha256=str(expected["config_sha256"]),
            )
            with self.assertRaises(LockedCompletionRequiredError):
                verify_current_state_against_locked(
                    observed,
                    current_git_head="e" * 40,
                    current_config_sha256=str(expected["config_sha256"]),
                )

            (output_dir / "results.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(LockedCompletionRequiredError):
                verify_locked_completion(output_dir)


class ClusterLabelPermutationTests(unittest.TestCase):
    def test_mapping_is_deterministic_balanced_and_cluster_constant(self) -> None:
        groups: list[int] = []
        labels: list[int] = []
        for group in range(8):
            for label in range(8):
                groups.extend([group, group])
                labels.extend([label, label])
        mapping_a = generate_cluster_label_mapping(
            permutation_index=17, analysis_seed=20260721
        )
        mapping_b = generate_cluster_label_mapping(
            permutation_index=17, analysis_seed=20260721
        )
        self.assertEqual(mapping_a, mapping_b)
        for values in mapping_a.values():
            self.assertEqual(set(values), set(range(8)))

        y = np.asarray(labels, dtype=np.int64)
        batch_ids = np.asarray(groups, dtype=np.int64)
        permuted = apply_cluster_label_mapping(y, batch_ids, mapping_a)
        for group in range(8):
            self.assertEqual(set(permuted[batch_ids == group]), set(range(8)))
            for label in range(8):
                cluster_values = np.unique(
                    permuted[(batch_ids == group) & (y == label)]
                )
                self.assertEqual(cluster_values.size, 1)

    def test_upper_tail_p_value_uses_plus_one_correction(self) -> None:
        observed = 0.8
        null = [0.1, 0.8, 0.9]
        self.assertAlmostEqual(monte_carlo_upper_p_value(observed, null), 0.75)


class SourceImageBoundaryTests(unittest.TestCase):
    def test_reciprocal_masks_are_complete_and_disjoint(self) -> None:
        records = [
            SimpleNamespace(replicate=1, sample_id="1/a"),
            SimpleNamespace(replicate=1, sample_id="1/b"),
            SimpleNamespace(replicate=2, sample_id="2/a"),
            SimpleNamespace(replicate=2, sample_id="2/b"),
        ]
        train, test = build_source_transfer_masks(
            records, train_replicate=1, test_replicate=2
        )
        self.assertFalse(np.any(train & test))
        self.assertTrue(np.all(train | test))
        self.assertEqual(np.flatnonzero(train).tolist(), [0, 1])
        self.assertEqual(np.flatnonzero(test).tolist(), [2, 3])

        overlapping = [
            SimpleNamespace(replicate=1, sample_id="duplicate"),
            SimpleNamespace(replicate=2, sample_id="duplicate"),
        ]
        with self.assertRaisesRegex(ValueError, "share sample identifiers"):
            build_source_transfer_masks(
                overlapping, train_replicate=1, test_replicate=2
            )


if __name__ == "__main__":
    unittest.main()
