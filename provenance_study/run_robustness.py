"""Post-lock robustness analyses for geographical-origin traceability.

This entry point is intentionally unavailable until the canonical locked
evaluation is complete and internally consistent.  Only after its completion,
configuration, Git, and result hashes are verified may this script hash/read
the full dataset.  The analyses are secondary stress and falsification checks;
they cannot alter the frozen primary predictor or effect gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from joblib import Parallel, delayed

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from provenance_study.core import (
    CLASS_NAMES,
    CONSTRUCTED_BATCHES,
    DEVELOPMENT_BATCHES,
    NUM_CLASSES,
    build_sg15_logistic_regression,
    build_sg15_rbf_svm,
    build_sg15_shrinkage_lda,
    decision_scores_to_probabilities,
    discover_manifest,
    equal_weight_probability_average,
    fit_decision_temperature,
    grouped_oof_decision_scores,
    grouped_oof_probabilities,
    load_csv_split,
    multiclass_metrics,
    sha256_file,
)
from provenance_study.run_locked_evaluation import (
    COMPLETE_STATE,
    MODEL_ROLES,
    PRIMARY_MODEL,
    _align_columns,
    _atomic_write_json,
    _atomic_write_text,
    _decorate_rows,
    _load_config,
    _read_json_object,
    _resolve_from_repo,
    _utc_now,
    _write_csv,
    assert_tracked_worktree_clean,
    build_snv_logistic_regression,
    environment_snapshot,
    make_prediction_rows,
    refuse_completed_output,
    validate_frozen_config,
    validate_probability_matrix,
)


STATUS_FILENAME = "execution_status.json"
SOURCE_MODEL_ORDER = (
    "snv_logistic_regression",
    "sg1_shrinkage_lda",
    "sg1_logistic_regression",
    "sg1_rbf_svm_group_temperature",
    PRIMARY_MODEL,
)
SOURCE_DIRECTIONS = (
    ("source_1_to_2", 1, 2),
    ("source_2_to_1", 2, 1),
)
PERMUTATION_STATISTIC = "development_grouped_oof_ensemble_balanced_accuracy"


class LockedCompletionRequiredError(RuntimeError):
    """Raised before data access when canonical locked evidence is unavailable."""


def verify_locked_completion(locked_output_dir: Path) -> dict[str, Any]:
    """Verify canonical completion and results hash without inspecting data files."""

    output_dir = Path(locked_output_dir)
    status_path = output_dir / STATUS_FILENAME
    results_path = output_dir / "results.json"
    status = _read_json_object(status_path)
    results = _read_json_object(results_path)
    if not status or status.get("state") != COMPLETE_STATE:
        raise LockedCompletionRequiredError(
            f"Canonical locked status is not {COMPLETE_STATE}: {status_path}"
        )
    if not results or results.get("execution_state") != COMPLETE_STATE:
        raise LockedCompletionRequiredError(
            f"Canonical locked results are not {COMPLETE_STATE}: {results_path}"
        )
    expected_results_hash = str(status.get("results_json_sha256", ""))
    observed_results_hash = sha256_file(results_path)
    if not expected_results_hash or observed_results_hash != expected_results_hash:
        raise LockedCompletionRequiredError("Locked results.json SHA-256 verification failed")
    for field in (
        "run_id",
        "git_head",
        "config_sha256",
        "manifest_sha256",
        "data_fingerprint_sha256",
    ):
        if not status.get(field) or status.get(field) != results.get(field):
            raise LockedCompletionRequiredError(
                f"Locked status/results disagree on {field}"
            )
    return results


def verify_current_state_against_locked(
    locked_results: Mapping[str, Any],
    *,
    current_git_head: str,
    current_config_sha256: str,
) -> None:
    """Require the exact pre-locked code/config state before robustness execution."""

    if str(locked_results["git_head"]) != str(current_git_head):
        raise LockedCompletionRequiredError(
            "Current Git HEAD differs from the canonical locked-evaluation HEAD"
        )
    if str(locked_results["config_sha256"]) != str(current_config_sha256):
        raise LockedCompletionRequiredError(
            "Current configuration hash differs from the canonical locked evaluation"
        )


def build_source_transfer_masks(
    records: Sequence[Any], *, train_replicate: int, test_replicate: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create disjoint whole-source-image boundaries and reject any overlap."""

    if {int(train_replicate), int(test_replicate)} != {1, 2}:
        raise ValueError("Source transfer must use opposite replicates 1 and 2")
    train_mask = np.asarray(
        [int(record.replicate) == int(train_replicate) for record in records], dtype=bool
    )
    test_mask = np.asarray(
        [int(record.replicate) == int(test_replicate) for record in records], dtype=bool
    )
    if not np.any(train_mask) or not np.any(test_mask):
        raise ValueError("Both source-image replicates must contain samples")
    if np.any(train_mask & test_mask) or np.any(~(train_mask | test_mask)):
        raise ValueError("Source-image transfer masks are not a complete disjoint partition")
    train_ids = {str(record.sample_id) for record, include in zip(records, train_mask) if include}
    test_ids = {str(record.sample_id) for record, include in zip(records, test_mask) if include}
    if train_ids & test_ids:
        raise ValueError("Training and test source images share sample identifiers")
    return train_mask, test_mask


def _validate_source_training_groups(y: np.ndarray, groups: np.ndarray) -> None:
    if set(groups.tolist()) != set(CONSTRUCTED_BATCHES):
        raise ValueError("A source-image training set must contain constructed batches 0--9")
    expected_classes = set(range(NUM_CLASSES))
    for batch in CONSTRUCTED_BATCHES:
        if set(y[groups == batch].tolist()) != expected_classes:
            raise ValueError(f"Source-image training batch {batch} lacks at least one origin")


def fit_source_models_and_predict_once(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_batches: np.ndarray,
    X_test: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], float]:
    """Fit one source image, calibrate within it, and predict the opposite image once."""

    classes = np.arange(NUM_CLASSES, dtype=np.int64)
    _validate_source_training_groups(y_train, train_batches)
    svm = build_sg15_rbf_svm()
    svm_oof = grouped_oof_decision_scores(
        svm,
        X_train,
        y_train,
        train_batches,
        group_order=CONSTRUCTED_BATCHES,
    )
    temperature = fit_decision_temperature(
        svm_oof.decision_scores,
        y_train,
        classes=classes,
        log_temperature_bounds=tuple(
            config["models"]["sg1_rbf_svm"]["temperature_log_bounds"]
        ),
    )
    estimators = {
        "snv_logistic_regression": build_snv_logistic_regression(config),
        "sg1_shrinkage_lda": build_sg15_shrinkage_lda(),
        "sg1_logistic_regression": build_sg15_logistic_regression(),
        "sg1_rbf_svm_group_temperature": svm,
    }
    probabilities: dict[str, np.ndarray] = {}
    for model_name, estimator in estimators.items():
        fitted = estimator.fit(X_train, y_train)
        if model_name == "sg1_rbf_svm_group_temperature":
            scores = _align_columns(
                fitted.decision_function(X_test), fitted.classes_, classes
            )
            values = decision_scores_to_probabilities(scores, temperature)
        else:
            values = _align_columns(fitted.predict_proba(X_test), fitted.classes_, classes)
        probabilities[model_name] = validate_probability_matrix(values, NUM_CLASSES)
    probabilities[PRIMARY_MODEL] = validate_probability_matrix(
        equal_weight_probability_average(
            [
                probabilities["sg1_shrinkage_lda"],
                probabilities["sg1_logistic_regression"],
                probabilities["sg1_rbf_svm_group_temperature"],
            ]
        ),
        NUM_CLASSES,
    )
    if tuple(probabilities) != SOURCE_MODEL_ORDER:
        raise AssertionError("Source-transfer model order drifted")
    return probabilities, float(temperature)


def source_transfer_analysis(
    X: np.ndarray,
    y: np.ndarray,
    records: Sequence[Any],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    """Run reciprocal whole-image transfer and direction-equal summaries."""

    features = np.asarray(X, dtype=np.float64)
    labels = np.asarray(y, dtype=np.int64)
    if features.shape[0] != labels.size or labels.size != len(records):
        raise ValueError("Source-transfer arrays and records are not aligned")
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    temperatures: dict[str, float] = {}
    per_direction_metrics: dict[tuple[str, str], dict[str, float | int]] = {}
    per_direction_recalls: dict[tuple[str, str, int], float] = {}

    for direction, train_replicate, test_replicate in SOURCE_DIRECTIONS:
        train_mask, test_mask = build_source_transfer_masks(
            records,
            train_replicate=train_replicate,
            test_replicate=test_replicate,
        )
        train_batches = np.asarray(
            [record.constructed_batch for record, include in zip(records, train_mask) if include],
            dtype=np.int64,
        )
        probabilities, temperature = fit_source_models_and_predict_once(
            features[train_mask],
            labels[train_mask],
            train_batches,
            features[test_mask],
            config,
        )
        temperatures[direction] = temperature
        test_records = [record for record, include in zip(records, test_mask) if include]
        metadata = [
            {
                "sample_index": record.sample_index,
                "sample_id": record.sample_id,
                "source_cube": record.source_cube,
                "source_replicate": record.replicate,
                "seed_id": record.seed_id,
                "relative_csv_path": record.relative_csv_path,
                "constructed_batch": record.constructed_batch,
            }
            for record in test_records
        ]
        direction_predictions = make_prediction_rows(
            probabilities,
            labels[test_mask],
            metadata,
            config["class_codes"],
        )
        for row in direction_predictions:
            prediction_rows.append(
                {
                    "analysis_role": "secondary_whole_source_image_stress_test",
                    "direction": direction,
                    "train_source_replicate": train_replicate,
                    "test_source_replicate": test_replicate,
                    "svm_training_oof_temperature": temperature,
                    **row,
                }
            )
        for model_name in SOURCE_MODEL_ORDER:
            values = probabilities[model_name]
            observed = multiclass_metrics(
                labels[test_mask], values, classes=np.arange(NUM_CLASSES), ece_bins=10
            )
            per_direction_metrics[(direction, model_name)] = observed
            metric_rows.append(
                {
                    "analysis_role": "secondary_whole_source_image_stress_test",
                    "scope": "direction_overall",
                    "direction": direction,
                    "train_source_replicate": train_replicate,
                    "test_source_replicate": test_replicate,
                    "model": model_name,
                    "model_role": MODEL_ROLES[model_name],
                    **observed,
                    "theta_equal_class_recall": float(observed["balanced_accuracy"]),
                    "label": "",
                    "class_code": "",
                    "recall": "",
                }
            )
            predictions = values.argmax(axis=1)
            for label in range(NUM_CLASSES):
                class_mask = labels[test_mask] == label
                recall = float(np.mean(predictions[class_mask] == label))
                per_direction_recalls[(direction, model_name, label)] = recall
                metric_rows.append(
                    {
                        "analysis_role": "secondary_whole_source_image_stress_test",
                        "scope": "direction_class",
                        "direction": direction,
                        "train_source_replicate": train_replicate,
                        "test_source_replicate": test_replicate,
                        "model": model_name,
                        "model_role": MODEL_ROLES[model_name],
                        "n": int(class_mask.sum()),
                        "accuracy": "",
                        "balanced_accuracy": "",
                        "macro_f1": "",
                        "negative_log_likelihood": "",
                        "multiclass_brier_score": "",
                        "expected_calibration_error": "",
                        "theta_equal_class_recall": "",
                        "label": label,
                        "class_code": config["class_codes"][label],
                        "recall": recall,
                    }
                )

    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "negative_log_likelihood",
        "multiclass_brier_score",
        "expected_calibration_error",
    )
    for model_name in SOURCE_MODEL_ORDER:
        direction_values = [
            per_direction_metrics[(direction, model_name)]
            for direction, _, _ in SOURCE_DIRECTIONS
        ]
        aggregate = {
            metric: float(np.mean([float(row[metric]) for row in direction_values]))
            for metric in metric_names
        }
        theta = aggregate["balanced_accuracy"]
        metric_rows.append(
            {
                "analysis_role": "secondary_whole_source_image_stress_test",
                "scope": "direction_equal_overall",
                "direction": "direction_equal",
                "train_source_replicate": "",
                "test_source_replicate": "",
                "model": model_name,
                "model_role": MODEL_ROLES[model_name],
                "n": int(
                    sum(
                        int(per_direction_metrics[(direction, model_name)]["n"])
                        for direction, _, _ in SOURCE_DIRECTIONS
                    )
                ),
                **aggregate,
                "theta_equal_class_and_direction_recall": theta,
                "theta_equal_class_recall": theta,
                "label": "",
                "class_code": "",
                "recall": "",
            }
        )
        for label in range(NUM_CLASSES):
            direction_equal_recall = float(
                np.mean(
                    [
                        per_direction_recalls[(direction, model_name, label)]
                        for direction, _, _ in SOURCE_DIRECTIONS
                    ]
                )
            )
            metric_rows.append(
                {
                    "analysis_role": "secondary_whole_source_image_stress_test",
                    "scope": "direction_equal_class",
                    "direction": "direction_equal",
                    "train_source_replicate": "",
                    "test_source_replicate": "",
                    "model": model_name,
                    "model_role": MODEL_ROLES[model_name],
                    "n": "",
                    "accuracy": "",
                    "balanced_accuracy": "",
                    "macro_f1": "",
                    "negative_log_likelihood": "",
                    "multiclass_brier_score": "",
                    "expected_calibration_error": "",
                    "theta_equal_class_recall": "",
                    "label": label,
                    "class_code": config["class_codes"][label],
                    "recall": direction_equal_recall,
                }
            )
    return metric_rows, prediction_rows, temperatures


def generate_cluster_label_mapping(
    *,
    permutation_index: int,
    analysis_seed: int,
    group_order: Sequence[int] = DEVELOPMENT_BATCHES,
    classes: Sequence[int] = tuple(range(NUM_CLASSES)),
) -> dict[int, tuple[int, ...]]:
    """Generate one independent origin-cluster permutation within every batch."""

    if permutation_index < 0:
        raise ValueError("permutation_index must be non-negative")
    class_array = np.asarray(classes, dtype=np.int64)
    if class_array.ndim != 1 or np.unique(class_array).size != class_array.size:
        raise ValueError("classes must be a unique one-dimensional sequence")
    mapping: dict[int, tuple[int, ...]] = {}
    for group in group_order:
        subseed = (
            int(analysis_seed)
            + (int(permutation_index) + 1) * 2_000_003
            + int(group) * 7_919
        )
        permuted = np.random.default_rng(subseed).permutation(class_array)
        mapping[int(group)] = tuple(int(value) for value in permuted)
    return mapping


def apply_cluster_label_mapping(
    y: Sequence[int],
    groups: Sequence[int],
    mapping: Mapping[int, Sequence[int]],
    *,
    classes: Sequence[int] = tuple(range(NUM_CLASSES)),
) -> np.ndarray:
    """Relabel whole origin-by-batch clusters while preserving class balance per batch."""

    labels = np.asarray(y, dtype=np.int64)
    batch_ids = np.asarray(groups, dtype=np.int64)
    class_array = np.asarray(classes, dtype=np.int64)
    if labels.ndim != 1 or batch_ids.ndim != 1 or labels.size != batch_ids.size:
        raise ValueError("y and groups must be aligned vectors")
    if set(batch_ids.tolist()) != set(mapping):
        raise ValueError("Mapping keys must equal the observed constructed batches")
    expected_classes = set(class_array.tolist())
    relabelled = np.full_like(labels, -1)
    for group, permutation in mapping.items():
        values = np.asarray(permutation, dtype=np.int64)
        if values.shape != class_array.shape or set(values.tolist()) != expected_classes:
            raise ValueError(f"Batch {group} mapping is not a class permutation")
        batch_mask = batch_ids == int(group)
        if set(labels[batch_mask].tolist()) != expected_classes:
            raise ValueError(f"Batch {group} does not contain every original class")
        positions = {int(label): index for index, label in enumerate(class_array)}
        relabelled[batch_mask] = np.asarray(
            [values[positions[int(label)]] for label in labels[batch_mask]], dtype=np.int64
        )
        if set(relabelled[batch_mask].tolist()) != expected_classes:
            raise AssertionError(f"Batch {group} relabelling lost class balance")
    if np.any(relabelled < 0):
        raise AssertionError("Cluster-label permutation left unassigned rows")
    return relabelled


def evaluate_development_oof_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    svm_temperature: float,
) -> tuple[dict[str, dict[str, float | int]], np.ndarray]:
    """Evaluate the frozen three-member OOF ensemble without fitting temperature."""

    classes = np.arange(NUM_CLASSES, dtype=np.int64)
    lda = grouped_oof_probabilities(
        build_sg15_shrinkage_lda(), X, y, groups, group_order=DEVELOPMENT_BATCHES
    ).probabilities
    lr = grouped_oof_probabilities(
        build_sg15_logistic_regression(), X, y, groups, group_order=DEVELOPMENT_BATCHES
    ).probabilities
    svm_scores = grouped_oof_decision_scores(
        build_sg15_rbf_svm(), X, y, groups, group_order=DEVELOPMENT_BATCHES
    ).decision_scores
    svm = decision_scores_to_probabilities(svm_scores, svm_temperature)
    ensemble = equal_weight_probability_average([lda, lr, svm])
    probabilities = {
        "sg1_shrinkage_lda": validate_probability_matrix(lda, NUM_CLASSES),
        "sg1_logistic_regression": validate_probability_matrix(lr, NUM_CLASSES),
        "sg1_rbf_svm_group_temperature": validate_probability_matrix(svm, NUM_CLASSES),
        PRIMARY_MODEL: validate_probability_matrix(ensemble, NUM_CLASSES),
    }
    metrics = {
        model: multiclass_metrics(y, values, classes=classes, ece_bins=10)
        for model, values in probabilities.items()
    }
    return metrics, probabilities[PRIMARY_MODEL]


def _mapping_json(mapping: Mapping[int, Sequence[int]]) -> str:
    return json.dumps(
        {str(group): list(values) for group, values in sorted(mapping.items())},
        sort_keys=True,
        separators=(",", ":"),
    )


def _permutation_worker(
    permutation_index: int,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    analysis_seed: int,
    svm_temperature: float,
) -> dict[str, Any]:
    mapping = generate_cluster_label_mapping(
        permutation_index=permutation_index,
        analysis_seed=analysis_seed,
    )
    permuted_y = apply_cluster_label_mapping(y, groups, mapping)
    metrics, _ = evaluate_development_oof_ensemble(
        X,
        permuted_y,
        groups,
        svm_temperature=svm_temperature,
    )
    mapping_json = _mapping_json(mapping)
    return {
        "row_type": "null_cluster_label_permutation",
        "permutation_index": permutation_index + 1,
        "statistic": PERMUTATION_STATISTIC,
        **metrics[PRIMARY_MODEL],
        "mapping_json": mapping_json,
        "mapping_sha256": hashlib.sha256(mapping_json.encode("utf-8")).hexdigest(),
    }


def monte_carlo_upper_p_value(observed: float, null_values: Sequence[float]) -> float:
    """Return the finite-simulation upper-tail p-value with the +1 correction."""

    null = np.asarray(null_values, dtype=np.float64)
    if not math.isfinite(observed) or null.ndim != 1 or null.size == 0:
        raise ValueError("Observed and null statistics must be finite and non-empty")
    if not np.all(np.isfinite(null)):
        raise ValueError("Null statistics must be finite")
    return float((1 + np.sum(null >= float(observed))) / (null.size + 1))


def development_cluster_label_permutation_analysis(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    config: Mapping[str, Any],
    *,
    jobs: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the frozen 200-repetition development-only falsification analysis."""

    if jobs == 0:
        raise ValueError("jobs cannot be zero")
    if set(np.asarray(groups, dtype=int).tolist()) != set(DEVELOPMENT_BATCHES):
        raise ValueError("Permutation analysis accepts development batches 0--7 only")
    repetitions = int(config["evaluation"]["label_permutation_repetitions"])
    if repetitions != 200:
        raise ValueError("Frozen label-permutation repetition count must remain 200")
    svm_temperature = float(
        config["primary_predictor"]["svm_temperature_frozen_development_value"]
    )
    observed_metrics, _ = evaluate_development_oof_ensemble(
        X, y, groups, svm_temperature=svm_temperature
    )
    observed_row = {
        "row_type": "observed_unpermuted",
        "permutation_index": 0,
        "statistic": PERMUTATION_STATISTIC,
        **observed_metrics[PRIMARY_MODEL],
        "mapping_json": "identity",
        "mapping_sha256": hashlib.sha256(b"identity").hexdigest(),
    }
    null_rows = Parallel(n_jobs=jobs, prefer="processes")(
        delayed(_permutation_worker)(
            index,
            np.asarray(X, dtype=np.float64),
            np.asarray(y, dtype=np.int64),
            np.asarray(groups, dtype=np.int64),
            int(config["analysis_seed"]),
            svm_temperature,
        )
        for index in range(repetitions)
    )
    null_values = np.asarray(
        [float(row["balanced_accuracy"]) for row in null_rows], dtype=np.float64
    )
    observed_value = float(observed_row["balanced_accuracy"])
    summary = {
        "analysis_role": "development_only_cluster_label_falsification",
        "statistic": PERMUTATION_STATISTIC,
        "observed": observed_value,
        "repetitions": repetitions,
        "upper_tail_p_value_plus_one": monte_carlo_upper_p_value(
            observed_value, null_values
        ),
        "null_mean": float(null_values.mean()),
        "null_standard_deviation": float(null_values.std(ddof=1)),
        "null_minimum": float(null_values.min()),
        "null_median": float(np.median(null_values)),
        "null_maximum": float(null_values.max()),
        "svm_temperature_source": "frozen_real_development_labels_not_refit_on_permutations",
        "svm_temperature": svm_temperature,
        "observed_member_metrics": observed_metrics,
    }
    return [observed_row, *null_rows], summary


def _combine_splits(development: Any, locked: Any) -> tuple[np.ndarray, np.ndarray, tuple[Any, ...]]:
    if not np.allclose(
        development.wavelengths, locked.wavelengths, rtol=0.0, atol=1e-6
    ):
        raise ValueError("Development and locked wavelength grids differ")
    indexed = [
        (record.sample_index, spectrum, int(label), record)
        for dataset in (development, locked)
        for spectrum, label, record in zip(dataset.X, dataset.y, dataset.records)
    ]
    indexed.sort(key=lambda item: item[0])
    indices = [item[0] for item in indexed]
    if len(indices) != len(set(indices)):
        raise ValueError("Development and locked splits overlap by sample index")
    return (
        np.asarray([item[1] for item in indexed], dtype=np.float64),
        np.asarray([item[2] for item in indexed], dtype=np.int64),
        tuple(item[3] for item in indexed),
    )


def _build_report(results: Mapping[str, Any]) -> str:
    transfer = results["source_transfer"]["direction_equal_metrics"][PRIMARY_MODEL]
    permutation = results["development_cluster_label_permutation"]
    return (
        "# 产地溯源稳健性与伪证分析报告\n\n"
        "本报告仅在主锁定评估完成后执行，不参与模型选择，也不改变主结果门槛。\n\n"
        "## 完整来源图像迁移压力检验\n\n"
        f"SG1 三成员等权集成的双方向、类别等权 θ 为 {transfer['theta']:.4%}；"
        "该数值衡量采集来源图像变化下的性能，不等同于外部年份、农场或设备验证。\n\n"
        "## 开发集 cluster-label 置换\n\n"
        f"未置换开发集 OOF balanced accuracy 为 {permutation['observed']:.4%}；"
        f"200 次构造批次内 cluster-label 置换的 +1 修正上尾 p 值为 "
        f"{permutation['upper_tail_p_value_plus_one']:.6f}。SVM 温度固定为真实开发标签"
        "阶段的冻结值，未在置换标签上重新拟合。\n\n"
        "## 结论边界\n\n"
        "这些结果是当前数据内的次要压力与伪证检查；它们不能扩展为未知批次、年份、"
        "农场或仪器的外部泛化声明。\n"
    )


def run_robustness(args: argparse.Namespace) -> Path:
    """Execute post-lock robustness, with every non-data guard preceding discovery."""

    repo_root = Path(args.repo_root).resolve()
    locked_output_dir = _resolve_from_repo(repo_root, Path(args.locked_output_dir))
    output_dir = _resolve_from_repo(repo_root, Path(args.output_dir))
    locked_results = verify_locked_completion(locked_output_dir)
    refuse_completed_output(output_dir)
    git_head = assert_tracked_worktree_clean(repo_root)
    config_path = _resolve_from_repo(repo_root, Path(args.config))
    config_sha256 = sha256_file(config_path)
    verify_current_state_against_locked(
        locked_results,
        current_git_head=git_head,
        current_config_sha256=config_sha256,
    )
    config = _load_config(config_path)
    validate_frozen_config(config)
    if not math.isclose(
        float(locked_results["svm_temperature"]),
        float(config["primary_predictor"]["svm_temperature_frozen_development_value"]),
        rel_tol=0.0,
        abs_tol=5e-10,
    ):
        raise LockedCompletionRequiredError(
            "Locked result SVM temperature differs from the frozen configuration"
        )

    started_at = _utc_now()
    run_id = f"robustness_{started_at.replace(':', '')}_{git_head[:12]}"
    status_path = output_dir / STATUS_FILENAME
    status: dict[str, Any] = {
        "state": "executing_running",
        "run_id": run_id,
        "started_at_utc": started_at,
        "completed_at_utc": None,
        "git_head": git_head,
        "config_sha256": config_sha256,
        "locked_run_id": locked_results["run_id"],
        "locked_results_json_sha256": sha256_file(locked_output_dir / "results.json"),
        "command": shlex.join([sys.executable, *sys.argv]),
        "environment": environment_snapshot(),
    }
    _atomic_write_json(status_path, status)

    try:
        data_root = _resolve_from_repo(repo_root, Path(args.data_root))
        # This is the first operation that discovers, hashes, or reads below data_root.
        manifest = discover_manifest(
            data_root,
            base_seed=int(config["analysis_seed"]),
            hash_files=True,
        )
        if manifest.data_fingerprint_sha256 != locked_results["data_fingerprint_sha256"]:
            raise LockedCompletionRequiredError(
                "Current data fingerprint differs from the canonical locked evaluation"
            )
        if manifest.manifest_sha256 != locked_results["manifest_sha256"]:
            raise LockedCompletionRequiredError(
                "Current manifest fingerprint differs from the canonical locked evaluation"
            )
        development = load_csv_split(
            manifest,
            split="development",
            expected_bands=int(config["expected_bands"]),
            verify_hashes=True,
        )
        locked = load_csv_split(
            manifest,
            split="locked",
            expected_bands=int(config["expected_bands"]),
            verify_hashes=True,
        )
        X_all, y_all, all_records = _combine_splits(development, locked)
        source_metrics, source_predictions, source_temperatures = source_transfer_analysis(
            X_all, y_all, all_records, config
        )
        development_groups = np.asarray(
            [record.constructed_batch for record in development.records], dtype=np.int64
        )
        permutation_rows, permutation_summary = (
            development_cluster_label_permutation_analysis(
                development.X,
                development.y,
                development_groups,
                config,
                jobs=int(args.jobs),
            )
        )

        context = {
            "run_id": run_id,
            "git_head": git_head,
            "config_sha256": config_sha256,
            "manifest_sha256": manifest.manifest_sha256,
            "data_fingerprint_sha256": manifest.data_fingerprint_sha256,
        }
        artifacts = {
            "source_transfer_metrics.csv": _decorate_rows(source_metrics, context),
            "predictions.csv": _decorate_rows(source_predictions, context),
            "permutation_null.csv": _decorate_rows(permutation_rows, context),
        }
        for filename, rows in artifacts.items():
            _write_csv(output_dir / filename, rows)

        direction_equal_metrics: dict[str, dict[str, float]] = {}
        for model_name in SOURCE_MODEL_ORDER:
            row = next(
                row
                for row in source_metrics
                if row["scope"] == "direction_equal_overall" and row["model"] == model_name
            )
            direction_equal_metrics[model_name] = {
                "theta": float(row["theta_equal_class_and_direction_recall"]),
                "accuracy": float(row["accuracy"]),
                "balanced_accuracy": float(row["balanced_accuracy"]),
                "macro_f1": float(row["macro_f1"]),
                "negative_log_likelihood": float(row["negative_log_likelihood"]),
            }
        results: dict[str, Any] = {
            "execution_state": COMPLETE_STATE,
            "run_id": run_id,
            "started_at_utc": started_at,
            "completed_at_utc": _utc_now(),
            "git_head": git_head,
            "config_sha256": config_sha256,
            "manifest_sha256": manifest.manifest_sha256,
            "data_fingerprint_sha256": manifest.data_fingerprint_sha256,
            "locked_run_id": locked_results["run_id"],
            "source_transfer": {
                "analysis_role": "secondary_whole_source_image_stress_test",
                "directions": [direction for direction, _, _ in SOURCE_DIRECTIONS],
                "svm_training_oof_temperatures": source_temperatures,
                "direction_equal_metrics": direction_equal_metrics,
            },
            "development_cluster_label_permutation": permutation_summary,
            "claim_boundary": (
                "secondary current-data stress/falsification only; no external "
                "farm/year/device generalization claim"
            ),
            "environment": status["environment"],
        }
        report_path = output_dir / "report.md"
        _atomic_write_text(report_path, _build_report(results))
        results["artifact_sha256"] = {
            filename: sha256_file(output_dir / filename)
            for filename in (*artifacts, "report.md")
        }
        results_path = output_dir / "results.json"
        _atomic_write_json(results_path, results)
        status.update(
            {
                "state": COMPLETE_STATE,
                "completed_at_utc": results["completed_at_utc"],
                "manifest_sha256": manifest.manifest_sha256,
                "data_fingerprint_sha256": manifest.data_fingerprint_sha256,
                "results_json_sha256": sha256_file(results_path),
            }
        )
        _atomic_write_json(status_path, status)
        return results_path
    except BaseException as exc:
        status.update(
            {
                "state": "executed_failure",
                "completed_at_utc": _utc_now(),
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        _atomic_write_json(status_path, status)
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    repository_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run post-lock source-transfer and cluster-label falsification analyses."
    )
    parser.add_argument("--repo-root", type=Path, default=repository_default)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=Path("provenance_study/config.json"))
    parser.add_argument(
        "--locked-output-dir",
        type=Path,
        default=Path("provenance_study/outputs/locked_evaluation"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("provenance_study/outputs/robustness"),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Deterministic joblib worker count; 1 is the reproducible low-memory default.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    results_path = run_robustness(args)
    print(f"Robustness analyses completed: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
