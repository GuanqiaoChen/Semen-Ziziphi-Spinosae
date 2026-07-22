#!/usr/bin/env python3
"""Run the locked-blind development analysis for origin traceability.

This entry point may enumerate locked paths and file-size metadata, but it does
not open locked CSV or MAT bytes.  It loads and hashes development CSV files
only.  The RBF-SVM outer-fold probabilities use strictly nested grouped
temperature fitting; its deployable temperature is fitted separately from the
complete eight-fold development OOF score matrix.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from provenance_study.core import (  # noqa: E402
    BASE_BATCH_SEED,
    CLASS_NAMES,
    DEVELOPMENT_BATCHES,
    EXPECTED_BANDS,
    LOCKED_BATCHES,
    LockedDataAccessError,
    Manifest,
    SampleRecord,
    SpectralDataset,
    StandardNormalVariate,
    build_sg15_logistic_regression,
    build_sg15_rbf_svm,
    build_sg15_shrinkage_lda,
    discover_manifest,
    equal_weight_probability_average,
    grouped_oof_probabilities,
    load_csv_split,
    multiclass_metrics,
    nested_grouped_oof_temperature_probabilities,
    sha256_file,
)


MODEL_SNV_LR = "snv_logistic_regression"
MODEL_SG_LDA = "sg1_shrinkage_lda"
MODEL_SG_LR = "sg1_logistic_regression"
MODEL_SG_SVM = "sg1_rbf_svm_group_temperature"
MODEL_ENSEMBLE = "batch_constrained_sg1_probability_ensemble"
MODEL_ORDER = (MODEL_SNV_LR, MODEL_SG_LDA, MODEL_SG_LR, MODEL_SG_SVM, MODEL_ENSEMBLE)
ENSEMBLE_MEMBERS = (MODEL_SG_LDA, MODEL_SG_LR, MODEL_SG_SVM)

EXPECTED_DEVELOPMENT_SAMPLES = 1012
EXPECTED_LOCKED_SAMPLES = 252
EXPECTED_ENSEMBLE_ERRORS = 18
EXPECTED_ENSEMBLE_BALANCED_ACCURACY = 0.982413
ENSEMBLE_BALANCED_ACCURACY_ATOL = 5e-6
DEFAULT_TEMPERATURE_ATOL = 1e-8

OUTPUT_FILENAMES = (
    "manifest.csv",
    "predictions.csv",
    "metrics.csv",
    "fold_metrics.csv",
    "temperature_calibration.json",
    "selection_summary.json",
    "report.md",
    "wavelengths.csv",
)


@dataclass(frozen=True)
class DevelopmentEvaluation:
    probabilities: Mapping[str, np.ndarray]
    metric_rows: tuple[dict[str, Any], ...]
    fold_metric_rows: tuple[dict[str, Any], ...]
    svm_final_temperature: float
    svm_outer_temperatures: tuple[tuple[int, float], ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration root must be an object: {path}")
    return value


def _require_equal(observed: Any, expected: Any, description: str) -> None:
    if observed != expected:
        raise ValueError(f"Frozen configuration mismatch for {description}: {observed!r} != {expected!r}")


def validate_frozen_config(config: Mapping[str, Any]) -> None:
    """Reject configuration drift before any spectrum is loaded."""

    _require_equal(int(config["analysis_seed"]), BASE_BATCH_SEED, "analysis_seed")
    _require_equal(int(config["expected_bands"]), EXPECTED_BANDS, "expected_bands")
    _require_equal(tuple(config["class_codes"]), CLASS_NAMES, "class_codes")
    batches = config["constructed_batches"]
    _require_equal(tuple(batches["development_indices"]), DEVELOPMENT_BATCHES, "development batches")
    _require_equal(tuple(batches["locked_test_indices"]), LOCKED_BATCHES, "locked batches")
    _require_equal(
        batches["subseed_formula"],
        "analysis_seed + label * 101 + source_replicate * 1009",
        "batch subseed formula",
    )
    _require_equal(
        batches["assignment"],
        "numeric_seed_id_sort_then_rng_permutation_rank_modulo_10",
        "batch assignment",
    )

    preprocessing = config["preprocessing"]
    _require_equal(int(preprocessing["savgol_window_length"]), 15, "SG window")
    _require_equal(int(preprocessing["savgol_polyorder"]), 2, "SG polyorder")
    _require_equal(int(preprocessing["savgol_derivative"]), 1, "SG derivative")
    models = config["models"]
    lda = models[MODEL_SG_LDA]
    _require_equal((lda["solver"], lda["shrinkage"]), ("lsqr", "auto"), "LDA")
    for model_name in (MODEL_SG_LR, MODEL_SNV_LR):
        specification = models[model_name]
        _require_equal(float(specification["C"]), 1.0, f"{model_name} C")
        _require_equal(specification["solver"], "lbfgs", f"{model_name} solver")
        _require_equal(int(specification["max_iter"]), 5000, f"{model_name} max_iter")
        _require_equal(float(specification["tol"]), 1e-4, f"{model_name} tol")
    svm = models["sg1_rbf_svm"]
    _require_equal(float(svm["C"]), 10.0, "SVM C")
    _require_equal(svm["kernel"], "rbf", "SVM kernel")
    _require_equal(svm["gamma"], "scale", "SVM gamma")
    _require_equal(bool(svm["probability"]), False, "SVM probability")
    _require_equal(svm["decision_function_shape"], "ovr", "SVM decision shape")
    _require_equal(tuple(map(float, svm["temperature_log_bounds"])), (-4.0, 4.0), "SVM T bounds")

    primary = config["primary_predictor"]
    _require_equal(primary["name"], MODEL_ENSEMBLE, "primary predictor")
    _require_equal(tuple(primary["members"]), ENSEMBLE_MEMBERS, "ensemble members")
    weights = np.asarray(primary["probability_weights"], dtype=np.float64)
    if weights.shape != (3,) or not np.allclose(weights, np.repeat(1.0 / 3.0, 3), atol=1e-15):
        raise ValueError("Primary ensemble must use exactly equal one-third weights")


def build_snv_logistic_regression(config: Mapping[str, Any]) -> Pipeline:
    """Construct SNV-LR with every trainable step inside the fold pipeline."""

    specification = config["models"][MODEL_SNV_LR]
    return Pipeline(
        [
            ("snv", StandardNormalVariate()),
            ("standardize", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(specification["C"]),
                    solver=str(specification["solver"]),
                    max_iter=int(specification["max_iter"]),
                    tol=float(specification["tol"]),
                    random_state=int(config["analysis_seed"]),
                ),
            ),
        ]
    )


def hash_development_csv_records(
    records: Sequence[SampleRecord],
) -> tuple[dict[str, str], str]:
    """Hash development CSVs only, rejecting mixed input before the first open."""

    ordered = tuple(sorted(records, key=lambda record: record.sample_index))
    if not ordered:
        raise ValueError("No development records supplied for hashing")
    nondevelopment = [record.sample_id for record in ordered if record.analysis_split != "development"]
    if nondevelopment:
        raise LockedDataAccessError(
            "Development hashing received a non-development record before file access; "
            f"first={nondevelopment[0]}"
        )
    if len({record.relative_csv_path for record in ordered}) != len(ordered):
        raise ValueError("Duplicate development CSV paths")

    per_file: dict[str, str] = {}
    aggregate = hashlib.sha256()
    for record in ordered:
        digest = sha256_file(record.csv_path)
        per_file[record.relative_csv_path] = digest
        aggregate.update(f"{record.relative_csv_path}\0{digest}\n".encode("utf-8"))
    return per_file, aggregate.hexdigest()


def _batch_ids(dataset: SpectralDataset) -> np.ndarray:
    batch_ids = np.asarray([record.constructed_batch for record in dataset.records], dtype=np.int64)
    if set(batch_ids.tolist()) != set(DEVELOPMENT_BATCHES):
        raise ValueError("Development records do not cover exactly constructed batches 0..7")
    if any(record.analysis_split != "development" for record in dataset.records):
        raise LockedDataAccessError("Loaded development dataset contains a locked record")
    return batch_ids


def _fold_rows(
    model_name: str,
    y: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    classes: np.ndarray,
    ece_bins: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch in DEVELOPMENT_BATCHES:
        mask = groups == batch
        metrics = multiclass_metrics(
            y[mask], probabilities[mask], classes=classes, ece_bins=ece_bins
        )
        rows.append(
            {
                "model": model_name,
                "evaluation_scope": "development_leave_one_constructed_batch_out",
                "held_out_batch": batch,
                **metrics,
                "errors": int(mask.sum() - round(float(metrics["accuracy"]) * int(mask.sum()))),
            }
        )
    return rows


def summarize_probability_models(
    y: np.ndarray,
    groups: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
    *,
    classes: np.ndarray,
    ece_bins: int,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Build stable global and grouped metric tables from aligned probabilities."""

    metric_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        if model_name not in probabilities:
            raise ValueError(f"Missing probability matrix for {model_name}")
        matrix = np.asarray(probabilities[model_name], dtype=np.float64)
        model_fold_rows = _fold_rows(model_name, y, matrix, groups, classes, ece_bins)
        metrics = multiclass_metrics(y, matrix, classes=classes, ece_bins=ece_bins)
        metric_rows.append(
            {
                "model": model_name,
                "evaluation_scope": "development_grouped_oof",
                **metrics,
                "equal_constructed_batch_accuracy": float(
                    np.mean([float(row["accuracy"]) for row in model_fold_rows])
                ),
                "errors": int(y.size - np.sum(classes[matrix.argmax(axis=1)] == y)),
            }
        )
        fold_rows.extend(model_fold_rows)
    return tuple(metric_rows), tuple(fold_rows)


def evaluate_development_models(
    dataset: SpectralDataset,
    config: Mapping[str, Any],
) -> DevelopmentEvaluation:
    """Execute the five frozen development predictors without locked access."""

    if dataset.analysis_split != "development":
        raise LockedDataAccessError("Development evaluation received another split")
    X = np.asarray(dataset.X, dtype=np.float64)
    y = np.asarray(dataset.y, dtype=np.int64)
    groups = _batch_ids(dataset)
    classes = np.arange(len(CLASS_NAMES), dtype=np.int64)
    if set(y.tolist()) != set(classes.tolist()):
        raise ValueError("Development dataset must contain all eight origin classes")

    snv_result = grouped_oof_probabilities(build_snv_logistic_regression(config), X, y, groups)
    lda_result = grouped_oof_probabilities(build_sg15_shrinkage_lda(), X, y, groups)
    lr_spec = config["models"][MODEL_SG_LR]
    lr_result = grouped_oof_probabilities(
        build_sg15_logistic_regression(
            C=float(lr_spec["C"]),
            max_iter=int(lr_spec["max_iter"]),
            tol=float(lr_spec["tol"]),
            random_state=int(config["analysis_seed"]),
        ),
        X,
        y,
        groups,
    )
    svm_result = nested_grouped_oof_temperature_probabilities(
        build_sg15_rbf_svm(), X, y, groups
    )
    for observed_classes in (snv_result.classes, lda_result.classes, lr_result.classes):
        if not np.array_equal(observed_classes, classes):
            raise AssertionError("OOF class order differs from frozen class order")

    probability_map: dict[str, np.ndarray] = {
        MODEL_SNV_LR: snv_result.probabilities,
        MODEL_SG_LDA: lda_result.probabilities,
        MODEL_SG_LR: lr_result.probabilities,
        MODEL_SG_SVM: svm_result.probabilities,
    }
    probability_map[MODEL_ENSEMBLE] = equal_weight_probability_average(
        [probability_map[member] for member in ENSEMBLE_MEMBERS]
    )
    metric_rows, fold_metric_rows = summarize_probability_models(
        y,
        groups,
        probability_map,
        classes=classes,
        ece_bins=int(config["evaluation"]["ece_bins"]),
    )
    return DevelopmentEvaluation(
        probabilities=probability_map,
        metric_rows=metric_rows,
        fold_metric_rows=fold_metric_rows,
        svm_final_temperature=float(svm_result.final_temperature),
        svm_outer_temperatures=svm_result.fold_temperatures,
    )


def build_manifest_rows(
    records: Sequence[SampleRecord],
    csv_hashes: Mapping[str, str],
    class_origins_zh: Sequence[str],
) -> list[dict[str, Any]]:
    """Create a development-only portable manifest table."""

    if any(record.analysis_split != "development" for record in records):
        raise LockedDataAccessError("Output manifest may contain development records only")
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.relative_csv_path not in csv_hashes:
            raise ValueError(f"Missing development CSV hash for {record.relative_csv_path}")
        rows.append(
            {
                "sample_index": record.sample_index,
                "sample_id": record.sample_id,
                "label": record.label,
                "class_code": record.class_name,
                "origin_zh": class_origins_zh[record.label],
                "source_cube": record.source_cube,
                "source_replicate": record.replicate,
                "seed_id": record.seed_id,
                "constructed_batch": record.constructed_batch,
                "analysis_split": record.analysis_split,
                "relative_csv_path": record.relative_csv_path,
                "relative_mat_path": record.relative_mat_path,
                "csv_path_sha256": record.csv_path_sha256,
                "mat_path_sha256": record.mat_path_sha256,
                "csv_size_bytes": record.csv_size_bytes,
                "mat_size_bytes": record.mat_size_bytes,
                "development_csv_sha256": csv_hashes[record.relative_csv_path],
                "mat_content_sha256": "not_read_in_development",
            }
        )
    return rows


def build_prediction_rows(
    dataset: SpectralDataset,
    probabilities: Mapping[str, np.ndarray],
    class_origins_zh: Sequence[str],
) -> list[dict[str, Any]]:
    """Create long model-by-sample predictions with all eight probabilities."""

    if dataset.analysis_split != "development" or any(
        record.analysis_split != "development" for record in dataset.records
    ):
        raise LockedDataAccessError("Prediction outputs may contain development samples only")
    rows: list[dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        matrix = np.asarray(probabilities[model_name], dtype=np.float64)
        if matrix.shape != (len(dataset.records), len(CLASS_NAMES)):
            raise ValueError(f"Unexpected probability shape for {model_name}: {matrix.shape}")
        if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError(f"Probability rows do not sum to one for {model_name}")
        predicted = matrix.argmax(axis=1)
        for row_index, record in enumerate(dataset.records):
            prediction = int(predicted[row_index])
            row: dict[str, Any] = {
                "model": model_name,
                "evaluation_scope": "development_grouped_oof",
                "sample_index": record.sample_index,
                "sample_id": record.sample_id,
                "true_label": record.label,
                "true_class_code": record.class_name,
                "true_origin_zh": class_origins_zh[record.label],
                "source_cube": record.source_cube,
                "source_replicate": record.replicate,
                "seed_id": record.seed_id,
                "constructed_batch": record.constructed_batch,
                "predicted_label": prediction,
                "predicted_class_code": CLASS_NAMES[prediction],
                "correct": int(prediction == record.label),
                "confidence": float(matrix[row_index, prediction]),
            }
            for class_index, class_code in enumerate(CLASS_NAMES):
                row[f"probability_{class_index}_{class_code}"] = float(
                    matrix[row_index, class_index]
                )
            rows.append(row)
    return rows


def _metric_by_model(rows: Sequence[Mapping[str, Any]], model_name: str) -> Mapping[str, Any]:
    matches = [row for row in rows if row["model"] == model_name]
    if len(matches) != 1:
        raise ValueError(f"Expected one metric row for {model_name}; observed {len(matches)}")
    return matches[0]


def _package_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
    }
    for package in ("joblib", "threadpoolctl"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def _git_provenance(repository_root: Path) -> dict[str, Any]:
    def command(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        commit = command("rev-parse", "HEAD")
        branch = command("branch", "--show-current")
        tracked_status = command("status", "--porcelain", "--untracked-files=no")
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": "unavailable", "branch": "unavailable", "tracked_worktree_clean": False}
    return {
        "commit": commit,
        "branch": branch,
        "tracked_worktree_clean": tracked_status == "",
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object is not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")


def _prepare_output_directory(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [name for name in OUTPUT_FILENAMES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Development outputs already exist ({existing}); pass --overwrite to replace known files"
        )


def _format_percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def build_chinese_report(
    metric_rows: Sequence[Mapping[str, Any]],
    *,
    final_temperature: float,
    development_fingerprint: str,
    access_audit: Mapping[str, Any],
) -> str:
    """Render an evidence-bounded report from already computed development rows."""

    lines = [
        "# 现有数据产地溯源开发集分析报告",
        "",
        "> 结果状态：本文件仅汇报构造批次 0–7 的新执行开发集分组 OOF 结果。构造批次 8–9 的光谱数值和文件字节均未读取，因此本文件不是最终锁定测试报告，也不构成外部批次验证。",
        "",
        "## 数据访问边界",
        "",
        f"- 开发样本数：{access_audit['development_samples']}；仅这些 CSV 被数值读取和内容哈希。",
        f"- 锁定样本路径枚举数：{access_audit['locked_paths_enumerated']}；锁定数值读取：{access_audit['locked_numeric_reads']}；锁定文件字节读取：{access_audit['locked_byte_reads']}。",
        f"- 开发 CSV 内容指纹：`{development_fingerprint}`。",
        "- MAT 文件在本阶段没有打开、解析或哈希。",
        "",
        "## 八折开发结果",
        "",
        "| 模型 | 平衡准确率 | Macro-F1 | 准确率 | 错分数 | NLL | Brier | ECE | 批次等权准确率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            "| {model} | {ba} | {f1} | {acc} | {errors} | {nll:.4f} | {brier:.4f} | {ece:.4f} | {batch_acc} |".format(
                model=row["model"],
                ba=_format_percent(float(row["balanced_accuracy"])),
                f1=_format_percent(float(row["macro_f1"])),
                acc=_format_percent(float(row["accuracy"])),
                errors=int(row["errors"]),
                nll=float(row["negative_log_likelihood"]),
                brier=float(row["multiclass_brier_score"]),
                ece=float(row["expected_calibration_error"]),
                batch_acc=_format_percent(float(row["equal_constructed_batch_accuracy"])),
            )
        )
    ensemble = _metric_by_model(metric_rows, MODEL_ENSEMBLE)
    lda = _metric_by_model(metric_rows, MODEL_SG_LDA)
    lines.extend(
        [
            "",
            "## 冻结选择",
            "",
            f"三成员等权融合在开发 OOF 上的平衡准确率为 {_format_percent(float(ensemble['balanced_accuracy']))}，共错分 {int(ensemble['errors'])} 粒；SG15–shrinkage LDA 为 {_format_percent(float(lda['balanced_accuracy']))}。这些数值用于冻结当前数据内候选方案，而不是估计跨年份、跨农场或跨设备推广性能。",
            f"SVM 的最终部署温度为 `{final_temperature:.10f}`。开发性能使用逐外层批次重新拟合的 nested 温度；该最终温度不会回填开发 OOF 性能。",
            "",
            "## 解释边界",
            "",
            "构造批次是导师授权的当前数据内分析单位，但不是新增的物理采收批次。现有结果支持当前数据内的八产地闭集溯源算法选择；锁定批次结果、来源图像互换压力测试及不确定性分析需由后续正式入口另行执行。",
            "",
        ]
    )
    return "\n".join(lines)


def run_development_analysis(
    *,
    data_root: Path,
    config_path: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run, validate, and persist the complete development-only analysis."""

    started = time.perf_counter()
    started_utc = _utc_now()
    config = _load_json(config_path)
    validate_frozen_config(config)

    # Blind discovery is intentionally explicit.  It may stat locked paths but
    # cannot hash or numerically parse any locked file.
    manifest: Manifest = discover_manifest(
        data_root,
        base_seed=int(config["analysis_seed"]),
        hash_files=False,
    )
    if manifest.hashes_complete or manifest.csv_content_sha256 or manifest.mat_content_sha256:
        raise AssertionError("Blind discovery unexpectedly produced file-content hashes")
    development_records = manifest.records_for_split("development")
    locked_records = manifest.records_for_split("locked")
    if len(development_records) != EXPECTED_DEVELOPMENT_SAMPLES:
        raise AssertionError(
            f"Expected {EXPECTED_DEVELOPMENT_SAMPLES} development samples; observed {len(development_records)}"
        )
    if len(locked_records) != EXPECTED_LOCKED_SAMPLES:
        raise AssertionError(
            f"Expected {EXPECTED_LOCKED_SAMPLES} locked samples; observed {len(locked_records)}"
        )

    # Two development-only hash passes bracket numerical loading and detect
    # concurrent data changes without ever opening locked bytes.
    hashes_before, fingerprint_before = hash_development_csv_records(development_records)
    dataset = load_csv_split(
        manifest,
        split="development",
        expected_bands=int(config["expected_bands"]),
        verify_hashes=False,
    )
    hashes_after, development_fingerprint = hash_development_csv_records(development_records)
    if hashes_before != hashes_after or fingerprint_before != development_fingerprint:
        raise RuntimeError("Development CSV content changed during analysis loading")
    if dataset.X.shape != (EXPECTED_DEVELOPMENT_SAMPLES, EXPECTED_BANDS):
        raise AssertionError(f"Unexpected development matrix shape: {dataset.X.shape}")

    evaluation = evaluate_development_models(dataset, config)
    expected_temperature = float(
        config["primary_predictor"]["svm_temperature_frozen_development_value"]
    )
    temperature_atol = float(
        config["primary_predictor"].get(
            "svm_temperature_assertion_atol", DEFAULT_TEMPERATURE_ATOL
        )
    )
    if not math.isclose(
        evaluation.svm_final_temperature,
        expected_temperature,
        rel_tol=0.0,
        abs_tol=temperature_atol,
    ):
        raise AssertionError(
            "Deployable SVM temperature drift: "
            f"observed={evaluation.svm_final_temperature:.12g}, "
            f"expected={expected_temperature:.12g}, atol={temperature_atol:g}"
        )

    ensemble_metrics = _metric_by_model(evaluation.metric_rows, MODEL_ENSEMBLE)
    reproduction_checks = {
        "development_n_expected": EXPECTED_DEVELOPMENT_SAMPLES,
        "development_n_observed": len(dataset.records),
        "ensemble_errors_expected": EXPECTED_ENSEMBLE_ERRORS,
        "ensemble_errors_observed": int(ensemble_metrics["errors"]),
        "ensemble_balanced_accuracy_reference": EXPECTED_ENSEMBLE_BALANCED_ACCURACY,
        "ensemble_balanced_accuracy_observed": float(ensemble_metrics["balanced_accuracy"]),
        "ensemble_balanced_accuracy_atol": ENSEMBLE_BALANCED_ACCURACY_ATOL,
        "svm_temperature_expected": expected_temperature,
        "svm_temperature_observed": evaluation.svm_final_temperature,
        "svm_temperature_atol": temperature_atol,
    }
    if int(ensemble_metrics["errors"]) != EXPECTED_ENSEMBLE_ERRORS:
        raise AssertionError(
            f"Ensemble prediction drift: expected 18 errors, observed {ensemble_metrics['errors']}"
        )
    if not math.isclose(
        float(ensemble_metrics["balanced_accuracy"]),
        EXPECTED_ENSEMBLE_BALANCED_ACCURACY,
        rel_tol=0.0,
        abs_tol=ENSEMBLE_BALANCED_ACCURACY_ATOL,
    ):
        raise AssertionError(
            "Ensemble balanced-accuracy drift: "
            f"observed={ensemble_metrics['balanced_accuracy']}, "
            f"reference={EXPECTED_ENSEMBLE_BALANCED_ACCURACY}"
        )

    access_audit = {
        "development_samples": len(development_records),
        "development_csv_numeric_reads": len(development_records),
        "development_csv_hash_passes": 2,
        "development_csv_byte_hash_reads": 2 * len(development_records),
        "development_mat_numeric_reads": 0,
        "development_mat_byte_reads": 0,
        "locked_paths_enumerated": len(locked_records),
        "locked_file_metadata_entries_enumerated": 2 * len(locked_records),
        "locked_numeric_reads": 0,
        "locked_byte_reads": 0,
        "locked_csv_hashes": 0,
        "locked_mat_hashes": 0,
        "enforcement": [
            "discover_manifest(hash_files=False)",
            "load_csv_split(split='development')",
            "hash_development_csv_records prevalidates every record before opening a file",
        ],
    }
    manifest_rows = build_manifest_rows(
        dataset.records, hashes_after, config["class_origins_zh"]
    )
    prediction_rows = build_prediction_rows(
        dataset, evaluation.probabilities, config["class_origins_zh"]
    )
    wavelengths_rows = [
        {"band_index": index, "wavelength_nm": float(wavelength)}
        for index, wavelength in enumerate(dataset.wavelengths)
    ]

    repository_root = Path(__file__).resolve().parents[1]
    runtime = {
        "started_utc": started_utc,
        "completed_utc": _utc_now(),
        "duration_seconds": float(time.perf_counter() - started),
        "platform": platform.platform(),
        "package_versions": _package_versions(),
        "git": _git_provenance(repository_root),
        "entrypoint_sha256": sha256_file(Path(__file__).resolve()),
        "core_sha256": sha256_file(Path(__file__).with_name("core.py")),
        "config_sha256": sha256_file(config_path),
    }
    temperature_payload = {
        "status": "newly_executed_development_only",
        "model": MODEL_SG_SVM,
        "probability_method": "softmax_of_ovr_decision_scores",
        "svm_internal_platt_probability": False,
        "temperature_objective": "multiclass_negative_log_likelihood",
        "log_temperature_bounds": config["models"]["sg1_rbf_svm"]["temperature_log_bounds"],
        "outer_performance_protocol": "outer batch excluded from all inner OOF temperature fits",
        "outer_fold_temperatures": [
            {"held_out_batch": batch, "temperature": temperature}
            for batch, temperature in evaluation.svm_outer_temperatures
        ],
        "deployment_temperature_protocol": "eight_fold_leave_one_development_batch_out_scores",
        "deployment_temperature": evaluation.svm_final_temperature,
        "frozen_expected_temperature": expected_temperature,
        "absolute_tolerance": temperature_atol,
        "absolute_difference": abs(evaluation.svm_final_temperature - expected_temperature),
        "assertion_passed": True,
        "locked_access": access_audit,
    }
    strongest_single = max(
        (
            row
            for row in evaluation.metric_rows
            if row["model"] != MODEL_ENSEMBLE
        ),
        key=lambda row: float(row["balanced_accuracy"]),
    )
    selection_summary = {
        "status": "newly_executed_development_only",
        "scientific_objective": "hyperspectral_geographical_origin_traceability_of_semen_ziziphi_spinosae",
        "primary_predictor": MODEL_ENSEMBLE,
        "ensemble_members": list(ENSEMBLE_MEMBERS),
        "ensemble_weights": [1.0 / 3.0] * 3,
        "analysis_units": "deterministically_constructed_batches",
        "development_batches": list(DEVELOPMENT_BATCHES),
        "locked_batches": list(LOCKED_BATCHES),
        "primary_metric": config["evaluation"]["primary_metric"],
        "primary_development_metrics": dict(ensemble_metrics),
        "strongest_single_model": dict(strongest_single),
        "balanced_accuracy_gain_over_strongest_single": float(
            ensemble_metrics["balanced_accuracy"] - strongest_single["balanced_accuracy"]
        ),
        "development_csv_content_sha256": development_fingerprint,
        "structural_manifest_sha256": manifest.manifest_sha256,
        "structural_assignment_fingerprint_sha256": manifest.data_fingerprint_sha256,
        "full_manifest_content_hashing_performed": False,
        "access_audit": access_audit,
        "reproduction_checks": reproduction_checks,
        "configuration": config,
        "runtime": runtime,
        "limitations": [
            "constructed batches are analytical units, not newly acquired physical lots",
            "locked batches were not opened and are not reported here",
            "no external year, farm, instrument, or acquisition-condition validation is available",
            "MAT cubes and spatial features are outside this development entry point",
        ],
    }

    _prepare_output_directory(output_dir, overwrite)
    manifest_fields = list(manifest_rows[0])
    prediction_fields = list(prediction_rows[0])
    metric_fields = list(evaluation.metric_rows[0])
    fold_fields = list(evaluation.fold_metric_rows[0])
    _write_csv(output_dir / "manifest.csv", manifest_rows, manifest_fields)
    _write_csv(output_dir / "predictions.csv", prediction_rows, prediction_fields)
    _write_csv(output_dir / "metrics.csv", evaluation.metric_rows, metric_fields)
    _write_csv(output_dir / "fold_metrics.csv", evaluation.fold_metric_rows, fold_fields)
    _write_csv(output_dir / "wavelengths.csv", wavelengths_rows, ("band_index", "wavelength_nm"))
    _write_json(output_dir / "temperature_calibration.json", temperature_payload)
    _write_json(output_dir / "selection_summary.json", selection_summary)
    report = build_chinese_report(
        evaluation.metric_rows,
        final_temperature=evaluation.svm_final_temperature,
        development_fingerprint=development_fingerprint,
        access_audit=access_audit,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return selection_summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    package_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run locked-blind grouped development analysis for geographical-origin traceability"
    )
    parser.add_argument("--data-root", type=Path, default=repository_root / "data")
    parser.add_argument("--config", type=Path, default=package_root / "config.json")
    parser.add_argument(
        "--output-dir", type=Path, default=package_root / "outputs" / "development"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the eight known development artifacts if they already exist",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_development_analysis(
        data_root=args.data_root.resolve(),
        config_path=args.config.resolve(),
        output_dir=args.output_dir.resolve(),
        overwrite=bool(args.overwrite),
    )
    metrics = summary["primary_development_metrics"]
    print(
        "Development analysis complete: "
        f"balanced_accuracy={metrics['balanced_accuracy']:.6f}, "
        f"errors={metrics['errors']}, output={args.output_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
