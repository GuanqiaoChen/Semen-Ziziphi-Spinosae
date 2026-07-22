"""Run the residual CNN on the eight development batches only.

This entry point deliberately has no locked-evaluation mode.  Manifest
discovery uses ``hash_files=False`` and only records assigned to constructed
batches 0--7 are passed to the development CSV loader.  Batches 8--9 are never
parsed, hashed, or supplied to the CNN evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import sklearn
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from provenance_study.cnn_baseline import (  # noqa: E402
    CNN_PARAMETER_COUNT,
    DEFAULT_CNN_CONFIG,
    CNNGroupedOOFResult,
    CNNTrainingConfig,
    evaluate_development_batches,
)
from provenance_study.core import (  # noqa: E402
    BASE_BATCH_SEED,
    CLASS_NAMES,
    DEVELOPMENT_BATCHES,
    EXPECTED_BANDS,
    EXPECTED_SOURCE_COUNTS,
    Manifest,
    discover_manifest,
    load_development_csv,
    multiclass_metrics,
    sha256_file,
)


DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "outputs" / "development_cnn"


def validate_development_batches(batches: Sequence[int]) -> tuple[int, ...]:
    """Accept exactly the fixed development batch IDs and nothing else."""

    requested = tuple(int(batch) for batch in batches)
    if len(requested) != len(DEVELOPMENT_BATCHES) or set(requested) != set(
        DEVELOPMENT_BATCHES
    ):
        raise ValueError(
            "The CNN development entry point permits exactly batches 0--7; "
            f"observed {sorted(set(requested))}"
        )
    return DEVELOPMENT_BATCHES


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV artifact: {path.name}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _environment_metadata(device: str | torch.device | None) -> dict[str, Any]:
    if device is None:
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved = torch.device(device)
    cuda_device_name = None
    if resolved.type == "cuda" and torch.cuda.is_available():
        cuda_device_name = torch.cuda.get_device_name(resolved)
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "requested_or_auto_device": str(resolved),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_device_name": cuda_device_name,
    }


def _prediction_rows(
    manifest: Manifest,
    records: Sequence[Any],
    y: np.ndarray,
    result: CNNGroupedOOFResult,
) -> list[dict[str, Any]]:
    del manifest  # Retained in the signature to make the artifact provenance explicit.
    predictions = result.classes[result.probabilities.argmax(axis=1)]
    rows: list[dict[str, Any]] = []
    for position, (record, actual, predicted, held_out) in enumerate(
        zip(records, y, predictions, result.held_out_batch, strict=True)
    ):
        row: dict[str, Any] = {
            "development_row": position,
            "sample_index": record.sample_index,
            "sample_id": record.sample_id,
            "source_cube": record.source_cube,
            "seed_id": record.seed_id,
            "constructed_batch": record.constructed_batch,
            "held_out_batch": int(held_out),
            "true_label": int(actual),
            "true_class": CLASS_NAMES[int(actual)],
            "predicted_label": int(predicted),
            "predicted_class": CLASS_NAMES[int(predicted)],
            "correct": int(actual == predicted),
        }
        row.update(
            {
                f"probability_{class_name}": float(result.probabilities[position, index])
                for index, class_name in enumerate(CLASS_NAMES)
            }
        )
        rows.append(row)
    return rows


def _fold_rows(result: CNNGroupedOOFResult) -> list[dict[str, Any]]:
    return [asdict(fold) for fold in result.folds]


def _render_report(results: Mapping[str, Any]) -> str:
    metrics = results["development_oof_metrics"]
    architecture = results["cnn"]["architecture"]
    return "\n".join(
        [
            "# 残差一维卷积网络开发集评估",
            "",
            f"- 执行状态：`{results['execution_status']}`",
            "- 分析目标：基于高光谱均值光谱的八产地溯源。",
            "- 数值分析范围：仅构建批次 0–7；批次 8–9 未读取。",
            f"- 锁定批次数值读取次数：**{results['data_access']['locked_numeric_reads']}**。",
            f"- 清单发现时文件内容哈希：`{results['manifest']['hashes_complete']}`。",
            "",
            "## 开发集 OOF 结果",
            "",
            f"- 样本数：{metrics['n']}",
            f"- 平衡准确率：{metrics['balanced_accuracy']:.6f}",
            f"- Macro-F1：{metrics['macro_f1']:.6f}",
            f"- 准确率：{metrics['accuracy']:.6f}",
            f"- 负对数似然：{metrics['negative_log_likelihood']:.6f}",
            f"- 多分类 Brier 分数：{metrics['multiclass_brier_score']:.6f}",
            f"- ECE：{metrics['expected_calibration_error']:.6f}",
            f"- 等权构建批次准确率：{results['equal_constructed_batch_accuracy']:.6f}",
            "",
            "## 模型与验证规则",
            "",
            f"模型为 {architecture['stages']} 阶段残差一维 CNN，共 "
            f"{architecture['parameter_count']:,} 个可训练参数。每个外层批次 g 的早停批次固定为 "
            "(g+1)%8；选定 epoch 后，从头在全部非外层开发批次上重训。",
            "",
            "## 解释边界",
            "",
            "该结果仅是当前数据构建批次 0–7 上的开发集 OOF 结果，不是锁定测试结果，也不是新产地、"
            "新年份、新物理批次或新仪器上的外部验证。构建批次不能等同于独立采收批次。",
            "",
        ]
    )


def run_development_cnn(
    data_root: Path,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    *,
    development_batches: Sequence[int] = DEVELOPMENT_BATCHES,
    expected_source_counts: Mapping[tuple[int, int], int] = EXPECTED_SOURCE_COUNTS,
    expected_bands: int = EXPECTED_BANDS,
    optimization_seed: int = 20260721,
    config: CNNTrainingConfig = DEFAULT_CNN_CONFIG,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Execute and serialize the development-only grouped CNN evaluation."""

    permitted_batches = validate_development_batches(development_batches)
    manifest = discover_manifest(
        Path(data_root),
        expected_source_counts=expected_source_counts,
        base_seed=BASE_BATCH_SEED,
        hash_files=False,
    )
    if manifest.hashes_complete:
        raise RuntimeError("Development manifest unexpectedly hashed file content")
    development_records = manifest.records_for_batches(permitted_batches)
    if not development_records or any(
        record.analysis_split != "development" for record in development_records
    ):
        raise RuntimeError("Non-development record reached the CNN development loader")
    dataset = load_development_csv(
        development_records,
        expected_bands=expected_bands,
        verify_hashes=False,
    )
    loaded_batches = tuple(sorted({record.constructed_batch for record in dataset.records}))
    if loaded_batches != permitted_batches:
        raise RuntimeError(
            f"Loaded batches {loaded_batches} do not equal development batches {permitted_batches}"
        )

    groups = np.asarray(
        [record.constructed_batch for record in dataset.records], dtype=np.int64
    )
    result = evaluate_development_batches(
        dataset.X,
        dataset.y,
        groups,
        optimization_seed=optimization_seed,
        config=config,
        device=device,
    )
    metrics = multiclass_metrics(
        dataset.y, result.probabilities, classes=result.classes, ece_bins=10
    )
    predictions = result.classes[result.probabilities.argmax(axis=1)]
    batch_accuracies = [
        float(np.mean(predictions[groups == batch] == dataset.y[groups == batch]))
        for batch in permitted_batches
    ]

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    prediction_path = output_directory / "predictions.csv"
    fold_path = output_directory / "folds.csv"
    results_path = output_directory / "results.json"
    report_path = output_directory / "report.md"
    _write_csv(
        prediction_path,
        _prediction_rows(manifest, dataset.records, dataset.y, result),
    )
    _write_csv(fold_path, _fold_rows(result))

    results: dict[str, Any] = {
        "execution_status": "executed_complete_development_only",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_objective": "eight_origin_hyperspectral_provenance_traceability",
        "data_access": {
            "permitted_constructed_batches": list(permitted_batches),
            "loaded_constructed_batches": list(loaded_batches),
            "locked_constructed_batches": [8, 9],
            "locked_numeric_reads": 0,
            "locked_records_loaded": 0,
            "loader": "load_development_csv",
        },
        "manifest": {
            "data_root": str(manifest.data_root),
            "n_all_path_records": len(manifest.records),
            "n_development_records_loaded": len(dataset.records),
            "manifest_sha256": manifest.manifest_sha256,
            "data_fingerprint_sha256": manifest.data_fingerprint_sha256,
            "loaded_csv_fingerprint_sha256": dataset.loaded_csv_fingerprint_sha256,
            "hashes_complete": manifest.hashes_complete,
            "csv_content_sha256": manifest.csv_content_sha256,
            "mat_content_sha256": manifest.mat_content_sha256,
        },
        "cnn": {
            "architecture": {
                "name": "three_stage_residual_1d_cnn",
                "stages": 3,
                "channels": [24, 48, 96],
                "residual_blocks_per_stage": 2,
                "block_kernel_sizes": [7, 5],
                "transition_kernel_size": 5,
                "global_pooling": "adaptive_average",
                "head_dropout": 0.30,
                "parameter_count": CNN_PARAMETER_COUNT,
            },
            "preprocessing": (
                f"SG{config.savgol_derivative}(window={config.savgol_window_length},"
                f"polyorder={config.savgol_polyorder})+training_partition_band_standardization"
            ),
            "constructed_batch_assignment_seed": BASE_BATCH_SEED,
            "training_config": asdict(config),
            "optimization_seed": optimization_seed,
            "outer_validation": "leave_one_development_batch_out",
            "inner_epoch_selection": "inner_validation_batch=(outer_batch+1)%8",
            "post_selection_refit": "fresh_initialization_on_all_non_outer_development_batches",
        },
        "development_oof_metrics": metrics,
        "constructed_batch_accuracies": batch_accuracies,
        "equal_constructed_batch_accuracy": float(np.mean(batch_accuracies)),
        "optimization_seed_stability_scope": "single_executed_optimization_seed",
        "elapsed_seconds": result.elapsed_seconds,
        "environment": _environment_metadata(device),
        "artifacts": {
            "predictions.csv": sha256_file(prediction_path),
            "folds.csv": sha256_file(fold_path),
        },
        "inference_boundary": (
            "Development OOF evidence conditional on constructed batches 0--7 only; "
            "not a locked test or external lot/year/instrument validation."
        ),
    }
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_render_report(results), encoding="utf-8")
    return results


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the development-only residual spectral CNN evaluation."
    )
    repository_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-root", type=Path, default=repository_root / "data")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument(
        "--development-batches",
        nargs="+",
        type=int,
        default=list(DEVELOPMENT_BATCHES),
        help="Must be exactly 0 1 2 3 4 5 6 7.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    run_development_cnn(
        arguments.data_root,
        arguments.output_dir,
        development_batches=arguments.development_batches,
        device=arguments.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
