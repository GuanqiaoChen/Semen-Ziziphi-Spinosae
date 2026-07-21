#!/usr/bin/env python3
"""Run leakage-aware baseline analyses on the currently available CSV spectra.

This module deliberately does not import or modify the legacy modelling scripts.  It
uses fixed, declared preprocessing/model combinations and never selects
hyperparameters using a held-out test partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC


PROTOCOL_LABELS = {
    "random_seed_holdout": "随机种子级 75/25 留出",
    "suffix_1_to_2": "采集立方体后缀 1→2",
    "suffix_2_to_1": "采集立方体后缀 2→1",
    "leave_one_cube_out": "留一采集立方体（16 折汇总）",
}


@dataclass(frozen=True)
class Dataset:
    """Loaded spectra and their mandatory hierarchy metadata."""

    X: np.ndarray
    y: np.ndarray
    wavelengths: np.ndarray
    manifest: pd.DataFrame
    fingerprint_sha256: str


class MultiplicativeScatterCorrection(BaseEstimator, TransformerMixin):
    """MSC using only the training-set mean spectrum as the reference.

    The reference is learned in ``fit`` and therefore cannot use test spectra.  For
    each spectrum x, least squares estimates ``x = intercept + slope * reference``;
    the corrected spectrum is ``(x - intercept) / slope``.
    """

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("MSC expects a two-dimensional sample-by-band matrix")
        self.reference_ = X.mean(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "reference_"):
            raise RuntimeError("MSC must be fitted before transform")
        X = np.asarray(X, dtype=float)
        reference = np.asarray(self.reference_, dtype=float)
        design = np.column_stack([np.ones(reference.size), reference])
        coefficients = np.linalg.lstsq(design, X.T, rcond=None)[0]
        intercept = coefficients[0]
        slope = coefficients[1]
        eps = np.finfo(float).eps
        safe_slope = np.where(np.abs(slope) < eps, np.where(slope < 0, -eps, eps), slope)
        return (X - intercept[:, None]) / safe_slope[:, None]


def snv_transform(X: np.ndarray) -> np.ndarray:
    """Standard normal variate transform, independently for each spectrum."""

    X = np.asarray(X, dtype=float)
    centered = X - X.mean(axis=1, keepdims=True)
    scale = X.std(axis=1, ddof=1, keepdims=True)
    scale = np.where(scale <= np.finfo(float).eps, 1.0, scale)
    return centered / scale


class SavitzkyGolayTransformer(BaseEstimator, TransformerMixin):
    """Sample-wise Savitzky--Golay smoothing or derivative."""

    def __init__(self, window_length: int = 11, polyorder: int = 2, deriv: int = 0):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        X = np.asarray(X)
        if self.window_length % 2 != 1:
            raise ValueError("Savitzky-Golay window_length must be odd")
        if self.window_length > X.shape[1]:
            raise ValueError("Savitzky-Golay window is longer than the spectrum")
        if self.polyorder >= self.window_length:
            raise ValueError("Savitzky-Golay polyorder must be smaller than window_length")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return savgol_filter(
            np.asarray(X, dtype=float),
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=self.deriv,
            axis=1,
            mode="interp",
        )


class PLSDAClassifier(BaseEstimator):
    """Fixed PLS-DA: one-hot PLS regression followed by response argmax.

    The internal scaler and PLS model are created and fitted anew for every
    training partition.  This matches the supplied legacy PLS-DA definition while
    preventing any test-partition information from entering preprocessing.
    """

    def __init__(self, n_components: int = 20, n_classes: int = 8, max_iter: int = 1000):
        self.n_components = n_components
        self.n_classes = n_classes
        self.max_iter = max_iter

    def fit(self, X: np.ndarray, y: np.ndarray):
        y = np.asarray(y, dtype=int)
        responses = np.zeros((y.size, self.n_classes), dtype=np.float32)
        responses[np.arange(y.size), y] = 1.0
        self.pipeline_ = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "pls",
                    PLSRegression(n_components=self.n_components, max_iter=self.max_iter),
                ),
            ]
        )
        self.pipeline_.fit(np.asarray(X, dtype=float), responses)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "pipeline_"):
            raise RuntimeError("PLSDAClassifier must be fitted before predict")
        responses = self.pipeline_.predict(np.asarray(X, dtype=float))
        return np.asarray(responses.argmax(axis=1), dtype=int)


def _cube_sort_key(path: Path) -> tuple[int, int]:
    left, right = path.name.split("-", maxsplit=1)
    return int(left), int(right)


def _legacy_lexical_csv_key(path: Path) -> str:
    """Match the lexical path ordering used by the supplied legacy scripts."""

    return path.name


def load_dataset(data_root: Path, expected_bands: int = 392) -> Dataset:
    """Load all ``data/<label>-<cube>/<seed>.csv`` mean spectra.

    Every CSV must have exactly two numeric columns (wavelength, reflectance) and a
    wavelength grid identical to the first file within numerical tolerance.
    """

    data_root = Path(data_root).resolve()
    cube_dirs = []
    for candidate in data_root.iterdir():
        if not candidate.is_dir():
            continue
        parts = candidate.name.split("-")
        if len(parts) == 2 and all(part.isdigit() for part in parts):
            cube_dirs.append(candidate)
    cube_dirs.sort(key=_cube_sort_key)
    if not cube_dirs:
        raise FileNotFoundError(f"No <label>-<cube> directories found under {data_root}")

    spectra: list[np.ndarray] = []
    labels: list[int] = []
    records: list[dict[str, Any]] = []
    reference_wavelengths: np.ndarray | None = None
    fingerprint = hashlib.sha256()

    for cube_dir in cube_dirs:
        label_text, replicate_text = cube_dir.name.split("-", maxsplit=1)
        label = int(label_text)
        replicate = int(replicate_text)
        csv_paths = sorted(cube_dir.glob("*.csv"), key=_legacy_lexical_csv_key)
        if not csv_paths:
            raise FileNotFoundError(f"No CSV spectra found in {cube_dir}")
        for csv_path in csv_paths:
            try:
                values = np.loadtxt(csv_path, delimiter=",", dtype=float)
            except ValueError as exc:
                raise ValueError(f"CSV is not a two-column numeric spectrum: {csv_path}") from exc
            if values.shape != (expected_bands, 2):
                raise ValueError(
                    f"Expected {expected_bands} rows and 2 columns in {csv_path}; got {values.shape}"
                )
            wavelengths = values[:, 0]
            reflectance = values[:, 1]
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Non-finite wavelength or reflectance found in {csv_path}")
            if np.any(np.diff(wavelengths) <= 0):
                raise ValueError(f"Wavelengths are not strictly increasing in {csv_path}")
            if reference_wavelengths is None:
                reference_wavelengths = wavelengths.copy()
            elif not np.allclose(wavelengths, reference_wavelengths, rtol=0.0, atol=1e-6):
                max_error = float(np.max(np.abs(wavelengths - reference_wavelengths)))
                raise ValueError(f"Wavelength grid mismatch ({max_error:g} nm) in {csv_path}")

            relative_path = csv_path.relative_to(data_root.parent).as_posix()
            file_digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
            fingerprint.update(relative_path.encode("utf-8"))
            fingerprint.update(file_digest.encode("ascii"))
            spectra.append(reflectance)
            labels.append(label)
            records.append(
                {
                    "sample_index": len(records),
                    "label": label,
                    "source_cube": cube_dir.name,
                    "cube_replicate": replicate,
                    "seed_id": csv_path.stem,
                    "relative_csv_path": relative_path,
                    "file_sha256": file_digest,
                }
            )

    assert reference_wavelengths is not None
    manifest = pd.DataFrame.from_records(records)
    observed_labels = sorted(manifest["label"].unique().tolist())
    expected_labels = list(range(max(observed_labels) + 1))
    if observed_labels != expected_labels:
        raise ValueError(f"Labels must be contiguous from zero; observed {observed_labels}")
    return Dataset(
        X=np.asarray(spectra, dtype=float),
        y=np.asarray(labels, dtype=int),
        wavelengths=reference_wavelengths,
        manifest=manifest,
        fingerprint_sha256=fingerprint.hexdigest(),
    )


def build_estimators(config: dict[str, Any]) -> dict[str, Any]:
    """Create all fixed estimators; there is intentionally no hyperparameter search."""

    seed = int(config["analysis_seed"])
    lr_cfg = config["models"]["logistic_regression"]
    svm_cfg = config["models"]["svm_rbf"]
    pls_cfg = config["models"]["pls_da"]
    rf_cfg = config["models"]["random_forest"]
    sg_cfg = config["preprocessing"]

    def lr() -> LogisticRegression:
        return LogisticRegression(
            C=float(lr_cfg["C"]),
            solver=str(lr_cfg["solver"]),
            max_iter=int(lr_cfg["max_iter"]),
            tol=float(lr_cfg["tol"]),
            random_state=seed,
        )

    def spectral_lr(preprocessor: Any) -> Pipeline:
        return Pipeline(
            [("preprocess", preprocessor), ("scale", StandardScaler()), ("classifier", lr())]
        )

    identity = FunctionTransformer(validate=False)
    return {
        # These first three reproduce the declared legacy random-holdout baselines.
        "raw_lr": spectral_lr(identity),
        "raw_svm": Pipeline(
            [
                ("preprocess", FunctionTransformer(validate=False)),
                ("scale", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        C=float(svm_cfg["C"]),
                        kernel="rbf",
                        gamma=svm_cfg["gamma"],
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "raw_pls_da": PLSDAClassifier(
            n_components=int(pls_cfg["n_components"]),
            n_classes=len(config["class_names"]),
            max_iter=int(pls_cfg["max_iter"]),
        ),
        "raw_rf": Pipeline(
            [
                ("preprocess", FunctionTransformer(validate=False)),
                ("scale", StandardScaler()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=int(rf_cfg["n_estimators"]),
                        max_depth=rf_cfg["max_depth"],
                        random_state=seed,
                        n_jobs=int(rf_cfg["n_jobs"]),
                    ),
                ),
            ]
        ),
        "snv_lr": spectral_lr(FunctionTransformer(snv_transform, validate=False)),
        "msc_lr": spectral_lr(MultiplicativeScatterCorrection()),
        "sg_smooth_lr": spectral_lr(
            SavitzkyGolayTransformer(
                window_length=int(sg_cfg["savgol_window_length"]),
                polyorder=int(sg_cfg["savgol_polyorder"]),
                deriv=0,
            )
        ),
        "sg_first_derivative_lr": spectral_lr(
            SavitzkyGolayTransformer(
                window_length=int(sg_cfg["savgol_window_length"]),
                polyorder=int(sg_cfg["savgol_polyorder"]),
                deriv=1,
            )
        ),
    }


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""

    if n <= 0:
        return math.nan, math.nan
    # 1.959963984540054 is the two-sided 95% standard-normal critical value.
    if not math.isclose(confidence, 0.95):
        from scipy.stats import norm

        z = float(norm.ppf(1 - (1 - confidence) / 2))
    else:
        z = 1.959963984540054
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - half_width, center + half_width


def _metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    return (
        float(accuracy_score(y_true, y_pred)),
        float(balanced_accuracy_score(y_true, y_pred)),
        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    )


def descriptive_bootstrap_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    repetitions: int,
    confidence: float,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """Naive sample bootstrap intervals, explicitly descriptive rather than inferential."""

    rng = np.random.default_rng(seed)
    n = y_true.size
    estimates = np.empty((repetitions, 3), dtype=float)
    for repetition in range(repetitions):
        indices = rng.integers(0, n, size=n)
        estimates[repetition] = _metric_values(y_true[indices], y_pred[indices])
    alpha = (1 - confidence) / 2
    bounds = np.quantile(estimates, [alpha, 1 - alpha], axis=0)
    names = ("accuracy", "balanced_accuracy", "macro_f1")
    return {name: (float(bounds[0, i]), float(bounds[1, i])) for i, name in enumerate(names)}


def metric_record(
    protocol: str,
    model: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    accuracy, balanced_accuracy, macro_f1 = _metric_values(y_true, y_pred)
    confidence = float(config["confidence_level"])
    wilson_low, wilson_high = wilson_interval(int(np.sum(y_true == y_pred)), y_true.size, confidence)
    bootstrap = descriptive_bootstrap_intervals(
        y_true,
        y_pred,
        repetitions=int(config["bootstrap_repetitions"]),
        confidence=confidence,
        seed=int(config["analysis_seed"]) + sum(ord(char) for char in protocol + model),
    )
    return {
        "protocol": protocol,
        "protocol_zh": PROTOCOL_LABELS[protocol],
        "model": model,
        "n_test_seeds": int(y_true.size),
        "n_correct": int(np.sum(y_true == y_pred)),
        "accuracy": accuracy,
        "accuracy_wilson_low": wilson_low,
        "accuracy_wilson_high": wilson_high,
        "accuracy_bootstrap_low": bootstrap["accuracy"][0],
        "accuracy_bootstrap_high": bootstrap["accuracy"][1],
        "balanced_accuracy": balanced_accuracy,
        "balanced_accuracy_bootstrap_low": bootstrap["balanced_accuracy"][0],
        "balanced_accuracy_bootstrap_high": bootstrap["balanced_accuracy"][1],
        "macro_f1": macro_f1,
        "macro_f1_bootstrap_low": bootstrap["macro_f1"][0],
        "macro_f1_bootstrap_high": bootstrap["macro_f1"][1],
        "interval_scope": "descriptive_seed_level_only",
    }


def _prediction_records(
    dataset: Dataset,
    indices: np.ndarray,
    predictions: np.ndarray,
    protocol: str,
    model: str,
    fold: str,
) -> list[dict[str, Any]]:
    selected = dataset.manifest.iloc[indices]
    records: list[dict[str, Any]] = []
    for row, prediction in zip(selected.itertuples(index=False), predictions, strict=True):
        records.append(
            {
                "protocol": protocol,
                "model": model,
                "fold": fold,
                "sample_index": int(row.sample_index),
                "label": int(row.label),
                "predicted_label": int(prediction),
                "correct": int(row.label == prediction),
                "source_cube": row.source_cube,
                "seed_id": row.seed_id,
                "relative_csv_path": row.relative_csv_path,
            }
        )
    return records


def _fit_predict(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConvergenceWarning)
        estimator.fit(X[train_indices], y[train_indices])
    return np.asarray(estimator.predict(X[test_indices]), dtype=int)


def run_protocols(dataset: Dataset, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run random holdout, reciprocal cube-domain tests, and leave-one-cube-out."""

    estimators = build_estimators(config)
    all_indices = np.arange(dataset.y.size)
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []

    random_train, random_test = train_test_split(
        all_indices,
        test_size=float(config["holdout_fraction"]),
        random_state=int(config["analysis_seed"]),
        stratify=dataset.y,
    )
    protocols: list[tuple[str, np.ndarray, np.ndarray]] = [
        # Preserve train_test_split's returned order.  The supplied legacy RF uses
        # bootstrap positions with a fixed RNG; sorting the rows is mathematically
        # irrelevant to LR/SVM but changes which physical samples those positions
        # identify and therefore prevents exact RF reproduction.
        ("random_seed_holdout", random_train, random_test),
        (
            "suffix_1_to_2",
            all_indices[dataset.manifest["cube_replicate"].to_numpy() == 1],
            all_indices[dataset.manifest["cube_replicate"].to_numpy() == 2],
        ),
        (
            "suffix_2_to_1",
            all_indices[dataset.manifest["cube_replicate"].to_numpy() == 2],
            all_indices[dataset.manifest["cube_replicate"].to_numpy() == 1],
        ),
    ]

    for protocol, train_indices, test_indices in protocols:
        train_cubes = set(dataset.manifest.iloc[train_indices]["source_cube"])
        test_cubes = set(dataset.manifest.iloc[test_indices]["source_cube"])
        if protocol != "random_seed_holdout" and train_cubes.intersection(test_cubes):
            raise AssertionError(f"Source cube leaked across grouped protocol {protocol}")
        for model_name, estimator in estimators.items():
            prediction = _fit_predict(estimator, dataset.X, dataset.y, train_indices, test_indices)
            metric_rows.append(
                metric_record(protocol, model_name, dataset.y[test_indices], prediction, config)
            )
            prediction_rows.extend(
                _prediction_records(dataset, test_indices, prediction, protocol, model_name, protocol)
            )

    cube_names = dataset.manifest["source_cube"].drop_duplicates().tolist()
    for model_name in estimators:
        pooled_indices: list[int] = []
        pooled_predictions: list[int] = []
        for cube_name in cube_names:
            test_indices = all_indices[dataset.manifest["source_cube"].to_numpy() == cube_name]
            train_indices = all_indices[dataset.manifest["source_cube"].to_numpy() != cube_name]
            estimator = build_estimators(config)[model_name]
            prediction = _fit_predict(estimator, dataset.X, dataset.y, train_indices, test_indices)
            pooled_indices.extend(test_indices.tolist())
            pooled_predictions.extend(prediction.tolist())
            fold_rows.append(
                {
                    "protocol": "leave_one_cube_out",
                    "model": model_name,
                    "held_out_cube": cube_name,
                    "held_out_label": int(dataset.y[test_indices][0]),
                    "n_test_seeds": int(test_indices.size),
                    "n_correct": int(np.sum(dataset.y[test_indices] == prediction)),
                    "accuracy": float(accuracy_score(dataset.y[test_indices], prediction)),
                }
            )
            prediction_rows.extend(
                _prediction_records(
                    dataset,
                    test_indices,
                    prediction,
                    "leave_one_cube_out",
                    model_name,
                    cube_name,
                )
            )
        order = np.argsort(np.asarray(pooled_indices))
        pooled_index_array = np.asarray(pooled_indices, dtype=int)[order]
        pooled_prediction_array = np.asarray(pooled_predictions, dtype=int)[order]
        metric_rows.append(
            metric_record(
                "leave_one_cube_out",
                model_name,
                dataset.y[pooled_index_array],
                pooled_prediction_array,
                config,
            )
        )

    return (
        pd.DataFrame.from_records(metric_rows),
        pd.DataFrame.from_records(prediction_rows),
        pd.DataFrame.from_records(fold_rows),
    )


def confusion_records(predictions: pd.DataFrame, n_classes: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels = np.arange(n_classes)
    for (protocol, model), frame in predictions.groupby(["protocol", "model"], sort=False):
        matrix = confusion_matrix(frame["label"], frame["predicted_label"], labels=labels)
        for actual in labels:
            for predicted in labels:
                rows.append(
                    {
                        "protocol": protocol,
                        "model": model,
                        "actual_label": int(actual),
                        "predicted_label": int(predicted),
                        "count": int(matrix[actual, predicted]),
                    }
                )
    return pd.DataFrame.from_records(rows)


def _format_percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a small DataFrame without pandas' optional ``tabulate`` package."""

    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    headers = [cell(column) for column in frame.columns]
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    rows.extend(
        "| " + " | ".join(cell(value) for value in record) + " |"
        for record in frame.itertuples(index=False, name=None)
    )
    return "\n".join(rows)


def build_markdown_report(
    dataset: Dataset,
    metrics: pd.DataFrame,
    folds: pd.DataFrame,
    config: dict[str, Any],
) -> str:
    """Generate a compact Chinese report directly from machine-readable results."""

    lines = [
        "# 现有数据条件下的诚实基线分析",
        "",
        "> 本报告由 `current_data_study/analyze.py` 自动生成。它只描述当前数据，不能把种子级分类性能解释为新产地、",
        "> 新批次或新仪器上的地理溯源能力。",
        "",
        "## 数据与分析单位",
        "",
        f"- 共读取 **{dataset.y.size}** 粒种子的 {dataset.X.shape[1]} 波段平均光谱。",
        f"- 共 **{dataset.manifest['source_cube'].nunique()}** 个采集立方体、{dataset.manifest['label'].nunique()} 个类别；每个类别只有两个立方体。",
        "- `source_cube` 是当前可恢复的最高层级；没有可核验的农场、独立批次、年份、供应商等元数据。",
        "- 数据指纹（SHA-256）：`" + dataset.fingerprint_sha256 + "`。",
        "",
        "## 固定分析方案",
        "",
        f"随机留出使用固定种子 {config['analysis_seed']}，按类别分层 75/25 划分。LR、RBF-SVM、PLS-DA、RF 参数与旧脚本声明一致；",
        "另以相同 LR 比较 SNV、MSC、Savitzky–Golay 平滑和一阶导数。所有训练型变换（MSC、StandardScaler）只在训练集拟合。",
        f"LR 统一采用严格数值收敛阈值 tol={config['models']['logistic_regression']['tol']:.0e}；全程不做超参数搜索，也不依据测试结果改参数。",
        "",
        "分组证据包括两向后缀验证（所有 `*-1` 训练、`*-2` 测试，随后反向）以及 16 折留一立方体汇总。",
        "这些方案消除了同一 `source_cube` 同时进入训练和测试的问题，但配对立方体仍可能来自同一商业批次，故仍非外部产地验证。",
        "",
        "## 主要结果",
        "",
    ]
    display = metrics.copy()
    display["准确率"] = display["accuracy"].map(_format_percent)
    display["平衡准确率"] = display["balanced_accuracy"].map(_format_percent)
    display["宏平均 F1"] = display["macro_f1"].map(_format_percent)
    display["准确率 Wilson 95% 区间"] = display.apply(
        lambda row: f"{_format_percent(row['accuracy_wilson_low'])}–{_format_percent(row['accuracy_wilson_high'])}",
        axis=1,
    )
    for protocol in PROTOCOL_LABELS:
        subset = display[display["protocol"] == protocol].sort_values(
            "accuracy", ascending=False
        )[
            ["model", "n_test_seeds", "准确率", "平衡准确率", "宏平均 F1", "准确率 Wilson 95% 区间"]
        ]
        lines.extend([f"### {PROTOCOL_LABELS[protocol]}", "", _markdown_table(subset), ""])

    random_metrics = metrics[metrics["protocol"] == "random_seed_holdout"].set_index("model")
    grouped = metrics[metrics["protocol"].isin(["suffix_1_to_2", "suffix_2_to_1"])]
    grouped_mean = grouped.groupby("model")["accuracy"].mean()
    best_random = random_metrics["accuracy"].idxmax()
    best_grouped = grouped_mean.idxmax()
    loco = metrics[metrics["protocol"] == "leave_one_cube_out"].set_index("model")
    lines.extend(
        [
            "## 审慎解读",
            "",
            f"- 随机种子级留出中最佳固定基线为 `{best_random}`（{_format_percent(random_metrics.loc[best_random, 'accuracy'])}）。",
            f"- 两向立方体后缀验证的平均准确率最高者为 `{best_grouped}`（{_format_percent(grouped_mean.loc[best_grouped])}）。",
            f"- `{best_grouped}` 的 16 折留一立方体汇总准确率为 {_format_percent(loco.loc[best_grouped, 'accuracy'])}。",
            "- 随机留出会让同一采集立方体中的种子同时进入训练与测试，容易利用共同照明、校准、采集会话和商业批次特征；不能称为“独立预测集”。",
            "- Wilson 与普通种子级 bootstrap 区间均把种子暂时当作独立观测，违反当前层级结构；它们只用于描述有限测试预测的不确定性，不是批次级或产地级推断区间。",
            "- 留一立方体每折只留出一个类别中的一个立方体，而训练中仍保留该类别的配对立方体；该设计只能评估有限采集域转移，不能估计未知农场、年份或供应商泛化。",
            "- 因每个类别仅两个立方体，无法同时可靠地分离“产地效应”“商业批次效应”和“采集立方体效应”。",
            "",
            "## 输出索引",
            "",
            "- `dataset_manifest.csv`：每粒种子的标签、采集立方体、种子编号、路径和文件哈希。",
            "- `metrics.csv`：各协议与模型的汇总指标和描述性区间。",
            "- `predictions.csv`：逐种子、逐模型的真实标签与预测标签。",
            "- `fold_metrics.csv`：16 折留一立方体的逐折准确率。",
            "- `confusion_matrices.csv`：长表格式混淆矩阵。",
            "- `results.json`：配置、运行环境、数据摘要和全部汇总指标。",
            "",
            "逐立方体准确率可从 `fold_metrics.csv` 审核；机器可读结果应作为后续论文表格和图形的唯一数字来源。",
            "",
        ]
    )
    return "\n".join(lines)


def _json_safe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def write_outputs(
    output_dir: Path,
    dataset: Dataset,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    folds: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    confusion = confusion_records(predictions, len(config["class_names"]))
    manifest = dataset.manifest.copy()
    manifest["class_name"] = manifest["label"].map(dict(enumerate(config["class_names"])))
    manifest.to_csv(output_dir / "dataset_manifest.csv", index=False)
    pd.DataFrame(
        {"band_index": np.arange(dataset.wavelengths.size), "wavelength_nm": dataset.wavelengths}
    ).to_csv(output_dir / "wavelengths.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    folds.to_csv(output_dir / "fold_metrics.csv", index=False)
    confusion.to_csv(output_dir / "confusion_matrices.csv", index=False)

    metadata = {
        "analysis_scope": "current_data_descriptive_analysis",
        "data_fingerprint_sha256": dataset.fingerprint_sha256,
        "data_summary": {
            "n_seeds": int(dataset.y.size),
            "n_bands": int(dataset.X.shape[1]),
            "n_source_cubes": int(dataset.manifest["source_cube"].nunique()),
            "n_classes": int(dataset.manifest["label"].nunique()),
            "seeds_per_cube": {
                str(key): int(value)
                for key, value in dataset.manifest.groupby("source_cube").size().items()
            },
        },
        "config": config,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "platform": platform.platform(),
        },
        "inference_warning": (
            "Seeds within a source cube are not independent provenance replicates; "
            "all reported intervals are descriptive seed-level intervals only."
        ),
        "metrics": _json_safe_records(metrics),
    }
    (output_dir / "results.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "report.md").write_text(
        build_markdown_report(dataset, metrics, folds, config), encoding="utf-8"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=script_dir.parent / "data")
    parser.add_argument("--config", type=Path, default=script_dir / "config.json")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "outputs")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(f"[1/3] Loading CSV spectra from {args.data_root}", flush=True)
    dataset = load_dataset(args.data_root, int(config["expected_bands"]))
    print(
        f"      {dataset.y.size} seeds, {dataset.X.shape[1]} bands, "
        f"{dataset.manifest['source_cube'].nunique()} source cubes",
        flush=True,
    )
    print("[2/3] Running fixed random and grouped validation protocols", flush=True)
    metrics, predictions, folds = run_protocols(dataset, config)
    print(f"[3/3] Writing deterministic outputs to {args.output_dir}", flush=True)
    write_outputs(args.output_dir, dataset, metrics, predictions, folds, config)
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
