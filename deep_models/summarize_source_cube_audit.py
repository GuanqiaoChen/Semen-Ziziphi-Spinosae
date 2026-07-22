#!/usr/bin/env python3
"""Locked post-processing for the preregistered current-data study.

This script consumes one *complete* output directory produced by
``deep_models/source_cube_audit.py``.  All estimands, comparisons,
bootstrap settings, calibration bins, and figures are fixed below; command-line
arguments can select paths only.  No result-dependent model or condition
selection is performed.

The bootstrap resamples the eight commercial-label pairs as intact blocks.  A
single 10,000 x 8 matrix of sampled label IDs is shared by every model and
condition, making all contrasts paired.  Its intervals are conditional on the
16 archived source cubes and must not be interpreted as new-lot intervals.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - reported by the executable
    raise SystemExit("matplotlib is required for publication figures") from exc


NUM_CLASSES = 8
CLASS_NAMES = ("HBS", "HBX", "HNA", "HNX", "NX", "SXD", "SXQ", "XJH")
DIRECTIONS = ("suffix_1_to_2", "suffix_2_to_1")
MODELS = ("snv_lr", "spectral_only", "fusion_net")
CONDITIONS = ("full", "spatial_shuffle", "mean_broadcast", "mask_only")
SEEDS = (42, 2024, 2025)
CALIBRATIONS = ("raw", "temperature_scaled")
PRIMARY_CALIBRATION = "temperature_scaled"
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 20260721
CONFIDENCE_LEVEL = 0.95
ECE_BINS = 10
CHANCE_BALANCED_ACCURACY = 1.0 / NUM_CLASSES

REQUIRED_INPUT_FILES = (
    "run_status.json",
    "results.json",
    "manifest.csv",
    "predictions.csv",
    "metrics.csv",
    "ensemble_predictions.csv",
    "ensemble_metrics.csv",
    "primary_estimands.csv",
    "spatial_mechanism_decision.json",
)


@dataclass(frozen=True)
class EffectSpec:
    effect_id: str
    label: str
    left_model: str
    left_condition: str
    right_model: str | None
    right_condition: str | None
    right_constant: float | None = None
    role: str = "secondary"


# Frozen from docs/来源立方体预注册分析方案.md Sections 2.2, 5, and 6.2.
EFFECT_SPECS = (
    EffectSpec(
        "primary_fusion_full_minus_spatial_shuffle",
        "fusion_net(full) - fusion_net(spatial_shuffle)",
        "fusion_net",
        "full",
        "fusion_net",
        "spatial_shuffle",
        role="primary",
    ),
    EffectSpec(
        "fusion_spatial_shuffle_minus_mean_broadcast",
        "fusion_net(spatial_shuffle) - fusion_net(mean_broadcast)",
        "fusion_net",
        "spatial_shuffle",
        "fusion_net",
        "mean_broadcast",
    ),
    EffectSpec(
        "fusion_full_minus_mean_broadcast",
        "fusion_net(full) - fusion_net(mean_broadcast)",
        "fusion_net",
        "full",
        "fusion_net",
        "mean_broadcast",
    ),
    EffectSpec(
        "fusion_mean_broadcast_minus_mask_only",
        "fusion_net(mean_broadcast) - fusion_net(mask_only)",
        "fusion_net",
        "mean_broadcast",
        "fusion_net",
        "mask_only",
    ),
    EffectSpec(
        "fusion_full_minus_spectral_only_full",
        "fusion_net(full) - spectral_only(full)",
        "fusion_net",
        "full",
        "spectral_only",
        "full",
    ),
    EffectSpec(
        "fusion_full_minus_snv_lr_full",
        "fusion_net(full) - snv_lr(full)",
        "fusion_net",
        "full",
        "snv_lr",
        "full",
        role="descriptive_baseline",
    ),
    EffectSpec(
        "fusion_mask_only_minus_chance",
        "fusion_net(mask_only) - chance (0.125)",
        "fusion_net",
        "mask_only",
        None,
        None,
        right_constant=CHANCE_BALANCED_ACCURACY,
    ),
)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to create an empty result table: {path.name}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probability_vector(row: Mapping[str, Any], calibration: str) -> np.ndarray:
    prefix = "calibrated_probability_" if calibration == "temperature_scaled" else "raw_probability_"
    vector = np.asarray([float(row[f"{prefix}{label}"]) for label in range(NUM_CLASSES)])
    if not np.all(np.isfinite(vector)):
        raise ValueError("Non-finite probability found")
    if np.any(vector < -1e-12) or np.any(vector > 1.0 + 1e-12):
        raise ValueError("Probability outside [0, 1]")
    if not np.isclose(vector.sum(), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError(f"Probabilities do not sum to one: {vector.sum():.12g}")
    return vector


def validate_complete_output(input_dir: Path) -> dict[str, Any]:
    """Reject partial or subset runs before any statistical computation."""

    missing = [name for name in REQUIRED_INPUT_FILES if not (input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete preregistered source-cube output; missing files: {missing}")
    run_status = read_json(input_dir / "run_status.json")
    results = read_json(input_dir / "results.json")
    if run_status.get("status") != "executed_complete":
        raise ValueError(f"run_status is not executed_complete: {run_status.get('status')!r}")
    if results.get("status") != "executed_complete":
        raise ValueError(f"results status is not executed_complete: {results.get('status')!r}")
    protocol = results.get("protocol", {})
    if tuple(protocol.get("directions", ())) != DIRECTIONS:
        raise ValueError("The completed run does not contain the two locked directions in order")
    if set(protocol.get("models", ())) != set(MODELS):
        raise ValueError(f"The completed run must contain exactly {MODELS}")
    if tuple(int(value) for value in protocol.get("seeds", ())) != SEEDS:
        raise ValueError(f"The completed run must contain locked seeds {SEEDS}")
    if set(protocol.get("counterfactuals", ())) != set(CONDITIONS):
        raise ValueError(f"The completed run must contain all counterfactuals {CONDITIONS}")

    predictions = read_csv_rows(input_dir / "predictions.csv")
    ensembles = read_csv_rows(input_dir / "ensemble_predictions.csv")
    metrics = read_csv_rows(input_dir / "metrics.csv")
    ensemble_metrics = read_csv_rows(input_dir / "ensemble_metrics.csv")
    expected_run_cells = {
        (direction, model, seed, condition)
        for direction in DIRECTIONS
        for model in MODELS
        for seed in SEEDS
        for condition in CONDITIONS
    }
    observed_run_cells = {
        (row["direction"], row["model"], int(row["seed"]), row["condition"])
        for row in predictions
    }
    if observed_run_cells != expected_run_cells:
        raise ValueError(
            "predictions.csv does not contain the complete locked matrix; "
            f"missing={sorted(expected_run_cells-observed_run_cells)}, "
            f"extra={sorted(observed_run_cells-expected_run_cells)}"
        )
    expected_ensemble_cells = {
        (direction, model, condition)
        for direction in DIRECTIONS
        for model in MODELS
        for condition in CONDITIONS
    }
    observed_ensemble_cells = {
        (row["direction"], row["model"], row["condition"]) for row in ensembles
    }
    if observed_ensemble_cells != expected_ensemble_cells:
        raise ValueError("ensemble_predictions.csv does not contain the complete locked matrix")

    for collection_name, rows, cell_keys in (
        ("predictions", predictions, ("direction", "model", "seed", "condition")),
        ("ensemble_predictions", ensembles, ("direction", "model", "condition")),
    ):
        grouped: dict[tuple[Any, ...], list[dict[str, str]]] = {}
        for row in rows:
            key = tuple(row[name] for name in cell_keys)
            grouped.setdefault(key, []).append(row)
            label = int(row["true_label"])
            if label not in range(NUM_CLASSES):
                raise ValueError(f"Invalid label in {collection_name}: {label}")
            if int(row["source_cube"].split("-")[0]) != label:
                raise ValueError(f"Label/source_cube mismatch in {collection_name}")
            for calibration in CALIBRATIONS:
                _probability_vector(row, calibration)
        reference_ids: dict[str, set[str]] = {}
        for key, cell_rows in grouped.items():
            direction = str(key[0])
            ids = {row["sample_id"] for row in cell_rows}
            if len(ids) != len(cell_rows):
                raise ValueError(f"Duplicate sample in {collection_name} cell {key}")
            labels = {int(row["true_label"]) for row in cell_rows}
            if labels != set(range(NUM_CLASSES)):
                raise ValueError(f"Cell {key} does not cover all eight labels")
            if direction in reference_ids and ids != reference_ids[direction]:
                raise ValueError(f"Models/conditions do not share samples in {direction}")
            reference_ids[direction] = ids

    expected_metric_cells = {
        (direction, model, seed, condition, calibration)
        for direction in DIRECTIONS
        for model in MODELS
        for seed in SEEDS
        for condition in CONDITIONS
        for calibration in CALIBRATIONS
    }
    observed_metric_cells = {
        (row["direction"], row["model"], int(row["seed"]), row["condition"], row["calibration"])
        for row in metrics
    }
    if observed_metric_cells != expected_metric_cells:
        raise ValueError("metrics.csv does not contain every seed/calibration cell")
    expected_ensemble_metric_cells = {
        (direction, model, condition, calibration)
        for direction in DIRECTIONS
        for model in MODELS
        for condition in CONDITIONS
        for calibration in CALIBRATIONS
    }
    observed_ensemble_metric_cells = {
        (row["direction"], row["model"], row["condition"], row["calibration"])
        for row in ensemble_metrics
    }
    if observed_ensemble_metric_cells != expected_ensemble_metric_cells:
        raise ValueError("ensemble_metrics.csv does not contain every locked cell")
    return {
        "run_status": run_status,
        "results": results,
        "predictions": predictions,
        "ensemble_predictions": ensembles,
        "metrics": metrics,
        "ensemble_metrics": ensemble_metrics,
    }


def balanced_accuracy_by_label(
    rows: Sequence[Mapping[str, Any]], calibration: str
) -> dict[tuple[str, str, str, int], float]:
    """Per-direction label recall; one label equals one test source cube."""

    grouped: dict[tuple[str, str, str, int], list[int]] = {}
    for row in rows:
        key = (str(row["direction"]), str(row["model"]), str(row["condition"]), int(row["true_label"]))
        predicted = int(np.argmax(_probability_vector(row, calibration)))
        grouped.setdefault(key, []).append(int(predicted == int(row["true_label"])))
    expected = {
        (direction, model, condition, label)
        for direction in DIRECTIONS
        for model in MODELS
        for condition in CONDITIONS
        for label in range(NUM_CLASSES)
    }
    if set(grouped) != expected:
        raise ValueError("Per-label recall input is not the complete ensemble matrix")
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def label_pair_values(
    recalls: Mapping[tuple[str, str, str, int], float],
    model: str,
    condition: str,
) -> np.ndarray:
    return np.asarray(
        [
            np.mean([recalls[(direction, model, condition, label)] for direction in DIRECTIONS])
            for label in range(NUM_CLASSES)
        ],
        dtype=float,
    )


def generate_block_draws(
    repetitions: int = BOOTSTRAP_REPETITIONS, seed: int = BOOTSTRAP_SEED
) -> np.ndarray:
    if repetitions <= 0:
        raise ValueError("Bootstrap repetitions must be positive")
    return np.random.default_rng(seed).integers(0, NUM_CLASSES, size=(repetitions, NUM_CLASSES))


def percentile_interval(values: np.ndarray, confidence: float = CONFIDENCE_LEVEL) -> tuple[float, float]:
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(np.asarray(values, dtype=float), (alpha, 1.0 - alpha))
    return float(low), float(high)


def bootstrap_theta_intervals(
    ensemble_rows: Sequence[Mapping[str, Any]], draws: np.ndarray
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for calibration in CALIBRATIONS:
        recalls = balanced_accuracy_by_label(ensemble_rows, calibration)
        for model in MODELS:
            for condition in CONDITIONS:
                blocks = label_pair_values(recalls, model, condition)
                bootstrap_values = blocks[draws].mean(axis=1)
                low, high = percentile_interval(bootstrap_values)
                records.append(
                    {
                        "estimand": "theta_two_direction_cube_equal_balanced_accuracy",
                        "model": model,
                        "condition": condition,
                        "calibration": calibration,
                        "observed_theta": float(blocks.mean()),
                        "bootstrap_mean": float(bootstrap_values.mean()),
                        "bootstrap_median": float(np.median(bootstrap_values)),
                        "conditional_ci_low": low,
                        "conditional_ci_high": high,
                        "confidence_level": CONFIDENCE_LEVEL,
                        "interval_method": "paired_label_block_bootstrap_percentile",
                        "bootstrap_repetitions": int(draws.shape[0]),
                        "bootstrap_seed": BOOTSTRAP_SEED,
                        "resampling_unit": "8 commercial-label pairs; both source-cube directions retained",
                        "scope": "conditional_on_the_16_archived_source_cubes",
                    }
                )
    return records


def _effect_blocks(
    recalls: Mapping[tuple[str, str, str, int], float], spec: EffectSpec
) -> np.ndarray:
    left = label_pair_values(recalls, spec.left_model, spec.left_condition)
    if spec.right_constant is not None:
        right = np.full(NUM_CLASSES, spec.right_constant, dtype=float)
    else:
        assert spec.right_model is not None and spec.right_condition is not None
        right = label_pair_values(recalls, spec.right_model, spec.right_condition)
    return left - right


def bootstrap_effect_intervals(
    ensemble_rows: Sequence[Mapping[str, Any]], draws: np.ndarray
) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    records: list[dict[str, Any]] = []
    cached: dict[str, dict[str, np.ndarray]] = {}
    for calibration in CALIBRATIONS:
        recalls = balanced_accuracy_by_label(ensemble_rows, calibration)
        cached[calibration] = {}
        for spec in EFFECT_SPECS:
            blocks = _effect_blocks(recalls, spec)
            bootstrap_values = blocks[draws].mean(axis=1)
            cached[calibration][spec.effect_id] = bootstrap_values
            low, high = percentile_interval(bootstrap_values)
            direction_values = []
            for direction in DIRECTIONS:
                left = np.mean(
                    [recalls[(direction, spec.left_model, spec.left_condition, label)] for label in range(NUM_CLASSES)]
                )
                if spec.right_constant is not None:
                    right = spec.right_constant
                else:
                    assert spec.right_model is not None and spec.right_condition is not None
                    right = np.mean(
                        [recalls[(direction, spec.right_model, spec.right_condition, label)] for label in range(NUM_CLASSES)]
                    )
                direction_values.append(float(left - right))
            records.append(
                {
                    "effect_id": spec.effect_id,
                    "effect_label": spec.label,
                    "role": spec.role,
                    "calibration": calibration,
                    "suffix_1_to_2_effect": direction_values[0],
                    "suffix_2_to_1_effect": direction_values[1],
                    "observed_two_direction_effect": float(blocks.mean()),
                    "bootstrap_mean": float(bootstrap_values.mean()),
                    "conditional_ci_low": low,
                    "conditional_ci_high": high,
                    "confidence_level": CONFIDENCE_LEVEL,
                    "interval_method": "paired_label_block_bootstrap_percentile",
                    "bootstrap_repetitions": int(draws.shape[0]),
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "resampling_unit": "8 paired commercial labels",
                    "scope": "conditional_on_the_16_archived_source_cubes",
                }
            )
    return records, cached


def exact_sign_flip_test(label_pair_effects: Sequence[float]) -> dict[str, Any]:
    effects = np.asarray(label_pair_effects, dtype=float)
    if effects.shape != (NUM_CLASSES,) or not np.all(np.isfinite(effects)):
        raise ValueError("The exact test requires eight finite label-pair effects")
    sign_matrix = np.asarray(list(itertools.product((-1.0, 1.0), repeat=NUM_CLASSES)))
    null = (sign_matrix * effects[None, :]).mean(axis=1)
    observed = float(effects.mean())
    tolerance = 1e-15
    p_greater = float(np.mean(null >= observed - tolerance))
    p_two_sided = float(np.mean(np.abs(null) >= abs(observed) - tolerance))
    return {
        "observed_effect": observed,
        "n_label_pairs": NUM_CLASSES,
        "n_exact_sign_assignments": int(null.size),
        "alternative_preregistered": "greater: fusion_net(full) > fusion_net(spatial_shuffle)",
        "p_value_greater": p_greater,
        "p_value_two_sided_sensitivity": p_two_sided,
        "test_scope": "exact sign-flip across eight observed label-pair effects; conditional, not new-lot inference",
        "null_distribution": null,
    }


def main_results_table(
    metric_rows: Sequence[Mapping[str, Any]], ensemble_metric_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    seed_lookup = {
        (row["direction"], row["model"], row["condition"], row["calibration"], int(row["seed"])): row
        for row in metric_rows
    }
    ensemble_lookup = {
        (row["direction"], row["model"], row["condition"], row["calibration"]): row
        for row in ensemble_metric_rows
    }
    output: list[dict[str, Any]] = []
    for calibration in CALIBRATIONS:
        for direction in DIRECTIONS:
            for model in MODELS:
                for condition in CONDITIONS:
                    seed_values = [
                        float(seed_lookup[(direction, model, condition, calibration, seed)]["balanced_accuracy"])
                        for seed in SEEDS
                    ]
                    ensemble = ensemble_lookup[(direction, model, condition, calibration)]
                    output.append(
                        {
                            "direction": direction,
                            "model": model,
                            "condition": condition,
                            "calibration": calibration,
                            "seed_42_balanced_accuracy": seed_values[0],
                            "seed_2024_balanced_accuracy": seed_values[1],
                            "seed_2025_balanced_accuracy": seed_values[2],
                            "seed_mean_balanced_accuracy": float(np.mean(seed_values)),
                            "seed_sd_balanced_accuracy": float(np.std(seed_values, ddof=1)),
                            "ensemble_accuracy": float(ensemble["accuracy"]),
                            "ensemble_balanced_accuracy": float(ensemble["balanced_accuracy"]),
                            "ensemble_macro_f1": float(ensemble["macro_f1"]),
                            "ensemble_nll": float(ensemble["nll"]),
                            "ensemble_brier": float(ensemble["brier"]),
                            "ensemble_ece_10": float(ensemble["ece_10"]),
                        }
                    )
    return output


def counterfactual_effect_table(
    metric_rows: Sequence[Mapping[str, Any]], ensemble_metric_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    lookup: dict[tuple[str, str, str, str, str], float] = {}
    for row in metric_rows:
        predictor = f"seed_{int(row['seed'])}"
        lookup[(predictor, row["direction"], row["model"], row["condition"], row["calibration"])] = float(
            row["balanced_accuracy"]
        )
    for row in ensemble_metric_rows:
        lookup[("probability_ensemble", row["direction"], row["model"], row["condition"], row["calibration"])] = float(
            row["balanced_accuracy"]
        )
    output: list[dict[str, Any]] = []
    predictors = tuple(f"seed_{seed}" for seed in SEEDS) + ("probability_ensemble",)
    for calibration in CALIBRATIONS:
        for spec in EFFECT_SPECS:
            for predictor in predictors:
                directional: list[float] = []
                for direction in DIRECTIONS:
                    left = lookup[(predictor, direction, spec.left_model, spec.left_condition, calibration)]
                    if spec.right_constant is not None:
                        right = spec.right_constant
                    else:
                        assert spec.right_model is not None and spec.right_condition is not None
                        right = lookup[(predictor, direction, spec.right_model, spec.right_condition, calibration)]
                    delta = left - right
                    directional.append(delta)
                    output.append(
                        {
                            "effect_id": spec.effect_id,
                            "effect_label": spec.label,
                            "role": spec.role,
                            "calibration": calibration,
                            "predictor": predictor,
                            "scope": direction,
                            "left_balanced_accuracy": left,
                            "right_balanced_accuracy": right,
                            "effect": delta,
                        }
                    )
                output.append(
                    {
                        "effect_id": spec.effect_id,
                        "effect_label": spec.label,
                        "role": spec.role,
                        "calibration": calibration,
                        "predictor": predictor,
                        "scope": "theta_two_direction_equal_weight",
                        "left_balanced_accuracy": "",
                        "right_balanced_accuracy": "",
                        "effect": float(np.mean(directional)),
                    }
                )
    return output


def reliability_rows_for_group(
    rows: Sequence[Mapping[str, Any]], calibration: str
) -> list[dict[str, Any]]:
    probabilities = np.asarray([_probability_vector(row, calibration) for row in rows])
    labels = np.asarray([int(row["true_label"]) for row in rows])
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    result: list[dict[str, Any]] = []
    for bin_index in range(ECE_BINS):
        lower = bin_index / ECE_BINS
        upper = (bin_index + 1) / ECE_BINS
        members = (confidence > lower) & (confidence <= upper)
        count = int(members.sum())
        result.append(
            {
                "bin_index": bin_index + 1,
                "lower_exclusive": lower,
                "upper_inclusive": upper,
                "n": count,
                "fraction": count / len(rows),
                "mean_confidence": float(confidence[members].mean()) if count else "",
                "accuracy": float(correct[members].mean()) if count else "",
                "absolute_gap": (
                    abs(float(correct[members].mean()) - float(confidence[members].mean()))
                    if count
                    else ""
                ),
            }
        )
    return result


def reliability_table(
    prediction_rows: Sequence[Mapping[str, Any]], ensemble_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    sources = (("seed", prediction_rows), ("probability_ensemble", ensemble_rows))
    for predictor_type, source_rows in sources:
        grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
        for row in source_rows:
            predictor = (
                f"seed_{int(row['seed'])}" if predictor_type == "seed" else "probability_ensemble"
            )
            key = (predictor, str(row["direction"]), str(row["model"]), str(row["condition"]))
            grouped.setdefault(key, []).append(row)
        for key, rows in sorted(grouped.items()):
            predictor, direction, model, condition = key
            for calibration in CALIBRATIONS:
                for bin_row in reliability_rows_for_group(rows, calibration):
                    output.append(
                        {
                            "predictor": predictor,
                            "direction": direction,
                            "model": model,
                            "condition": condition,
                            "calibration": calibration,
                            **bin_row,
                        }
                    )
    return output


def confusion_table(ensemble_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for direction in DIRECTIONS:
        for model in MODELS:
            selected = [
                row
                for row in ensemble_rows
                if row["direction"] == direction
                and row["model"] == model
                and row["condition"] == "full"
            ]
            matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
            for row in selected:
                actual = int(row["true_label"])
                predicted = int(np.argmax(_probability_vector(row, PRIMARY_CALIBRATION)))
                matrix[actual, predicted] += 1
            row_totals = matrix.sum(axis=1, keepdims=True)
            normalized = np.divide(
                matrix,
                row_totals,
                out=np.zeros_like(matrix, dtype=float),
                where=row_totals > 0,
            )
            for actual in range(NUM_CLASSES):
                for predicted in range(NUM_CLASSES):
                    output.append(
                        {
                            "direction": direction,
                            "model": model,
                            "condition": "full",
                            "calibration": PRIMARY_CALIBRATION,
                            "true_label": actual,
                            "true_class_name": CLASS_NAMES[actual],
                            "predicted_label": predicted,
                            "predicted_class_name": CLASS_NAMES[predicted],
                            "count": int(matrix[actual, predicted]),
                            "row_fraction": float(normalized[actual, predicted]),
                        }
                    )
    return output


def _set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_figure(fig: Any, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _direction_label(direction: str) -> str:
    """Render a locked suffix-transfer identifier without leaking code tokens."""

    if direction not in DIRECTIONS:
        raise ValueError(f"Unknown locked direction: {direction}")
    return direction.removeprefix("suffix_").replace("_to_", " → ")


def plot_main_performance(main_rows: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    selected = [
        row
        for row in main_rows
        if row["condition"] == "full" and row["calibration"] == PRIMARY_CALIBRATION
    ]
    lookup = {(row["direction"], row["model"]): row for row in selected}
    labels = ("SNV–LR", "Spectral net", "Fusion net")
    seed_colors = ("#0072B2", "#E69F00", "#009E73")
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), sharey=True)
    for axis, direction in zip(axes, DIRECTIONS, strict=True):
        x = np.arange(len(MODELS))
        for model_index, model in enumerate(MODELS):
            row = lookup[(direction, model)]
            values = [
                float(row["seed_42_balanced_accuracy"]),
                float(row["seed_2024_balanced_accuracy"]),
                float(row["seed_2025_balanced_accuracy"]),
            ]
            for seed_index, value in enumerate(values):
                axis.scatter(
                    model_index + (seed_index - 1) * 0.07,
                    value,
                    s=25,
                    color=seed_colors[seed_index],
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=3,
                    label=f"seed {SEEDS[seed_index]}" if model_index == 0 else None,
                )
            axis.scatter(
                model_index,
                float(row["ensemble_balanced_accuracy"]),
                marker="D",
                s=42,
                color="black",
                zorder=4,
                label="probability ensemble" if model_index == 0 else None,
            )
        axis.axhline(CHANCE_BALANCED_ACCURACY, color="0.6", linestyle="--", linewidth=0.8)
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.set_title(_direction_label(direction))
        axis.set_ylim(0.0, 1.02)
        axis.grid(axis="y", color="0.9", linewidth=0.7)
    axes[0].set_ylabel("Balanced accuracy")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Locked full-input performance across optimization seeds", y=1.13, fontsize=11)
    fig.tight_layout()
    _save_figure(fig, output_dir, "figure_main_performance")


def plot_counterfactuals(main_rows: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    lookup = {
        (row["direction"], row["model"], row["condition"], row["calibration"]): row
        for row in main_rows
    }
    condition_labels = ("Full", "Spatial\nshuffle", "Mean\nbroadcast", "Mask\nonly")
    colors = ("#0072B2", "#D55E00")
    markers = ("o", "s")
    fig, axis = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(CONDITIONS))
    for direction_index, direction in enumerate(DIRECTIONS):
        ensemble_values = []
        for condition_index, condition in enumerate(CONDITIONS):
            row = lookup[(direction, "fusion_net", condition, PRIMARY_CALIBRATION)]
            seed_values = [
                float(row["seed_42_balanced_accuracy"]),
                float(row["seed_2024_balanced_accuracy"]),
                float(row["seed_2025_balanced_accuracy"]),
            ]
            ensemble_values.append(float(row["ensemble_balanced_accuracy"]))
            jitter = np.asarray((-0.045, 0.0, 0.045)) + (direction_index - 0.5) * 0.16
            axis.scatter(
                condition_index + jitter,
                seed_values,
                s=18,
                color=colors[direction_index],
                alpha=0.45,
                edgecolor="none",
            )
        axis.plot(
            x + (direction_index - 0.5) * 0.16,
            ensemble_values,
            color=colors[direction_index],
            marker=markers[direction_index],
            linewidth=1.5,
            markersize=5,
            label=_direction_label(direction),
        )
    axis.axhline(CHANCE_BALANCED_ACCURACY, color="0.6", linestyle="--", linewidth=0.8, label="chance")
    axis.set_xticks(x, condition_labels)
    axis.set_ylabel("Balanced accuracy")
    axis.set_ylim(0.0, 1.02)
    axis.set_title("Fusion-net counterfactual ladder (points: seeds; lines: ensembles)")
    axis.grid(axis="y", color="0.9", linewidth=0.7)
    axis.legend(frameon=False, ncol=3, loc="lower left")
    fig.tight_layout()
    _save_figure(fig, output_dir, "figure_counterfactual_effects")


def plot_calibration(reliability_rows: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    colors = {"raw": "#D55E00", "temperature_scaled": "#0072B2"}
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.8), sharex=True, sharey=True)
    for row_index, direction in enumerate(DIRECTIONS):
        for column_index, model in enumerate(MODELS):
            axis = axes[row_index, column_index]
            for calibration in CALIBRATIONS:
                selected = [
                    row
                    for row in reliability_rows
                    if row["predictor"] == "probability_ensemble"
                    and row["direction"] == direction
                    and row["model"] == model
                    and row["condition"] == "full"
                    and row["calibration"] == calibration
                    and row["n"] != 0
                ]
                axis.plot(
                    [float(row["mean_confidence"]) for row in selected],
                    [float(row["accuracy"]) for row in selected],
                    marker="o",
                    markersize=3,
                    linewidth=1.1,
                    color=colors[calibration],
                    label="temperature scaled" if calibration == "temperature_scaled" else "raw",
                )
            axis.plot((0, 1), (0, 1), color="0.55", linestyle="--", linewidth=0.8)
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            axis.grid(color="0.92", linewidth=0.6)
            if row_index == 0:
                axis.set_title(("SNV–LR", "Spectral net", "Fusion net")[column_index])
            if column_index == 0:
                axis.set_ylabel(f"{_direction_label(direction)}\nObserved accuracy")
            if row_index == 1:
                axis.set_xlabel("Mean confidence")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Full-input probability-ensemble reliability", y=1.04, fontsize=11)
    fig.tight_layout()
    _save_figure(fig, output_dir, "figure_calibration_reliability")


def plot_confusions(confusion_rows: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(9.4, 6.2), sharex=True, sharey=True)
    for row_index, direction in enumerate(DIRECTIONS):
        for column_index, model in enumerate(MODELS):
            selected = [
                row for row in confusion_rows if row["direction"] == direction and row["model"] == model
            ]
            matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=float)
            for row in selected:
                matrix[int(row["true_label"]), int(row["predicted_label"])] = float(row["row_fraction"])
            axis = axes[row_index, column_index]
            image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
            for actual in range(NUM_CLASSES):
                for predicted in range(NUM_CLASSES):
                    value = matrix[actual, predicted]
                    if value >= 0.05:
                        axis.text(
                            predicted,
                            actual,
                            f"{value:.2f}",
                            ha="center",
                            va="center",
                            fontsize=5.5,
                            color="white" if value > 0.55 else "black",
                        )
            if row_index == 0:
                axis.set_title(("SNV–LR", "Spectral net", "Fusion net")[column_index])
            axis.set_xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=45, ha="right")
            axis.set_yticks(range(NUM_CLASSES), CLASS_NAMES)
            if column_index == 0:
                axis.set_ylabel(f"{_direction_label(direction)}\nTrue class")
            if row_index == 1:
                axis.set_xlabel("Predicted class")
    fig.subplots_adjust(
        left=0.08,
        right=0.88,
        bottom=0.11,
        top=0.93,
        wspace=0.12,
        hspace=0.18,
    )
    colorbar_axis = fig.add_axes((0.91, 0.19, 0.015, 0.62))
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Row-normalized fraction")
    fig.suptitle("Full-input calibrated probability-ensemble confusion matrices", y=0.99, fontsize=11)
    _save_figure(fig, output_dir, "figure_ensemble_confusion_matrices")


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    rendered = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    rendered.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(rendered)


def build_summary(
    theta_rows: Sequence[Mapping[str, Any]],
    effect_rows: Sequence[Mapping[str, Any]],
    sign_flip: Mapping[str, Any],
    mechanism: Mapping[str, Any],
) -> str:
    primary_theta = [
        row
        for row in theta_rows
        if row["calibration"] == PRIMARY_CALIBRATION and row["condition"] == "full"
    ]
    primary_effect = next(
        row
        for row in effect_rows
        if row["effect_id"] == "primary_fusion_full_minus_spatial_shuffle"
        and row["calibration"] == PRIMARY_CALIBRATION
    )
    theta_table = _markdown_table(
        ("模型 / Model", "θ", "条件性95%区间 / Conditional 95% interval"),
        [
            (
                row["model"],
                _percent(float(row["observed_theta"])),
                f"{_percent(float(row['conditional_ci_low']))}–{_percent(float(row['conditional_ci_high']))}",
            )
            for row in primary_theta
        ],
    )
    effect_table = _markdown_table(
        ("效应 / Effect", "双向效应 / Effect", "条件性95%区间 / Conditional 95% interval"),
        [
            (
                row["effect_label"],
                _percent(float(row["observed_two_direction_effect"])),
                f"{_percent(float(row['conditional_ci_low']))}–{_percent(float(row['conditional_ci_high']))}",
            )
            for row in effect_rows
            if row["calibration"] == PRIMARY_CALIBRATION
        ],
    )
    support = bool(mechanism.get("limited_support_for_spatial_arrangement", False))
    chinese_decision = "达到预声明的有限空间排列支持门槛" if support else "未达到预声明的有限空间排列支持门槛"
    english_decision = "met the preregistered limited-support gate" if support else "did not meet the preregistered limited-support gate"
    return f"""# 当前数据来源立方体审计后处理摘要 / Source-cube-isolated current-data post-processing summary

## 中文审计摘要

本后处理只读取已标记为 `executed_complete` 的完整固定矩阵。主要预测器是三个预声明优化种子的温度校准概率平均集成；主要估计量是两个来源立方体迁移方向等权的 balanced accuracy（θ）。全部模型与反事实条件使用同一个由固定随机种子 `{BOOTSTRAP_SEED}` 生成的 {BOOTSTRAP_REPETITIONS:,} 次八标签对block重采样矩阵；区间为固定95%百分位区间。可靠性数据严格使用预注册的10个等宽置信度箱。

{theta_table}

{effect_table}

主要 `fusion_net(full) − fusion_net(spatial_shuffle)` 效应为 {_percent(float(primary_effect['observed_two_direction_effect']))}，条件性95%区间为 {_percent(float(primary_effect['conditional_ci_low']))}–{_percent(float(primary_effect['conditional_ci_high']))}。八标签对精确单侧sign-flip检验 `p={float(sign_flip['p_value_greater']):.6g}`，双侧敏感性值 `p={float(sign_flip['p_value_two_sided_sensitivity']):.6g}`。依据训练入口已冻结的方向、效应量和5/6 seed稳定性联合规则，本次结果**{chinese_decision}**。

上述区间与检验只条件于当前16个存档来源立方体。它们不是新农场、新批次、年份、供应商、仪器或实验室的总体区间；三个优化种子也不是生物重复。即使空间门槛通过，也只能说明当前配对来源立方体迁移中模型使用了可用的前景内部排列信息，不能解释为地理产地组织结构、化学机制或外部泛化。

## English audit summary

This post-processor accepts only an `executed_complete` locked matrix. The primary predictor is the temperature-scaled probability ensemble across the three preregistered optimization seeds. The primary estimand is balanced accuracy averaged equally over the two reciprocal source-cube directions. Every model and counterfactual shares one {BOOTSTRAP_REPETITIONS:,}-draw, eight-label-pair block-bootstrap index matrix generated with seed `{BOOTSTRAP_SEED}`; intervals are fixed 95% percentile intervals. Reliability data use the preregistered ten equal-width confidence bins.

{theta_table}

{effect_table}

The primary `fusion_net(full) − fusion_net(spatial_shuffle)` effect was {_percent(float(primary_effect['observed_two_direction_effect']))}, with a conditional 95% interval of {_percent(float(primary_effect['conditional_ci_low']))}–{_percent(float(primary_effect['conditional_ci_high']))}. The exact eight-label-pair sign-flip result was `p={float(sign_flip['p_value_greater']):.6g}` for the preregistered greater alternative and `p={float(sign_flip['p_value_two_sided_sensitivity']):.6g}` for the two-sided sensitivity analysis. Under the frozen joint direction/effect-size/5-of-6-seed rule, the result **{english_decision}**.

All intervals and tests are conditional on the 16 archived source cubes. They are not population intervals for new farms, lots, years, suppliers, instruments, or laboratories. Optimization seeds are not biological replicates. Passing the spatial gate would support only the use of within-foreground arrangement in these reciprocal cube transfers; it would not establish geographical tissue structure, chemical causality, or external generalization.

## Audit artifacts

- `bootstrap_label_pair_indices.csv`: the single shared resampling matrix;
- `bootstrap_theta_intervals.csv` and `bootstrap_effect_intervals.csv`: conditional intervals;
- `primary_sign_flip_label_effects.csv`, `primary_sign_flip_null_distribution.csv`, and `primary_sign_flip_test.json`;
- `main_results.csv`, `counterfactual_effects.csv`, `reliability_data.csv`, and `ensemble_confusion_matrices.csv`;
- four publication figures, each in PNG and PDF;
- `postprocessing_manifest.json`: input hashes, locked constants, and generated-file hashes.
"""


def summarize(input_dir: Path, output_dir: Path) -> None:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite or mix post-processing outputs: {output_dir}")
    bundle = validate_complete_output(input_dir)
    output_dir.mkdir(parents=True)

    draws = generate_block_draws()
    draw_rows = [
        {"bootstrap_replicate": index + 1, **{f"label_pair_position_{position + 1}": int(label) for position, label in enumerate(row)}}
        for index, row in enumerate(draws)
    ]
    theta_intervals = bootstrap_theta_intervals(bundle["ensemble_predictions"], draws)
    effect_intervals, _ = bootstrap_effect_intervals(bundle["ensemble_predictions"], draws)

    calibrated_recalls = balanced_accuracy_by_label(bundle["ensemble_predictions"], PRIMARY_CALIBRATION)
    primary_spec = EFFECT_SPECS[0]
    primary_blocks = _effect_blocks(calibrated_recalls, primary_spec)
    sign_flip = exact_sign_flip_test(primary_blocks)
    sign_flip_json = {key: value for key, value in sign_flip.items() if key != "null_distribution"}
    label_effect_rows = [
        {
            "label": label,
            "class_name": CLASS_NAMES[label],
            "paired_direction_mean_full_minus_spatial_shuffle": float(primary_blocks[label]),
        }
        for label in range(NUM_CLASSES)
    ]
    null_rows = [
        {"sign_assignment_index": index, "null_mean_effect": float(value)}
        for index, value in enumerate(sign_flip["null_distribution"])
    ]

    main_rows = main_results_table(bundle["metrics"], bundle["ensemble_metrics"])
    counterfactual_rows = counterfactual_effect_table(bundle["metrics"], bundle["ensemble_metrics"])
    reliability_rows = reliability_table(bundle["predictions"], bundle["ensemble_predictions"])
    confusion_rows = confusion_table(bundle["ensemble_predictions"])

    write_csv_rows(output_dir / "bootstrap_label_pair_indices.csv", draw_rows)
    write_csv_rows(output_dir / "bootstrap_theta_intervals.csv", theta_intervals)
    write_csv_rows(output_dir / "bootstrap_effect_intervals.csv", effect_intervals)
    write_csv_rows(output_dir / "primary_sign_flip_label_effects.csv", label_effect_rows)
    write_csv_rows(output_dir / "primary_sign_flip_null_distribution.csv", null_rows)
    write_json(output_dir / "primary_sign_flip_test.json", sign_flip_json)
    write_csv_rows(output_dir / "main_results.csv", main_rows)
    write_csv_rows(output_dir / "counterfactual_effects.csv", counterfactual_rows)
    write_csv_rows(output_dir / "reliability_data.csv", reliability_rows)
    write_csv_rows(output_dir / "ensemble_confusion_matrices.csv", confusion_rows)

    _set_publication_style()
    plot_main_performance(main_rows, output_dir)
    plot_counterfactuals(main_rows, output_dir)
    plot_calibration(reliability_rows, output_dir)
    plot_confusions(confusion_rows, output_dir)

    mechanism = read_json(input_dir / "spatial_mechanism_decision.json")
    (output_dir / "summary.md").write_text(
        build_summary(theta_intervals, effect_intervals, sign_flip_json, mechanism), encoding="utf-8"
    )

    input_hashes = {name: sha256_file(input_dir / name) for name in REQUIRED_INPUT_FILES}
    generated_names = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    output_hashes = {
        name: sha256_file(output_dir / name)
        for name in generated_names
        if name != "postprocessing_manifest.json"
    }
    write_json(
        output_dir / "postprocessing_manifest.json",
        {
            "status": "executed_complete",
            "analysis_definition": "frozen_before_reading_results; no model or condition selection",
            "input_directory": str(input_dir),
            "input_file_sha256": input_hashes,
            "locked_settings": {
                "directions": list(DIRECTIONS),
                "models": list(MODELS),
                "conditions": list(CONDITIONS),
                "seeds": list(SEEDS),
                "primary_calibration": PRIMARY_CALIBRATION,
                "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "resampling_unit": "eight commercial-label pairs with both directions retained",
                "effect_ids": [spec.effect_id for spec in EFFECT_SPECS],
                "ece_bins": ECE_BINS,
            },
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "matplotlib": matplotlib.__version__,
            },
            "output_file_sha256": output_hashes,
            "inference_scope": "conditional_on_the_16_archived_source_cubes_only",
        },
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Complete source-cube audit output directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="New post-processing directory (default: <input-dir>/postprocessing)",
    )
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = args.input_dir / "postprocessing"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summarize(args.input_dir, args.output_dir)
    print(f"Auditable post-processing outputs: {args.output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
