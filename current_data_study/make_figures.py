#!/usr/bin/env python3
"""Create deterministic, manuscript-oriented figures from locked analysis outputs.

The figures visualize current-data diagnostics only.  They do not add models,
select a preferred pipeline, or imply that seed-level observations are
independent provenance replicates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze import load_dataset


MODEL_ORDER = [
    "raw_rf",
    "raw_svm",
    "raw_lr",
    "sg_smooth_lr",
    "msc_lr",
    "sg_first_derivative_lr",
    "snv_lr",
    "raw_pls_da",
]
MODEL_LABELS = {
    "raw_lr": "Raw LR",
    "raw_svm": "RBF-SVM",
    "raw_pls_da": "PLS-DA",
    "raw_rf": "Random forest",
    "snv_lr": "SNV–LR",
    "msc_lr": "MSC–LR",
    "sg_smooth_lr": "SG smooth–LR",
    "sg_first_derivative_lr": "SG derivative–LR",
}
CLASS_COLORS = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB", "#000000"]


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def performance_figure(metrics: pd.DataFrame, output_dir: Path) -> None:
    """Compare models across the three validation constructions without error bars."""

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2), constrained_layout=True)
    random = metrics[metrics["protocol"] == "random_seed_holdout"].set_index("model")
    forward = metrics[metrics["protocol"] == "suffix_1_to_2"].set_index("model")
    reverse = metrics[metrics["protocol"] == "suffix_2_to_1"].set_index("model")
    loco = metrics[metrics["protocol"] == "leave_one_cube_out"].set_index("model")
    order = [model for model in MODEL_ORDER if model in random.index]
    positions = np.arange(len(order))

    values = 100 * random.loc[order, "accuracy"].to_numpy()
    axes[0].barh(positions, values, color="#4477AA")
    axes[0].axvline(96.84, color="#AA3377", linestyle="--", linewidth=1.5, label="Legacy HS3I-Net (96.84%)")
    axes[0].set(yticks=positions, yticklabels=[MODEL_LABELS[m] for m in order], xlim=(40, 100), xlabel="Accuracy (%)", title="A  Random seed holdout")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    for y, value in zip(positions, values, strict=True):
        axes[0].text(value + 0.6, y, f"{value:.2f}", va="center", fontsize=7.5)

    first = 100 * forward.loc[order, "accuracy"].to_numpy()
    second = 100 * reverse.loc[order, "accuracy"].to_numpy()
    for y, left, right in zip(positions, first, second, strict=True):
        axes[1].plot([left, right], [y, y], color="#BBBBBB", linewidth=1.2, zorder=1)
    axes[1].scatter(first, positions, color="#228833", label="Train *-1 → test *-2", zorder=2)
    axes[1].scatter(second, positions, color="#EE6677", marker="s", label="Train *-2 → test *-1", zorder=2)
    axes[1].set(yticks=positions, yticklabels=[MODEL_LABELS[m] for m in order], xlim=(40, 100), xlabel="Accuracy (%)", title="B  Reciprocal source-cube tests")
    axes[1].legend(frameon=False, fontsize=7.5, loc="lower right")

    loco_values = 100 * loco.loc[order, "accuracy"].to_numpy()
    axes[2].barh(positions, loco_values, color="#CCBB44")
    axes[2].set(yticks=positions, yticklabels=[MODEL_LABELS[m] for m in order], xlim=(40, 100), xlabel="Pooled accuracy (%)", title="C  Leave one source cube out")
    for y, value in zip(positions, loco_values, strict=True):
        axes[2].text(value + 0.6, y, f"{value:.2f}", va="center", fontsize=7.5)

    for ax in axes:
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("Performance depends on preprocessing and validation construction", fontsize=12)
    fig.text(
        0.5,
        -0.015,
        "Seed-level metrics are descriptive; the grouped tests still do not constitute independent-lot or geographical-origin validation.",
        ha="center",
        fontsize=8,
    )
    _save(fig, output_dir, "figure_performance_by_protocol")


def spectra_figure(data_root: Path, class_names: list[str], output_dir: Path) -> None:
    """Show each class's two source-cube mean spectra and within-cube spread."""

    dataset = load_dataset(data_root)
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.4), sharex=True, sharey=True, constrained_layout=True)
    for label, ax in enumerate(axes.flat):
        for suffix, linestyle in ((1, "-"), (2, "--")):
            cube = f"{label}-{suffix}"
            selected = dataset.manifest["source_cube"].to_numpy() == cube
            spectra = dataset.X[selected]
            mean = spectra.mean(axis=0)
            sd = spectra.std(axis=0, ddof=1)
            color = CLASS_COLORS[label]
            ax.plot(dataset.wavelengths, mean, color=color, linestyle=linestyle, linewidth=1.2, label=cube)
            ax.fill_between(dataset.wavelengths, mean - sd, mean + sd, color=color, alpha=0.10)
        ax.set_title(f"{label}: {class_names[label]}")
        ax.grid(color="#E5E5E5", linewidth=0.5)
        ax.legend(frameon=False, fontsize=7)
    for ax in axes[-1, :]:
        ax.set_xlabel("Measured wavelength (nm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Reflectance")
    fig.suptitle("Paired source cubes within each archived commercial label", fontsize=12)
    fig.text(
        0.5,
        -0.015,
        "Lines are source-cube means; shading is ±1 SD across seeds and is descriptive rather than a batch-level interval.",
        ha="center",
        fontsize=8,
    )
    _save(fig, output_dir, "figure_source_cube_spectra")


def confusion_figure(predictions: pd.DataFrame, class_names: list[str], output_dir: Path) -> None:
    """Plot row-normalized SNV-LR confusion matrices for grouped protocols."""

    protocols = [
        ("suffix_1_to_2", "A  Train *-1 → test *-2"),
        ("suffix_2_to_1", "B  Train *-2 → test *-1"),
        ("leave_one_cube_out", "C  Leave one source cube out"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), constrained_layout=True)
    image = None
    for ax, (protocol, title) in zip(axes, protocols, strict=True):
        frame = predictions[
            (predictions["protocol"] == protocol) & (predictions["model"] == "snv_lr")
        ]
        matrix = pd.crosstab(frame["label"], frame["predicted_label"]).reindex(
            index=range(len(class_names)), columns=range(len(class_names)), fill_value=0
        ).to_numpy(dtype=float)
        row_totals = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, row_totals, out=np.zeros_like(matrix), where=row_totals > 0)
        image = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(class_names, fontsize=7)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Archived label")
        ax.set_title(title)
        for row in range(len(class_names)):
            for column in range(len(class_names)):
                value = normalized[row, column]
                if value >= 0.05:
                    ax.text(
                        column,
                        row,
                        f"{100 * value:.0f}",
                        ha="center",
                        va="center",
                        color="white" if value > 0.55 else "black",
                        fontsize=6.5,
                    )
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, shrink=0.78, pad=0.02)
    colorbar.set_label("Row-normalized recall")
    fig.suptitle("SNV–LR errors change across source-cube validation constructions", fontsize=12)
    _save(fig, output_dir, "figure_snv_lr_grouped_confusions")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=script_dir.parent / "data")
    parser.add_argument("--output-root", type=Path, default=script_dir / "outputs")
    parser.add_argument("--figure-dir", type=Path, default=script_dir / "figures")
    parser.add_argument("--config", type=Path, default=script_dir / "config.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _style()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(args.output_root / "metrics.csv")
    predictions = pd.read_csv(args.output_root / "predictions.csv")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    class_names = list(config["class_names"])
    performance_figure(metrics, args.figure_dir)
    spectra_figure(args.data_root, class_names, args.figure_dir)
    confusion_figure(predictions, class_names, args.figure_dir)
    metrics.to_csv(args.figure_dir / "figure_performance_source.csv", index=False)
    print(f"Wrote figures and source table to {args.figure_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
