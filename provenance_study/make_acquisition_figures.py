#!/usr/bin/env python3
"""Publication figures for the acquisition-robust provenance study (development).

Reads the machine tables in ``outputs/development_acquisition_calibration/`` and
recomputes the reliability curve and per-origin recall from development batches
0-7 only (no locked batch is read).  Colours use a CVD-validated Okabe-Ito
subset.  Every figure is written as 300-dpi PNG, vector PDF, and a source CSV.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.metrics import recall_score  # noqa: E402

from provenance_study.core import discover_manifest, load_csv_split  # noqa: E402
from provenance_study.acquisition_robust_pipeline import (  # noqa: E402
    _self_standardize,
    _sg1,
    fit_temperature,
    softmax_temperature,
)

CLASS = ("HBS", "HBX", "HNA", "HNX", "NX", "SXD", "SXQ", "XJH")
BLUE, VERMILLION, GREEN, GRAY, INK = "#0072B2", "#D55E00", "#009E73", "#B0B0B0", "#222222"
plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.dpi": 120,
})
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "provenance_study" / "outputs" / "development_acquisition_calibration"
FIG = OUT / "figures"


def _read_csv(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _save(fig, stem: str, source_rows: list[dict]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    if source_rows:
        keys = list(source_rows[0].keys())
        with (FIG / f"{stem}_source.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(source_rows)


def _load_dev():
    man = discover_manifest(ROOT / "data")
    dev = load_csv_split(man, split="development", verify_hashes=False)
    rep = np.array([r.replicate for r in dev.records])
    batch = np.array([r.constructed_batch for r in dev.records])
    return dev.X, dev.y, rep, batch


def _crosscube_lda(X, y, rep, batch, *, target_norm, calibrate):
    """Return pooled (y_true, proba) over both directions for sg1-lda."""
    ys, ps = [], []
    for tr, te in ((1, 2), (2, 1)):
        itr, ite = rep == tr, rep == te
        F_tr, mean, scale = _self_standardize(_sg1(X[itr]))
        F_te = _self_standardize(_sg1(X[ite]))[0] if target_norm else (_sg1(X[ite]) - mean) / scale
        model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(F_tr, y[itr])
        logits = model.predict_log_proba(F_te)
        T = 1.0
        if calibrate:
            cl, cy = [], []
            for b in np.unique(batch[itr]):
                inner = itr & (batch != b)
                held = itr & (batch == b)
                Fi, _, _ = _self_standardize(_sg1(X[inner]))
                Fh, _, _ = _self_standardize(_sg1(X[held]))
                fm = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Fi, y[inner])
                cl.append(fm.predict_log_proba(Fh)); cy.append(y[held])
            T = fit_temperature(np.vstack(cl), np.concatenate(cy))
        ys.append(y[ite]); ps.append(softmax_temperature(logits, T))
    return np.concatenate(ys), np.vstack(ps)


# --------------------------------------------------------------------------- #
def figure1(X, y, rep, batch) -> None:
    cross = _read_csv("cross_cube_metrics.csv")
    lobo = _read_csv("lobo_reference_metrics.csv")
    adapt = _read_csv("adaptation_metrics.csv")
    lobo_ba = float([r for r in lobo if r["calibration"] == "uncalibrated"][0]["balanced_accuracy"])

    reps = ["raw", "snv", "msc", "sg1", "sg2"]
    ba = {r["representation"]: float(r["balanced_accuracy"])
          for r in cross if r["classifier"] == "lda" and r["calibration"] == "uncalibrated"}
    src = float([r for r in adapt if r["method"] == "source_standardize"][0]["balanced_accuracy"])
    tgt = float([r for r in adapt if r["method"] == "target_standardize"][0]["balanced_accuracy"])
    snv_lr = float([r for r in cross
                    if r["representation"] == "snv" and r["classifier"] == "lr"
                    and r["calibration"] == "uncalibrated"][0]["balanced_accuracy"])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    colors = [VERMILLION if r == "sg1" else GRAY for r in reps]
    vals = [ba[r] for r in reps]
    axA.bar(reps, vals, color=colors, width=0.62)
    axA.axhline(lobo_ba, ls="--", lw=1.5, color=INK)
    axA.text(4.4, lobo_ba + 0.004, f"same-domain LOBO {lobo_ba:.3f}", ha="right", color=INK, fontsize=8.5)
    for i, v in enumerate(vals):
        axA.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=8.5, color=INK)
    axA.set_ylim(0.80, 1.0); axA.set_ylabel("cross-cube balanced accuracy")
    axA.set_title("A  Representation drives cross-acquisition transfer (LDA)", fontsize=10, loc="left")

    steps = ["SNV-LR", "SG1-LDA\n(source std)", "SG1-LDA\n(+target norm\n+calibration)"]
    svals = [snv_lr, src, tgt]
    scolors = [GRAY, BLUE, VERMILLION]
    axB.bar(steps, svals, color=scolors, width=0.6)
    axB.axhline(lobo_ba, ls="--", lw=1.5, color=INK)
    for i, v in enumerate(svals):
        axB.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=9, color=INK)
    axB.annotate("", xy=(2, tgt), xytext=(0, snv_lr),
                 arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.6))
    axB.text(1.0, (snv_lr + tgt) / 2 + 0.02, f"+{(tgt - snv_lr) * 100:.1f} pp",
             color=GREEN, fontsize=9.5, ha="center", fontweight="bold")
    axB.set_ylim(0.80, 1.0); axB.set_ylabel("cross-cube balanced accuracy")
    axB.set_title("B  Deployment pipeline vs weak baseline", fontsize=10, loc="left")
    fig.tight_layout()
    _save(fig, "figure1_cross_acquisition_representation",
          [{"representation": r, "cross_cube_balanced_accuracy": ba[r]} for r in reps]
          + [{"representation": s, "cross_cube_balanced_accuracy": v} for s, v in zip(steps, svals)])


def figure2(X, y, rep, batch) -> None:
    y_u, p_u = _crosscube_lda(X, y, rep, batch, target_norm=False, calibrate=False)
    y_c, p_c = _crosscube_lda(X, y, rep, batch, target_norm=False, calibrate=True)
    head = _read_csv("headline_effect.csv")

    def reliability(yt, p, n_bins=10):
        conf = p.max(1); correct = (p.argmax(1) == yt).astype(float)
        edges = np.linspace(0, 1, n_bins + 1); xs, ys, ws = [], [], []
        for i in range(n_bins):
            m = (conf > edges[i]) & (conf <= edges[i + 1])
            if m.sum() >= 5:
                xs.append(conf[m].mean()); ys.append(correct[m].mean()); ws.append(m.mean())
        return np.array(xs), np.array(ys)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    axA.plot([0, 1], [0, 1], ls=":", color=GRAY, lw=1.2)
    xu, yu = reliability(y_u, p_u); xc, yc = reliability(y_c, p_c)
    axA.plot(xu, yu, "o-", color=BLUE, lw=2, ms=6, label="uncalibrated")
    axA.plot(xc, yc, "s-", color=VERMILLION, lw=2, ms=6, label="calibrated")
    axA.set_xlim(0.4, 1.02); axA.set_ylim(0.4, 1.02)
    axA.set_xlabel("confidence"); axA.set_ylabel("accuracy")
    axA.legend(frameon=False, loc="upper left", fontsize=9)
    axA.set_title("A  Cross-cube reliability (SG1-LDA)", fontsize=10, loc="left")

    # ECE / NLL small multiples with bootstrap-CI on the reduction
    cross = _read_csv("cross_cube_metrics.csv")
    row_u = [r for r in cross if r["representation"] == "sg1" and r["classifier"] == "lda"
             and r["calibration"] == "uncalibrated"][0]
    row_c = [r for r in cross if r["representation"] == "sg1" and r["classifier"] == "lda"
             and r["calibration"] == "shift_aware_temperature"][0]
    metrics = [("expected_calibration_error", "ECE"), ("negative_log_likelihood", "NLL")]
    positions = [0, 1]
    for base, (mkey, mlab) in zip((0, 1.6), metrics):
        u = float(row_u[mkey]); c = float(row_c[mkey])
        axB.bar(base, u, width=0.6, color=BLUE)
        axB.bar(base + 0.7, c, width=0.6, color=VERMILLION)
        axB.text(base, u + max(u, c) * 0.02, f"{u:.3f}", ha="center", fontsize=8.5)
        axB.text(base + 0.7, c + max(u, c) * 0.02, f"{c:.3f}", ha="center", fontsize=8.5)
        axB.text(base + 0.35, -0.02 * 0.25, mlab, ha="center", va="top", fontsize=9)
    axB.set_xticks([]); axB.set_ylabel("value (lower is better)")
    axB.set_title("B  Calibration repair under shift", fontsize=10, loc="left")
    axB.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE),
                        plt.Rectangle((0, 0), 1, 1, color=VERMILLION)],
               labels=["uncalibrated", "calibrated"], frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    _save(fig, "figure2_calibration_repair",
          [{"bin_confidence": float(a), "bin_accuracy": float(b), "series": s}
           for s, xs, ys in (("uncalibrated", xu, yu), ("calibrated", xc, yc))
           for a, b in zip(xs, ys)])


def figure3(X, y, rep, batch) -> None:
    def per_origin(target_norm):
        recs = []
        for tr, te in ((1, 2), (2, 1)):
            itr, ite = rep == tr, rep == te
            F_tr, mean, scale = _self_standardize(_sg1(X[itr]))
            F_te = _self_standardize(_sg1(X[ite]))[0] if target_norm else (_sg1(X[ite]) - mean) / scale
            m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(F_tr, y[itr])
            recs.append(recall_score(y[ite], m.predict(F_te), labels=range(8), average=None, zero_division=0))
        return np.mean(recs, 0)

    src = per_origin(False); tgt = per_origin(True)
    order = np.argsort(src)  # worst origins first
    labels = [CLASS[c] for c in order]
    xpos = np.arange(8)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 4.0), gridspec_kw={"width_ratios": [1.5, 1]})
    axA.bar(xpos - 0.2, src[order], width=0.38, color=BLUE, label="source std")
    axA.bar(xpos + 0.2, tgt[order], width=0.38, color=VERMILLION, label="+ target norm")
    axA.set_xticks(xpos); axA.set_xticklabels(labels, rotation=0)
    axA.set_ylim(0.5, 1.02); axA.set_ylabel("cross-cube recall")
    axA.legend(frameon=False, fontsize=9, loc="lower right")
    axA.set_title("A  Per-origin rescue by target normalization", fontsize=10, loc="left")

    abl = _read_csv("negative_ablation.csv")
    ref = [r for r in abl if r["feature"] == "sg1_reference"][0]
    cov = [r for r in abl if r["feature"] == "sg1_plus_covariance"][0]
    for base, key, lab, scale in ((0, "cross_cube_balanced_accuracy", "balanced\naccuracy", 1),
                                  (1.6, "cross_cube_negative_log_likelihood", "NLL", 1)):
        rv = float(ref[key]); cv = float(cov[key])
        axB.bar(base, rv, width=0.6, color=GREEN)
        axB.bar(base + 0.7, cv, width=0.6, color=GRAY)
        axB.text(base, rv + max(rv, cv) * 0.02, f"{rv:.3f}", ha="center", fontsize=8.5)
        axB.text(base + 0.7, cv + max(rv, cv) * 0.02, f"{cv:.3f}", ha="center", fontsize=8.5)
        axB.text(base + 0.35, -0.03 * max(rv, cv), lab, ha="center", va="top", fontsize=9)
    axB.set_xticks([]); axB.set_ylabel("value")
    axB.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=GREEN),
                        plt.Rectangle((0, 0), 1, 1, color=GRAY)],
               labels=["SG1 spectrum", "+ pixel covariance"], frameon=False, fontsize=9, loc="upper right")
    axB.set_title("B  Covariance features are harmful (negative)", fontsize=10, loc="left")
    fig.tight_layout()
    _save(fig, "figure3_origin_rescue_and_negatives",
          [{"origin": CLASS[c], "recall_source_std": float(src[c]), "recall_target_norm": float(tgt[c])}
           for c in range(8)])


def main() -> int:
    X, y, rep, batch = _load_dev()
    figure1(X, y, rep, batch)
    figure2(X, y, rep, batch)
    figure3(X, y, rep, batch)
    print(f"Wrote figures to {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
