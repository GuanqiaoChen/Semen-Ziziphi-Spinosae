"""
SVM 和 PLS-DA
5 折交叉验证对比实验（修正版）
================================

特征：
    单粒 SZR 个体的前景平均光谱，392 维。

主要修正：
    1. 增加 ensure_hwc()，兼容 H × W × bands 和 bands × H × W；
    2. 增加 mask 方向检查，必要时自动转置 mask；
    3. 前景平均光谱与 corrected 1D-CNN 的提取逻辑保持一致；
    4. 增加 outputs/paper_figures 文件夹创建，避免保存图片时报错；
    5. 清理无用的混淆矩阵重建函数和未使用变量；
    6. PLS-DA 使用 one-hot 标签进行 PLSRegression，预测时取最大响应类别。

运行：
    python fig3_svm_plsda_fixed.py
"""

import os
import glob
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from datetime import datetime as _dt

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    import scipy.io as sio


# ═══════════════════════════════════════════════
# 0. 配置
# ═══════════════════════════════════════════════

SEED        = 42
DATA_ROOT   = "cube"
NUM_CLASSES = 8
N_BANDS     = 392
USE_MASK    = True
N_FOLDS     = 5
PREDICTION_SIZE = 0.25  # 3:1 split; 25% held out as independent prediction set

CLASS_NAMES = ["HBS", "HBX", "HNA", "HNX", "NX", "SXD", "SXQ", "XJH"]

SAVE_DIR = os.path.join(
    "outputs",
    "svm_plsda_holdout_prediction_5fold",
    f"run_{_dt.now().strftime('%Y%m%d_%H%M%S')}",
)

SVM_CONFIG = {
    "C": 10.0,
    "kernel": "rbf",
    "gamma": "scale",
}
PLSDA_CONFIG = {
    "n_components": 20,
}

random.seed(SEED)
np.random.seed(SEED)


# ═══════════════════════════════════════════════
# 1. 特征提取
# ═══════════════════════════════════════════════

def ensure_hwc(data: np.ndarray, n_bands: int = N_BANDS) -> np.ndarray:
    """
    将 patch 数据统一为 H × W × bands。
    兼容：
        H × W × bands
        bands × H × W
    """
    data = np.asarray(data, dtype=np.float32)

    if data.ndim != 3:
        raise ValueError(f"Expected a 3D hyperspectral patch, but got shape {data.shape}")

    if data.shape[-1] == n_bands:
        return data

    if data.shape[0] == n_bands:
        return np.transpose(data, (1, 2, 0))

    raise ValueError(
        f"Cannot infer spectral dimension from shape {data.shape}. "
        f"Expected one dimension to be {n_bands}."
    )


def extract_features(mat_path: str, use_mask: bool = True) -> np.ndarray:
    """
    提取单个 .mat 文件的前景平均光谱，返回 shape = (392,)。

    与 corrected 1D-CNN 保持一致：
        只对 mask 内前景像素求均值，不把背景 0 纳入平均。
    """
    if HAS_H5PY:
        with h5py.File(mat_path, "r") as f:
            data = f["patch_chw"][()].astype(np.float32)
            mask = f["crop_mask"][()].astype(np.float32) if use_mask and "crop_mask" in f else None
    else:
        mat = sio.loadmat(mat_path)
        data = mat["patch_chw"].astype(np.float32)
        mask = mat["crop_mask"].astype(np.float32) if use_mask and "crop_mask" in mat else None

    data = ensure_hwc(data, N_BANDS)
    data = np.clip(data, 0.0, 1.0)

    if use_mask and mask is not None:
        mask = np.asarray(mask, dtype=np.float32).squeeze()

        if mask.shape != data.shape[:2]:
            if mask.T.shape == data.shape[:2]:
                mask = mask.T
            else:
                raise ValueError(
                    f"Mask shape {mask.shape} does not match data spatial shape {data.shape[:2]} "
                    f"for file: {mat_path}"
                )

        foreground = mask > 0
        if foreground.sum() > 0:
            pixels = data[foreground]  # n_pixels × bands
        else:
            pixels = data.reshape(-1, data.shape[-1])
    else:
        pixels = data.reshape(-1, data.shape[-1])

    spec = pixels.mean(axis=0).astype(np.float32)

    if spec.shape[0] != N_BANDS:
        raise ValueError(f"Extracted spectrum length {spec.shape[0]} != expected {N_BANDS}")

    return np.clip(spec, 0.0, 1.0).astype(np.float32)


def load_dataset(root_dir: str, use_mask: bool = True):
    samples = []

    for folder in sorted(glob.glob(os.path.join(root_dir, "*-*"))):
        try:
            label = int(os.path.basename(folder).split("-")[0])
        except ValueError:
            continue

        for p in sorted(glob.glob(os.path.join(folder, "*.mat"))):
            samples.append((p, label))

    if len(samples) == 0:
        raise FileNotFoundError(f"No .mat files found under {root_dir}")

    print(f"[Data] 共 {len(samples)} 个样本，提取特征中...")
    X = np.array([extract_features(p, use_mask) for p, _ in samples])
    y = np.array([lbl for _, lbl in samples])

    print(f"[Data] 特征矩阵 shape: {X.shape}")
    for c, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {(y == c).sum()}")

    return X, y


# ═══════════════════════════════════════════════
# 2. PLS-DA 封装
# ═══════════════════════════════════════════════

class PLSDAClassifier:
    """
    PLS-DA implemented as PLS regression with one-hot encoded class labels.
    Class assignment is determined by the maximum predicted response.
    """

    def __init__(self, n_components: int = 20, n_classes: int = 8):
        self.n_classes = n_classes
        self.pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pls", PLSRegression(n_components=n_components, max_iter=1000)),
        ])

    def fit(self, X, y):
        Y = np.zeros((len(y), self.n_classes), dtype=np.float32)
        for i, c in enumerate(y):
            Y[i, c] = 1.0
        self.pipe.fit(X, Y)
        return self

    def predict(self, X):
        scores = self.pipe.predict(X)
        return scores.argmax(axis=1)


# ═══════════════════════════════════════════════
# 3. 混淆矩阵绘图
# ═══════════════════════════════════════════════

def plot_cm(cm, title, save_path, class_names):
    """Blues colormap；零值不显示；颜色条显示原始数量。"""
    n = len(class_names)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5

    fig, ax = plt.subplots(figsize=(9, 7.5), facecolor="white")
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0, vmax=cm.max())
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=12, labelpad=10)
    ax.set_ylabel("True label", fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=13, pad=14)

    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            if val == 0:
                continue
            color = "white" if val > thresh else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=10, color=color)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"  图已保存 → {save_path}")




def compute_metrics(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def save_split_info(y, dev_idx, pred_idx, save_dir):
    dev_idx = np.array(dev_idx, dtype=int)
    pred_idx = np.array(pred_idx, dtype=int)
    np.savez(
        os.path.join(save_dir, "dataset_split_indices.npz"),
        development_indices=dev_idx,
        prediction_indices=pred_idx,
    )
    path = os.path.join(save_dir, "dataset_split_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("Dataset split summary\n")
        f.write(f"Seed={SEED}\n")
        f.write(f"Prediction set ratio={PREDICTION_SIZE:.2f}\n")
        f.write(f"Model-development set size={len(dev_idx)}\n")
        f.write(f"Independent prediction set size={len(pred_idx)}\n\n")
        f.write("Class\tDevelopment\tPrediction\tTotal\n")
        for c, name in enumerate(CLASS_NAMES):
            dev_n = int((y[dev_idx] == c).sum())
            pred_n = int((y[pred_idx] == c).sum())
            f.write(f"{name}\t{dev_n}\t{pred_n}\t{dev_n + pred_n}\n")
    print(f"[OK] Dataset split summary saved → {path}")


def append_model_report(lines, model_name, fold_metrics, y_true_internal, y_pred_internal, pred_metrics, y_true_pred, y_pred_pred):
    accs = [m["acc"] for m in fold_metrics]
    precs = [m["precision"] for m in fold_metrics]
    recs = [m["recall"] for m in fold_metrics]
    f1s = [m["f1"] for m in fold_metrics]
    internal_acc_all = accuracy_score(y_true_internal, y_pred_internal)

    lines += [
        f"── {model_name} ──",
        "Internal 5-fold CV on model-development set:",
    ]
    for i, m in enumerate(fold_metrics, 1):
        lines.append(
            f"  Fold {i}: acc={m['acc']:.4f}  "
            f"prec={m['precision']:.4f}  "
            f"rec={m['recall']:.4f}  "
            f"f1={m['f1']:.4f}"
        )

    lines += [
        "Internal 5-fold Mean ± Std:",
        f"  Accuracy  : {np.mean(accs):.4f} ± {np.std(accs):.4f}",
        f"  Precision : {np.mean(precs):.4f} ± {np.std(precs):.4f}",
        f"  Recall    : {np.mean(recs):.4f} ± {np.std(recs):.4f}",
        f"  F1        : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}",
        f"Internal 5-fold overall accuracy (folds combined): {internal_acc_all:.4f}",
        "Internal 5-fold overall per-class report:",
        classification_report(
            y_true_internal,
            y_pred_internal,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        ),
        "Independent prediction set results:",
        f"  Accuracy  : {pred_metrics['acc']:.4f}",
        f"  Precision : {pred_metrics['precision']:.4f}",
        f"  Recall    : {pred_metrics['recall']:.4f}",
        f"  F1        : {pred_metrics['f1']:.4f}",
        "Independent prediction set per-class report:",
        classification_report(
            y_true_pred,
            y_pred_pred,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        ),
        "",
    ]
    return lines


def build_model(model_name):
    if model_name == "SVM":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(**SVM_CONFIG, random_state=SEED)),
        ])
    if model_name == "PLS-DA":
        return PLSDAClassifier(PLSDA_CONFIG["n_components"], NUM_CLASSES)
    raise ValueError(f"Unknown model: {model_name}")


# ═══════════════════════════════════════════════
# 4. 主流程
# ═══════════════════════════════════════════════


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs("outputs/paper_figures", exist_ok=True)

    print(f"[Config] SEED={SEED}  Folds={N_FOLDS}")
    print(f"[Config] split=3:1  prediction_size={PREDICTION_SIZE:.2f}")
    print("[Config] 特征：前景平均光谱 392维")

    X, y = load_dataset(DATA_ROOT, USE_MASK)
    all_idx = np.arange(len(y))

    # 与 3D-CNN 脚本保持一致：先划出 independent prediction set
    dev_idx, pred_idx, dev_lbl, pred_lbl = train_test_split(
        all_idx,
        y,
        test_size=PREDICTION_SIZE,
        random_state=SEED,
        shuffle=True,
        stratify=y,
    )
    dev_idx = np.array(dev_idx, dtype=int)
    pred_idx = np.array(pred_idx, dtype=int)
    dev_lbl = np.array(dev_lbl, dtype=int)
    pred_lbl = np.array(pred_lbl, dtype=int)

    print(f"[Split] Model-development set: {len(dev_idx)} samples")
    print(f"[Split] Independent prediction set: {len(pred_idx)} samples")
    save_split_info(y, dev_idx, pred_idx, SAVE_DIR)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    model_names = ["SVM", "PLS-DA"]
    results = {
        name: {
            "true_internal": [],
            "pred_internal": [],
            "fold_metrics": [],
            "true_prediction": [],
            "pred_prediction": [],
            "prediction_metrics": None,
        }
        for name in model_names
    }

    # 只在 model-development set 内部做 5-fold CV
    for fold, (train_pos, test_pos) in enumerate(skf.split(dev_idx, dev_lbl), start=1):
        train_idx = dev_idx[train_pos]
        test_idx = dev_idx[test_pos]

        print("\n" + "=" * 50)
        print(f"  Internal Fold {fold}/{N_FOLDS}  train={len(train_idx)}  test={len(test_idx)}")
        print("=" * 50)

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        for model_name in model_names:
            model = build_model(model_name)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = compute_metrics(y_test, y_pred)

            print(
                f"  {model_name:<6} Fold {fold}: "
                f"acc={metrics['acc']:.4f}  "
                f"prec={metrics['precision']:.4f}  "
                f"rec={metrics['recall']:.4f}  "
                f"f1={metrics['f1']:.4f}"
            )

            results[model_name]["true_internal"].extend(y_test.tolist())
            results[model_name]["pred_internal"].extend(y_pred.tolist())
            results[model_name]["fold_metrics"].append(metrics)

    # 用全部 development set 训练 final model，再在 independent prediction set 上评价
    print("\n" + "=" * 50)
    print("Independent Prediction Set Evaluation")
    print("=" * 50)

    for model_name in model_names:
        final_model = build_model(model_name)
        final_model.fit(X[dev_idx], y[dev_idx])
        y_pred_pred = final_model.predict(X[pred_idx])
        pred_metrics = compute_metrics(y[pred_idx], y_pred_pred)

        results[model_name]["true_prediction"] = y[pred_idx].tolist()
        results[model_name]["pred_prediction"] = y_pred_pred.tolist()
        results[model_name]["prediction_metrics"] = pred_metrics

        print(
            f"  {model_name:<6} Prediction set: "
            f"acc={pred_metrics['acc']:.4f}  "
            f"prec={pred_metrics['precision']:.4f}  "
            f"rec={pred_metrics['recall']:.4f}  "
            f"f1={pred_metrics['f1']:.4f}"
        )

    report_lines = [
        "SVM & PLS-DA: Model-development 5-fold CV + Independent Prediction Set",
        "Feature: foreground mean spectrum, 392 dimensions",
        f"Seed={SEED}  Folds={N_FOLDS}",
        f"Split: model-development={1-PREDICTION_SIZE:.2f}, prediction={PREDICTION_SIZE:.2f}",
        f"Development samples={len(dev_idx)}  Prediction samples={len(pred_idx)}",
        f"SVM: C={SVM_CONFIG['C']} kernel={SVM_CONFIG['kernel']} gamma={SVM_CONFIG['gamma']}",
        f"PLS-DA: n_components={PLSDA_CONFIG['n_components']}",
        "=" * 70,
        "",
    ]

    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)

    for model_name in model_names:
        res = results[model_name]
        y_true_internal = np.array(res["true_internal"])
        y_pred_internal = np.array(res["pred_internal"])
        y_true_pred = np.array(res["true_prediction"])
        y_pred_pred = np.array(res["pred_prediction"])
        fm = res["fold_metrics"]
        pred_metrics = res["prediction_metrics"]

        accs = [m["acc"] for m in fm]
        precs = [m["precision"] for m in fm]
        recs = [m["recall"] for m in fm]
        f1s = [m["f1"] for m in fm]

        cm_internal = confusion_matrix(y_true_internal, y_pred_internal, labels=list(range(NUM_CLASSES)))
        cm_prediction = confusion_matrix(y_true_pred, y_pred_pred, labels=list(range(NUM_CLASSES)))

        print("\n" + "=" * 50)
        print(f"{model_name} Summary")
        print("=" * 50)
        for i, m in enumerate(fm, 1):
            print(
                f"  Fold {i}: acc={m['acc']:.4f} "
                f"prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f}"
            )
        print(
            f"  Internal Mean ± Std: Acc={np.mean(accs):.4f}±{np.std(accs):.4f} "
            f"F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}"
        )
        print(
            f"  Prediction set: Acc={pred_metrics['acc']:.4f} "
            f"F1={pred_metrics['f1']:.4f}"
        )

        report_lines = append_model_report(
            report_lines,
            model_name,
            fm,
            y_true_internal,
            y_pred_internal,
            pred_metrics,
            y_true_pred,
            y_pred_pred,
        )

        safe_name = model_name.lower().replace("-", "").replace(" ", "_")
        plot_cm(
            cm_internal,
            title=f"{model_name} — Internal 5-fold CV Overall (Acc={accuracy_score(y_true_internal, y_pred_internal):.1%})",
            save_path=os.path.join(SAVE_DIR, f"cm_internal_5fold_overall_{safe_name}.png"),
            class_names=CLASS_NAMES,
        )
        plot_cm(
            cm_prediction,
            title=f"{model_name} — Independent Prediction Set (Acc={pred_metrics['acc']:.1%})",
            save_path=os.path.join(SAVE_DIR, f"cm_independent_prediction_set_{safe_name}.png"),
            class_names=CLASS_NAMES,
        )
        # 兼容原先论文作图文件夹：保存 prediction set 混淆矩阵副本
        plot_cm(
            cm_prediction,
            title=f"{model_name} — Independent Prediction Set (Acc={pred_metrics['acc']:.1%})",
            save_path=os.path.join("outputs", "paper_figures", f"cm_prediction_{safe_name}.png"),
            class_names=CLASS_NAMES,
        )

    report_path = os.path.join(SAVE_DIR, "results_svm_plsda_holdout_prediction_5fold.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n[Done] 文本报告 → {report_path}")
    print(f"[Done] 输出文件夹 → {SAVE_DIR}")


if __name__ == "__main__":
    main()
