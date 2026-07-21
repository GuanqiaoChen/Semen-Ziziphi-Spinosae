from __future__ import annotations

"""
Corrected 1D-CNN for hyperspectral mean-spectrum classification
with model-development 5-fold CV + independent prediction set
================================================================

主要修正：
1. 平均光谱提取：只对 mask 内的前景像素求均值，而不是背景置 0 后对整个 patch 求均值。
2. 数据增强：删除 wavelength flip，因为近红外光谱的波长顺序具有物理意义，不能反转。
3. 保留与 3D-CNN 尽量一致的训练设置：
   - StratifiedKFold 5-fold
   - AdamW
   - cosine annealing warm restarts
   - label smoothing
   - mixup
   - dropout
   - TTA=5
4. 不使用 TTA；只输出 no-TTA 结果，便于与 3D-CNN no-TTA 主结果公平对比。

运行：
    python 1d_cnn_kfold_corrected.py
"""

import os
import glob
import random
from datetime import datetime as _dt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════
# 0. 基础配置
# ═══════════════════════════════════════════════

SEED = 42
DATA_ROOT = "cube"

NUM_CLASSES = 8
N_BANDS = 392
N_FOLDS = 5
PREDICTION_SIZE = 0.25  # 3:1 split; 25% held out as independent prediction set

CLASS_NAMES = ["HBS", "HBX", "HNA", "HNX", "NX", "SXD", "SXQ", "XJH"]

# 训练参数：尽量与 3D-CNN 保持一致
BATCH_SIZE = 32
EPOCHS = 360
WARMUP_EPOCHS = 10
LR = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.1
MIXUP_ALPHA = 0.3
DROPOUT = 0.35
SGDR_T0 = 50
SGDR_T_MULT = 2
# TTA removed: direct prediction only
TTA_TIMES = 1

# 1D 光谱增强参数：保留强度扰动和微小噪声，不做波长翻转
AUG_SCALE_MIN = 0.90
AUG_SCALE_MAX = 1.10
AUG_OFFSET_RANGE = 0.005
AUG_NOISE_STD = 0.003

# 是否做训练集统计量标准化
# 为了与原 3D-CNN 输入 0-1 反射率更接近，默认 False。
# 如果想进一步提高 1D 光谱模型性能，可改为 True。
STANDARDIZE_WITH_TRAIN_STATS = False

# 数据读取缓存：1D 平均光谱很小，建议缓存，可明显减少重复读 mat 的时间
CACHE_SPECTRA = True

NUM_WORKERS = min(4, os.cpu_count() or 1)

SAVE_DIR = os.path.join(
    "outputs",
    "1d_cnn_holdout_prediction_5fold_no_tta",
    f"run_{_dt.now().strftime('%Y%m%d_%H%M%S')}",
)


# ═══════════════════════════════════════════════
# 1. 兼容 h5py / scipy.io 读取 mat 文件
# ═══════════════════════════════════════════════

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    h5py = None
    HAS_H5PY = False

try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    sio = None
    HAS_SCIPY = False


def seed_everything(seed: int) -> None:
    """固定随机种子，尽量保证结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_mat_patch(path: str):
    """
    读取单个 .mat 文件中的 patch_chw 和 crop_mask。

    期望：
        patch_chw: H × W × bands，或 bands × H × W
        crop_mask: H × W
    """
    data = None
    mask = None

    # 优先尝试 h5py，适用于 v7.3 mat
    if HAS_H5PY:
        try:
            with h5py.File(path, "r") as f:
                data = f["patch_chw"][()].astype(np.float32)
                if "crop_mask" in f:
                    mask = f["crop_mask"][()].astype(np.float32)
        except Exception:
            data = None
            mask = None

    # h5py 失败则尝试 scipy.io，适用于普通 mat
    if data is None:
        if not HAS_SCIPY:
            raise ImportError(
                "Cannot read .mat file. Please install scipy or h5py."
            )
        mat = sio.loadmat(path)
        data = mat["patch_chw"].astype(np.float32)
        mask = mat["crop_mask"].astype(np.float32) if "crop_mask" in mat else None

    data = np.asarray(data, dtype=np.float32)
    if mask is not None:
        mask = np.asarray(mask, dtype=np.float32).squeeze()

    return data, mask


def ensure_hwc(data: np.ndarray, n_bands: int = N_BANDS) -> np.ndarray:
    """
    将 patch 数据统一为 H × W × bands。
    原脚本默认 patch_chw 实际是 H × W × λ。
    这里做一个稳健处理，防止文件维度顺序不一致。
    """
    if data.ndim != 3:
        raise ValueError(f"Expected 3D patch, but got shape {data.shape}")

    # H × W × bands
    if data.shape[-1] == n_bands:
        return data

    # bands × H × W
    if data.shape[0] == n_bands:
        return np.transpose(data, (1, 2, 0))

    raise ValueError(
        f"Cannot infer spectral dimension from shape {data.shape}. "
        f"Expected one dimension to be {n_bands}."
    )


def extract_foreground_mean_spectrum(
    data: np.ndarray,
    mask: np.ndarray | None,
    n_bands: int = N_BANDS,
) -> np.ndarray:
    """
    提取真正的前景平均光谱。

    关键修正：
    不再使用 data * mask 后对整个 H × W 求均值；
    而是只选取 mask 内的前景像素，然后求均值。
    """
    data = ensure_hwc(data, n_bands=n_bands)
    data = np.clip(data, 0.0, 1.0)

    if mask is not None:
        mask = np.asarray(mask).squeeze()

        # 如果 mask 方向和 data 前两维相反，尝试转置
        if mask.shape != data.shape[:2]:
            if mask.T.shape == data.shape[:2]:
                mask = mask.T
            else:
                raise ValueError(
                    f"Mask shape {mask.shape} does not match data spatial shape {data.shape[:2]}"
                )

        foreground = mask > 0

        if foreground.sum() > 0:
            pixels = data[foreground]  # n_pixels × bands
        else:
            pixels = data.reshape(-1, data.shape[-1])
    else:
        pixels = data.reshape(-1, data.shape[-1])

    spectrum = pixels.mean(axis=0).astype(np.float32)

    if spectrum.shape[0] != n_bands:
        raise ValueError(
            f"Extracted spectrum length {spectrum.shape[0]} != expected {n_bands}"
        )

    return spectrum


# ═══════════════════════════════════════════════
# 2. 数据集
# ═══════════════════════════════════════════════

class RawSpectralDataset(Dataset):
    """
    只负责：
    1. 扫描文件；
    2. 读取标签；
    3. 提取前景平均光谱；
    4. 可选缓存。

    不做增强，也不做标准化。
    """

    def __init__(
        self,
        root_dir: str,
        use_mask: bool = True,
        cache_spectra: bool = True,
    ):
        self.root_dir = root_dir
        self.use_mask = use_mask
        self.cache_spectra = cache_spectra

        self.samples: list[tuple[str, int]] = []
        self.labels: list[int] = []

        for folder in sorted(glob.glob(os.path.join(root_dir, "*-*"))):
            base = os.path.basename(folder)
            try:
                label = int(base.split("-")[0])
            except ValueError:
                continue

            for p in sorted(glob.glob(os.path.join(folder, "*.mat"))):
                self.samples.append((p, label))
                self.labels.append(label)

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No .mat files found under {root_dir}. "
                f"Please check DATA_ROOT."
            )

        unique_labels = sorted(set(self.labels))
        expected_labels = list(range(NUM_CLASSES))
        if unique_labels != expected_labels:
            raise ValueError(
                f"Labels should be {expected_labels}, but got {unique_labels}. "
                f"Please check folder names, e.g. 0-HBS, 1-HBX, ..."
            )

        self._spectra_cache = None
        if self.cache_spectra:
            print("[Data] Caching foreground mean spectra ...")
            self._spectra_cache = []
            for i in range(len(self.samples)):
                self._spectra_cache.append(self._load_spectrum_by_index(i))
            self._spectra_cache = np.stack(self._spectra_cache, axis=0)
            print(f"[Data] Cached spectra shape: {self._spectra_cache.shape}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_spectrum_by_index(self, idx: int) -> np.ndarray:
        path, _ = self.samples[idx]
        data, mask = read_mat_patch(path)
        if not self.use_mask:
            mask = None
        spectrum = extract_foreground_mean_spectrum(data, mask, n_bands=N_BANDS)
        return spectrum

    def get_raw_spectrum(self, idx: int) -> np.ndarray:
        if self._spectra_cache is not None:
            return self._spectra_cache[idx].copy()
        return self._load_spectrum_by_index(idx)

    def __getitem__(self, idx: int):
        spectrum = self.get_raw_spectrum(idx)
        label = self.labels[idx]
        return spectrum, label


class SpectralViewDataset(Dataset):
    """
    在 RawSpectralDataset 基础上增加：
    1. 数据增强；
    2. 可选标准化；
    3. 转换为 torch tensor。

    输入给 1D-CNN 的形状：
        tensor: 1 × N_BANDS
    """

    def __init__(
        self,
        raw_dataset: RawSpectralDataset,
        augment: bool = False,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ):
        self.raw_dataset = raw_dataset
        self.augment = augment

        self.mean = None if mean is None else mean.astype(np.float32)
        self.std = None if std is None else std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, idx: int):
        spectrum, label = self.raw_dataset[idx]
        spectrum = spectrum.astype(np.float32).copy()

        # 光谱增强：只做物理上合理的小幅强度扰动，不做 wavelength flip
        if self.augment:
            scale = np.random.uniform(AUG_SCALE_MIN, AUG_SCALE_MAX)
            offset = np.random.uniform(-AUG_OFFSET_RANGE, AUG_OFFSET_RANGE)
            noise = np.random.normal(0.0, AUG_NOISE_STD, size=spectrum.shape).astype(np.float32)

            spectrum = spectrum * scale + offset + noise
            spectrum = np.clip(spectrum, 0.0, 1.0)

        # 可选：只使用训练集统计量进行标准化，避免数据泄漏
        if self.mean is not None and self.std is not None:
            spectrum = (spectrum - self.mean) / self.std

        tensor = torch.from_numpy(spectrum).unsqueeze(0)  # 1 × N_BANDS
        return tensor, label


def compute_train_mean_std(
    raw_dataset: RawSpectralDataset,
    train_indices: np.ndarray,
):
    """基于训练折计算均值和标准差，防止测试集信息泄漏。"""
    spectra = np.stack(
        [raw_dataset.get_raw_spectrum(int(i)) for i in train_indices],
        axis=0,
    )
    mean = spectra.mean(axis=0).astype(np.float32)
    std = spectra.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    return mean, std


# ═══════════════════════════════════════════════
# 3. 1D-CNN 模型
# ═══════════════════════════════════════════════

class CNN1D(nn.Module):
    """
    1D-CNN baseline:
    输入为单粒酸枣仁的前景平均光谱，shape = batch × 1 × 392。
    """

    def __init__(
        self,
        num_bands: int = N_BANDS,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.MaxPool1d(2),
        )

        self.avgpool = nn.AdaptiveAvgPool1d(4)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256 * 4, 512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


# ═══════════════════════════════════════════════
# 4. 训练与预测
# ═══════════════════════════════════════════════

def mixup_batch(x, y, alpha: float, device):
    if alpha <= 0:
        return x, y, y, 1.0

    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=device)

    x_mix = lam * x + (1.0 - lam) * x[idx]
    y_a = y
    y_b = y[idx]

    return x_mix, y_a, y_b, lam


def build_optimizer_and_scheduler(model):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    warmup = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda ep: (ep + 1) / WARMUP_EPOCHS if ep < WARMUP_EPOCHS else 1.0,
    )

    sgdr = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=SGDR_T0,
        T_mult=SGDR_T_MULT,
        eta_min=LR * 0.01,
    )

    return optimizer, warmup, sgdr


def train_one_fold(
    model,
    train_loader,
    device,
    fold: int,
    save_dir: str,
    use_amp: bool = True,
):
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimizer, warmup, sgdr = build_optimizer_and_scheduler(model)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        monitor_correct = 0
        n = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                if MIXUP_ALPHA > 0 and epoch > WARMUP_EPOCHS:
                    x_mix, y_a, y_b, lam = mixup_batch(x, y, MIXUP_ALPHA, device)
                    logits = model(x_mix)
                    loss = (
                        lam * criterion(logits, y_a)
                        + (1.0 - lam) * criterion(logits, y_b)
                    )
                else:
                    logits = model(x)
                    loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * y.size(0)

            # mixup 阶段该 train_acc 只是监控指标，不作为论文结果
            monitor_correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)

        if epoch <= WARMUP_EPOCHS:
            warmup.step()
        else:
            sgdr.step()

        epoch_loss = total_loss / n
        monitor_acc = monitor_correct / n
        history.append((epoch, epoch_loss, monitor_acc))

        if epoch == 1 or epoch % 60 == 0 or epoch == EPOCHS:
            print(
                f"    Epoch {epoch:3d}/{EPOCHS}  "
                f"loss={epoch_loss:.4f}  monitor_acc={monitor_acc:.3f}"
            )

    model_path = os.path.join(save_dir, f"fold{fold}_final.pth")
    torch.save(model.state_dict(), model_path)
    return model, history


@torch.no_grad()
def predict_no_tta(
    model,
    dataset,
    indices,
    device,
    batch_size: int = 64,
):
    model.eval()

    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    y_true = []
    y_pred = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(1).cpu().numpy()

        y_true.extend(y.numpy().tolist())
        y_pred.extend(pred.tolist())

    return np.array(y_true), np.array(y_pred)



def compute_metrics(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
    }


# ═══════════════════════════════════════════════
# 5. 保存结果与绘图
# ═══════════════════════════════════════════════

def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    save_path,
    title,
    normalize: bool = False,
):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
    )

    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        cm_show = cm.astype(np.float64) / row_sum
    else:
        cm_show = cm

    fig, ax = plt.subplots(figsize=(9, 7.5), facecolor="white")
    im = ax.imshow(cm_show, interpolation="nearest", cmap="Blues")

    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=12, labelpad=10)
    ax.set_ylabel("True label", fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=13, pad=14)

    threshold = cm_show.max() / 2.0 if cm_show.max() > 0 else 0.5

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                text = f"{cm_show[i, j]:.2f}"
            else:
                text = str(cm[i, j])

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=10,
                color="white" if cm_show[i, j] > threshold else "black",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()



def plot_training_curve(history, fold, save_dir):
    """
    Plot training loss and training monitor accuracy curves for one fold.
    history format: [(epoch, train_loss, train_monitor_acc), ...]
    """
    epochs = [x[0] for x in history]
    train_loss = [x[1] for x in history]
    train_acc = [x[2] for x in history]

    # Combined figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='white')

    axes[0].plot(epochs, train_loss, linewidth=1.8, label='Train loss')
    axes[0].set_title(f'Fold {fold} Training Loss', fontsize=13)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, linewidth=1.8, label='Train monitor accuracy')
    axes[1].set_title(f'Fold {fold} Training Monitor Accuracy', fontsize=13)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    combined_path = os.path.join(save_dir, f'training_curve_fold{fold}.png')
    plt.savefig(combined_path, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()

    # Separate loss curve
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    ax.plot(epochs, train_loss, linewidth=1.8, label='Train loss')
    ax.set_title(f'Fold {fold} Training Loss Curve', fontsize=13)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    loss_path = os.path.join(save_dir, f'fold{fold}_train_loss_curve.png')
    plt.savefig(loss_path, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()

    # Separate accuracy curve
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    ax.plot(epochs, train_acc, linewidth=1.8, label='Train monitor accuracy')
    ax.set_title(f'Fold {fold} Training Monitor Accuracy Curve', fontsize=13)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    acc_path = os.path.join(save_dir, f'fold{fold}_train_accuracy_curve.png')
    plt.savefig(acc_path, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f'  [OK] Training curves saved: {combined_path}')



def plot_overall_training_curves(all_histories, save_dir):
    """
    Plot overall mean ± SD training curves across all 5 folds.
    all_histories format:
        [
            [(epoch, train_loss, train_monitor_acc), ...],
            ...
        ]
    """
    if len(all_histories) == 0:
        return

    min_len = min(len(h) for h in all_histories)
    histories = [h[:min_len] for h in all_histories]

    epochs = np.array([x[0] for x in histories[0]])
    loss_arr = np.array([[x[1] for x in h] for h in histories], dtype=np.float64)
    acc_arr = np.array([[x[2] for x in h] for h in histories], dtype=np.float64)

    loss_mean = loss_arr.mean(axis=0)
    loss_std = loss_arr.std(axis=0)
    acc_mean = acc_arr.mean(axis=0)
    acc_std = acc_arr.std(axis=0)

    # Combined figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='white')

    axes[0].plot(epochs, loss_mean, linewidth=1.8, label='Mean train loss')
    axes[0].fill_between(epochs, loss_mean - loss_std, loss_mean + loss_std,
                         alpha=0.2, label='±1 SD')
    axes[0].set_title('Overall Training Loss Across 5 Folds', fontsize=13)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, acc_mean, linewidth=1.8, label='Mean train monitor accuracy')
    axes[1].fill_between(epochs, acc_mean - acc_std, acc_mean + acc_std,
                         alpha=0.2, label='±1 SD')
    axes[1].set_title('Overall Training Monitor Accuracy Across 5 Folds', fontsize=13)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    combined_path = os.path.join(save_dir, 'overall_training_curves_5fold.png')
    plt.savefig(combined_path, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()

    # Separate loss curve
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    ax.plot(epochs, loss_mean, linewidth=1.8, label='Mean train loss')
    ax.fill_between(epochs, loss_mean - loss_std, loss_mean + loss_std,
                    alpha=0.2, label='±1 SD')
    ax.set_title('Overall Training Loss Across 5 Folds', fontsize=13)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    loss_path = os.path.join(save_dir, 'overall_train_loss_curve_5fold.png')
    plt.savefig(loss_path, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()

    # Separate accuracy curve
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    ax.plot(epochs, acc_mean, linewidth=1.8, label='Mean train monitor accuracy')
    ax.fill_between(epochs, acc_mean - acc_std, acc_mean + acc_std,
                    alpha=0.2, label='±1 SD')
    ax.set_title('Overall Training Monitor Accuracy Across 5 Folds', fontsize=13)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    acc_path = os.path.join(save_dir, 'overall_train_accuracy_curve_5fold.png')
    plt.savefig(acc_path, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()

    # Save numerical values
    values_path = os.path.join(save_dir, 'overall_training_curves_5fold_values.txt')
    with open(values_path, 'w', encoding='utf-8') as f:
        f.write('epoch\ttrain_loss_mean\ttrain_loss_sd\ttrain_acc_mean\ttrain_acc_sd\n')
        for ep, lm, ls, am, astd in zip(epochs, loss_mean, loss_std, acc_mean, acc_std):
            f.write(f'{int(ep)}\t{lm:.6f}\t{ls:.6f}\t{am:.6f}\t{astd:.6f}\n')

    print(f'[OK] Overall 5-fold training curves saved → {combined_path}')


def save_confusion_values(y_true, y_pred, save_path):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
    )
    row_sum = cm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    cm_norm = cm.astype(np.float64) / row_sum

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("Confusion Matrix Counts\n")
        f.write("True\\Pred\t" + "\t".join(CLASS_NAMES) + "\n")

        for i, row in enumerate(cm):
            f.write(CLASS_NAMES[i] + "\t" + "\t".join(map(str, row)) + "\n")

        f.write("\nConfusion Matrix Row-normalized\n")
        f.write("True\\Pred\t" + "\t".join(CLASS_NAMES) + "\n")

        for i, row in enumerate(cm_norm):
            f.write(
                CLASS_NAMES[i]
                + "\t"
                + "\t".join([f"{v:.4f}" for v in row])
                + "\n"
            )


def summarize_fold_metrics(fold_metrics):
    keys = ["acc", "precision", "recall", "f1"]
    summary = {}

    for k in keys:
        values = np.array([m[k] for m in fold_metrics], dtype=np.float64)
        summary[k] = (values.mean(), values.std())

    return summary


def print_summary(name, fold_metrics):
    summary = summarize_fold_metrics(fold_metrics)

    print(f"\n{name}")
    print("-" * 55)

    for i, m in enumerate(fold_metrics, 1):
        print(
            f"  Fold {i}: "
            f"acc={m['acc']:.4f}  "
            f"precision={m['precision']:.4f}  "
            f"recall={m['recall']:.4f}  "
            f"f1={m['f1']:.4f}"
        )

    print("\n  Mean ± Std:")
    print(f"  Accuracy : {summary['acc'][0]:.4f} ± {summary['acc'][1]:.4f}")
    print(f"  Precision: {summary['precision'][0]:.4f} ± {summary['precision'][1]:.4f}")
    print(f"  Recall   : {summary['recall'][0]:.4f} ± {summary['recall'][1]:.4f}")
    print(f"  F1-score : {summary['f1'][0]:.4f} ± {summary['f1'][1]:.4f}")


def save_report(
    save_path,
    fold_metrics_no_tta,
    y_true_no_tta,
    y_pred_no_tta,
):
    summary_no_tta = summarize_fold_metrics(fold_metrics_no_tta)
    NL = chr(10)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("Corrected 1D-CNN 5-Fold Cross-Validation Results without TTA" + NL)
        f.write("=" * 70 + NL + NL)

        f.write("Main corrections:" + NL)
        f.write("1. Mean spectrum was calculated using foreground pixels only." + NL)
        f.write("2. Wavelength flipping was removed from spectral augmentation." + NL)
        f.write("3. TTA was disabled; direct prediction was used for all test folds." + NL + NL)

        f.write("Configuration:" + NL)
        f.write(f"SEED={SEED}" + NL)
        f.write(f"DATA_ROOT={DATA_ROOT}" + NL)
        f.write(f"NUM_CLASSES={NUM_CLASSES}" + NL)
        f.write(f"N_BANDS={N_BANDS}" + NL)
        f.write(f"N_FOLDS={N_FOLDS}" + NL)
        f.write(f"BATCH_SIZE={BATCH_SIZE}" + NL)
        f.write(f"EPOCHS={EPOCHS}" + NL)
        f.write(f"LR={LR}" + NL)
        f.write(f"WEIGHT_DECAY={WEIGHT_DECAY}" + NL)
        f.write(f"LABEL_SMOOTH={LABEL_SMOOTH}" + NL)
        f.write(f"MIXUP_ALPHA={MIXUP_ALPHA}" + NL)
        f.write(f"DROPOUT={DROPOUT}" + NL)
        f.write(f"SGDR_T0={SGDR_T0}" + NL)
        f.write(f"SGDR_T_MULT={SGDR_T_MULT}" + NL)
        f.write("TTA=disabled" + NL)
        f.write(f"STANDARDIZE_WITH_TRAIN_STATS={STANDARDIZE_WITH_TRAIN_STATS}" + NL + NL)

        f.write("Per-fold results without TTA:" + NL)
        for i, m in enumerate(fold_metrics_no_tta, 1):
            f.write(
                f"  Fold {i}: "
                f"acc={m['acc']:.4f}, "
                f"precision={m['precision']:.4f}, "
                f"recall={m['recall']:.4f}, "
                f"f1={m['f1']:.4f}" + NL
            )

        f.write(NL + "Mean ± Std without TTA:" + NL)
        for k, label in [
            ("acc", "Accuracy"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1", "F1-score"),
        ]:
            f.write(f"  {label}: {summary_no_tta[k][0]:.4f} ± {summary_no_tta[k][1]:.4f}" + NL)

        f.write(NL + "Overall classification report without TTA:" + NL)
        f.write(
            classification_report(
                y_true_no_tta,
                y_pred_no_tta,
                labels=list(range(NUM_CLASSES)),
                target_names=CLASS_NAMES,
                digits=4,
                zero_division=0,
            )
        )





def save_split_info(raw_dataset, dev_idx, pred_idx, save_dir):
    """Save model-development / prediction-set indices and class distribution."""
    dev_idx = np.array(dev_idx, dtype=int)
    pred_idx = np.array(pred_idx, dtype=int)

    np.savez(
        os.path.join(save_dir, "dataset_split_indices.npz"),
        development_indices=dev_idx,
        prediction_indices=pred_idx,
    )

    labels = np.array(raw_dataset.labels)
    path = os.path.join(save_dir, "dataset_split_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("Dataset split summary\n")
        f.write(f"Seed={SEED}\n")
        f.write(f"Prediction set ratio={PREDICTION_SIZE:.2f}\n")
        f.write(f"Model-development set size={len(dev_idx)}\n")
        f.write(f"Independent prediction set size={len(pred_idx)}\n\n")
        f.write("Class\tDevelopment\tPrediction\tTotal\n")
        for c, name in enumerate(CLASS_NAMES):
            dev_n = int((labels[dev_idx] == c).sum())
            pred_n = int((labels[pred_idx] == c).sum())
            f.write(f"{name}\t{dev_n}\t{pred_n}\t{dev_n + pred_n}\n")
    print(f"[OK] Dataset split summary saved → {path}")


def write_holdout_report(
    save_path,
    fold_metrics,
    y_true_internal,
    y_pred_internal,
    pred_metrics,
    y_true_prediction,
    y_pred_prediction,
    dev_idx,
    pred_idx,
):
    summary = summarize_fold_metrics(fold_metrics)
    NL = chr(10)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("Corrected 1D-CNN: Model-development 5-fold CV + Independent Prediction Set" + NL)
        f.write("=" * 80 + NL + NL)
        f.write("Configuration:" + NL)
        f.write(f"SEED={SEED}" + NL)
        f.write(f"Split: model-development={1-PREDICTION_SIZE:.2f}, prediction={PREDICTION_SIZE:.2f}" + NL)
        f.write(f"Development samples={len(dev_idx)}  Prediction samples={len(pred_idx)}" + NL)
        f.write(f"N_FOLDS={N_FOLDS}  EPOCHS={EPOCHS}  BATCH_SIZE={BATCH_SIZE}" + NL)
        f.write(f"LR={LR}  WEIGHT_DECAY={WEIGHT_DECAY}  DROPOUT={DROPOUT}" + NL)
        f.write(f"LABEL_SMOOTH={LABEL_SMOOTH}  MIXUP_ALPHA={MIXUP_ALPHA}" + NL)
        f.write(f"SGDR_T0={SGDR_T0}  SGDR_T_MULT={SGDR_T_MULT}" + NL)
        f.write("TTA=disabled" + NL)
        f.write(f"STANDARDIZE_WITH_TRAIN_STATS={STANDARDIZE_WITH_TRAIN_STATS}" + NL + NL)

        f.write("Internal 5-fold CV on model-development set:" + NL)
        for i, m in enumerate(fold_metrics, 1):
            f.write(
                f"  Fold {i}: acc={m['acc']:.4f}  "
                f"precision={m['precision']:.4f}  "
                f"recall={m['recall']:.4f}  "
                f"f1={m['f1']:.4f}" + NL
            )
        f.write(NL + "Internal 5-fold Mean ± Std:" + NL)
        for k, label in [
            ("acc", "Accuracy"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1", "F1-score"),
        ]:
            f.write(f"  {label}: {summary[k][0]:.4f} ± {summary[k][1]:.4f}" + NL)

        internal_acc_all = accuracy_score(y_true_internal, y_pred_internal)
        f.write(NL + f"Internal 5-fold overall accuracy (folds combined): {internal_acc_all:.4f}" + NL + NL)
        f.write("Internal 5-fold overall per-class report:" + NL)
        f.write(classification_report(
            y_true_internal,
            y_pred_internal,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        ))

        f.write(NL + NL + "Independent prediction set results:" + NL)
        f.write(f"  Accuracy  : {pred_metrics['acc']:.4f}" + NL)
        f.write(f"  Precision : {pred_metrics['precision']:.4f}" + NL)
        f.write(f"  Recall    : {pred_metrics['recall']:.4f}" + NL)
        f.write(f"  F1        : {pred_metrics['f1']:.4f}" + NL + NL)
        f.write("Independent prediction set per-class report:" + NL)
        f.write(classification_report(
            y_true_prediction,
            y_pred_prediction,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        ))


# ═══════════════════════════════════════════════
# 6. 主流程
# ═══════════════════════════════════════════════


def main():
    seed_everything(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print("=" * 70)
    print("Corrected 1D-CNN: development-set 5-fold CV + independent prediction set")
    print("=" * 70)
    print(f"[Config] device={device}")
    print(f"[Config] seed={SEED}, folds={N_FOLDS}")
    print(f"[Config] split=3:1, prediction_size={PREDICTION_SIZE:.2f}")
    print(f"[Config] epochs={EPOCHS}, batch_size={BATCH_SIZE}, lr={LR}")
    print(f"[Config] dropout={DROPOUT}, mixup={MIXUP_ALPHA}, label_smooth={LABEL_SMOOTH}")
    print("[Config] TTA=disabled")
    print(f"[Config] standardize={STANDARDIZE_WITH_TRAIN_STATS}")
    print(f"[Config] save_dir={SAVE_DIR}")

    # 检查模型结构：不修改 1D-CNN baseline 结构
    with torch.no_grad():
        tmp_model = CNN1D(N_BANDS, NUM_CLASSES, DROPOUT).to(device)
        dummy = torch.zeros(2, 1, N_BANDS, device=device)
        out = tmp_model(dummy)
        params = sum(p.numel() for p in tmp_model.parameters() if p.requires_grad)
        print(f"[Model] dummy forward output shape: {out.shape}")
        print(f"[Model] trainable parameters: {params:,}")
        del tmp_model

    raw_dataset = RawSpectralDataset(
        DATA_ROOT,
        use_mask=True,
        cache_spectra=CACHE_SPECTRA,
    )

    all_indices = np.arange(len(raw_dataset))
    all_labels = np.array(raw_dataset.labels)

    print(f"[Data] total samples: {len(raw_dataset)}")
    for c in range(NUM_CLASSES):
        print(f"  {CLASS_NAMES[c]}: {(all_labels == c).sum()}")

    # 与 3D-CNN 脚本保持一致：先按 3:1 stratified split 划出 independent prediction set
    dev_idx, pred_idx, dev_lbl, pred_lbl = train_test_split(
        all_indices,
        all_labels,
        test_size=PREDICTION_SIZE,
        random_state=SEED,
        shuffle=True,
        stratify=all_labels,
    )
    dev_idx = np.array(dev_idx, dtype=int)
    pred_idx = np.array(pred_idx, dtype=int)
    dev_lbl = np.array(dev_lbl, dtype=int)
    pred_lbl = np.array(pred_lbl, dtype=int)

    print(f"[Split] Model-development set: {len(dev_idx)} samples")
    print(f"[Split] Independent prediction set: {len(pred_idx)} samples")
    save_split_info(raw_dataset, dev_idx, pred_idx, SAVE_DIR)

    skf = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    fold_metrics = []
    all_histories = []
    all_true_internal = []
    all_pred_internal = []

    # 只在 model-development set 内部做 5-fold CV
    for fold, (train_pos, test_pos) in enumerate(
        skf.split(dev_idx, dev_lbl),
        start=1,
    ):
        train_idx = dev_idx[train_pos]
        test_idx = dev_idx[test_pos]

        print("\n" + "=" * 70)
        print(f"Internal Fold {fold}/{N_FOLDS} | train={len(train_idx)} | test={len(test_idx)}")
        print("=" * 70)

        seed_everything(SEED + fold * 100)

        if STANDARDIZE_WITH_TRAIN_STATS:
            mean, std = compute_train_mean_std(raw_dataset, train_idx)
        else:
            mean, std = None, None

        train_dataset = SpectralViewDataset(raw_dataset, augment=True, mean=mean, std=std)
        test_dataset = SpectralViewDataset(raw_dataset, augment=False, mean=mean, std=std)

        train_loader = DataLoader(
            Subset(train_dataset, train_idx.tolist()),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=(NUM_WORKERS > 0),
        )

        model = CNN1D(N_BANDS, NUM_CLASSES, DROPOUT).to(device)
        model, history = train_one_fold(
            model=model,
            train_loader=train_loader,
            device=device,
            fold=fold,
            save_dir=SAVE_DIR,
            use_amp=use_amp,
        )
        all_histories.append(history)

        y_true, y_pred = predict_no_tta(model, test_dataset, test_idx.tolist(), device, batch_size=64)
        metrics = compute_metrics(y_true, y_pred)
        fold_metrics.append(metrics)
        all_true_internal.extend(y_true.tolist())
        all_pred_internal.extend(y_pred.tolist())

        print(
            f"[Internal Fold {fold}] "
            f"acc={metrics['acc']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1']:.4f}"
        )

        log_path = os.path.join(SAVE_DIR, f"fold{fold}_train_log.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("epoch\tloss\tmonitor_acc\n")
            for ep, loss, acc in history:
                f.write(f"{ep}\t{loss:.6f}\t{acc:.6f}\n")
        plot_training_curve(history, fold, SAVE_DIR)

    plot_overall_training_curves(all_histories, SAVE_DIR)

    all_true_internal = np.array(all_true_internal)
    all_pred_internal = np.array(all_pred_internal)

    print("\n" + "=" * 70)
    print_summary("Internal 5-Fold Summary on Model-Development Set — Corrected 1D-CNN", fold_metrics)
    print("\nInternal 5-fold overall report:")
    print(classification_report(
        all_true_internal,
        all_pred_internal,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    ))

    plot_confusion_matrix(
        all_true_internal,
        all_pred_internal,
        CLASS_NAMES,
        os.path.join(SAVE_DIR, "cm_internal_5fold_overall_counts.png"),
        title="Corrected 1D-CNN Internal 5-fold Overall Confusion Matrix",
        normalize=False,
    )
    plot_confusion_matrix(
        all_true_internal,
        all_pred_internal,
        CLASS_NAMES,
        os.path.join(SAVE_DIR, "cm_internal_5fold_overall_normalized.png"),
        title="Corrected 1D-CNN Internal 5-fold Overall Confusion Matrix",
        normalize=True,
    )
    save_confusion_values(
        all_true_internal,
        all_pred_internal,
        os.path.join(SAVE_DIR, "cm_internal_5fold_overall_values.txt"),
    )

    # 用全部 model-development set 训练 final model，再只在 independent prediction set 上评价
    print("\n" + "=" * 70)
    print("Final model training on full model-development set")
    print("=" * 70)

    seed_everything(SEED + 9999)
    if STANDARDIZE_WITH_TRAIN_STATS:
        final_mean, final_std = compute_train_mean_std(raw_dataset, dev_idx)
    else:
        final_mean, final_std = None, None

    final_train_dataset = SpectralViewDataset(raw_dataset, augment=True, mean=final_mean, std=final_std)
    final_test_dataset = SpectralViewDataset(raw_dataset, augment=False, mean=final_mean, std=final_std)

    final_train_loader = DataLoader(
        Subset(final_train_dataset, dev_idx.tolist()),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    final_model = CNN1D(N_BANDS, NUM_CLASSES, DROPOUT).to(device)
    final_model, final_history = train_one_fold(
        model=final_model,
        train_loader=final_train_loader,
        device=device,
        fold="final_development",
        save_dir=SAVE_DIR,
        use_amp=use_amp,
    )

    final_log_path = os.path.join(SAVE_DIR, "final_development_train_log.txt")
    with open(final_log_path, "w", encoding="utf-8") as f:
        f.write("epoch\tloss\tmonitor_acc\n")
        for ep, loss, acc in final_history:
            f.write(f"{ep}\t{loss:.6f}\t{acc:.6f}\n")
    plot_training_curve(final_history, "final_development", SAVE_DIR)

    print("\n" + "=" * 70)
    print("Independent Prediction Set Evaluation")
    print("=" * 70)

    y_true_pred, y_pred_pred = predict_no_tta(
        final_model,
        final_test_dataset,
        pred_idx.tolist(),
        device,
        batch_size=64,
    )
    pred_metrics = compute_metrics(y_true_pred, y_pred_pred)

    print(
        f"Prediction set: acc={pred_metrics['acc']:.4f}, "
        f"precision={pred_metrics['precision']:.4f}, "
        f"recall={pred_metrics['recall']:.4f}, "
        f"f1={pred_metrics['f1']:.4f}"
    )
    print(classification_report(
        y_true_pred,
        y_pred_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    ))

    plot_confusion_matrix(
        y_true_pred,
        y_pred_pred,
        CLASS_NAMES,
        os.path.join(SAVE_DIR, "cm_independent_prediction_set_counts.png"),
        title=f"Corrected 1D-CNN Independent Prediction Set (Acc={pred_metrics['acc']:.1%})",
        normalize=False,
    )
    plot_confusion_matrix(
        y_true_pred,
        y_pred_pred,
        CLASS_NAMES,
        os.path.join(SAVE_DIR, "cm_independent_prediction_set_normalized.png"),
        title=f"Corrected 1D-CNN Independent Prediction Set (Acc={pred_metrics['acc']:.1%})",
        normalize=True,
    )
    save_confusion_values(
        y_true_pred,
        y_pred_pred,
        os.path.join(SAVE_DIR, "cm_independent_prediction_set_values.txt"),
    )

    report_path = os.path.join(SAVE_DIR, "results_1d_cnn_holdout_prediction_5fold.txt")
    write_holdout_report(
        report_path,
        fold_metrics,
        all_true_internal,
        all_pred_internal,
        pred_metrics,
        y_true_pred,
        y_pred_pred,
        dev_idx,
        pred_idx,
    )

    print(f"\n[Done] Results saved to: {SAVE_DIR}")
    print(f"[Done] Text report: {report_path}")


if __name__ == "__main__":
    main()
