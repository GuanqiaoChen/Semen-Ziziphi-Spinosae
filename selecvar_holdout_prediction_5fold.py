"""
3D-CNN + Selecvar：开发集 5 折交叉验证 + 独立 prediction set
=========================================================
模型结构与超参数保持不变，只改变数据集划分方式：

1) 先用 stratified hold-out 将全部样本按 3:1 划分为：
   - model-development set: 75%  用于模型开发和内部 5-fold CV
   - independent prediction set: 25%  只用于最终预测评价

2) 在 model-development set 内部进行 StratifiedKFold(n_splits=5)，
   报告内部交叉验证性能。

3) 用全部 model-development set 重新训练 final model，
   再在 independent prediction set 上进行最终测试。

注意：此脚本不修改 HSI3DCNN + Selecvar 模型结构和训练超参数。
运行：python selecvar_holdout_prediction_5fold.py
"""

import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from collections import defaultdict
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.model_selection import StratifiedKFold, train_test_split
from datetime import datetime as _dt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    import scipy.io as sio
    HAS_H5PY = False


# ═══════════════════════════════════════════════
# 0. 配置（与 hsi_3dcnn_selecvar.py 完全一致）
# ═══════════════════════════════════════════════

SEED           = 42
DATA_ROOT      = 'cube'
NUM_CLASSES    = 8
N_BANDS        = 392
PATCH_SIZE     = 32
USE_MASK       = True
N_FOLDS        = 5
PREDICTION_SIZE = 0.25        # 3:1 split; 25% samples are held out as independent prediction set

# ── 训练超参数（照搬原脚本）──
BATCH_SIZE     = 32
EPOCHS         = 360          # warmup(10) + SGDR: 50+100+200
WARMUP_EPOCHS  = 10
LR_MAIN        = 3e-4
LR_SELECVAR    = 1e-3
WEIGHT_DECAY   = 1e-4
LAMBDA_SPARSE  = 5e-5
LAMBDA_SMOOTH  = 1e-5
LABEL_SMOOTH   = 0.1
MIXUP_ALPHA    = 0.3
DROPOUT        = 0.35
SGDR_T0        = 50
SGDR_T_MULT    = 2
# TTA disabled: direct prediction only
TTA_TIMES      = 1

WAVE_START     = 949.764
WAVE_END       = 1650.855
CLASS_NAMES    = ['HBS', 'HBX', 'HNA', 'HNX', 'NX', 'SXD', 'SXQ', 'XJH']

SAVE_DIR = os.path.join('outputs', 'selecvar_holdout_prediction_5fold_no_tta',
                        f'run_{_dt.now().strftime("%Y%m%d_%H%M%S")}')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ═══════════════════════════════════════════════
# 1. 数据集（与原脚本相同）
# ═══════════════════════════════════════════════

class HSIDataset(Dataset):
    def __init__(self, root_dir, use_mask=True, augment=False):
        self.use_mask = use_mask
        self.augment  = augment
        self.samples  = []

        for folder in sorted(glob.glob(os.path.join(root_dir, '*-*'))):
            try:
                label = int(os.path.basename(folder).split('-')[0])
            except ValueError:
                continue
            for p in sorted(glob.glob(os.path.join(folder, '*.mat'))):
                self.samples.append((p, label))

    def _load(self, path):
        if HAS_H5PY:
            with h5py.File(path, 'r') as f:
                data = f['patch_chw'][()].astype(np.float32)
                mask = f['crop_mask'][()].astype(np.float32) if self.use_mask else None
        else:
            mat  = sio.loadmat(path)
            data = mat['patch_chw'].astype(np.float32)
            mask = mat['crop_mask'].astype(np.float32) if self.use_mask else None
        return data, mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data, mask  = self._load(path)

        if self.use_mask and mask is not None:
            data = data * mask[:, :, np.newaxis]
            data = np.clip(data, 0., 1.)

        # HWλ → (1, λ, H, W)
        data   = np.transpose(data, (2, 0, 1))[np.newaxis]
        tensor = torch.from_numpy(data)

        if self.augment:
            if torch.rand(1) > 0.5:
                tensor = torch.flip(tensor, dims=[2])
            if torch.rand(1) > 0.5:
                tensor = torch.flip(tensor, dims=[3])
            scale  = 0.9 + 0.2 * torch.rand(1).item()
            tensor = torch.clamp(tensor * scale, 0., 1.)

        return tensor, label


# ═══════════════════════════════════════════════
# 2. 模型（与 hsi_3dcnn_selecvar.py 完全一致）
# ═══════════════════════════════════════════════

class Selecvar(nn.Module):
    def __init__(self, n_bands, lambda_sparse=5e-5, lambda_smooth=1e-5):
        super().__init__()
        self.n_bands       = n_bands
        self.lambda_sparse = lambda_sparse
        self.lambda_smooth = lambda_smooth
        self.weight        = nn.Parameter(torch.randn(n_bands) * 0.01)

    def forward(self, x):
        w = F.softplus(self.weight).view(1, 1, self.n_bands, 1, 1)
        return x * w

    def hoyer_loss(self):
        w = F.softplus(self.weight)
        return self.lambda_sparse * w.sum() / (w.pow(2).sum().sqrt() + 1e-8)

    def smoothness_loss(self):
        w    = F.softplus(self.weight)
        diff = w[1:] - w[:-1]
        return self.lambda_smooth * diff.pow(2).sum()

    def regularization_loss(self):
        return self.hoyer_loss() + self.smoothness_loss()

    def get_weights(self):
        w = F.softplus(self.weight).detach().cpu().numpy()
        return w / (w.max() + 1e-8)


class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch,
                 kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)):
        super().__init__()
        self.conv1    = nn.Conv3d(in_ch, out_ch, kernel_size,
                                  stride=stride, padding=padding, bias=False)
        self.bn1      = nn.BatchNorm3d(out_ch)
        self.act      = nn.GELU()
        self.conv2    = nn.Conv3d(out_ch, out_ch, kernel_size,
                                  stride=1, padding=padding, bias=False)
        self.bn2      = nn.BatchNorm3d(out_ch)
        strides       = stride if isinstance(stride, (list,tuple)) else [stride]*3
        need_proj     = (in_ch != out_ch) or any(s != 1 for s in strides)
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm3d(out_ch),
        ) if need_proj else nn.Identity()

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + self.shortcut(x))


class HSI3DCNN(nn.Module):
    def __init__(self, num_classes=8, n_bands=392,
                 lambda_sparse=5e-5, lambda_smooth=1e-5, dropout=0.35):
        super().__init__()
        self.selecvar = Selecvar(n_bands, lambda_sparse, lambda_smooth)
        self.block1   = ResBlock3D(1,  16,  (11,3,3), (2,1,1), (5,1,1))
        self.pool1    = nn.MaxPool3d((2,1,1))
        self.block2   = ResBlock3D(16, 32,  (7,3,3),  (2,1,1), (3,1,1))
        self.pool2    = nn.MaxPool3d((2,1,1))
        self.block3   = ResBlock3D(32, 64,  (3,3,3),  (1,1,1), (1,1,1))
        self.pool3    = nn.MaxPool3d((2,2,2))
        self.block4   = ResBlock3D(64, 128, (3,3,3),  (1,1,1), (1,1,1))
        self.gap      = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.selecvar(x)
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.block4(x)
        return self.classifier(self.gap(x))

    def spectral_importance(self):
        return self.selecvar.get_weights()


# ═══════════════════════════════════════════════
# 3. 训练工具
# ═══════════════════════════════════════════════

def build_optimizers(model):
    """双优化器：主干 LR_MAIN，Selecvar LR_SELECVAR（照搬原脚本）"""
    selecvar_params = list(model.selecvar.parameters())
    selecvar_ids    = {id(p) for p in selecvar_params}
    main_params     = [p for p in model.parameters() if id(p) not in selecvar_ids]
    optimizer = torch.optim.AdamW([
        {'params': main_params,     'lr': LR_MAIN},
        {'params': selecvar_params, 'lr': LR_SELECVAR},
    ], weight_decay=WEIGHT_DECAY)
    return optimizer


def build_schedulers(optimizer):
    warmup = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda ep: (ep+1)/WARMUP_EPOCHS if ep < WARMUP_EPOCHS else 1.0)
    sgdr   = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=SGDR_T0, T_mult=SGDR_T_MULT,
        eta_min=LR_MAIN * 0.01)
    return warmup, sgdr


def mixup_batch(x, y, alpha, device):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=device)
    return lam*x + (1-lam)*x[idx], y, y[idx], lam


def train_one_fold(model, train_loader, device, fold, save_dir):
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimizer = build_optimizers(model)
    warmup, sgdr = build_schedulers(optimizer)
    log = []

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss, correct, n = 0., 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if MIXUP_ALPHA > 0 and epoch > WARMUP_EPOCHS:
                x, ya, yb, lam = mixup_batch(x, y, MIXUP_ALPHA, device)
                optimizer.zero_grad()
                logits = model(x)
                loss   = (lam * criterion(logits, ya)
                          + (1-lam) * criterion(logits, yb)
                          + model.selecvar.regularization_loss())
                pred_y = ya if lam >= 0.5 else yb
            else:
                optimizer.zero_grad()
                logits = model(x)
                loss   = (criterion(logits, y)
                          + model.selecvar.regularization_loss())
                pred_y = y

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct    += (logits.argmax(1) == pred_y).sum().item()
            n          += len(y)

        if epoch <= WARMUP_EPOCHS:
            warmup.step()
        else:
            sgdr.step()

        log.append((epoch, total_loss/n, correct/n))
        if epoch % 60 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{EPOCHS}  "
                  f"loss={total_loss/n:.4f}  train={correct/n:.3f}")

    tag = f'fold{fold}' if isinstance(fold, int) else str(fold)
    torch.save(model.state_dict(), os.path.join(save_dir, f'{tag}_final.pth'))
    return model, log


# ═══════════════════════════════════════════════
# 4. TTA（与无 Selecvar 版本相同）
# ═══════════════════════════════════════════════

def predict_with_tta(model, dataset, indices, device,
                     tta_times=TTA_TIMES, batch_size=32):
    model.eval()
    aug_ds  = HSIDataset(DATA_ROOT, use_mask=USE_MASK, augment=True)
    base_ds = HSIDataset(DATA_ROOT, use_mask=USE_MASK, augment=False)
    all_true  = [dataset.samples[i][1] for i in indices]
    probs_sum = np.zeros((len(indices), NUM_CLASSES))

    for t in range(tta_times):
        ds     = Subset(aug_ds if t > 0 else base_ds, indices)
        loader = DataLoader(ds, batch_size=batch_size,
                            shuffle=False, num_workers=4)
        probs  = []
        with torch.no_grad():
            for x, _ in loader:
                probs.append(
                    F.softmax(model(x.to(device)), dim=1).cpu().numpy())
        probs_sum += np.vstack(probs)

    return np.array(all_true), probs_sum.argmax(axis=1)


def predict_no_tta(model, dataset, indices, device, batch_size=32):
    """
    Direct prediction without test-time augmentation.
    indices 可以是内部 CV 的 held-out fold，也可以是最终 independent prediction set。
    """
    model.eval()
    all_true = [dataset.samples[int(i)][1] for i in indices]
    preds = []

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            logits = model(x)
            preds.extend(logits.argmax(1).cpu().numpy().tolist())

    return np.array(all_true), np.array(preds)


# ═══════════════════════════════════════════════
# 5. 可视化
# ═══════════════════════════════════════════════

def plot_confusion_matrix(cm, title, save_path):
    n      = cm.shape[0]
    thresh = cm.max() / 2.
    fig, ax = plt.subplots(figsize=(9, 7.5), facecolor='white')
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(CLASS_NAMES, fontsize=11)
    ax.set_xlabel('Predicted label', fontsize=12, labelpad=10)
    ax.set_ylabel('True label',      fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=13, pad=14)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                    fontsize=10,
                    color='white' if cm[i,j] > thresh else 'black')
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()



def plot_training_curve(log, fold, save_dir):
    """
    Plot training loss and training accuracy curves for one fold.
    log format: [(epoch, train_loss, train_acc), ...]
    """
    epochs = [x[0] for x in log]
    train_loss = [x[1] for x in log]
    train_acc = [x[2] for x in log]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='white')

    axes[0].plot(epochs, train_loss, linewidth=1.8, label='Train loss')
    axes[0].set_title(f'Fold {fold} Training Loss', fontsize=13)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, linewidth=1.8, label='Train accuracy')
    axes[1].set_title(f'Fold {fold} Training Accuracy', fontsize=13)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    combined_path = os.path.join(save_dir, f'training_curve_fold{fold}.png')
    plt.savefig(combined_path, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()

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

    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    ax.plot(epochs, train_acc, linewidth=1.8, label='Train accuracy')
    ax.set_title(f'Fold {fold} Training Accuracy Curve', fontsize=13)
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


def plot_overall_training_curves(all_logs, save_dir):
    """
    Plot overall mean ± SD training curves across all 5 folds.
    all_logs format:
        [
            [(epoch, train_loss, train_acc), ...],
            ...
        ]
    """
    if len(all_logs) == 0:
        return

    min_len = min(len(log) for log in all_logs)
    logs = [log[:min_len] for log in all_logs]

    epochs = np.array([x[0] for x in logs[0]])
    loss_arr = np.array([[x[1] for x in log] for log in logs], dtype=np.float64)
    acc_arr = np.array([[x[2] for x in log] for log in logs], dtype=np.float64)

    loss_mean = loss_arr.mean(axis=0)
    loss_std = loss_arr.std(axis=0)
    acc_mean = acc_arr.mean(axis=0)
    acc_std = acc_arr.std(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='white')

    axes[0].plot(epochs, loss_mean, linewidth=1.8, label='Mean train loss')
    axes[0].fill_between(epochs, loss_mean - loss_std, loss_mean + loss_std,
                         alpha=0.2, label='±1 SD')
    axes[0].set_title('Overall Training Loss Across 5 Folds', fontsize=13)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, acc_mean, linewidth=1.8, label='Mean train accuracy')
    axes[1].fill_between(epochs, acc_mean - acc_std, acc_mean + acc_std,
                         alpha=0.2, label='±1 SD')
    axes[1].set_title('Overall Training Accuracy Across 5 Folds', fontsize=13)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    combined_path = os.path.join(save_dir, 'overall_training_curves_5fold.png')
    plt.savefig(combined_path, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()

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

    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    ax.plot(epochs, acc_mean, linewidth=1.8, label='Mean train accuracy')
    ax.fill_between(epochs, acc_mean - acc_std, acc_mean + acc_std,
                    alpha=0.2, label='±1 SD')
    ax.set_title('Overall Training Accuracy Across 5 Folds', fontsize=13)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    acc_path = os.path.join(save_dir, 'overall_train_accuracy_curve_5fold.png')
    plt.savefig(acc_path, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()

    values_path = os.path.join(save_dir, 'overall_training_curves_5fold_values.txt')
    with open(values_path, 'w', encoding='utf-8') as f:
        f.write('epoch\ttrain_loss_mean\ttrain_loss_sd\ttrain_acc_mean\ttrain_acc_sd\n')
        for ep, lm, ls, am, astd in zip(epochs, loss_mean, loss_std, acc_mean, acc_std):
            f.write(f'{int(ep)}\t{lm:.6f}\t{ls:.6f}\t{am:.6f}\t{astd:.6f}\n')

    print(f'[OK] Overall 5-fold training curves saved → {combined_path}')



def plot_selecvar_avg(all_weights, save_path):
    """5折 Selecvar 权重均值曲线。"""
    wavelengths = np.linspace(WAVE_START, WAVE_END, N_BANDS)
    weights_arr = np.array(all_weights)     # (5, 392)
    mean_w      = weights_arr.mean(axis=0)
    std_w       = weights_arr.std(axis=0)

    fig, ax = plt.subplots(figsize=(12, 4), facecolor='white')
    ax.fill_between(wavelengths, mean_w - std_w, mean_w + std_w,
                    alpha=0.2, color='#1A5FA8')
    ax.plot(wavelengths, mean_w, color='#1A5FA8', linewidth=1.8,
            label='Mean band importance (5-fold)')
    for i, w in enumerate(weights_arr):
        ax.plot(wavelengths, w, color='#1A5FA8', linewidth=0.5,
                alpha=0.3)

    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Normalized importance', fontsize=12)
    ax.set_title('Selecvar Band Importance — 5-fold Average', fontsize=13)
    ax.set_xlim(WAVE_START, WAVE_END)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2)
    for sp in ax.spines.values(): sp.set_linewidth(0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[OK] Selecvar 均值曲线 → {save_path}')



def save_split_info(dataset, dev_idx, pred_idx, save_dir):
    """Save model-development / prediction-set indices and class distribution."""
    def _count(indices):
        counts = {name: 0 for name in CLASS_NAMES}
        for i in indices:
            label = int(dataset.samples[int(i)][1])
            counts[CLASS_NAMES[label]] += 1
        return counts

    os.makedirs(save_dir, exist_ok=True)
    dev_idx = np.array(dev_idx, dtype=int)
    pred_idx = np.array(pred_idx, dtype=int)
    np.savez(os.path.join(save_dir, 'dataset_split_indices.npz'),
             development_indices=dev_idx,
             prediction_indices=pred_idx)

    path = os.path.join(save_dir, 'dataset_split_summary.txt')
    dev_counts = _count(dev_idx)
    pred_counts = _count(pred_idx)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('Dataset split summary\n')
        f.write(f'Seed={SEED}\n')
        f.write(f'Prediction set ratio={PREDICTION_SIZE:.2f}\n')
        f.write(f'Model-development set size={len(dev_idx)}\n')
        f.write(f'Independent prediction set size={len(pred_idx)}\n\n')
        f.write('Class\tDevelopment\tPrediction\tTotal\n')
        for name in CLASS_NAMES:
            f.write(f'{name}\t{dev_counts[name]}\t{pred_counts[name]}\t{dev_counts[name] + pred_counts[name]}\n')
    print(f'[OK] Dataset split summary saved → {path}')


# ═══════════════════════════════════════════════
# 6. 主流程
# ═══════════════════════════════════════════════

def main():
    seed_everything(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'[Config] device={device}  seed={SEED}  folds={N_FOLDS}')
    print(f'[Config] split=3:1  prediction_size={PREDICTION_SIZE:.2f}')
    print(f'[Config] epochs={EPOCHS}  lr_main={LR_MAIN}  lr_selecvar={LR_SELECVAR}')
    print(f'[Config] SGDR T0={SGDR_T0} T_mult={SGDR_T_MULT}  TTA=disabled')

    # dummy forward 确认结构：模型结构不变
    with torch.no_grad():
        m_tmp  = HSI3DCNN(NUM_CLASSES, N_BANDS, LAMBDA_SPARSE,
                           LAMBDA_SMOOTH, DROPOUT).to(device)
        dummy  = torch.zeros(2, 1, N_BANDS, PATCH_SIZE, PATCH_SIZE, device=device)
        print(f'[Model] dummy forward → {m_tmp(dummy).shape}')
        params = sum(p.numel() for p in m_tmp.parameters() if p.requires_grad)
        print(f'[Model] 可训练参数：{params:,}')
        del m_tmp

    # 数据
    base_ds = HSIDataset(DATA_ROOT, use_mask=USE_MASK, augment=False)
    all_idx = np.arange(len(base_ds))
    all_lbl = np.array([base_ds.samples[i][1] for i in all_idx])
    print(f'[Data] 共 {len(base_ds)} 个样本，{NUM_CLASSES} 类')

    # 第一步：先划出 independent prediction set；该集合不参与内部 5-fold CV 和调参
    dev_idx, pred_idx, dev_lbl, pred_lbl = train_test_split(
        all_idx,
        all_lbl,
        test_size=PREDICTION_SIZE,
        random_state=SEED,
        shuffle=True,
        stratify=all_lbl
    )
    dev_idx = np.array(dev_idx, dtype=int)
    pred_idx = np.array(pred_idx, dtype=int)
    dev_lbl = np.array(dev_lbl, dtype=int)
    pred_lbl = np.array(pred_lbl, dtype=int)

    print(f'[Split] Model-development set: {len(dev_idx)} samples')
    print(f'[Split] Independent prediction set: {len(pred_idx)} samples')
    save_split_info(base_ds, dev_idx, pred_idx, SAVE_DIR)

    # 第二步：只在 development set 内部做 5-fold CV
    skf             = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                      random_state=SEED)
    fold_metrics    = []
    all_logs        = []
    all_true_global = []
    all_pred_global = []
    all_weights     = []   # 每折 Selecvar 权重，用于最终均值曲线

    for fold, (train_pos, test_pos) in enumerate(
            skf.split(dev_idx, dev_lbl), start=1):

        train_idx = dev_idx[train_pos]
        test_idx  = dev_idx[test_pos]

        print(f'\n{"="*55}')
        print(f'  Internal Fold {fold}/{N_FOLDS}  train={len(train_idx)}  test={len(test_idx)}')
        print(f'{"="*55}')

        aug_ds  = HSIDataset(DATA_ROOT, use_mask=USE_MASK, augment=True)
        train_loader = DataLoader(
            Subset(aug_ds, train_idx.tolist()),
            batch_size=BATCH_SIZE, shuffle=True,
            num_workers=4, pin_memory=True)

        # 每折独立种子（模型结构和超参数不变）
        seed_everything(SEED + fold * 100)
        model = HSI3DCNN(NUM_CLASSES, N_BANDS, LAMBDA_SPARSE,
                         LAMBDA_SMOOTH, DROPOUT).to(device)

        model, log = train_one_fold(model, train_loader, device, fold, SAVE_DIR)
        all_logs.append(log)

        # 保存每折训练日志和训练曲线
        train_log_path = os.path.join(SAVE_DIR, f'fold{fold}_train_log.txt')
        with open(train_log_path, 'w', encoding='utf-8') as f:
            f.write('epoch\ttrain_loss\ttrain_acc\n')
            for ep, loss, acc in log:
                f.write(f'{ep}\t{loss:.6f}\t{acc:.6f}\n')
        plot_training_curve(log, fold, SAVE_DIR)

        # 保存本折 Selecvar 权重
        all_weights.append(model.spectral_importance())

        # No-TTA 内部折测试
        y_true, y_pred = predict_no_tta(
            model, base_ds, test_idx.tolist(), device)
        all_true_global.extend(y_true.tolist())
        all_pred_global.extend(y_pred.tolist())

        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec  = recall_score(   y_true, y_pred, average='macro', zero_division=0)
        f1   = f1_score(       y_true, y_pred, average='macro', zero_division=0)
        cm   = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
        fold_metrics.append({'acc':acc, 'precision':prec, 'recall':rec, 'f1':f1})

        print(f'  Internal Fold {fold} test  acc={acc:.4f}  '
              f'prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}')

        plot_confusion_matrix(
            cm, f'3D-CNN+Selecvar Internal Fold {fold} (Acc={acc:.1%})',
            os.path.join(SAVE_DIR, f'cm_internal_fold{fold}.png'))

    # 绘制 development-set 5 折总体训练曲线（mean ± SD）
    plot_overall_training_curves(all_logs, SAVE_DIR)

    # ── 内部 5-fold CV 汇总 ──
    print(f'\n{"="*55}')
    print('Internal 5-Fold Summary on Model-Development Set — 3D-CNN + Selecvar')
    print(f'{"="*55}')

    accs  = [m['acc']       for m in fold_metrics]
    precs = [m['precision'] for m in fold_metrics]
    recs  = [m['recall']    for m in fold_metrics]
    f1s   = [m['f1']        for m in fold_metrics]

    for i, m in enumerate(fold_metrics, 1):
        print(f'  Fold {i}:  acc={m["acc"]:.4f}  prec={m["precision"]:.4f}  '
              f'rec={m["recall"]:.4f}  f1={m["f1"]:.4f}')

    print(f'\n  内部 5-fold 均值 ± 标准差：')
    print(f'  Accuracy  : {np.mean(accs):.4f} ± {np.std(accs):.4f}')
    print(f'  Precision : {np.mean(precs):.4f} ± {np.std(precs):.4f}')
    print(f'  Recall    : {np.mean(recs):.4f} ± {np.std(recs):.4f}')
    print(f'  F1        : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}')

    y_true_all = np.array(all_true_global)
    y_pred_all = np.array(all_pred_global)
    cm_all     = confusion_matrix(y_true_all, y_pred_all,
                                  labels=list(range(NUM_CLASSES)))
    acc_all    = accuracy_score(y_true_all, y_pred_all)
    print(f'\n  内部 5-fold 合并准确率: {acc_all:.4f}')
    print(classification_report(y_true_all, y_pred_all,
                                 target_names=CLASS_NAMES, digits=4))

    plot_confusion_matrix(
        cm_all,
        f'3D-CNN+Selecvar Internal 5-fold Overall (Acc={acc_all:.1%})',
        os.path.join(SAVE_DIR, 'cm_internal_5fold_overall.png'))

    # Selecvar 均值重要性曲线：只来自内部 5-fold 模型
    plot_selecvar_avg(
        all_weights,
        os.path.join(SAVE_DIR, 'selecvar_importance_avg_internal_5fold.png'))

    np.save(os.path.join(SAVE_DIR, 'selecvar_weights_internal_5fold.npy'),
            np.array(all_weights))
    print(f'[OK] Selecvar 权重已保存（internal 5-fold）→ '
          f'{SAVE_DIR}/selecvar_weights_internal_5fold.npy')

    # 第三步：用全部 development set 训练 final model，再只在 independent prediction set 上最终评价
    print(f'\n{"="*55}')
    print('Final model training on full model-development set')
    print(f'{"="*55}')

    aug_ds = HSIDataset(DATA_ROOT, use_mask=USE_MASK, augment=True)
    final_train_loader = DataLoader(
        Subset(aug_ds, dev_idx.tolist()),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True)

    seed_everything(SEED + 9999)
    final_model = HSI3DCNN(NUM_CLASSES, N_BANDS, LAMBDA_SPARSE,
                           LAMBDA_SMOOTH, DROPOUT).to(device)
    final_model, final_log = train_one_fold(final_model, final_train_loader,
                                            device, 'final_development', SAVE_DIR)

    final_log_path = os.path.join(SAVE_DIR, 'final_development_train_log.txt')
    with open(final_log_path, 'w', encoding='utf-8') as f:
        f.write('epoch\ttrain_loss\ttrain_acc\n')
        for ep, loss, acc in final_log:
            f.write(f'{ep}\t{loss:.6f}\t{acc:.6f}\n')
    plot_training_curve(final_log, 'final_development', SAVE_DIR)

    print(f'\n{"="*55}')
    print('Independent Prediction Set Evaluation')
    print(f'{"="*55}')

    y_true_pred, y_pred_pred = predict_no_tta(
        final_model, base_ds, pred_idx.tolist(), device)

    pred_acc  = accuracy_score(y_true_pred, y_pred_pred)
    pred_prec = precision_score(y_true_pred, y_pred_pred, average='macro', zero_division=0)
    pred_rec  = recall_score(   y_true_pred, y_pred_pred, average='macro', zero_division=0)
    pred_f1   = f1_score(       y_true_pred, y_pred_pred, average='macro', zero_division=0)
    pred_cm   = confusion_matrix(y_true_pred, y_pred_pred,
                                 labels=list(range(NUM_CLASSES)))

    print(f'  Prediction set  acc={pred_acc:.4f}  '
          f'prec={pred_prec:.4f}  rec={pred_rec:.4f}  f1={pred_f1:.4f}')
    print(classification_report(y_true_pred, y_pred_pred,
                                 target_names=CLASS_NAMES, digits=4))

    plot_confusion_matrix(
        pred_cm,
        f'3D-CNN+Selecvar Independent Prediction Set (Acc={pred_acc:.1%})',
        os.path.join(SAVE_DIR, 'cm_independent_prediction_set.png'))

    # 保存最终模型 Selecvar 权重
    final_weight_path = os.path.join(SAVE_DIR, 'selecvar_weights_final_development.npy')
    np.save(final_weight_path, final_model.spectral_importance())
    print(f'[OK] Final model Selecvar 权重 → {final_weight_path}')

    # 文本报告
    report_path = os.path.join(SAVE_DIR, 'results_selecvar_holdout_prediction_5fold.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('3D-CNN + Selecvar: Model-development 5-fold CV + Independent Prediction Set\n')
        f.write(f'Seed={SEED}  Folds={N_FOLDS}  Epochs={EPOCHS}  TTA=disabled\n')
        f.write(f'Split: model-development={(1-PREDICTION_SIZE):.2f}, prediction={PREDICTION_SIZE:.2f}\n')
        f.write(f'Development samples={len(dev_idx)}  Prediction samples={len(pred_idx)}\n')
        f.write(f'LR_main={LR_MAIN}  LR_selecvar={LR_SELECVAR}  dropout={DROPOUT}\n')
        f.write(f'SGDR T0={SGDR_T0} T_mult={SGDR_T_MULT}\n')
        f.write('='*70 + '\n\nInternal 5-fold CV on model-development set:\n')
        for i, m in enumerate(fold_metrics, 1):
            f.write(f'  Fold {i}: acc={m["acc"]:.4f}  '
                    f'prec={m["precision"]:.4f}  '
                    f'rec={m["recall"]:.4f}  '
                    f'f1={m["f1"]:.4f}\n')
        f.write(f'\nInternal 5-fold Mean ± Std:\n'
                f'  Accuracy  : {np.mean(accs):.4f} ± {np.std(accs):.4f}\n'
                f'  Precision : {np.mean(precs):.4f} ± {np.std(precs):.4f}\n'
                f'  Recall    : {np.mean(recs):.4f} ± {np.std(recs):.4f}\n'
                f'  F1        : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}\n'
                f'\nInternal 5-fold overall accuracy (folds combined): {acc_all:.4f}\n\n')
        f.write('Internal 5-fold overall per-class report:\n')
        f.write(classification_report(y_true_all, y_pred_all,
                                       target_names=CLASS_NAMES, digits=4))
        f.write('\n\nIndependent prediction set results:\n')
        f.write(f'  Accuracy  : {pred_acc:.4f}\n')
        f.write(f'  Precision : {pred_prec:.4f}\n')
        f.write(f'  Recall    : {pred_rec:.4f}\n')
        f.write(f'  F1        : {pred_f1:.4f}\n\n')
        f.write('Independent prediction set per-class report:\n')
        f.write(classification_report(y_true_pred, y_pred_pred,
                                       target_names=CLASS_NAMES, digits=4))
    print(f'[Done] 报告 → {report_path}')


if __name__ == '__main__':
    main()
