"""
3D-CNN without SelecVar：开发集 5 折交叉验证 + 独立 prediction set
=================================================================
用途：
1) 与 3D-CNN + SelecVar 版本使用完全相同的数据划分方式：
   - 先按 3:1 进行 stratified hold-out split
   - model-development set: 75%，用于模型开发和内部 5-fold CV
   - independent prediction set: 25%，只用于最终预测评价
2) 只去掉 SelecVar wavelength reweighting module；其余 3D-CNN 主干结构、
   训练超参数、数据增强、Mixup、label smoothing、SGDR 等保持一致。
3) 输出内部 5-fold CV 结果、最终 independent prediction set 结果、混淆矩阵、
   训练曲线、数据划分摘要和完整文本报告。

运行：python no_selecvar_holdout_prediction_5fold.py
"""

import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
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
# 0. 配置：与 SelecVar hold-out 版本保持一致
# ═══════════════════════════════════════════════

SEED            = 42
DATA_ROOT       = 'cube'
NUM_CLASSES     = 8
N_BANDS         = 392
PATCH_SIZE      = 32
USE_MASK        = True
N_FOLDS         = 5
PREDICTION_SIZE = 0.25       # 3:1 split; 25% independent prediction set

# ── 训练超参数：与 SelecVar 版本主干训练保持一致 ──
BATCH_SIZE      = 32
EPOCHS          = 360        # warmup(10) + SGDR: 50+100+200
WARMUP_EPOCHS   = 10
LR_MAIN         = 3e-4
WEIGHT_DECAY    = 1e-4
LABEL_SMOOTH    = 0.1
MIXUP_ALPHA     = 0.3
DROPOUT         = 0.35
SGDR_T0         = 50
SGDR_T_MULT     = 2
TTA_TIMES       = 1          # no TTA

CLASS_NAMES     = ['HBS', 'HBX', 'HNA', 'HNX', 'NX', 'SXD', 'SXQ', 'XJH']

SAVE_DIR = os.path.join(
    'outputs',
    'no_selecvar_holdout_prediction_5fold_no_tta',
    f'run_{_dt.now().strftime("%Y%m%d_%H%M%S")}'
)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════
# 1. 数据集：与原脚本相同
# ═══════════════════════════════════════════════

class HSIDataset(Dataset):
    def __init__(self, root_dir, use_mask=True, augment=False):
        self.use_mask = use_mask
        self.augment = augment
        self.samples = []

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
            mat = sio.loadmat(path)
            data = mat['patch_chw'].astype(np.float32)
            mask = mat['crop_mask'].astype(np.float32) if self.use_mask else None
        return data, mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data, mask = self._load(path)

        if self.use_mask and mask is not None:
            data = data * mask[:, :, np.newaxis]
            data = np.clip(data, 0., 1.)

        # H W λ → (1, λ, H, W)
        data = np.transpose(data, (2, 0, 1))[np.newaxis]
        tensor = torch.from_numpy(data)

        if self.augment:
            if torch.rand(1) > 0.5:
                tensor = torch.flip(tensor, dims=[2])
            if torch.rand(1) > 0.5:
                tensor = torch.flip(tensor, dims=[3])
            scale = 0.9 + 0.2 * torch.rand(1).item()
            tensor = torch.clamp(tensor * scale, 0., 1.)

        return tensor, label


# ═══════════════════════════════════════════════
# 2. 模型：3D-CNN without SelecVar
# ═══════════════════════════════════════════════

class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch,
                 kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.act = nn.GELU()
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size,
                               stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        strides = stride if isinstance(stride, (list, tuple)) else [stride] * 3
        need_proj = (in_ch != out_ch) or any(s != 1 for s in strides)
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm3d(out_ch),
        ) if need_proj else nn.Identity()

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + self.shortcut(x))


class HSI3DCNN_NoSelecVar(nn.Module):
    """3D-CNN backbone without SelecVar wavelength reweighting module."""
    def __init__(self, num_classes=8, dropout=0.35):
        super().__init__()
        # No SelecVar here: input is directly passed into block1.
        self.block1 = ResBlock3D(1, 16, (11, 3, 3), (2, 1, 1), (5, 1, 1))
        self.pool1 = nn.MaxPool3d((2, 1, 1))
        self.block2 = ResBlock3D(16, 32, (7, 3, 3), (2, 1, 1), (3, 1, 1))
        self.pool2 = nn.MaxPool3d((2, 1, 1))
        self.block3 = ResBlock3D(32, 64, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((2, 2, 2))
        self.block4 = ResBlock3D(64, 128, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.block4(x)
        return self.classifier(self.gap(x))


# ═══════════════════════════════════════════════
# 3. 训练工具
# ═══════════════════════════════════════════════

def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=LR_MAIN, weight_decay=WEIGHT_DECAY)


def build_schedulers(optimizer):
    warmup = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda ep: (ep + 1) / WARMUP_EPOCHS if ep < WARMUP_EPOCHS else 1.0
    )
    sgdr = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=SGDR_T0, T_mult=SGDR_T_MULT, eta_min=LR_MAIN * 0.01
    )
    return warmup, sgdr


def mixup_batch(x, y, alpha, device):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def train_model(model, train_loader, device, tag, save_dir):
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimizer = build_optimizer(model)
    warmup, sgdr = build_schedulers(optimizer)
    log = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, n = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            if MIXUP_ALPHA > 0 and epoch > WARMUP_EPOCHS:
                x_mix, ya, yb, lam = mixup_batch(x, y, MIXUP_ALPHA, device)
                logits = model(x_mix)
                loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
                # This is only a monitoring accuracy under Mixup, not final evaluation accuracy.
                pred_y = ya if lam >= 0.5 else yb
            else:
                logits = model(x)
                loss = criterion(logits, y)
                pred_y = y

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == pred_y).sum().item()
            n += len(y)

        if epoch <= WARMUP_EPOCHS:
            warmup.step()
        else:
            sgdr.step()

        log.append((epoch, total_loss / n, correct / n))
        if epoch % 60 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{EPOCHS}  loss={total_loss/n:.4f}  train_monitor={correct/n:.3f}")

    torch.save(model.state_dict(), os.path.join(save_dir, f'{tag}_final.pth'))
    return model, log


# ═══════════════════════════════════════════════
# 4. 预测与评估
# ═══════════════════════════════════════════════

def predict_no_tta(model, dataset, indices, device, batch_size=32):
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
            logits = model(x.to(device))
            preds.extend(logits.argmax(1).cpu().numpy().tolist())

    return np.array(all_true), np.array(preds)


def calc_metrics(y_true, y_pred):
    return {
        'acc': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
    }


# ═══════════════════════════════════════════════
# 5. 可视化与保存工具
# ═══════════════════════════════════════════════

def plot_confusion_matrix(cm, title, save_path):
    n = cm.shape[0]
    thresh = cm.max() / 2.0
    fig, ax = plt.subplots(figsize=(9, 7.5), facecolor='white')
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(CLASS_NAMES, fontsize=11)
    ax.set_xlabel('Predicted label', fontsize=12, labelpad=10)
    ax.set_ylabel('True label', fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=13, pad=14)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=10,
                    color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_training_curve(log, tag, save_dir):
    epochs = [x[0] for x in log]
    train_loss = [x[1] for x in log]
    train_acc = [x[2] for x in log]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='white')
    axes[0].plot(epochs, train_loss, linewidth=1.8, label='Train loss')
    axes[0].set_title(f'{tag} Training Loss', fontsize=13)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, linewidth=1.8, label='Train monitor accuracy')
    axes[1].set_title(f'{tag} Training Monitor Accuracy', fontsize=13)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    combined_path = os.path.join(save_dir, f'training_curve_{tag}.png')
    plt.savefig(combined_path, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    ax.plot(epochs, train_loss, linewidth=1.8, label='Train loss')
    ax.set_title(f'{tag} Training Loss Curve', fontsize=13)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{tag}_train_loss_curve.png'),
                dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    ax.plot(epochs, train_acc, linewidth=1.8, label='Train monitor accuracy')
    ax.set_title(f'{tag} Training Monitor Accuracy Curve', fontsize=13)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{tag}_train_monitor_accuracy_curve.png'),
                dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_overall_training_curves(all_logs, save_dir):
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

    values_path = os.path.join(save_dir, 'overall_training_curves_5fold_values.txt')
    with open(values_path, 'w', encoding='utf-8') as f:
        f.write('epoch\ttrain_loss_mean\ttrain_loss_sd\ttrain_monitor_acc_mean\ttrain_monitor_acc_sd\n')
        for ep, lm, ls, am, astd in zip(epochs, loss_mean, loss_std, acc_mean, acc_std):
            f.write(f'{int(ep)}\t{lm:.6f}\t{ls:.6f}\t{am:.6f}\t{astd:.6f}\n')


def save_training_log(log, tag, save_dir):
    path = os.path.join(save_dir, f'{tag}_train_log.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('epoch\ttrain_loss\ttrain_monitor_acc\n')
        for ep, loss, acc in log:
            f.write(f'{ep}\t{loss:.6f}\t{acc:.6f}\n')


def save_split_summary(dataset, dev_idx, pred_idx, save_dir):
    dev_set = set(int(i) for i in dev_idx)
    pred_set = set(int(i) for i in pred_idx)
    path = os.path.join(save_dir, 'dataset_split_summary.txt')

    with open(path, 'w', encoding='utf-8') as f:
        f.write('Dataset split summary for 3D-CNN without SelecVar\n')
        f.write(f'Seed={SEED}; split=3:1; prediction_size={PREDICTION_SIZE}\n')
        f.write('Origin\tTotal\tModel-development set\tIndependent prediction set\n')
        for cls_id, cls_name in enumerate(CLASS_NAMES):
            all_cls = [i for i, (_, y) in enumerate(dataset.samples) if y == cls_id]
            dev_n = sum(i in dev_set for i in all_cls)
            pred_n = sum(i in pred_set for i in all_cls)
            f.write(f'{cls_name}\t{len(all_cls)}\t{dev_n}\t{pred_n}\n')
        f.write(f'Total\t{len(dataset)}\t{len(dev_idx)}\t{len(pred_idx)}\n')

    print(f'[OK] Dataset split summary saved → {path}')


# ═══════════════════════════════════════════════
# 6. 主流程
# ═══════════════════════════════════════════════

def main():
    seed_everything(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'[Config] device={device}  seed={SEED}  folds={N_FOLDS}')
    print(f'[Config] split=3:1  prediction_size={PREDICTION_SIZE}')
    print(f'[Config] epochs={EPOCHS}  lr_main={LR_MAIN}')
    print(f'[Config] SGDR T0={SGDR_T0} T_mult={SGDR_T_MULT}  TTA=disabled')
    print('[Model] 3D-CNN without SelecVar')

    with torch.no_grad():
        m_tmp = HSI3DCNN_NoSelecVar(NUM_CLASSES, DROPOUT).to(device)
        dummy = torch.zeros(2, 1, N_BANDS, PATCH_SIZE, PATCH_SIZE, device=device)
        print(f'[Model] dummy forward → {m_tmp(dummy).shape}')
        params = sum(p.numel() for p in m_tmp.parameters() if p.requires_grad)
        print(f'[Model] 可训练参数：{params:,}')
        del m_tmp

    base_ds = HSIDataset(DATA_ROOT, use_mask=USE_MASK, augment=False)
    aug_ds = HSIDataset(DATA_ROOT, use_mask=USE_MASK, augment=True)
    all_idx = np.arange(len(base_ds))
    all_lbl = np.array([base_ds.samples[i][1] for i in all_idx])
    print(f'[Data] 共 {len(base_ds)} 个样本，{NUM_CLASSES} 类')

    dev_idx, pred_idx = train_test_split(
        all_idx,
        test_size=PREDICTION_SIZE,
        stratify=all_lbl,
        random_state=SEED,
        shuffle=True
    )
    dev_idx = np.array(sorted(dev_idx))
    pred_idx = np.array(sorted(pred_idx))
    dev_lbl = all_lbl[dev_idx]

    print(f'[Split] Model-development set: {len(dev_idx)} samples')
    print(f'[Split] Independent prediction set: {len(pred_idx)} samples')

    save_split_summary(base_ds, dev_idx, pred_idx, SAVE_DIR)
    np.savez(
        os.path.join(SAVE_DIR, 'dataset_split_indices.npz'),
        dev_idx=dev_idx,
        pred_idx=pred_idx,
        seed=np.array([SEED]),
        prediction_size=np.array([PREDICTION_SIZE])
    )

    # ───────────────────────────────────────────
    # A. Internal 5-fold CV within development set
    # ───────────────────────────────────────────
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_metrics = []
    all_logs = []
    all_true_internal = []
    all_pred_internal = []

    for fold, (tr_pos, te_pos) in enumerate(skf.split(dev_idx, dev_lbl), start=1):
        train_idx = dev_idx[tr_pos]
        test_idx = dev_idx[te_pos]

        print(f'\n{"="*65}')
        print(f'  Internal Fold {fold}/{N_FOLDS}  train={len(train_idx)}  test={len(test_idx)}')
        print(f'{"="*65}')

        train_loader = DataLoader(
            Subset(aug_ds, train_idx),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        seed_everything(SEED + fold * 100)
        model = HSI3DCNN_NoSelecVar(NUM_CLASSES, DROPOUT).to(device)
        tag = f'internal_fold{fold}'
        model, log = train_model(model, train_loader, device, tag, SAVE_DIR)
        all_logs.append(log)
        save_training_log(log, tag, SAVE_DIR)
        plot_training_curve(log, tag, SAVE_DIR)

        y_true, y_pred = predict_no_tta(model, base_ds, list(test_idx), device, BATCH_SIZE)
        metrics = calc_metrics(y_true, y_pred)
        fold_metrics.append(metrics)
        all_true_internal.extend(y_true.tolist())
        all_pred_internal.extend(y_pred.tolist())

        print(f'  Internal Fold {fold} test  acc={metrics["acc"]:.4f}  '
              f'prec={metrics["precision"]:.4f}  rec={metrics["recall"]:.4f}  f1={metrics["f1"]:.4f}')

        cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
        plot_confusion_matrix(
            cm,
            f'3D-CNN without SelecVar Internal Fold {fold} (Acc={metrics["acc"]:.1%})',
            os.path.join(SAVE_DIR, f'cm_internal_fold{fold}.png')
        )

    plot_overall_training_curves(all_logs, SAVE_DIR)

    accs = [m['acc'] for m in fold_metrics]
    precs = [m['precision'] for m in fold_metrics]
    recs = [m['recall'] for m in fold_metrics]
    f1s = [m['f1'] for m in fold_metrics]

    y_true_internal = np.array(all_true_internal)
    y_pred_internal = np.array(all_pred_internal)
    cm_internal = confusion_matrix(y_true_internal, y_pred_internal, labels=list(range(NUM_CLASSES)))
    internal_overall = calc_metrics(y_true_internal, y_pred_internal)

    print(f'\n{"="*65}')
    print('Internal 5-Fold Summary on Model-Development Set — 3D-CNN without SelecVar')
    print(f'{"="*65}')
    for i, m in enumerate(fold_metrics, 1):
        print(f'  Fold {i}: acc={m["acc"]:.4f}  prec={m["precision"]:.4f}  rec={m["recall"]:.4f}  f1={m["f1"]:.4f}')
    print('\n  Internal 5-fold Mean ± Std:')
    print(f'  Accuracy  : {np.mean(accs):.4f} ± {np.std(accs):.4f}')
    print(f'  Precision : {np.mean(precs):.4f} ± {np.std(precs):.4f}')
    print(f'  Recall    : {np.mean(recs):.4f} ± {np.std(recs):.4f}')
    print(f'  F1        : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}')
    print(f'\n  Internal 5-fold overall merged accuracy: {internal_overall["acc"]:.4f}')
    print(classification_report(y_true_internal, y_pred_internal, target_names=CLASS_NAMES, digits=4))

    plot_confusion_matrix(
        cm_internal,
        f'3D-CNN without SelecVar Internal 5-fold Overall (Acc={internal_overall["acc"]:.1%})',
        os.path.join(SAVE_DIR, 'cm_internal_5fold_overall.png')
    )

    # ───────────────────────────────────────────
    # B. Train final model on full development set
    # ───────────────────────────────────────────
    print(f'\n{"="*65}')
    print(f'Final model training on full model-development set: {len(dev_idx)} samples')
    print(f'{"="*65}')

    seed_everything(SEED + 9999)
    final_model = HSI3DCNN_NoSelecVar(NUM_CLASSES, DROPOUT).to(device)
    final_loader = DataLoader(
        Subset(aug_ds, dev_idx),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    final_model, final_log = train_model(final_model, final_loader, device, 'final_development', SAVE_DIR)
    save_training_log(final_log, 'final_development', SAVE_DIR)
    plot_training_curve(final_log, 'final_development', SAVE_DIR)

    # ───────────────────────────────────────────
    # C. Independent prediction set evaluation
    # ───────────────────────────────────────────
    print(f'\n{"="*65}')
    print(f'Independent Prediction Set Evaluation: {len(pred_idx)} samples')
    print(f'{"="*65}')

    y_true_pred, y_pred_pred = predict_no_tta(final_model, base_ds, list(pred_idx), device, BATCH_SIZE)
    pred_metrics = calc_metrics(y_true_pred, y_pred_pred)
    cm_pred = confusion_matrix(y_true_pred, y_pred_pred, labels=list(range(NUM_CLASSES)))

    print(f'  Prediction set acc={pred_metrics["acc"]:.4f}  '
          f'prec={pred_metrics["precision"]:.4f}  '
          f'rec={pred_metrics["recall"]:.4f}  '
          f'f1={pred_metrics["f1"]:.4f}')
    print(classification_report(y_true_pred, y_pred_pred, target_names=CLASS_NAMES, digits=4))

    plot_confusion_matrix(
        cm_pred,
        f'3D-CNN without SelecVar Independent Prediction Set (Acc={pred_metrics["acc"]:.1%})',
        os.path.join(SAVE_DIR, 'cm_independent_prediction_set.png')
    )

    # ───────────────────────────────────────────
    # D. Save final report
    # ───────────────────────────────────────────
    report_path = os.path.join(SAVE_DIR, 'results_no_selecvar_holdout_prediction_5fold.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('3D-CNN without SelecVar: Hold-out Prediction Set + Internal 5-Fold CV\n')
        f.write(f'Seed={SEED}; Split=3:1; Prediction_size={PREDICTION_SIZE}\n')
        f.write(f'Architecture=3D-CNN backbone without SelecVar; Epochs={EPOCHS}; TTA=disabled\n')
        f.write(f'LR_main={LR_MAIN}; Dropout={DROPOUT}; Weight_decay={WEIGHT_DECAY}\n')
        f.write(f'SGDR T0={SGDR_T0}; T_mult={SGDR_T_MULT}; Mixup_alpha={MIXUP_ALPHA}; Label_smooth={LABEL_SMOOTH}\n')
        f.write('=' * 70 + '\n\n')

        f.write('Dataset split:\n')
        f.write(f'  Total samples: {len(base_ds)}\n')
        f.write(f'  Model-development set: {len(dev_idx)}\n')
        f.write(f'  Independent prediction set: {len(pred_idx)}\n\n')

        f.write('Internal 5-fold CV on model-development set:\n')
        for i, m in enumerate(fold_metrics, 1):
            f.write(f'  Fold {i}: acc={m["acc"]:.4f}  precision={m["precision"]:.4f}  '
                    f'recall={m["recall"]:.4f}  f1={m["f1"]:.4f}\n')
        f.write('\nInternal 5-fold Mean ± Std:\n')
        f.write(f'  Accuracy  : {np.mean(accs):.4f} ± {np.std(accs):.4f}\n')
        f.write(f'  Precision : {np.mean(precs):.4f} ± {np.std(precs):.4f}\n')
        f.write(f'  Recall    : {np.mean(recs):.4f} ± {np.std(recs):.4f}\n')
        f.write(f'  F1        : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}\n')
        f.write(f'\nInternal 5-fold overall merged accuracy: {internal_overall["acc"]:.4f}\n\n')
        f.write('Internal 5-fold overall per-class report:\n')
        f.write(classification_report(y_true_internal, y_pred_internal, target_names=CLASS_NAMES, digits=4))

        f.write('\n\nIndependent prediction set results:\n')
        f.write(f'  Accuracy  : {pred_metrics["acc"]:.4f}\n')
        f.write(f'  Precision : {pred_metrics["precision"]:.4f}\n')
        f.write(f'  Recall    : {pred_metrics["recall"]:.4f}\n')
        f.write(f'  F1        : {pred_metrics["f1"]:.4f}\n\n')
        f.write('Independent prediction set per-class report:\n')
        f.write(classification_report(y_true_pred, y_pred_pred, target_names=CLASS_NAMES, digits=4))

    print(f'\n[Done] All outputs saved in → {SAVE_DIR}')
    print(f'[Done] Report → {report_path}')


if __name__ == '__main__':
    main()
