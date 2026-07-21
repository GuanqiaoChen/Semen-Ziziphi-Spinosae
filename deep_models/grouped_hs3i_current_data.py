#!/usr/bin/env python3
"""Leakage-aware 3-D HSI experiments for the currently available data.

The only confirmatory comparisons implemented here are reciprocal source-cube
transfers: all ``*-1`` cubes train a model evaluated on all ``*-2`` cubes, and
vice versa.  Hyperparameters are fixed in this file, and no test observation is
used for early stopping, checkpoint selection, or hyperparameter selection.

This module intentionally does not implement Grad-CAM.  The supplied backbone's
late activations have very broad spectral receptive fields, so assigning them to
individual wavelengths would imply unsupported wavelength precision.  SelecVar
gate weights, when requested, are exported against the exact CSV wavelength grid
and must be interpreted as model parameters rather than chemical evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# These settings are declared before any test evaluation and are never selected
# from test-set performance.  CLI options choose prespecified model/negative-
# control cells, seeds, and execution resources; they do not tune these values.
EXPECTED_BANDS = 392
EXPECTED_PATCH_SIZE = 32
NUM_CLASSES = 8
CLASS_NAMES = ("HBS", "HBX", "HNA", "HNX", "NX", "SXD", "SXQ", "XJH")

BATCH_SIZE = 16
EPOCHS = 360
WARMUP_EPOCHS = 10
LR_MAIN = 3e-4
LR_SELECVAR = 1e-3
MIN_LR_FACTOR = 0.01
WEIGHT_DECAY = 1e-4
LAMBDA_SPARSE = 5e-5
LAMBDA_SMOOTH = 1e-5
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.3
DROPOUT = 0.35
GRAD_CLIP_NORM = 5.0

MODEL_CHOICES = ("hs3i", "no_selecvar")
CONDITION_CHOICES = ("full", "spatial_shuffle", "mask_only")
DEFAULT_SEEDS = (42, 2024, 2025)
CUBE_PATTERN = re.compile(r"^(?P<label>\d+)-(?P<suffix>[12])$")


@dataclass(frozen=True)
class Sample:
    """One seed patch with mandatory acquisition hierarchy metadata."""

    sample_index: int
    label: int
    source_cube: str
    cube_suffix: int
    seed_id: str
    mat_path: Path
    csv_path: Path
    relative_mat_path: str
    relative_csv_path: str

    @property
    def sample_id(self) -> str:
        return f"{self.source_cube}/{self.seed_id}"


@dataclass(frozen=True)
class FixedHyperparameters:
    expected_bands: int = EXPECTED_BANDS
    expected_patch_size: int = EXPECTED_PATCH_SIZE
    batch_size: int = BATCH_SIZE
    epochs: int = EPOCHS
    warmup_epochs: int = WARMUP_EPOCHS
    lr_main: float = LR_MAIN
    lr_selecvar: float = LR_SELECVAR
    min_lr_factor: float = MIN_LR_FACTOR
    weight_decay: float = WEIGHT_DECAY
    lambda_sparse: float = LAMBDA_SPARSE
    lambda_smooth: float = LAMBDA_SMOOTH
    label_smoothing: float = LABEL_SMOOTHING
    mixup_alpha: float = MIXUP_ALPHA
    dropout: float = DROPOUT
    grad_clip_norm: float = GRAD_CLIP_NORM


def _natural_seed_key(path: Path) -> tuple[int, int | str]:
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.stem)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_spectrum_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        values = np.loadtxt(path, delimiter=",", dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"CSV is not a headerless two-column numeric spectrum: {path}") from exc
    if values.shape != (EXPECTED_BANDS, 2):
        raise ValueError(
            f"Expected ({EXPECTED_BANDS}, 2) in {path}, observed {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite wavelength or reflectance value in {path}")
    wavelengths = values[:, 0]
    if np.any(np.diff(wavelengths) <= 0):
        raise ValueError(f"Wavelengths are not strictly increasing in {path}")
    return wavelengths, values[:, 1]


def discover_samples(data_root: Path) -> tuple[list[Sample], np.ndarray, dict[str, str]]:
    """Discover paired MAT/CSV samples and validate the exact wavelength grid."""

    data_root = data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist or is not a directory: {data_root}")

    cube_dirs: list[tuple[int, int, Path]] = []
    for candidate in data_root.iterdir():
        if not candidate.is_dir():
            continue
        match = CUBE_PATTERN.fullmatch(candidate.name)
        if match:
            cube_dirs.append((int(match.group("label")), int(match.group("suffix")), candidate))
    cube_dirs.sort(key=lambda item: (item[0], item[1]))
    if not cube_dirs:
        raise FileNotFoundError(f"No <label>-<1|2> source-cube directories under {data_root}")

    observed_cubes = {(label, suffix) for label, suffix, _ in cube_dirs}
    expected_cubes = {(label, suffix) for label in range(NUM_CLASSES) for suffix in (1, 2)}
    if observed_cubes != expected_cubes:
        missing = sorted(expected_cubes - observed_cubes)
        extra = sorted(observed_cubes - expected_cubes)
        raise ValueError(f"Expected exactly labels 0..7 with suffixes 1 and 2; missing={missing}, extra={extra}")

    samples: list[Sample] = []
    wavelength_reference: np.ndarray | None = None
    csv_content_digest = hashlib.sha256()
    manifest_digest = hashlib.sha256()

    for label, suffix, cube_dir in cube_dirs:
        mat_paths = sorted(cube_dir.glob("*.mat"), key=_natural_seed_key)
        csv_paths = sorted(cube_dir.glob("*.csv"), key=_natural_seed_key)
        mat_by_stem = {path.stem: path for path in mat_paths}
        csv_by_stem = {path.stem: path for path in csv_paths}
        if not mat_by_stem:
            raise FileNotFoundError(f"No MAT patches in source cube {cube_dir}")
        if mat_by_stem.keys() != csv_by_stem.keys():
            missing_csv = sorted(mat_by_stem.keys() - csv_by_stem.keys())
            missing_mat = sorted(csv_by_stem.keys() - mat_by_stem.keys())
            raise ValueError(
                f"MAT/CSV stem mismatch in {cube_dir}: missing_csv={missing_csv}, "
                f"missing_mat={missing_mat}"
            )

        for mat_path in mat_paths:
            csv_path = csv_by_stem[mat_path.stem]
            wavelengths, _ = _read_spectrum_csv(csv_path)
            if wavelength_reference is None:
                wavelength_reference = wavelengths.copy()
            elif not np.allclose(wavelengths, wavelength_reference, rtol=0.0, atol=1e-6):
                maximum_error = float(np.max(np.abs(wavelengths - wavelength_reference)))
                raise ValueError(
                    f"Wavelength-grid mismatch in {csv_path}; max absolute error={maximum_error:g} nm"
                )

            relative_mat = mat_path.relative_to(data_root).as_posix()
            relative_csv = csv_path.relative_to(data_root).as_posix()
            csv_sha = _sha256_file(csv_path)
            csv_content_digest.update(relative_csv.encode("utf-8"))
            csv_content_digest.update(csv_sha.encode("ascii"))
            manifest_line = (
                f"{label},{suffix},{relative_mat},{mat_path.stat().st_size},"
                f"{relative_csv},{csv_path.stat().st_size}\n"
            )
            manifest_digest.update(manifest_line.encode("utf-8"))
            samples.append(
                Sample(
                    sample_index=len(samples),
                    label=label,
                    source_cube=cube_dir.name,
                    cube_suffix=suffix,
                    seed_id=mat_path.stem,
                    mat_path=mat_path,
                    csv_path=csv_path,
                    relative_mat_path=relative_mat,
                    relative_csv_path=relative_csv,
                )
            )

    assert wavelength_reference is not None
    return samples, wavelength_reference, {
        "csv_content_sha256": csv_content_digest.hexdigest(),
        "manifest_sha256": manifest_digest.hexdigest(),
        "manifest_hash_scope": "relative paths and file sizes; CSV contents are hashed separately",
    }


def _stable_transform_seed(run_seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(f"{run_seed}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _orient_patch_hwb(raw: np.ndarray, mask: np.ndarray, path: Path) -> np.ndarray:
    """Convert MATLAB/HDF5 axis conventions to H x W x wavelength."""

    raw = np.asarray(raw, dtype=np.float32).squeeze()
    if raw.ndim != 3:
        raise ValueError(f"patch_chw must be three-dimensional in {path}; observed {raw.shape}")
    band_axes = [axis for axis, size in enumerate(raw.shape) if size == EXPECTED_BANDS]
    if len(band_axes) != 1:
        raise ValueError(
            f"Cannot identify unique {EXPECTED_BANDS}-band axis in {path}; observed {raw.shape}"
        )
    cube = np.moveaxis(raw, band_axes[0], -1)
    if cube.shape[:2] != mask.shape:
        if cube.shape[:2][::-1] == mask.shape:
            cube = np.transpose(cube, (1, 0, 2))
        else:
            raise ValueError(
                f"patch/mask spatial mismatch in {path}: patch={cube.shape}, mask={mask.shape}"
            )
    if cube.shape[:2] != (EXPECTED_PATCH_SIZE, EXPECTED_PATCH_SIZE):
        raise ValueError(
            f"Expected {EXPECTED_PATCH_SIZE}x{EXPECTED_PATCH_SIZE} patch in {path}; "
            f"observed {cube.shape[:2]}"
        )
    return cube


def load_mat_patch(sample: Sample) -> tuple[np.ndarray, np.ndarray]:
    """Read a v7.3 MAT patch without consulting any test-set outcome."""

    with h5py.File(sample.mat_path, "r") as handle:
        missing = {"patch_chw", "crop_mask"} - set(handle.keys())
        if missing:
            raise KeyError(f"Missing datasets {sorted(missing)} in {sample.mat_path}")
        raw = np.asarray(handle["patch_chw"][()], dtype=np.float32)
        mask = np.asarray(handle["crop_mask"][()], dtype=np.float32).squeeze()
    if mask.ndim != 2:
        raise ValueError(f"crop_mask must be two-dimensional in {sample.mat_path}; observed {mask.shape}")
    mask = mask > 0.5
    if not np.any(mask):
        raise ValueError(f"Empty foreground mask in {sample.mat_path}")
    cube = _orient_patch_hwb(raw, mask, sample.mat_path)
    if not np.all(np.isfinite(cube)):
        raise ValueError(f"Non-finite patch value in {sample.mat_path}")
    return cube, mask


class HSIPatchDataset(Dataset):
    """Dataset with prespecified full-image and spatial falsification conditions."""

    def __init__(
        self,
        samples: Sequence[Sample],
        condition: str,
        augment: bool,
        transform_seed: int,
    ) -> None:
        if condition not in CONDITION_CHOICES:
            raise ValueError(f"Unknown condition: {condition}")
        self.samples = list(samples)
        self.condition = condition
        self.augment = augment
        self.transform_seed = transform_seed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        sample = self.samples[index]
        cube, mask = load_mat_patch(sample)
        cube = np.clip(cube, 0.0, 1.0)
        cube = cube * mask[:, :, None]

        if self.condition == "spatial_shuffle":
            # Shuffle complete foreground spectra, not individual bands.  This
            # preserves the multivariate pixel spectra and silhouette while
            # destroying their within-seed spatial arrangement.
            foreground = np.flatnonzero(mask.reshape(-1))
            rng = np.random.default_rng(_stable_transform_seed(self.transform_seed, sample.sample_id))
            shuffled = cube.reshape(-1, EXPECTED_BANDS).copy()
            shuffled[foreground] = shuffled[rng.permutation(foreground)]
            cube = shuffled.reshape(cube.shape)
        elif self.condition == "mask_only":
            # Repeated binary mask removes spectral/intensity information and
            # retains only silhouette, area, orientation, and patch placement.
            cube = np.repeat(mask[:, :, None].astype(np.float32), EXPECTED_BANDS, axis=2)

        tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(cube, (2, 0, 1))))
        tensor = tensor.unsqueeze(0)  # 1 x wavelength x H x W
        if self.augment:
            if torch.rand(()) > 0.5:
                tensor = torch.flip(tensor, dims=(2,))
            if torch.rand(()) > 0.5:
                tensor = torch.flip(tensor, dims=(3,))
            if self.condition != "mask_only":
                scale = 0.9 + 0.2 * torch.rand(()).item()
                tensor = torch.clamp(tensor * scale, 0.0, 1.0)
        return tensor, sample.label, index


class SelecVar(nn.Module):
    """Positive trainable wavelength gates retained from the legacy model."""

    def __init__(self, n_bands: int) -> None:
        super().__init__()
        self.n_bands = n_bands
        self.weight = nn.Parameter(torch.randn(n_bands) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = F.softplus(self.weight).view(1, 1, self.n_bands, 1, 1)
        return x * gates

    def regularization_loss(self) -> torch.Tensor:
        gates = F.softplus(self.weight)
        hoyer = LAMBDA_SPARSE * gates.sum() / (gates.square().sum().sqrt() + 1e-8)
        smooth = LAMBDA_SMOOTH * (gates[1:] - gates[:-1]).square().sum()
        return hoyer + smooth

    def normalized_weights(self) -> np.ndarray:
        gates = F.softplus(self.weight).detach().cpu().numpy()
        return gates / (gates.max() + 1e-8)


class ResBlock3D(nn.Module):
    """Residual 3-D block matching the supplied HS3I-Net backbone."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int],
        stride: tuple[int, int, int],
        padding: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        needs_projection = in_channels != out_channels or any(value != 1 for value in stride)
        self.shortcut: nn.Module = (
            nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )
            if needs_projection
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.activation(x + residual)


class GroupedHSI3DCNN(nn.Module):
    """Legacy spectral-spatial backbone with an explicit SelecVar switch."""

    def __init__(self, use_selecvar: bool) -> None:
        super().__init__()
        self.use_selecvar = use_selecvar
        self.selecvar: SelecVar | None = SelecVar(EXPECTED_BANDS) if use_selecvar else None
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
            nn.Dropout(DROPOUT),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(DROPOUT * 0.5),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.selecvar is not None:
            x = self.selecvar(x)
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.block4(x)
        return self.classifier(self.gap(x))

    def regularization_loss(self) -> torch.Tensor:
        if self.selecvar is None:
            return next(self.parameters()).new_zeros(())
        return self.selecvar.regularization_loss()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_optimizer(model: GroupedHSI3DCNN) -> torch.optim.Optimizer:
    if model.selecvar is None:
        return torch.optim.AdamW(model.parameters(), lr=LR_MAIN, weight_decay=WEIGHT_DECAY)
    gate_parameters = list(model.selecvar.parameters())
    gate_ids = {id(parameter) for parameter in gate_parameters}
    backbone_parameters = [parameter for parameter in model.parameters() if id(parameter) not in gate_ids]
    return torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": LR_MAIN},
            {"params": gate_parameters, "lr": LR_SELECVAR},
        ],
        weight_decay=WEIGHT_DECAY,
    )


def _lr_multiplier(epoch_index: int) -> float:
    if epoch_index < WARMUP_EPOCHS:
        return float(epoch_index + 1) / float(WARMUP_EPOCHS)
    progress = (epoch_index - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR_FACTOR + (1.0 - MIN_LR_FACTOR) * cosine


def mixup_batch(
    inputs: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
    permutation = torch.randperm(inputs.size(0), device=inputs.device)
    mixed = lam * inputs + (1.0 - lam) * inputs[permutation]
    return mixed, targets, targets[permutation], lam


def make_loader(
    dataset: HSIPatchDataset,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def train_fixed_model(
    model: GroupedHSI3DCNN,
    train_loader: DataLoader,
    device: torch.device,
) -> list[dict[str, float | int]]:
    """Fit for exactly EPOCHS; there is no validation or checkpoint selection."""

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_multiplier)
    history: list[dict[str, float | int]] = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_classification_loss = 0.0
        total_regularization_loss = 0.0
        seen = 0
        for inputs, targets, _ in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if epoch > WARMUP_EPOCHS:
                mixed, targets_a, targets_b, lam = mixup_batch(inputs, targets)
                logits = model(mixed)
                classification_loss = (
                    lam * criterion(logits, targets_a) + (1.0 - lam) * criterion(logits, targets_b)
                )
            else:
                logits = model(inputs)
                classification_loss = criterion(logits, targets)
            regularization_loss = model.regularization_loss()
            loss = classification_loss + regularization_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            batch_n = targets.size(0)
            seen += batch_n
            total_loss += float(loss.detach().item()) * batch_n
            total_classification_loss += float(classification_loss.detach().item()) * batch_n
            total_regularization_loss += float(regularization_loss.detach().item()) * batch_n

        history.append(
            {
                "epoch": epoch,
                "loss": total_loss / seen,
                "classification_loss": total_classification_loss / seen,
                "regularization_loss": total_regularization_loss / seen,
                "lr_main": float(optimizer.param_groups[0]["lr"]),
                "lr_selecvar": (
                    float(optimizer.param_groups[1]["lr"]) if len(optimizer.param_groups) > 1 else float("nan")
                ),
            }
        )
        scheduler.step()
    return history


def classification_metrics(
    true_labels: Sequence[int], predicted_labels: Sequence[int]
) -> dict[str, Any]:
    true = np.asarray(true_labels, dtype=np.int64)
    predicted = np.asarray(predicted_labels, dtype=np.int64)
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for actual, estimate in zip(true, predicted, strict=True):
        confusion[actual, estimate] += 1

    class_rows: list[dict[str, Any]] = []
    for label in range(NUM_CLASSES):
        true_positive = int(confusion[label, label])
        false_positive = int(confusion[:, label].sum() - true_positive)
        false_negative = int(confusion[label, :].sum() - true_positive)
        support = int(confusion[label, :].sum())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_rows.append(
            {
                "label": label,
                "class_name": CLASS_NAMES[label],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    accuracy = float(np.mean(true == predicted))
    return {
        "n_test": int(true.size),
        "n_correct": int(np.sum(true == predicted)),
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean([row["recall"] for row in class_rows])),
        "macro_precision": float(np.mean([row["precision"] for row in class_rows])),
        "macro_recall": float(np.mean([row["recall"] for row in class_rows])),
        "macro_f1": float(np.mean([row["f1"] for row in class_rows])),
        "confusion_matrix_rows_true_columns_predicted": confusion.tolist(),
        "per_class": class_rows,
    }


@torch.no_grad()
def evaluate_once(
    model: GroupedHSI3DCNN,
    test_loader: DataLoader,
    test_samples: Sequence[Sample],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    prediction_rows: list[dict[str, Any]] = []
    true_labels: list[int] = []
    predicted_labels: list[int] = []

    for inputs, targets, indices in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        logits = model(inputs)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        predictions = np.argmax(probabilities, axis=1)
        for row_index, dataset_index in enumerate(indices.tolist()):
            sample = test_samples[dataset_index]
            actual = int(targets[row_index].item())
            estimate = int(predictions[row_index])
            row: dict[str, Any] = {
                "sample_index": sample.sample_index,
                "sample_id": sample.sample_id,
                "source_cube": sample.source_cube,
                "cube_suffix": sample.cube_suffix,
                "seed_id": sample.seed_id,
                "relative_mat_path": sample.relative_mat_path,
                "true_label": actual,
                "true_class": CLASS_NAMES[actual],
                "predicted_label": estimate,
                "predicted_class": CLASS_NAMES[estimate],
                "correct": int(actual == estimate),
            }
            for label, probability in enumerate(probabilities[row_index]):
                row[f"probability_class_{label}"] = float(probability)
            prediction_rows.append(row)
            true_labels.append(actual)
            predicted_labels.append(estimate)
    return classification_metrics(true_labels, predicted_labels), prediction_rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    if not rows and fieldnames is None:
        raise ValueError(f"Cannot infer columns for empty CSV: {path}")
    columns = list(fieldnames) if fieldnames is not None else list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _split_samples(samples: Sequence[Sample], train_suffix: int) -> tuple[list[Sample], list[Sample]]:
    test_suffix = 2 if train_suffix == 1 else 1
    train = [sample for sample in samples if sample.cube_suffix == train_suffix]
    test = [sample for sample in samples if sample.cube_suffix == test_suffix]
    train_cubes = {sample.source_cube for sample in train}
    test_cubes = {sample.source_cube for sample in test}
    if train_cubes & test_cubes:
        raise AssertionError(f"Source-cube leakage detected: {sorted(train_cubes & test_cubes)}")
    expected_labels = set(range(NUM_CLASSES))
    if {sample.label for sample in train} != expected_labels:
        raise ValueError(f"Training suffix {train_suffix} does not contain every label")
    if {sample.label for sample in test} != expected_labels:
        raise ValueError(f"Test suffix {test_suffix} does not contain every label")
    return train, test


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but torch.cuda.is_available() is false")
    return device


def _aggregate_metric_rows(metric_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in metric_rows:
        key = (row["direction"], row["model"], row["condition"])
        grouped.setdefault(key, []).append(row)
    aggregates: list[dict[str, Any]] = []
    metric_names = ("accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1")
    for (direction, model_name, condition), rows in sorted(grouped.items()):
        aggregate: dict[str, Any] = {
            "direction": direction,
            "model": model_name,
            "condition": condition,
            "n_seeds": len(rows),
            "seeds": ";".join(str(row["seed"]) for row in rows),
        }
        for metric_name in metric_names:
            values = np.asarray([row[metric_name] for row in rows], dtype=np.float64)
            aggregate[f"{metric_name}_mean"] = float(values.mean())
            aggregate[f"{metric_name}_sd"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
            aggregate[f"{metric_name}_min"] = float(values.min())
            aggregate[f"{metric_name}_max"] = float(values.max())
        aggregates.append(aggregate)
    return aggregates


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reciprocal source-cube-isolated HS3I analyses on current MAT patches."
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing 0-1 ... 7-2")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="New output directory (default: deep_models/outputs/grouped_<UTC timestamp>)",
    )
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_CHOICES, default=list(MODEL_CHOICES),
        help="Prespecified architecture cells to execute",
    )
    parser.add_argument(
        "--conditions", nargs="+", choices=CONDITION_CHOICES, default=list(CONDITION_CHOICES),
        help="Full input and/or prespecified negative controls",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
        help="Independent training seeds; at least two unique values are required",
    )
    parser.add_argument("--device", default="auto", help="PyTorch device, e.g. auto, cpu, cuda, cuda:0")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count")
    parser.add_argument(
        "--save-checkpoints", action="store_true", help="Also retain final (not test-selected) state_dict files"
    )
    args = parser.parse_args(argv)
    args.models = list(dict.fromkeys(args.models))
    args.conditions = list(dict.fromkeys(args.conditions))
    args.seeds = list(dict.fromkeys(args.seeds))
    if len(args.seeds) < 2:
        parser.error("--seeds requires at least two unique values")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started_at = datetime.now(timezone.utc)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (Path(__file__).resolve().parent / "outputs" / f"grouped_{started_at:%Y%m%dT%H%M%SZ}")
    )
    if output_dir.exists():
        raise FileExistsError(f"Refusing to mix or overwrite results in existing directory: {output_dir}")
    output_dir.mkdir(parents=True)

    samples, wavelengths, fingerprints = discover_samples(args.data_root)
    device = _resolve_device(args.device)
    hyperparameters = FixedHyperparameters()

    manifest_rows = [
        {
            "sample_index": sample.sample_index,
            "sample_id": sample.sample_id,
            "label": sample.label,
            "class_name": CLASS_NAMES[sample.label],
            "source_cube": sample.source_cube,
            "cube_suffix": sample.cube_suffix,
            "seed_id": sample.seed_id,
            "relative_mat_path": sample.relative_mat_path,
            "relative_csv_path": sample.relative_csv_path,
        }
        for sample in samples
    ]
    _write_csv(output_dir / "manifest.csv", manifest_rows)
    _write_csv(
        output_dir / "wavelengths.csv",
        [
            {"band_index_zero_based": index, "wavelength_nm": float(wavelength)}
            for index, wavelength in enumerate(wavelengths)
        ],
    )

    metric_rows: list[dict[str, Any]] = []
    metric_details: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []

    total_runs = 2 * len(args.models) * len(args.conditions) * len(args.seeds)
    run_number = 0
    for train_suffix in (1, 2):
        test_suffix = 2 if train_suffix == 1 else 1
        direction = f"suffix_{train_suffix}_to_{test_suffix}"
        train_samples, test_samples = _split_samples(samples, train_suffix)
        for model_name in args.models:
            for condition in args.conditions:
                for seed in args.seeds:
                    run_number += 1
                    run_id = f"{direction}__{model_name}__{condition}__seed_{seed}"
                    print(f"[{run_number}/{total_runs}] {run_id}", flush=True)
                    seed_everything(seed)
                    train_dataset = HSIPatchDataset(
                        train_samples, condition=condition, augment=True, transform_seed=seed
                    )
                    test_dataset = HSIPatchDataset(
                        test_samples, condition=condition, augment=False, transform_seed=seed
                    )
                    train_loader = make_loader(
                        train_dataset, shuffle=True, seed=seed, num_workers=args.num_workers
                    )
                    # Constructing the loader does not read test MAT patches.  It
                    # is iterated exactly once, only after fixed-length fitting.
                    test_loader = make_loader(
                        test_dataset, shuffle=False, seed=seed, num_workers=args.num_workers
                    )
                    model = GroupedHSI3DCNN(use_selecvar=model_name == "hs3i").to(device)
                    history = train_fixed_model(model, train_loader, device)
                    metrics, run_predictions = evaluate_once(model, test_loader, test_samples, device)

                    common = {
                        "run_id": run_id,
                        "direction": direction,
                        "train_suffix": train_suffix,
                        "test_suffix": test_suffix,
                        "model": model_name,
                        "condition": condition,
                        "seed": seed,
                    }
                    for row in history:
                        history_rows.append({**common, **row})
                    for row in run_predictions:
                        prediction_rows.append({**common, **row})

                    flat_metrics = {
                        **common,
                        "n_train": len(train_samples),
                        "n_test": metrics["n_test"],
                        "n_correct": metrics["n_correct"],
                        "accuracy": metrics["accuracy"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "macro_precision": metrics["macro_precision"],
                        "macro_recall": metrics["macro_recall"],
                        "macro_f1": metrics["macro_f1"],
                    }
                    metric_rows.append(flat_metrics)
                    metric_details.append({**common, "n_train": len(train_samples), **metrics})

                    if model.selecvar is not None:
                        weights = model.selecvar.normalized_weights()
                        for band_index, (wavelength, weight) in enumerate(zip(wavelengths, weights, strict=True)):
                            gate_rows.append(
                                {
                                    **common,
                                    "band_index_zero_based": band_index,
                                    "wavelength_nm": float(wavelength),
                                    "normalized_gate_weight": float(weight),
                                }
                            )
                    if args.save_checkpoints:
                        torch.save(
                            {
                                "run_id": run_id,
                                "model_state_dict": model.state_dict(),
                                "fixed_hyperparameters": asdict(hyperparameters),
                                "class_names": CLASS_NAMES,
                                "wavelengths_nm": wavelengths,
                            },
                            output_dir / f"{run_id}.pt",
                        )
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

    aggregates = _aggregate_metric_rows(metric_rows)
    _write_csv(output_dir / "metrics_by_run.csv", metric_rows)
    _write_csv(output_dir / "metrics_seed_aggregate.csv", aggregates)
    _write_csv(output_dir / "predictions.csv", prediction_rows)
    _write_csv(output_dir / "training_history.csv", history_rows)
    if gate_rows:
        _write_csv(output_dir / "selecvar_gate_weights.csv", gate_rows)

    finished_at = datetime.now(timezone.utc)
    report = {
        "schema_version": "1.0",
        "status": "executed",
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "protocol": {
            "primary_splits": ["suffix_1_to_2", "suffix_2_to_1"],
            "experimental_unit_for_split": "source_cube",
            "test_use": "single evaluation after fixed-epoch training; no validation, tuning, or checkpoint selection",
            "models": args.models,
            "conditions": args.conditions,
            "seeds": args.seeds,
        },
        "fixed_hyperparameters": asdict(hyperparameters),
        "dataset": {
            "data_root": str(args.data_root.resolve()),
            "n_samples": len(samples),
            "n_source_cubes": len({sample.source_cube for sample in samples}),
            "samples_by_source_cube": {
                cube: sum(sample.source_cube == cube for sample in samples)
                for cube in sorted({sample.source_cube for sample in samples})
            },
            **fingerprints,
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "h5py": h5py.__version__,
            "device": str(device),
        },
        "metric_details_by_run": metric_details,
        "seed_aggregates": aggregates,
        "interpretation_warnings": [
            "Each test class contains only one source cube, so seed-level observations are clustered technical subsamples.",
            "Suffix transfer is not an external geographical-origin validation unless independent lots and provenance are established.",
            "Variation across training seeds is not a confidence interval for new lots, years, suppliers, or instruments.",
            "SelecVar gate weights are model parameters, not wavelength-resolved chemical or causal evidence.",
            "The spatial_shuffle and mask_only cells are falsification controls; they do not repair limited biological replication.",
        ],
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Completed {total_runs} fixed-protocol runs. Outputs: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
