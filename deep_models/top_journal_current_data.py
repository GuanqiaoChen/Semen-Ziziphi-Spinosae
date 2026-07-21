#!/usr/bin/env python3
"""Rigorous, auditable current-data study for an 8 GB consumer GPU.

This entry point is independent of the legacy 3-D scripts.  It implements two
reciprocal source-cube tests (suffix 1 -> 2 and suffix 2 -> 1), performs model
selection and temperature calibration only inside the training suffix, and
touches test patches only after a model has been locked.  A trained model is then
evaluated, without refitting, under full, spatial-shuffle, mask-only, and
mean-broadcast inputs.

The protocol cannot create biological replication absent from the data.  Each
test class still contains one acquisition cube, so results are acquisition-cube
transfer diagnostics rather than external geographical-origin validation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import h5py
except ImportError:  # Allows pure protocol tests on machines without HDF5.
    h5py = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset as TorchDataset
except ImportError:  # Allows pure protocol tests on machines without PyTorch.
    torch = None
    nn = None
    F = None
    DataLoader = None
    TorchDataset = object

try:
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
except ImportError:  # Reported clearly when the executable analysis is invoked.
    sklearn = None
    LogisticRegression = None
    StandardScaler = None


EXPECTED_BANDS = 392
EXPECTED_PATCH_SIZE = 32
NUM_CLASSES = 8
CLASS_NAMES = ("HBS", "HBX", "HNA", "HNX", "NX", "SXD", "SXQ", "XJH")
CUBE_PATTERN = re.compile(r"^(?P<label>\d+)-(?P<suffix>[12])$")

MODEL_CHOICES = ("snv_lr", "spectral_only", "fusion_net")
COUNTERFACTUALS = ("full", "spatial_shuffle", "mask_only", "mean_broadcast")
DEFAULT_SEEDS = (42, 2024, 2025)
VALIDATION_FRACTION = 0.20
COUNTERFACTUAL_SEED = 9173

# Prespecified compute-aware neural settings.  None is selected from test data.
BATCH_SIZE = 32
MAX_EPOCHS = 60
MIN_EPOCHS = 12
EARLY_STOPPING_PATIENCE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
GRADIENT_CLIP_NORM = 5.0
LR_CANDIDATES = (0.01, 0.1, 1.0, 10.0, 100.0)
TEMPERATURE_GRID = np.exp(np.linspace(math.log(0.25), math.log(4.0), 301))
ECE_BINS = 10


@dataclass(frozen=True)
class Sample:
    sample_index: int
    label: int
    source_cube: str
    cube_suffix: int
    seed_id: str
    mat_path: Path
    csv_path: Path
    relative_mat_path: str
    relative_csv_path: str
    mat_sha256: str = ""
    csv_sha256: str = ""

    @property
    def sample_id(self) -> str:
        return f"{self.source_cube}/{self.seed_id}"


def _natural_key(path: Path) -> tuple[int, int | str]:
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


def read_spectrum_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if values.shape != (EXPECTED_BANDS, 2):
        raise ValueError(f"Expected ({EXPECTED_BANDS}, 2) in {path}; observed {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite CSV value in {path}")
    wavelengths = values[:, 0]
    if np.any(np.diff(wavelengths) <= 0):
        raise ValueError(f"Wavelengths are not strictly increasing in {path}")
    return wavelengths, values[:, 1]


def discover_samples(data_root: Path) -> tuple[list[Sample], np.ndarray, dict[str, str]]:
    """Validate the 16-cube hierarchy, MAT/CSV pairs, and real wavelength grid."""

    data_root = data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root is not a directory: {data_root}")
    cube_dirs: list[tuple[int, int, Path]] = []
    for candidate in data_root.iterdir():
        match = CUBE_PATTERN.fullmatch(candidate.name) if candidate.is_dir() else None
        if match:
            cube_dirs.append((int(match.group("label")), int(match.group("suffix")), candidate))
    cube_dirs.sort(key=lambda item: (item[0], item[1]))
    observed = {(label, suffix) for label, suffix, _ in cube_dirs}
    expected = {(label, suffix) for label in range(NUM_CLASSES) for suffix in (1, 2)}
    if observed != expected:
        raise ValueError(
            f"Expected exactly 0-1 ... 7-2; missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )

    samples: list[Sample] = []
    wavelength_reference: np.ndarray | None = None
    csv_digest = hashlib.sha256()
    mat_digest = hashlib.sha256()
    manifest_digest = hashlib.sha256()
    for label, suffix, cube_dir in cube_dirs:
        mats = sorted(cube_dir.glob("*.mat"), key=_natural_key)
        csv_by_stem = {path.stem: path for path in cube_dir.glob("*.csv")}
        mat_by_stem = {path.stem: path for path in mats}
        if not mats or mat_by_stem.keys() != csv_by_stem.keys():
            raise ValueError(
                f"Missing data or MAT/CSV stem mismatch in {cube_dir}: "
                f"MAT-only={sorted(mat_by_stem.keys()-csv_by_stem.keys())}, "
                f"CSV-only={sorted(csv_by_stem.keys()-mat_by_stem.keys())}"
            )
        for mat_path in mats:
            csv_path = csv_by_stem[mat_path.stem]
            wavelengths, _ = read_spectrum_csv(csv_path)
            if wavelength_reference is None:
                wavelength_reference = wavelengths.copy()
            elif not np.allclose(wavelengths, wavelength_reference, rtol=0.0, atol=1e-6):
                error = float(np.max(np.abs(wavelengths - wavelength_reference)))
                raise ValueError(f"Wavelength mismatch in {csv_path}; max error={error:g} nm")
            relative_mat = mat_path.relative_to(data_root).as_posix()
            relative_csv = csv_path.relative_to(data_root).as_posix()
            mat_sha256 = _sha256_file(mat_path)
            csv_sha256 = _sha256_file(csv_path)
            mat_digest.update(relative_mat.encode("utf-8"))
            mat_digest.update(mat_sha256.encode("ascii"))
            csv_digest.update(relative_csv.encode("utf-8"))
            csv_digest.update(csv_sha256.encode("ascii"))
            manifest_digest.update(
                (
                    f"{label},{suffix},{relative_mat},{mat_path.stat().st_size},{mat_sha256},"
                    f"{relative_csv},{csv_path.stat().st_size},{csv_sha256}\n"
                ).encode()
            )
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
                    mat_sha256=mat_sha256,
                    csv_sha256=csv_sha256,
                )
            )
    assert wavelength_reference is not None
    return samples, wavelength_reference, {
        "csv_content_sha256": csv_digest.hexdigest(),
        "mat_content_sha256": mat_digest.hexdigest(),
        "manifest_sha256": manifest_digest.hexdigest(),
    }


def reciprocal_source_cube_split(
    samples: Sequence[Sample], train_suffix: int
) -> tuple[list[Sample], list[Sample]]:
    """Keep every acquisition cube wholly on one side of the final test split."""

    if train_suffix not in (1, 2):
        raise ValueError("train_suffix must be 1 or 2")
    test_suffix = 2 if train_suffix == 1 else 1
    development = [sample for sample in samples if sample.cube_suffix == train_suffix]
    test = [sample for sample in samples if sample.cube_suffix == test_suffix]
    development_cubes = {sample.source_cube for sample in development}
    test_cubes = {sample.source_cube for sample in test}
    if development_cubes & test_cubes:
        raise AssertionError("Source-cube leakage in reciprocal split")
    expected_labels = set(range(NUM_CLASSES))
    if {sample.label for sample in development} != expected_labels:
        raise ValueError("Development suffix lacks at least one class")
    if {sample.label for sample in test} != expected_labels:
        raise ValueError("Test suffix lacks at least one class")
    return development, test


def _stable_int(*parts: object) -> int:
    value = ":".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "little")


def internal_validation_split(
    development: Sequence[Sample], seed: int, fraction: float = VALIDATION_FRACTION
) -> tuple[list[Sample], list[Sample]]:
    """Split seeds within each development cube solely for training control.

    With only one development cube per class, a source-cube-independent validation
    split is mathematically unavailable.  This validation set may select epochs,
    LR C, and temperature, but it is never reported as external performance.
    """

    if not 0.0 < fraction < 0.5:
        raise ValueError("Internal validation fraction must be between 0 and 0.5")
    by_cube: dict[str, list[Sample]] = {}
    for sample in development:
        by_cube.setdefault(sample.source_cube, []).append(sample)
    train: list[Sample] = []
    validation: list[Sample] = []
    for cube, cube_samples in sorted(by_cube.items()):
        ordered = sorted(cube_samples, key=lambda sample: _natural_key(sample.mat_path))
        rng = np.random.default_rng(_stable_int("internal-validation", seed, cube))
        permutation = rng.permutation(len(ordered))
        n_validation = max(1, int(round(len(ordered) * fraction)))
        n_validation = min(n_validation, len(ordered) - 1)
        validation_indices = set(permutation[:n_validation].tolist())
        for index, sample in enumerate(ordered):
            (validation if index in validation_indices else train).append(sample)
    train_ids = {sample.sample_id for sample in train}
    validation_ids = {sample.sample_id for sample in validation}
    if train_ids & validation_ids or train_ids | validation_ids != {
        sample.sample_id for sample in development
    }:
        raise AssertionError("Invalid internal train/validation partition")
    return train, validation


def snv(spectra: np.ndarray) -> np.ndarray:
    spectra = np.asarray(spectra, dtype=np.float64)
    one_dimensional = spectra.ndim == 1
    if one_dimensional:
        spectra = spectra[None, :]
    centered = spectra - spectra.mean(axis=1, keepdims=True)
    scale = spectra.std(axis=1, ddof=1, keepdims=True)
    scale = np.where(scale <= np.finfo(np.float64).eps, 1.0, scale)
    transformed = centered / scale
    return transformed[0] if one_dimensional else transformed


def apply_counterfactual(
    cube: np.ndarray,
    mask: np.ndarray,
    condition: str,
    sample_id: str,
    transform_seed: int = COUNTERFACTUAL_SEED,
) -> np.ndarray:
    """Apply a deterministic, complete-spectrum spatial intervention."""

    if condition not in COUNTERFACTUALS:
        raise ValueError(f"Unknown counterfactual condition: {condition}")
    cube = np.asarray(cube, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if cube.ndim != 3 or cube.shape[:2] != mask.shape:
        raise ValueError("cube must be H x W x B and match the two-dimensional mask")
    result = np.clip(cube, 0.0, 1.0) * mask[:, :, None]
    foreground = np.flatnonzero(mask.reshape(-1))
    if not foreground.size:
        raise ValueError("Counterfactual input has an empty foreground")
    if condition == "spatial_shuffle":
        flat = result.reshape(-1, result.shape[-1]).copy()
        rng = np.random.default_rng(_stable_int(transform_seed, sample_id))
        flat[foreground] = flat[rng.permutation(foreground)]
        result = flat.reshape(result.shape)
    elif condition == "mask_only":
        result = np.repeat(mask[:, :, None].astype(np.float32), result.shape[-1], axis=2)
    elif condition == "mean_broadcast":
        mean_spectrum = result[mask].mean(axis=0)
        result = np.zeros_like(result)
        result[mask] = mean_spectrum
    return result


def _orient_hwb(raw: np.ndarray, mask: np.ndarray, path: Path) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32).squeeze()
    if raw.ndim != 3:
        raise ValueError(f"patch_chw is not 3-D in {path}: {raw.shape}")
    band_axes = [axis for axis, size in enumerate(raw.shape) if size == EXPECTED_BANDS]
    if len(band_axes) != 1:
        raise ValueError(f"Cannot identify one {EXPECTED_BANDS}-band axis in {path}: {raw.shape}")
    cube = np.moveaxis(raw, band_axes[0], -1)
    if cube.shape[:2] != mask.shape and cube.shape[:2][::-1] == mask.shape:
        cube = np.transpose(cube, (1, 0, 2))
    if cube.shape != (*mask.shape, EXPECTED_BANDS):
        raise ValueError(f"Patch/mask mismatch in {path}: cube={cube.shape}, mask={mask.shape}")
    if mask.shape != (EXPECTED_PATCH_SIZE, EXPECTED_PATCH_SIZE):
        raise ValueError(f"Expected 32x32 mask in {path}; observed {mask.shape}")
    return cube


def load_patch(sample: Sample) -> tuple[np.ndarray, np.ndarray]:
    if h5py is None:
        raise RuntimeError("h5py is required to read MATLAB v7.3 patches")
    with h5py.File(sample.mat_path, "r") as handle:
        if "patch_chw" not in handle or "crop_mask" not in handle:
            raise KeyError(f"Missing patch_chw or crop_mask in {sample.mat_path}")
        raw = handle["patch_chw"][()]
        mask = np.asarray(handle["crop_mask"][()]).squeeze() > 0.5
    if mask.ndim != 2 or not np.any(mask):
        raise ValueError(f"Invalid crop_mask in {sample.mat_path}")
    cube = _orient_hwb(raw, mask, sample.mat_path)
    if not np.all(np.isfinite(cube)):
        raise ValueError(f"Non-finite patch value in {sample.mat_path}")
    return cube, mask


def compare_patch_mean_to_csv(
    cube: np.ndarray, mask: np.ndarray, csv_spectrum: np.ndarray
) -> dict[str, float]:
    """Compare the un-clipped MAT foreground mean with its exported CSV mean."""

    cube = np.asarray(cube, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    csv_spectrum = np.asarray(csv_spectrum, dtype=np.float64)
    if cube.shape != (*mask.shape, csv_spectrum.size):
        raise ValueError("MAT cube, mask, and CSV spectrum shapes are inconsistent")
    mat_mean = cube[mask].mean(axis=0)
    difference = mat_mean - csv_spectrum
    return {
        "max_absolute_difference": float(np.max(np.abs(difference))),
        "mean_absolute_difference": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
    }


def audit_mat_csv_mean_consistency(
    samples: Sequence[Sample], absolute_tolerance: float = 1e-5
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute a representation-integrity audit; never use it to tune a model."""

    rows: list[dict[str, Any]] = []
    for sample in samples:
        cube, mask = load_patch(sample)
        _, csv_spectrum = read_spectrum_csv(sample.csv_path)
        comparison = compare_patch_mean_to_csv(cube, mask, csv_spectrum)
        rows.append(
            {
                "sample_index": sample.sample_index,
                "sample_id": sample.sample_id,
                "source_cube": sample.source_cube,
                "label": sample.label,
                **comparison,
                "absolute_tolerance": absolute_tolerance,
                "within_tolerance": int(comparison["max_absolute_difference"] <= absolute_tolerance),
            }
        )
    global_maximum = max(row["max_absolute_difference"] for row in rows)
    summary = {
        "status": "passed" if global_maximum <= absolute_tolerance else "failed",
        "n_samples_checked": len(rows),
        "absolute_tolerance": absolute_tolerance,
        "global_max_absolute_difference": global_maximum,
        "n_outside_tolerance": sum(not bool(row["within_tolerance"]) for row in rows),
        "purpose": "representation integrity only; not model selection or tuning",
    }
    return rows, summary


class HSIFusionDataset(TorchDataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        condition: str,
        augment: bool,
        augmentation_seed: int,
    ) -> None:
        self.samples = list(samples)
        self.condition = condition
        self.augment = augment
        self.augmentation_seed = augmentation_seed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        if torch is None:
            raise RuntimeError("PyTorch is required for fusion datasets")
        sample = self.samples[index]
        cube, mask = load_patch(sample)
        cube = apply_counterfactual(cube, mask, self.condition, sample.sample_id)
        cube_tensor = torch.from_numpy(
            np.ascontiguousarray(np.transpose(cube, (2, 0, 1)))
        ).float()
        mask_tensor = torch.from_numpy(mask.astype(np.float32)[None, ...])
        if self.augment:
            if torch.rand(()) > 0.5:
                cube_tensor = torch.flip(cube_tensor, dims=(1,))
                mask_tensor = torch.flip(mask_tensor, dims=(1,))
            if torch.rand(()) > 0.5:
                cube_tensor = torch.flip(cube_tensor, dims=(2,))
                mask_tensor = torch.flip(mask_tensor, dims=(2,))
            scale = 0.9 + 0.2 * torch.rand(()).item()
            cube_tensor = torch.clamp(cube_tensor * scale, 0.0, 1.0)
        mask_bool = mask_tensor[0] > 0.5
        mean_spectrum = cube_tensor[:, mask_bool].mean(dim=1).numpy()
        spectrum_tensor = torch.from_numpy(snv(mean_spectrum).astype(np.float32))
        return cube_tensor, spectrum_tensor, mask_tensor, sample.label, index


class SpectralOnlyDataset(TorchDataset):
    """CSV-backed spectral data; dummy spatial tensors are never consumed."""

    def __init__(self, samples: Sequence[Sample], condition: str) -> None:
        self.samples = list(samples)
        self.condition = condition

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        if torch is None:
            raise RuntimeError("PyTorch is required for neural spectral data")
        sample = self.samples[index]
        _, spectrum = read_spectrum_csv(sample.csv_path)
        if self.condition == "mask_only":
            spectrum = np.ones_like(spectrum)
        spectrum_tensor = torch.from_numpy(snv(spectrum).astype(np.float32))
        # EfficientSpectralSpatialFusion does not access these when
        # use_spatial=False.  Their small shapes avoid needless MAT I/O.
        dummy_cube = torch.zeros((1, 1, 1), dtype=torch.float32)
        dummy_mask = torch.zeros((1, 1, 1), dtype=torch.float32)
        return dummy_cube, spectrum_tensor, dummy_mask, sample.label, index


_ModuleBase = nn.Module if nn is not None else object


class Residual1D(_ModuleBase):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 5, padding=2 * dilation, dilation=dilation, bias=False)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, 5, padding=2 * dilation, dilation=dilation, bias=False)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x):
        residual = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.gelu(x + residual)


class SpectralEncoder(_ModuleBase):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 32, 9, stride=2, padding=4, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            Residual1D(32, 1),
            nn.Conv1d(32, 64, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            Residual1D(64, 2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.GELU(),
        )

    def forward(self, spectrum):
        return self.layers(spectrum.unsqueeze(1))


class Residual2D(_ModuleBase):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.gelu(x + residual)


class EfficientSpectralSpatialFusion(_ModuleBase):
    """Factorized fusion: local spectral projection + 2-D spatial encoder.

    The 392->16 1x1 projection operates on every pixel spectrum but avoids the
    enormous wavelength-depth activation tensor of the legacy 3-D network.
    """

    def __init__(self, use_spatial: bool = True) -> None:
        super().__init__()
        self.use_spatial = use_spatial
        self.spectral = SpectralEncoder()
        self.band_projection = nn.Sequential(
            nn.Conv2d(EXPECTED_BANDS, 16, 1, bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
        )
        self.spatial = nn.Sequential(
            Residual2D(17, 32, 2),
            Residual2D(32, 48, 2),
            Residual2D(48, 64, 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 96),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(96, NUM_CLASSES),
        )
        if not use_spatial:
            for parameter in self.band_projection.parameters():
                parameter.requires_grad = False
            for parameter in self.spatial.parameters():
                parameter.requires_grad = False

    def forward(self, cube, spectrum, mask):
        spectral_embedding = self.spectral(spectrum)
        if self.use_spatial:
            projected = self.band_projection(cube)
            spatial_embedding = self.spatial(torch.cat((projected, mask), dim=1))
        else:
            # The complete architecture is still instantiated.  Consequently,
            # equal run seeds give spectral_only and fusion_net exactly the same
            # initial spectral-branch and classifier parameters.  A fixed zero
            # spatial embedding makes this a paired neural spectral ablation.
            spatial_embedding = torch.zeros_like(spectral_embedding)
        return self.classifier(torch.cat((spectral_embedding, spatial_embedding), dim=1))


def softmax_numpy(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    scaled -= scaled.max(axis=1, keepdims=True)
    exponentiated = np.exp(scaled)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def multiclass_metrics(labels: Sequence[int], probabilities: np.ndarray) -> dict[str, Any]:
    labels_array = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities.argmax(axis=1)
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for actual, predicted in zip(labels_array, predictions, strict=True):
        confusion[actual, predicted] += 1
    precision: list[float] = []
    recall: list[float] = []
    f1: list[float] = []
    for label in range(NUM_CLASSES):
        tp = int(confusion[label, label])
        fp = int(confusion[:, label].sum() - tp)
        fn = int(confusion[label, :].sum() - tp)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    clipped = np.clip(probabilities, 1e-12, 1.0)
    nll = float(-np.log(clipped[np.arange(labels_array.size), labels_array]).mean())
    one_hot = np.eye(NUM_CLASSES, dtype=np.float64)[labels_array]
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    confidence = probabilities.max(axis=1)
    correct = predictions == labels_array
    ece = 0.0
    for lower in np.linspace(0.0, 1.0, ECE_BINS + 1)[:-1]:
        upper = lower + 1.0 / ECE_BINS
        members = (confidence > lower) & (confidence <= upper)
        if np.any(members):
            ece += float(members.mean()) * abs(float(correct[members].mean()) - float(confidence[members].mean()))
    return {
        "n": int(labels_array.size),
        "n_correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(np.mean(recall)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "nll": nll,
        "brier": brier,
        "ece_10": ece,
        "confusion_matrix_rows_true_columns_predicted": confusion.tolist(),
    }


def select_temperature(validation_logits: np.ndarray, validation_labels: Sequence[int]) -> float:
    labels = np.asarray(validation_labels, dtype=np.int64)
    losses = []
    for temperature in TEMPERATURE_GRID:
        probabilities = softmax_numpy(validation_logits, float(temperature))
        losses.append(-np.log(np.clip(probabilities[np.arange(labels.size), labels], 1e-12, 1.0)).mean())
    return float(TEMPERATURE_GRID[int(np.argmin(losses))])


def _spectral_matrix(samples: Sequence[Sample], condition: str = "full") -> np.ndarray:
    spectra: list[np.ndarray] = []
    for sample in samples:
        _, spectrum = read_spectrum_csv(sample.csv_path)
        if condition == "mask_only":
            spectrum = np.ones_like(spectrum)
        spectra.append(snv(spectrum))
    return np.asarray(spectra, dtype=np.float64)


def fit_snv_lr(
    train_samples: Sequence[Sample], validation_samples: Sequence[Sample], seed: int
) -> tuple[Any, Any, float, list[dict[str, Any]], float]:
    if LogisticRegression is None or StandardScaler is None:
        raise RuntimeError("scikit-learn is required for the SNV-LR baseline")
    x_train = _spectral_matrix(train_samples)
    y_train = np.asarray([sample.label for sample in train_samples])
    x_validation = _spectral_matrix(validation_samples)
    y_validation = np.asarray([sample.label for sample in validation_samples])
    scaler = StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_validation_scaled = scaler.transform(x_validation)
    candidates: list[dict[str, Any]] = []
    fitted: dict[float, Any] = {}
    for c_value in LR_CANDIDATES:
        classifier = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=5000,
            random_state=seed,
        ).fit(x_train_scaled, y_train)
        logits = classifier.decision_function(x_validation_scaled)
        metrics = multiclass_metrics(y_validation, softmax_numpy(logits))
        candidates.append({"candidate_C": c_value, **{key: metrics[key] for key in ("accuracy", "macro_f1", "nll")}})
        fitted[c_value] = classifier
    selected = sorted(candidates, key=lambda row: (-row["macro_f1"], row["nll"], row["candidate_C"]))[0]
    selected_c = float(selected["candidate_C"])
    classifier = fitted[selected_c]
    validation_logits = classifier.decision_function(x_validation_scaled)
    temperature = select_temperature(validation_logits, y_validation)
    return scaler, classifier, temperature, candidates, selected_c


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def paired_initialization_sha256(model: Any) -> str:
    """Hash the spectral branch and shared classifier before optimization."""

    digest = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state):
        if key.startswith("spectral.") or key.startswith("classifier."):
            digest.update(key.encode("utf-8"))
            digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def seed_worker(worker_id: int) -> None:
    del worker_id
    value = torch.initial_seed() % (2**32)
    np.random.seed(value)
    random.seed(value)


def make_loader(dataset, shuffle: bool, seed: int, workers: int):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def _autocast(enabled: bool):
    return torch.cuda.amp.autocast(enabled=enabled)


def neural_logits(model, loader, device, amp_enabled: bool) -> tuple[np.ndarray, np.ndarray, list[int]]:
    model.eval()
    logits_rows: list[np.ndarray] = []
    labels: list[int] = []
    indices: list[int] = []
    with torch.no_grad():
        for cube, spectrum, mask, target, index in loader:
            cube = cube.to(device, non_blocking=True)
            spectrum = spectrum.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with _autocast(amp_enabled):
                output = model(cube, spectrum, mask)
            logits_rows.append(output.float().cpu().numpy())
            labels.extend(int(value) for value in target.tolist())
            indices.extend(int(value) for value in index.tolist())
    return np.concatenate(logits_rows), np.asarray(labels), indices


def checkpoint_is_better(
    macro_f1: float, nll: float, best_macro_f1: float, best_nll: float
) -> bool:
    """Locked lexicographic rule; exact ties are false so earliest epoch remains."""

    return macro_f1 > best_macro_f1 + 1e-12 or (
        abs(macro_f1 - best_macro_f1) <= 1e-12 and nll < best_nll - 1e-12
    )


def _neural_dataset(
    samples: Sequence[Sample], model_name: str, condition: str, augment: bool, seed: int
):
    if model_name == "spectral_only":
        return SpectralOnlyDataset(samples, condition)
    if model_name == "fusion_net":
        return HSIFusionDataset(samples, condition, augment, seed)
    raise ValueError(f"Not a neural model: {model_name}")


def train_neural(
    train_samples: Sequence[Sample],
    validation_samples: Sequence[Sample],
    model_name: str,
    seed: int,
    device,
    workers: int,
    amp_enabled: bool,
) -> tuple[Any, float, list[dict[str, Any]], int, float, str]:
    seed_everything(seed)
    train_dataset = _neural_dataset(train_samples, model_name, "full", True, seed)
    validation_dataset = _neural_dataset(validation_samples, model_name, "full", False, seed)
    train_loader = make_loader(train_dataset, True, seed, workers)
    validation_loader = make_loader(validation_dataset, False, seed, workers)
    model = EfficientSpectralSpatialFusion(use_spatial=model_name == "fusion_net").to(device)
    initialization_sha256 = paired_initialization_sha256(model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    labels = np.asarray([sample.label for sample in train_samples])
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    weights = labels.size / (NUM_CLASSES * counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
        label_smoothing=LABEL_SMOOTHING,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    best_state = None
    best_epoch = 0
    best_macro_f1 = -math.inf
    best_nll = math.inf
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for cube, spectrum, mask, target, _ in train_loader:
            cube = cube.to(device, non_blocking=True)
            spectrum = spectrum.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(amp_enabled):
                logits = model(cube, spectrum, mask)
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().item()) * target.size(0)
            seen += target.size(0)

        validation_logits, validation_labels, _ = neural_logits(
            model, validation_loader, device, amp_enabled
        )
        validation_metrics = multiclass_metrics(validation_labels, softmax_numpy(validation_logits))
        scheduler.step(validation_metrics["nll"])
        improved = checkpoint_is_better(
            validation_metrics["macro_f1"], validation_metrics["nll"], best_macro_f1, best_nll
        )
        if improved:
            best_macro_f1 = validation_metrics["macro_f1"]
            best_nll = validation_metrics["nll"]
            best_epoch = epoch
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / seen,
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_macro_f1": validation_metrics["macro_f1"],
                "validation_nll": validation_metrics["nll"],
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "selected_so_far": int(improved),
            }
        )
        if epoch >= MIN_EPOCHS and stale_epochs >= EARLY_STOPPING_PATIENCE:
            break

    training_seconds = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("No neural checkpoint was selected")
    model.load_state_dict(best_state)
    validation_logits, validation_labels, _ = neural_logits(model, validation_loader, device, amp_enabled)
    temperature = select_temperature(validation_logits, validation_labels)
    return model, temperature, history, best_epoch, training_seconds, initialization_sha256


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(path)


def prediction_records(
    samples: Sequence[Sample], logits: np.ndarray, temperature: float, common: dict[str, Any]
) -> list[dict[str, Any]]:
    raw = softmax_numpy(logits)
    calibrated = softmax_numpy(logits, temperature)
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        row: dict[str, Any] = {
            **common,
            "sample_index": sample.sample_index,
            "sample_id": sample.sample_id,
            "source_cube": sample.source_cube,
            "cube_suffix": sample.cube_suffix,
            "seed_id": sample.seed_id,
            "true_label": sample.label,
            "raw_predicted_label": int(raw[index].argmax()),
            "calibrated_predicted_label": int(calibrated[index].argmax()),
            "temperature": temperature,
        }
        for label in range(NUM_CLASSES):
            row[f"logit_{label}"] = float(logits[index, label])
            row[f"raw_probability_{label}"] = float(raw[index, label])
            row[f"calibrated_probability_{label}"] = float(calibrated[index, label])
        rows.append(row)
    return rows


def metric_records(
    labels: Sequence[int], logits: np.ndarray, temperature: float, common: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scalar_rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for calibration, probabilities in (
        ("raw", softmax_numpy(logits)),
        ("temperature_scaled", softmax_numpy(logits, temperature)),
    ):
        metrics = multiclass_metrics(labels, probabilities)
        scalar_rows.append(
            {
                **common,
                "calibration": calibration,
                **{key: metrics[key] for key in (
                    "n", "n_correct", "accuracy", "balanced_accuracy", "macro_precision",
                    "macro_recall", "macro_f1", "nll", "brier", "ece_10"
                )},
            }
        )
        details.append({**common, "calibration": calibration, **metrics})
    return scalar_rows, details


def cube_metric_records(prediction_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    keys = ("run_id", "direction", "model", "seed", "condition", "source_cube")
    for row in prediction_rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    result: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        labels = [int(row["true_label"]) for row in rows]
        for calibration, prefix in (("raw", "raw_probability_"), ("temperature_scaled", "calibrated_probability_")):
            probabilities = np.asarray(
                [[float(row[f"{prefix}{label}"]) for label in range(NUM_CLASSES)] for row in rows]
            )
            metrics = multiclass_metrics(labels, probabilities)
            result.append(
                {
                    **dict(zip(keys, key, strict=True)),
                    "calibration": calibration,
                    "n": metrics["n"],
                    "n_correct": metrics["n_correct"],
                    "accuracy": metrics["accuracy"],
                    "nll": metrics["nll"],
                    "brier": metrics["brier"],
                    "ece_10": metrics["ece_10"],
                }
            )
    return result


def counterfactual_pairs(prediction_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["run_id"], row["sample_id"], row["condition"]): row for row in prediction_rows
    }
    pairs: list[dict[str, Any]] = []
    for (run_id, sample_id, condition), row in sorted(by_key.items()):
        if condition == "full":
            continue
        reference = by_key[(run_id, sample_id, "full")]
        label = int(row["true_label"])
        full_probability = float(reference[f"calibrated_probability_{label}"])
        counterfactual_probability = float(row[f"calibrated_probability_{label}"])
        pairs.append(
            {
                "run_id": run_id,
                "direction": row["direction"],
                "model": row["model"],
                "seed": row["seed"],
                "sample_id": sample_id,
                "source_cube": row["source_cube"],
                "true_label": label,
                "condition": condition,
                "full_predicted_label": reference["calibrated_predicted_label"],
                "counterfactual_predicted_label": row["calibrated_predicted_label"],
                "prediction_changed": int(
                    reference["calibrated_predicted_label"] != row["calibrated_predicted_label"]
                ),
                "full_true_class_probability": full_probability,
                "counterfactual_true_class_probability": counterfactual_probability,
                "true_class_probability_delta": counterfactual_probability - full_probability,
            }
        )
    return pairs


def aggregate_seed_metrics(metric_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("direction", "model", "condition", "calibration")
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in metric_rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    output: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        record: dict[str, Any] = {**dict(zip(keys, key, strict=True)), "n_seeds": len(rows)}
        for metric in ("accuracy", "balanced_accuracy", "macro_f1", "nll", "brier", "ece_10"):
            values = np.asarray([row[metric] for row in rows])
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            record[f"{metric}_min"] = float(values.min())
            record[f"{metric}_max"] = float(values.max())
        output.append(record)
    return output


def probability_ensemble_predictions(
    prediction_rows: Sequence[dict[str, Any]], expected_seeds: Sequence[int]
) -> list[dict[str, Any]]:
    """Average probabilities across prespecified seeds before computing metrics."""

    group_keys = ("direction", "model", "condition", "sample_id")
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in prediction_rows:
        grouped.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    expected = set(expected_seeds)
    ensemble_rows: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        observed = {int(row["seed"]) for row in rows}
        if observed != expected or len(rows) != len(expected):
            raise ValueError(
                f"Incomplete seed ensemble for {key}: expected={sorted(expected)}, observed={sorted(observed)}"
            )
        first = rows[0]
        raw = np.mean(
            [[float(row[f"raw_probability_{label}"]) for label in range(NUM_CLASSES)] for row in rows],
            axis=0,
        )
        calibrated = np.mean(
            [[float(row[f"calibrated_probability_{label}"]) for label in range(NUM_CLASSES)] for row in rows],
            axis=0,
        )
        direction, model_name, condition, sample_id = key
        record: dict[str, Any] = {
            "run_id": f"{direction}__{model_name}__probability_ensemble",
            "direction": direction,
            "model": model_name,
            "seed": "probability_ensemble",
            "n_ensemble_seeds": len(rows),
            "condition": condition,
            "sample_index": first["sample_index"],
            "sample_id": sample_id,
            "source_cube": first["source_cube"],
            "cube_suffix": first["cube_suffix"],
            "seed_id": first["seed_id"],
            "true_label": first["true_label"],
            "raw_predicted_label": int(np.argmax(raw)),
            "calibrated_predicted_label": int(np.argmax(calibrated)),
        }
        for label in range(NUM_CLASSES):
            record[f"raw_probability_{label}"] = float(raw[label])
            record[f"calibrated_probability_{label}"] = float(calibrated[label])
        ensemble_rows.append(record)
    return ensemble_rows


def ensemble_metric_records(
    ensemble_rows: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    keys = ("direction", "model", "condition")
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in ensemble_rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    scalar_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        labels = [int(row["true_label"]) for row in rows]
        for calibration, prefix in (
            ("raw", "raw_probability_"),
            ("temperature_scaled", "calibrated_probability_"),
        ):
            probabilities = np.asarray(
                [[float(row[f"{prefix}{label}"]) for label in range(NUM_CLASSES)] for row in rows]
            )
            calculated = multiclass_metrics(labels, probabilities)
            common = {**dict(zip(keys, key, strict=True)), "calibration": calibration}
            scalar_rows.append(
                {
                    **common,
                    **{name: calculated[name] for name in (
                        "n", "n_correct", "accuracy", "balanced_accuracy", "macro_precision",
                        "macro_recall", "macro_f1", "nll", "brier", "ece_10"
                    )},
                }
            )
            detail_rows.append({**common, **calculated})
    return scalar_rows, detail_rows


def compute_primary_estimands(
    ensemble_metric_rows: Sequence[dict[str, Any]],
    seed_metric_rows: Sequence[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute preregistered cube-equal theta and the spatial mechanism gate."""

    lookup = {
        (row["direction"], row["model"], row["condition"], row["calibration"]): row
        for row in ensemble_metric_rows
    }
    directions = ("suffix_1_to_2", "suffix_2_to_1")
    theta_rows: list[dict[str, Any]] = []
    models = sorted({row["model"] for row in ensemble_metric_rows})
    for model_name in models:
        for calibration in ("raw", "temperature_scaled"):
            directional = [
                float(lookup[(direction, model_name, "full", calibration)]["balanced_accuracy"])
                for direction in directions
            ]
            theta_rows.append(
                {
                    "estimand": "theta_two_direction_cube_equal_balanced_accuracy",
                    "model": model_name,
                    "calibration": calibration,
                    "suffix_1_to_2_balanced_accuracy": directional[0],
                    "suffix_2_to_1_balanced_accuracy": directional[1],
                    "theta": float(np.mean(directional)),
                    "estimand_role": "primary" if calibration == "temperature_scaled" else "sensitivity_raw",
                    "weighting": "8 class/source-cubes equal within each direction; two directions equal",
                    "probability_estimator": "prespecified-seed probability ensemble",
                }
            )

    mechanism: dict[str, Any]
    if any(row["model"] == "fusion_net" for row in ensemble_metric_rows):
        deltas = []
        direction_records = []
        for direction in directions:
            full = float(
                lookup[(direction, "fusion_net", "full", "temperature_scaled")]["balanced_accuracy"]
            )
            shuffled = float(
                lookup[(direction, "fusion_net", "spatial_shuffle", "temperature_scaled")]["balanced_accuracy"]
            )
            delta = full - shuffled
            deltas.append(delta)
            direction_records.append(
                {"direction": direction, "full_balanced_accuracy": full, "spatial_shuffle_balanced_accuracy": shuffled, "delta_full_minus_shuffle": delta}
            )
        mean_delta = float(np.mean(deltas))
        both_positive = all(delta > 0.0 for delta in deltas)
        seed_direction_deltas: list[dict[str, Any]] = []
        if seed_metric_rows is not None:
            seed_lookup = {
                (
                    row["direction"], int(row["seed"]), row["model"], row["condition"],
                    row["calibration"],
                ): row
                for row in seed_metric_rows
            }
            observed_seeds = sorted(
                {
                    int(row["seed"])
                    for row in seed_metric_rows
                    if row["model"] == "fusion_net"
                }
            )
            for direction in directions:
                for seed in observed_seeds:
                    full_seed = float(
                        seed_lookup[
                            (direction, seed, "fusion_net", "full", "temperature_scaled")
                        ]["balanced_accuracy"]
                    )
                    shuffled_seed = float(
                        seed_lookup[
                            (
                                direction, seed, "fusion_net", "spatial_shuffle",
                                "temperature_scaled",
                            )
                        ]["balanced_accuracy"]
                    )
                    seed_direction_deltas.append(
                        {
                            "direction": direction,
                            "seed": seed,
                            "full_balanced_accuracy": full_seed,
                            "spatial_shuffle_balanced_accuracy": shuffled_seed,
                            "delta_full_minus_shuffle": full_seed - shuffled_seed,
                        }
                    )
        positive_seed_direction_deltas = sum(
            row["delta_full_minus_shuffle"] > 0.0 for row in seed_direction_deltas
        )
        seed_stability_passed = (
            len(seed_direction_deltas) == 6 and positive_seed_direction_deltas >= 5
        )
        mechanism = {
            "estimand": "delta_spatial_arrangement_full_minus_spatial_shuffle",
            "model": "fusion_net",
            "calibration": "temperature_scaled_probability_ensemble",
            "directional_results": direction_records,
            "mean_delta": mean_delta,
            "directions_have_same_positive_sign": both_positive,
            "direction_by_seed_deltas": seed_direction_deltas,
            "positive_direction_by_seed_deltas": positive_seed_direction_deltas,
            "required_positive_direction_by_seed_deltas": "at least 5 of 6",
            "seed_stability_passed": seed_stability_passed,
            "minimum_effect_threshold": 0.02,
            "limited_support_for_spatial_arrangement": bool(
                both_positive and mean_delta >= 0.02 and seed_stability_passed
            ),
            "decision_rule": (
                "both ensemble directional deltas positive; equal-weight ensemble mean delta >=0.02; "
                "and at least 5/6 direction-by-seed deltas positive"
            ),
        }
    else:
        mechanism = {
            "estimand": "delta_spatial_arrangement_full_minus_spatial_shuffle",
            "status": "not_computed_fusion_net_not_requested",
        }
    return theta_rows, mechanism


def git_execution_state(repository_root: Path) -> dict[str, Any]:
    """Capture the exact committed code state before any output is created."""

    def run_git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return completed.stdout.strip()

    try:
        commit = run_git("rev-parse", "HEAD")
        branch = run_git("branch", "--show-current")
        status = run_git("status", "--porcelain=v1", "--untracked-files=all")
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "available": True,
        "commit": commit,
        "branch": branch,
        "worktree_clean_before_run": not bool(status),
        "porcelain_status_before_run": status.splitlines(),
    }


def _runtime_versions(device: Any) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "h5py": getattr(h5py, "__version__", None),
        "torch": getattr(torch, "__version__", None),
        "sklearn": getattr(sklearn, "__version__", None),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def _resolve_device(requested: str):
    if torch is None:
        raise RuntimeError("PyTorch is required for fusion_net")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Current-data source-cube-isolated top-journal audit")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=list(MODEL_CHOICES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-checkpoints", action="store_true")
    args = parser.parse_args(argv)
    args.models = list(dict.fromkeys(args.models))
    args.seeds = list(dict.fromkeys(args.seeds))
    if len(args.seeds) != 3:
        parser.error("The locked protocol requires exactly three unique seeds")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    return args


def run_analysis(
    args: argparse.Namespace,
    output_dir: Path,
    execution_context: dict[str, Any] | None = None,
) -> None:
    samples, wavelengths, fingerprints = discover_samples(args.data_root)
    neural_requested = any(model in args.models for model in ("spectral_only", "fusion_net"))
    if neural_requested and torch is None:
        raise RuntimeError("Neural models require PyTorch")
    if "fusion_net" in args.models and h5py is None:
        raise RuntimeError("fusion_net requires h5py")
    if "snv_lr" in args.models and LogisticRegression is None:
        raise RuntimeError("snv_lr requires scikit-learn")
    device = _resolve_device(args.device) if neural_requested else None
    amp_enabled = bool(device is not None and device.type == "cuda" and not args.no_amp)

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
            "mat_sha256": sample.mat_sha256,
            "csv_sha256": sample.csv_sha256,
        }
        for sample in samples
    ]
    _write_csv(output_dir / "manifest.csv", manifest_rows)
    _write_csv(
        output_dir / "wavelengths.csv",
        [{"band_index_zero_based": index, "wavelength_nm": float(value)} for index, value in enumerate(wavelengths)],
    )
    if "fusion_net" in args.models:
        consistency_rows, consistency_summary = audit_mat_csv_mean_consistency(samples)
        _write_csv(output_dir / "mat_csv_mean_consistency.csv", consistency_rows)
        _write_json(output_dir / "mat_csv_mean_consistency.json", consistency_summary)
        if consistency_summary["status"] != "passed":
            raise ValueError(
                "MAT foreground means do not match paired CSV spectra within 1e-5; "
                "see mat_csv_mean_consistency.csv"
            )
    else:
        consistency_summary = {
            "status": "not_run_fusion_net_not_requested",
            "purpose": "representation integrity only; not model selection or tuning",
        }

    predictions: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    metric_details: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    run_inventory: list[dict[str, Any]] = []
    initialization_fingerprints: dict[tuple[str, int, str], str] = {}

    for train_suffix in (1, 2):
        test_suffix = 2 if train_suffix == 1 else 1
        direction = f"suffix_{train_suffix}_to_{test_suffix}"
        development, test_samples = reciprocal_source_cube_split(samples, train_suffix)
        for seed in args.seeds:
            train_samples, validation_samples = internal_validation_split(development, seed)
            for role, role_samples in (
                ("development_train", train_samples),
                ("development_validation", validation_samples),
                ("locked_test", test_samples),
            ):
                for sample in role_samples:
                    split_rows.append(
                        {
                            "direction": direction,
                            "seed": seed,
                            "role": role,
                            "sample_id": sample.sample_id,
                            "source_cube": sample.source_cube,
                            "label": sample.label,
                        }
                    )

            for model_name in args.models:
                run_id = f"{direction}__{model_name}__seed_{seed}"
                print(f"Starting {run_id}", flush=True)
                started = time.perf_counter()
                if model_name == "snv_lr":
                    initialization_sha256 = None
                    scaler, model, temperature, candidates, selected_c = fit_snv_lr(
                        train_samples, validation_samples, seed
                    )
                    training_seconds = time.perf_counter() - started
                    parameter_count = int(model.coef_.size + model.intercept_.size)
                    for candidate in candidates:
                        selections.append(
                            {
                                "run_id": run_id,
                                "direction": direction,
                                "model": model_name,
                                "seed": seed,
                                "selection_type": "C",
                                "candidate_C": candidate["candidate_C"],
                                "selected_epoch": "",
                                "validation_accuracy": candidate["accuracy"],
                                "validation_macro_f1": candidate["macro_f1"],
                                "validation_nll": candidate["nll"],
                                "selected": int(candidate["candidate_C"] == selected_c),
                            }
                        )
                    validation_logits = model.decision_function(
                        scaler.transform(_spectral_matrix(validation_samples))
                    )
                else:
                    model, temperature, history, best_epoch, training_seconds, initialization_sha256 = train_neural(
                        train_samples, validation_samples, model_name, seed, device, args.num_workers, amp_enabled
                    )
                    parameter_count = sum(
                        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
                    )
                    initialization_fingerprints[(direction, seed, model_name)] = initialization_sha256
                    for row in history:
                        histories.append({"run_id": run_id, "direction": direction, "model": model_name, "seed": seed, **row})
                    selected_history = next(row for row in history if row["epoch"] == best_epoch)
                    selections.append(
                        {
                            "run_id": run_id,
                            "direction": direction,
                            "model": model_name,
                            "seed": seed,
                            "selection_type": "epoch",
                            "candidate_C": "",
                            "selected_epoch": best_epoch,
                            "validation_accuracy": selected_history["validation_accuracy"],
                            "validation_macro_f1": selected_history["validation_macro_f1"],
                            "validation_nll": selected_history["validation_nll"],
                            "selected": 1,
                        }
                    )
                    validation_loader = make_loader(
                        _neural_dataset(validation_samples, model_name, "full", False, seed),
                        False, seed, args.num_workers
                    )
                    validation_logits, _, _ = neural_logits(model, validation_loader, device, amp_enabled)
                    if not args.no_checkpoints:
                        torch.save(
                            {
                                "run_id": run_id,
                                "state_dict": model.state_dict(),
                                "temperature": temperature,
                                "wavelengths_nm": wavelengths,
                                "protocol": "training-suffix internal validation; test untouched",
                            },
                            output_dir / f"{run_id}.pt",
                        )

                validation_labels = [sample.label for sample in validation_samples]
                for calibration, probabilities in (
                    ("raw", softmax_numpy(validation_logits)),
                    ("temperature_scaled", softmax_numpy(validation_logits, temperature)),
                ):
                    validation_metrics = multiclass_metrics(validation_labels, probabilities)
                    calibration_rows.append(
                        {
                            "run_id": run_id,
                            "direction": direction,
                            "model": model_name,
                            "seed": seed,
                            "temperature": temperature,
                            "calibration": calibration,
                            **{key: validation_metrics[key] for key in ("nll", "brier", "ece_10", "accuracy", "macro_f1")},
                        }
                    )

                for condition in COUNTERFACTUALS:
                    evaluation_started = time.perf_counter()
                    if model_name == "snv_lr":
                        x_test = scaler.transform(_spectral_matrix(test_samples, condition))
                        test_logits = model.decision_function(x_test)
                    else:
                        test_dataset = _neural_dataset(test_samples, model_name, condition, False, seed)
                        test_loader = make_loader(test_dataset, False, seed, args.num_workers)
                        test_logits, test_labels_array, indices = neural_logits(
                            model, test_loader, device, amp_enabled
                        )
                        if indices != list(range(len(test_samples))):
                            raise AssertionError("Test loader did not preserve manifest order")
                        if test_labels_array.tolist() != [sample.label for sample in test_samples]:
                            raise AssertionError("Test labels do not match the locked manifest")
                    evaluation_seconds = time.perf_counter() - evaluation_started
                    common = {
                        "run_id": run_id,
                        "direction": direction,
                        "train_suffix": train_suffix,
                        "test_suffix": test_suffix,
                        "model": model_name,
                        "seed": seed,
                        "condition": condition,
                    }
                    predictions.extend(prediction_records(test_samples, test_logits, temperature, common))
                    scalar, detailed = metric_records(
                        [sample.label for sample in test_samples], test_logits, temperature,
                        {**common, "n_train": len(train_samples), "n_validation": len(validation_samples),
                         "training_seconds": training_seconds, "evaluation_seconds": evaluation_seconds,
                         "parameter_count": parameter_count},
                    )
                    metrics.extend(scalar)
                    metric_details.extend(detailed)

                run_inventory.append(
                    {
                        "run_id": run_id,
                        "direction": direction,
                        "model": model_name,
                        "seed": seed,
                        "training_seconds": training_seconds,
                        "parameter_count": parameter_count,
                        "temperature": temperature,
                        "paired_spectral_classifier_initialization_sha256": initialization_sha256,
                    }
                )
                _write_csv(output_dir / "predictions.csv", predictions)
                _write_csv(output_dir / "metrics.csv", metrics)
                _write_csv(output_dir / "training_history.csv", histories)
                _write_csv(output_dir / "model_selection.csv", selections)
                _write_csv(output_dir / "development_validation_calibration.csv", calibration_rows)
                _write_json(
                    output_dir / "run_status.json",
                    {
                        "status": "running",
                        "execution": execution_context,
                        "completed_runs": run_inventory,
                        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
                if model_name in ("spectral_only", "fusion_net") and device.type == "cuda":
                    del model
                    torch.cuda.empty_cache()

    if all(model in args.models for model in ("spectral_only", "fusion_net")):
        for direction in ("suffix_1_to_2", "suffix_2_to_1"):
            for seed in args.seeds:
                spectral_hash = initialization_fingerprints[(direction, seed, "spectral_only")]
                fusion_hash = initialization_fingerprints[(direction, seed, "fusion_net")]
                if spectral_hash != fusion_hash:
                    raise AssertionError(
                        f"Shared initialization check failed for {direction}, seed={seed}"
                    )
        shared_initialization_verified = True
    else:
        shared_initialization_verified = None

    cube_rows = cube_metric_records(predictions)
    paired_rows = counterfactual_pairs(predictions)
    aggregate_rows = aggregate_seed_metrics(metrics)
    ensemble_predictions = probability_ensemble_predictions(predictions, args.seeds)
    ensemble_metrics, ensemble_metric_details = ensemble_metric_records(ensemble_predictions)
    ensemble_cube_rows = cube_metric_records(ensemble_predictions)
    ensemble_paired_rows = counterfactual_pairs(ensemble_predictions)
    theta_rows, spatial_mechanism = compute_primary_estimands(ensemble_metrics, metrics)
    _write_csv(output_dir / "splits.csv", split_rows)
    _write_csv(output_dir / "cube_metrics.csv", cube_rows)
    _write_csv(output_dir / "counterfactual_pairs.csv", paired_rows)
    _write_csv(output_dir / "metrics_seed_aggregate.csv", aggregate_rows)
    _write_csv(output_dir / "ensemble_predictions.csv", ensemble_predictions)
    _write_csv(output_dir / "ensemble_metrics.csv", ensemble_metrics)
    _write_csv(output_dir / "ensemble_cube_metrics.csv", ensemble_cube_rows)
    _write_csv(output_dir / "ensemble_counterfactual_pairs.csv", ensemble_paired_rows)
    _write_csv(output_dir / "primary_estimands.csv", theta_rows)
    _write_json(output_dir / "spatial_mechanism_decision.json", spatial_mechanism)
    results = {
        "status": "executed_complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution": execution_context,
        "protocol": {
            "directions": ["suffix_1_to_2", "suffix_2_to_1"],
            "models": args.models,
            "seeds": args.seeds,
            "counterfactuals": list(COUNTERFACTUALS),
            "test_policy": "test cubes evaluated only after internal selection and calibration; never used for tuning",
            "internal_validation_limitation": "seed-level validation within each development cube because only one cube per class is available",
            "epoch_selection_rule": "maximum development-validation macro-F1; ties use minimum NLL; exact ties retain the earliest epoch",
            "shared_pairing": "spectral_only and fusion_net use identical splits, seeds, spectral branch, classifier architecture, and initial spectral/classifier parameters",
            "shared_initialization_verified": shared_initialization_verified,
            "neural_hyperparameters": {
                "batch_size": BATCH_SIZE,
                "max_epochs": MAX_EPOCHS,
                "min_epochs": MIN_EPOCHS,
                "patience": EARLY_STOPPING_PATIENCE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "label_smoothing": LABEL_SMOOTHING,
            },
            "lr_C_candidates": list(LR_CANDIDATES),
        },
        "dataset": {
            "n_samples": len(samples),
            "n_source_cubes": len({sample.source_cube for sample in samples}),
            **fingerprints,
            "mat_csv_mean_consistency": consistency_summary,
        },
        "software": _runtime_versions(device) if device is not None else {
            "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
            "sklearn": getattr(sklearn, "__version__", None), "device": None,
        },
        "run_inventory": run_inventory,
        "metrics_with_confusion_matrices": metric_details,
        "ensemble_metrics_with_confusion_matrices": ensemble_metric_details,
        "seed_aggregates": aggregate_rows,
        "primary_estimands": theta_rows,
        "spatial_mechanism_decision": spatial_mechanism,
        "limitations": [
            "Each test class contains only one acquisition cube; seed predictions are clustered technical subsamples.",
            "The reciprocal suffix test is not external farm, lot, harvest-year, supplier, or instrument validation.",
            "Internal validation is seed-level within a cube and is used only for training control, not as generalization evidence.",
            "Temperature calibration is learned on the internal validation seeds and may not transfer under cube shift.",
            "Counterfactual effects diagnose model dependence but do not establish chemical causality.",
        ],
    }
    _write_json(output_dir / "results.json", results)
    _write_json(
        output_dir / "run_status.json",
        {
            "status": "executed_complete",
            "execution": execution_context,
            "completed_runs": run_inventory,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    repository_root = Path(__file__).resolve().parents[1]
    execution = {
        "started_at_utc": started_at.isoformat(),
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "git": git_execution_state(repository_root),
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else Path(__file__).resolve().parent / "outputs" / f"top_journal_{timestamp}"
    )
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite or mix output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    _write_json(
        output_dir / "run_status.json",
        {
            "status": "initializing",
            "execution": execution,
            "arguments": vars(args)
            | {"data_root": str(args.data_root), "output_dir": str(output_dir)},
        },
    )
    try:
        run_analysis(args, output_dir, execution)
    except Exception as exc:
        _write_json(
            output_dir / "run_status.json",
            {
                "status": "failed",
                "execution": execution,
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.perf_counter() - started_clock,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    completed_at = datetime.now(timezone.utc)
    execution.update(
        {
            "completed_at_utc": completed_at.isoformat(),
            "elapsed_seconds": time.perf_counter() - started_clock,
        }
    )
    results_path = output_dir / "results.json"
    with results_path.open("r", encoding="utf-8") as handle:
        results = json.load(handle)
    results["execution"] = execution
    _write_json(results_path, results)
    _write_json(
        output_dir / "run_status.json",
        {
            "status": "executed_complete",
            "execution": execution,
            "completed_runs": results["run_inventory"],
        },
    )
    print(f"Complete auditable outputs: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
