"""Deterministic residual 1-D CNN baseline for origin classification.

The module contains the exact architecture and training rules used in the
development-only CNN benchmark.  It deliberately accepts in-memory arrays and
has no data-discovery or file-reading code.  Callers therefore retain control
over the development/locked-data boundary.

Two fitting routes are provided:

* :func:`evaluate_development_batches` performs the strict eight-fold grouped
  development evaluation.  For outer batch ``g``, batch ``(g + 1) % 8`` is
  used only to choose the epoch.  A fresh network is then trained on every
  non-outer development batch for exactly that many epochs.
* :func:`fit_full_development_cnn` fits the deployment model for the frozen
  88-epoch duration.  :func:`predict_external_probabilities` applies its
  training-fitted preprocessing to an external feature matrix.

Savitzky--Golay differentiation is sample-wise and has no fitted state.  Every
band standardizer is fitted only on the applicable training partition.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import savgol_filter
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss

from .core import DEVELOPMENT_BATCHES, NUM_CLASSES


CNN_PARAMETER_COUNT = 321_776
FULL_DEVELOPMENT_EPOCHS = 88


@dataclass(frozen=True)
class CNNTrainingConfig:
    """Frozen optimization and regularization settings for the CNN baseline."""

    max_epochs: int = 120
    min_epochs: int = 25
    patience: int = 18
    batch_size: int = 64
    learning_rate: float = 5e-4
    weight_decay: float = 1e-3
    label_smoothing: float = 0.05
    mixup_alpha: float = 0.20
    noise_std: float = 0.01
    gradient_clip_norm: float = 5.0
    scheduler_minimum_fraction: float = 0.02
    savgol_window_length: int = 15
    savgol_polyorder: int = 2
    savgol_derivative: int = 1

    def validate(self) -> None:
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be positive")
        if not 1 <= self.min_epochs <= self.max_epochs:
            raise ValueError("min_epochs must be in [1, max_epochs]")
        if self.patience < 1 or self.batch_size < 1:
            raise ValueError("patience and batch_size must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if self.mixup_alpha <= 0.0 or self.noise_std < 0.0:
            raise ValueError("mixup_alpha must be positive and noise_std non-negative")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        if not 0.0 < self.scheduler_minimum_fraction <= 1.0:
            raise ValueError("scheduler_minimum_fraction must be in (0, 1]")
        if self.savgol_window_length < 3 or self.savgol_window_length % 2 == 0:
            raise ValueError("savgol_window_length must be an odd integer of at least three")
        if not 0 <= self.savgol_polyorder < self.savgol_window_length:
            raise ValueError("savgol_polyorder must be below the window length")
        if not 0 <= self.savgol_derivative <= self.savgol_polyorder:
            raise ValueError("savgol_derivative must not exceed savgol_polyorder")


DEFAULT_CNN_CONFIG = CNNTrainingConfig()


@dataclass(frozen=True)
class BandStandardizer:
    """Immutable, training-fitted per-band population standardizer."""

    mean: np.ndarray
    scale: np.ndarray
    n_samples_seen: int

    @classmethod
    def fit(cls, X_train: np.ndarray) -> "BandStandardizer":
        values = _validate_feature_matrix(X_train, "X_train")
        mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[scale < 1e-8] = 1.0
        return cls(mean=mean, scale=scale, n_samples_seen=int(values.shape[0]))

    def transform(self, X: np.ndarray) -> np.ndarray:
        values = _validate_feature_matrix(X, "X")
        if values.shape[1] != self.mean.size:
            raise ValueError(
                f"Expected {self.mean.size} spectral features; observed {values.shape[1]}"
            )
        return ((values - self.mean) / self.scale).astype(np.float32)


@dataclass(frozen=True)
class CNNFoldResult:
    outer_batch: int
    inner_validation_batch: int
    selected_epoch: int
    early_stopping_epochs_run: int
    inner_validation_balanced_accuracy: float
    outer_balanced_accuracy: float
    outer_macro_f1: float


@dataclass(frozen=True)
class CNNGroupedOOFResult:
    probabilities: np.ndarray
    classes: np.ndarray
    held_out_batch: np.ndarray
    folds: tuple[CNNFoldResult, ...]
    optimization_seed: int
    parameter_count: int
    elapsed_seconds: float


@dataclass(frozen=True)
class FittedCNN:
    """A full-development network together with its fitted preprocessing."""

    model: "ResidualSpectralCNN"
    standardizer: BandStandardizer
    classes: np.ndarray
    optimization_seed: int
    epochs: int
    raw_band_count: int
    training_config: CNNTrainingConfig

    def predict_proba(
        self, X_external: np.ndarray, *, batch_size: int | None = None
    ) -> np.ndarray:
        return predict_external_probabilities(self, X_external, batch_size=batch_size)


class ResidualBlock1D(nn.Module):
    """Two-convolution residual block used at every spectral resolution."""

    def __init__(self, channels: int, dropout: float = 0.05) -> None:
        super().__init__()
        groups = 8 if channels % 8 == 0 else 4
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=7, padding=3, bias=False
        )
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=5, padding=2, bias=False
        )
        self.norm2 = nn.GroupNorm(groups, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        residual = X
        X = F.gelu(self.norm1(self.conv1(X)))
        X = self.dropout(X)
        X = self.norm2(self.conv2(X))
        return F.gelu(X + residual)


class ResidualSpectralCNN(nn.Module):
    """Three-stage residual spectral CNN with global average pooling."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 24, kernel_size=9, padding=4, bias=False),
            nn.GroupNorm(8, 24),
            nn.GELU(),
        )
        self.stage1 = nn.Sequential(ResidualBlock1D(24), ResidualBlock1D(24))
        self.down2 = nn.Sequential(
            nn.Conv1d(24, 48, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 48),
            nn.GELU(),
        )
        self.stage2 = nn.Sequential(ResidualBlock1D(48), ResidualBlock1D(48))
        self.down3 = nn.Sequential(
            nn.Conv1d(48, 96, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 96),
            nn.GELU(),
        )
        self.stage3 = nn.Sequential(ResidualBlock1D(96), ResidualBlock1D(96))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(96, NUM_CLASSES),
        )
        observed = count_trainable_parameters(self)
        if observed != CNN_PARAMETER_COUNT:
            raise RuntimeError(
                f"CNN architecture drift: expected {CNN_PARAMETER_COUNT} parameters, "
                f"observed {observed}"
            )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.stage1(self.stem(X))
        X = self.stage2(self.down2(X))
        X = self.stage3(self.down3(X))
        return self.head(X)


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def seed_everything(seed: int) -> None:
    """Set Python, NumPy, CPU, and CUDA state for deterministic optimization."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    # ``torch.device('cuda')`` and a module's concrete ``cuda:0`` device refer
    # to the same accelerator, but PyTorch device equality treats them as
    # distinct objects.  Canonicalize an unspecified CUDA index before the
    # strict model/input device check so the public ``--device cuda`` CLI works
    # without weakening protection against a genuine cross-device mismatch.
    if resolved.type == "cuda" and resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


def _validate_feature_matrix(X: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(X, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty sample-by-band matrix")
    if np.any(~np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    return values


def preprocess_sg1(
    X: np.ndarray, *, config: CNNTrainingConfig = DEFAULT_CNN_CONFIG
) -> np.ndarray:
    """Apply the frozen sample-wise SG first-derivative transformation."""

    config.validate()
    values = _validate_feature_matrix(X, "X")
    if values.shape[1] < config.savgol_window_length:
        raise ValueError(
            f"At least {config.savgol_window_length} bands are required; "
            f"observed {values.shape[1]}"
        )
    return savgol_filter(
        values,
        window_length=config.savgol_window_length,
        polyorder=config.savgol_polyorder,
        deriv=config.savgol_derivative,
        axis=1,
    ).astype(np.float32)


def _validate_labels(y: Sequence[int], n_samples: int) -> np.ndarray:
    labels = np.asarray(y, dtype=np.int64)
    if labels.ndim != 1 or labels.size != n_samples:
        raise ValueError("y must be one-dimensional and aligned with X")
    expected = np.arange(NUM_CLASSES, dtype=np.int64)
    if not np.array_equal(np.unique(labels), expected):
        raise ValueError(
            f"CNN fitting requires every class {expected.tolist()}; "
            f"observed {np.unique(labels).tolist()}"
        )
    return labels


def _class_weights(y: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(np.float32)
    if np.any(counts == 0):
        raise ValueError("Every training partition must contain every class")
    weights = y.size / (NUM_CLASSES * counts)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _epoch_batches(
    n_samples: int, *, batch_size: int, seed: int, epoch: int
) -> tuple[np.ndarray, ...]:
    order = np.random.default_rng(seed + epoch * 10_007).permutation(n_samples)
    return tuple(
        order[start : start + batch_size]
        for start in range(0, n_samples, batch_size)
    )


def _training_epoch(
    model: ResidualSpectralCNN,
    optimizer: torch.optim.Optimizer,
    X: np.ndarray,
    y: np.ndarray,
    weights: torch.Tensor,
    device: torch.device,
    seed: int,
    epoch: int,
    config: CNNTrainingConfig,
) -> float:
    model.train()
    generator = np.random.default_rng(seed + 50_021 * epoch)
    total_loss = 0.0
    seen = 0
    for indices in _epoch_batches(
        len(y), batch_size=config.batch_size, seed=seed, epoch=epoch
    ):
        inputs = torch.as_tensor(X[indices], device=device).unsqueeze(1)
        targets = torch.as_tensor(y[indices], device=device)
        noise = torch.as_tensor(
            generator.normal(0.0, config.noise_std, size=inputs.shape),
            dtype=inputs.dtype,
            device=device,
        )
        inputs = inputs + noise
        mixing_weight = float(generator.beta(config.mixup_alpha, config.mixup_alpha))
        pairing = torch.as_tensor(
            generator.permutation(len(indices)), dtype=torch.long, device=device
        )
        mixed_inputs = (
            mixing_weight * inputs + (1.0 - mixing_weight) * inputs[pairing]
        )
        logits = model(mixed_inputs)
        loss_a = F.cross_entropy(
            logits,
            targets,
            weight=weights,
            label_smoothing=config.label_smoothing,
        )
        loss_b = F.cross_entropy(
            logits,
            targets[pairing],
            weight=weights,
            label_smoothing=config.label_smoothing,
        )
        loss = mixing_weight * loss_a + (1.0 - mixing_weight) * loss_b
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()
        total_loss += float(loss.detach()) * len(indices)
        seen += len(indices)
    return total_loss / seen


@torch.inference_mode()
def predict_standardized_probabilities(
    model: ResidualSpectralCNN,
    X_standardized: np.ndarray,
    *,
    batch_size: int = 256,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Predict class probabilities from an already standardized matrix."""

    values = _validate_feature_matrix(X_standardized, "X_standardized")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model_device = next(model.parameters()).device
    resolved = model_device if device is None else _resolve_device(device)
    if resolved != model_device:
        raise ValueError(
            f"Model is on {model_device}, but prediction device {resolved} was requested"
        )
    model.eval()
    outputs: list[np.ndarray] = []
    for start in range(0, len(values), batch_size):
        inputs = torch.as_tensor(
            values[start : start + batch_size], device=resolved
        ).unsqueeze(1)
        outputs.append(torch.softmax(model(inputs), dim=1).cpu().numpy())
    probabilities = np.concatenate(outputs).astype(np.float64)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    if probabilities.shape != (len(values), NUM_CLASSES):
        raise RuntimeError("CNN produced an unexpected probability shape")
    return probabilities


def _new_optimizer_and_scheduler(
    model: ResidualSpectralCNN, config: CNNTrainingConfig
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.CosineAnnealingLR]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.max_epochs,
        eta_min=config.learning_rate * config.scheduler_minimum_fraction,
    )
    return optimizer, scheduler


def _select_epoch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    config: CNNTrainingConfig,
) -> tuple[int, int, float]:
    seed_everything(seed)
    model = ResidualSpectralCNN().to(device)
    weights = _class_weights(y_train, device)
    optimizer, scheduler = _new_optimizer_and_scheduler(model, config)
    best_epoch = 1
    best_balanced_accuracy = -np.inf
    best_nll = np.inf
    stale_epochs = 0
    epochs_run = 0
    for epoch in range(1, config.max_epochs + 1):
        _training_epoch(
            model,
            optimizer,
            X_train,
            y_train,
            weights,
            device,
            seed,
            epoch,
            config,
        )
        scheduler.step()
        probabilities = predict_standardized_probabilities(
            model, X_validation, device=device
        )
        predictions = probabilities.argmax(axis=1)
        balanced_accuracy = float(
            balanced_accuracy_score(y_validation, predictions)
        )
        negative_log_likelihood = float(
            log_loss(y_validation, probabilities, labels=np.arange(NUM_CLASSES))
        )
        improved = balanced_accuracy > best_balanced_accuracy + 1e-12 or (
            abs(balanced_accuracy - best_balanced_accuracy) <= 1e-12
            and negative_log_likelihood < best_nll
        )
        if improved:
            best_epoch = epoch
            best_balanced_accuracy = balanced_accuracy
            best_nll = negative_log_likelihood
            stale_epochs = 0
        else:
            stale_epochs += 1
        epochs_run = epoch
        if epoch >= config.min_epochs and stale_epochs >= config.patience:
            break
    return best_epoch, epochs_run, best_balanced_accuracy


def _fit_fixed_epochs(
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    device: torch.device,
    seed: int,
    config: CNNTrainingConfig,
) -> ResidualSpectralCNN:
    if not 1 <= epochs <= config.max_epochs:
        raise ValueError("epochs must be in [1, config.max_epochs]")
    seed_everything(seed)
    model = ResidualSpectralCNN().to(device)
    weights = _class_weights(y, device)
    optimizer, scheduler = _new_optimizer_and_scheduler(model, config)
    for epoch in range(1, epochs + 1):
        _training_epoch(
            model,
            optimizer,
            X,
            y,
            weights,
            device,
            seed,
            epoch,
            config,
        )
        scheduler.step()
    model.eval()
    return model


def evaluate_development_batches(
    X: np.ndarray,
    y: Sequence[int],
    constructed_batches: Sequence[int],
    *,
    optimization_seed: int = 20260721,
    config: CNNTrainingConfig = DEFAULT_CNN_CONFIG,
    device: str | torch.device | None = None,
) -> CNNGroupedOOFResult:
    """Run nested, leakage-controlled OOF evaluation on batches 0--7 only.

    The function rejects any group set other than the eight fixed development
    batches.  It consequently cannot silently evaluate batches 8--9.
    """

    started = time.perf_counter()
    config.validate()
    raw = _validate_feature_matrix(X, "X")
    labels = _validate_labels(y, raw.shape[0])
    groups = np.asarray(constructed_batches, dtype=np.int64)
    if groups.ndim != 1 or groups.size != labels.size:
        raise ValueError("constructed_batches must be one-dimensional and aligned with X")
    expected_groups = np.asarray(DEVELOPMENT_BATCHES, dtype=np.int64)
    if set(groups.tolist()) != set(expected_groups.tolist()):
        raise ValueError(
            "CNN development evaluation requires exactly constructed batches 0--7; "
            f"observed {sorted(set(groups.tolist()))}"
        )
    expected_classes = set(range(NUM_CLASSES))
    for group in expected_groups:
        if set(labels[groups == group].tolist()) != expected_classes:
            raise ValueError(f"Development batch {group} does not contain every class")

    transformed = preprocess_sg1(raw, config=config)
    probabilities = np.full((len(labels), NUM_CLASSES), np.nan, dtype=np.float64)
    held_out_batch = np.full(len(labels), -1, dtype=np.int64)
    fold_results: list[CNNFoldResult] = []
    resolved_device = _resolve_device(device)

    for outer_batch in expected_groups:
        inner_validation_batch = (int(outer_batch) + 1) % len(expected_groups)
        outer_test = groups == outer_batch
        outer_train = ~outer_test
        inner_validation = groups == inner_validation_batch
        inner_train = outer_train & ~inner_validation
        if (
            np.any(inner_train & inner_validation)
            or np.any(inner_train & outer_test)
            or np.any(inner_validation & outer_test)
        ):
            raise RuntimeError("Constructed-batch leakage in nested CNN split")

        inner_standardizer = BandStandardizer.fit(transformed[inner_train])
        X_inner_train = inner_standardizer.transform(transformed[inner_train])
        X_inner_validation = inner_standardizer.transform(
            transformed[inner_validation]
        )
        selection_seed = optimization_seed + int(outer_batch) * 1_009 + 17
        selected_epoch, epochs_run, inner_balanced_accuracy = _select_epoch(
            X_inner_train,
            labels[inner_train],
            X_inner_validation,
            labels[inner_validation],
            device=resolved_device,
            seed=selection_seed,
            config=config,
        )

        outer_standardizer = BandStandardizer.fit(transformed[outer_train])
        X_outer_train = outer_standardizer.transform(transformed[outer_train])
        X_outer_test = outer_standardizer.transform(transformed[outer_test])
        final_seed = optimization_seed + int(outer_batch) * 1_009 + 503
        model = _fit_fixed_epochs(
            X_outer_train,
            labels[outer_train],
            epochs=selected_epoch,
            device=resolved_device,
            seed=final_seed,
            config=config,
        )
        fold_probabilities = predict_standardized_probabilities(
            model, X_outer_test, device=resolved_device
        )
        probabilities[outer_test] = fold_probabilities
        held_out_batch[outer_test] = outer_batch
        fold_predictions = fold_probabilities.argmax(axis=1)
        fold_results.append(
            CNNFoldResult(
                outer_batch=int(outer_batch),
                inner_validation_batch=inner_validation_batch,
                selected_epoch=selected_epoch,
                early_stopping_epochs_run=epochs_run,
                inner_validation_balanced_accuracy=inner_balanced_accuracy,
                outer_balanced_accuracy=float(
                    balanced_accuracy_score(labels[outer_test], fold_predictions)
                ),
                outer_macro_f1=float(
                    f1_score(
                        labels[outer_test],
                        fold_predictions,
                        labels=np.arange(NUM_CLASSES),
                        average="macro",
                        zero_division=0,
                    )
                ),
            )
        )
        del model
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()

    if np.any(~np.isfinite(probabilities)) or np.any(held_out_batch < 0):
        raise RuntimeError("Incomplete grouped CNN OOF predictions")
    return CNNGroupedOOFResult(
        probabilities=probabilities,
        classes=np.arange(NUM_CLASSES, dtype=np.int64),
        held_out_batch=held_out_batch,
        folds=tuple(fold_results),
        optimization_seed=int(optimization_seed),
        parameter_count=CNN_PARAMETER_COUNT,
        elapsed_seconds=float(time.perf_counter() - started),
    )


def fit_full_development_cnn(
    X_development: np.ndarray,
    y_development: Sequence[int],
    *,
    epochs: int = FULL_DEVELOPMENT_EPOCHS,
    optimization_seed: int = 20260721,
    config: CNNTrainingConfig = DEFAULT_CNN_CONFIG,
    device: str | torch.device | None = None,
) -> FittedCNN:
    """Fit the full-development CNN for the frozen 88 epochs by default."""

    config.validate()
    raw = _validate_feature_matrix(X_development, "X_development")
    labels = _validate_labels(y_development, raw.shape[0])
    transformed = preprocess_sg1(raw, config=config)
    standardizer = BandStandardizer.fit(transformed)
    standardized = standardizer.transform(transformed)
    resolved_device = _resolve_device(device)
    model = _fit_fixed_epochs(
        standardized,
        labels,
        epochs=epochs,
        device=resolved_device,
        seed=optimization_seed,
        config=config,
    )
    return FittedCNN(
        model=model,
        standardizer=standardizer,
        classes=np.arange(NUM_CLASSES, dtype=np.int64),
        optimization_seed=int(optimization_seed),
        epochs=int(epochs),
        raw_band_count=int(raw.shape[1]),
        training_config=config,
    )


def predict_external_probabilities(
    fitted: FittedCNN,
    X_external: np.ndarray,
    *,
    batch_size: int | None = None,
) -> np.ndarray:
    """Apply a full-development model without refitting any external-data state."""

    raw = _validate_feature_matrix(X_external, "X_external")
    if raw.shape[1] != fitted.raw_band_count:
        raise ValueError(
            f"Expected {fitted.raw_band_count} raw bands; observed {raw.shape[1]}"
        )
    transformed = preprocess_sg1(raw, config=fitted.training_config)
    standardized = fitted.standardizer.transform(transformed)
    prediction_batch_size = (
        fitted.training_config.batch_size if batch_size is None else batch_size
    )
    return predict_standardized_probabilities(
        fitted.model,
        standardized,
        batch_size=prediction_batch_size,
    )
