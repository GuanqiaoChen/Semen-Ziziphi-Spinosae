"""Core primitives for leakage-controlled geographical-origin modelling.

Manifest discovery inspects directory entries and file metadata, but never
parses a CSV or MAT numeric payload.  Content hashing is explicit and disabled
by default so development cannot open locked file bytes.  Numerical CSV loading
is a separate, split-explicit operation.  This separation makes accidental
access to locked spectra testable and auditable.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_array, check_is_fitted


EXPECTED_BANDS = 392
NUM_CLASSES = 8
CLASS_NAMES = ("HBS", "HBX", "HNA", "HNX", "NX", "SXD", "SXQ", "XJH")
BASE_BATCH_SEED = 20260721
CONSTRUCTED_BATCHES = tuple(range(10))
DEVELOPMENT_BATCHES = tuple(range(8))
LOCKED_BATCHES = (8, 9)
SplitName = Literal["development", "locked"]

# These are repository-data expectations, not parameters estimated from labels.
EXPECTED_SOURCE_COUNTS: Mapping[tuple[int, int], int] = {
    **{(label, replicate): 80 for label in range(NUM_CLASSES) for replicate in (1, 2)},
    (4, 1): 90,
    (7, 2): 54,
}

_SOURCE_DIRECTORY_PATTERN = re.compile(r"^(?P<label>\d+)-(?P<replicate>\d+)$")


class LockedDataAccessError(ValueError):
    """Raised before I/O when a development loader receives locked records."""


@dataclass(frozen=True)
class SampleRecord:
    """One seed and its deterministic analysis assignment."""

    sample_index: int
    label: int
    class_name: str
    replicate: int
    source_cube: str
    seed_id: str
    constructed_batch: int
    analysis_split: SplitName
    csv_path: Path
    mat_path: Path
    relative_csv_path: str
    relative_mat_path: str
    csv_path_sha256: str
    mat_path_sha256: str
    csv_size_bytes: int
    mat_size_bytes: int
    csv_sha256: str
    mat_sha256: str
    record_sha256: str

    @property
    def sample_id(self) -> str:
        return f"{self.source_cube}/{self.seed_id}"

    def as_serializable_dict(self) -> dict[str, Any]:
        """Return a path-portable row suitable for CSV or JSON artifacts."""

        row = asdict(self)
        row["csv_path"] = str(self.csv_path)
        row["mat_path"] = str(self.mat_path)
        return row


@dataclass(frozen=True)
class Manifest:
    """Validated 16-source manifest plus content and assignment fingerprints."""

    data_root: Path
    records: tuple[SampleRecord, ...]
    manifest_sha256: str
    csv_content_sha256: str
    mat_content_sha256: str
    data_fingerprint_sha256: str
    hashes_complete: bool

    def records_for_split(self, split: SplitName) -> tuple[SampleRecord, ...]:
        split = _validate_split_name(split)
        return tuple(record for record in self.records if record.analysis_split == split)

    def records_for_batches(self, batches: Iterable[int]) -> tuple[SampleRecord, ...]:
        requested = frozenset(int(batch) for batch in batches)
        unknown = requested - frozenset(CONSTRUCTED_BATCHES)
        if unknown:
            raise ValueError(f"Unknown constructed batches: {sorted(unknown)}")
        return tuple(record for record in self.records if record.constructed_batch in requested)


@dataclass(frozen=True)
class SpectralDataset:
    """Numerical spectra loaded from exactly one explicit analysis split."""

    X: np.ndarray
    y: np.ndarray
    wavelengths: np.ndarray
    records: tuple[SampleRecord, ...]
    analysis_split: SplitName
    loaded_csv_fingerprint_sha256: str


@dataclass(frozen=True)
class OOFProbabilityResult:
    probabilities: np.ndarray
    classes: np.ndarray
    held_out_batch: np.ndarray
    fold_metrics: tuple[dict[str, float | int], ...]


@dataclass(frozen=True)
class OOFDecisionResult:
    decision_scores: np.ndarray
    classes: np.ndarray
    held_out_batch: np.ndarray


@dataclass(frozen=True)
class TemperatureCalibrationResult:
    """Cross-fitted development probabilities and the deployable final T."""

    probabilities: np.ndarray
    final_temperature: float
    fold_temperatures: tuple[tuple[int, float], ...]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without interpreting its payload."""

    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_split_name(split: str) -> SplitName:
    if split not in ("development", "locked"):
        raise ValueError("split must be explicitly 'development' or 'locked'")
    return split  # type: ignore[return-value]


def _numeric_stem(path: Path) -> int:
    if not path.stem.isdecimal():
        raise ValueError(f"Seed filename stem must be an unsigned integer: {path.name}")
    return int(path.stem)


def _validate_expected_counts(
    expected_source_counts: Mapping[tuple[int, int], int],
) -> dict[tuple[int, int], int]:
    expected_keys = {(label, replicate) for label in range(NUM_CLASSES) for replicate in (1, 2)}
    normalized = {tuple(key): int(value) for key, value in expected_source_counts.items()}
    if set(normalized) != expected_keys:
        raise ValueError(
            "expected_source_counts must define exactly labels 0..7 and replicates 1..2; "
            f"missing={sorted(expected_keys - set(normalized))}, "
            f"extra={sorted(set(normalized) - expected_keys)}"
        )
    if any(count <= 0 for count in normalized.values()):
        raise ValueError("Every expected source count must be positive")
    return normalized


def _paired_paths(cube_dir: Path) -> list[tuple[str, Path, Path]]:
    csv_paths = list(cube_dir.glob("*.csv"))
    mat_paths = list(cube_dir.glob("*.mat"))
    csv_by_stem = {path.stem: path for path in csv_paths}
    mat_by_stem = {path.stem: path for path in mat_paths}
    if len(csv_by_stem) != len(csv_paths) or len(mat_by_stem) != len(mat_paths):
        raise ValueError(f"Duplicate filename stems in {cube_dir}")
    if set(csv_by_stem) != set(mat_by_stem):
        raise ValueError(
            f"CSV/MAT stem mismatch in {cube_dir}: "
            f"CSV-only={sorted(set(csv_by_stem) - set(mat_by_stem))}, "
            f"MAT-only={sorted(set(mat_by_stem) - set(csv_by_stem))}"
        )
    normalized_ids: dict[int, str] = {}
    for stem in csv_by_stem:
        numeric_id = _numeric_stem(csv_by_stem[stem])
        if numeric_id in normalized_ids:
            raise ValueError(
                f"Numerically duplicate seed IDs in {cube_dir}: "
                f"{normalized_ids[numeric_id]!r} and {stem!r}"
            )
        normalized_ids[numeric_id] = stem
    return [
        (stem, csv_by_stem[stem], mat_by_stem[stem])
        for stem in sorted(csv_by_stem, key=lambda value: (int(value), value))
    ]


def discover_manifest(
    data_root: Path,
    *,
    expected_source_counts: Mapping[tuple[int, int], int] = EXPECTED_SOURCE_COUNTS,
    base_seed: int = BASE_BATCH_SEED,
    hash_files: bool = False,
) -> Manifest:
    """Discover paths, validate structure, and assign batches without numeric parsing.

    Within every label-by-source directory, numeric seed IDs are sorted.  A
    deterministic vector of randomized ranks is generated with
    ``base_seed + label*101 + replicate*1009``; rank modulo ten is the constructed
    batch.  Batches 0--7 are development and batches 8--9 are locked.

    The default ``hash_files=False`` is mandatory during blind development: it
    permits path enumeration and file-size inspection but does not open locked
    file bytes.  Only the authorized final evaluation should explicitly request
    ``hash_files=True``; hashing then reads opaque bytes without parsing values.
    With hashing disabled, content fingerprints are intentionally blank and
    ``hashes_complete`` is false.
    """

    data_root = Path(data_root).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root is not a directory: {data_root}")
    counts = _validate_expected_counts(expected_source_counts)

    observed_dirs: dict[tuple[int, int], Path] = {}
    malformed_numeric_dirs: list[str] = []
    for candidate in data_root.iterdir():
        if not candidate.is_dir():
            continue
        match = _SOURCE_DIRECTORY_PATTERN.fullmatch(candidate.name)
        if match is None:
            continue
        key = (int(match.group("label")), int(match.group("replicate")))
        if key not in counts:
            malformed_numeric_dirs.append(candidate.name)
        elif key in observed_dirs:
            raise ValueError(f"Duplicate source directory for {key}: {candidate}")
        else:
            observed_dirs[key] = candidate
    if malformed_numeric_dirs:
        raise ValueError(f"Unexpected numeric source directories: {sorted(malformed_numeric_dirs)}")
    if set(observed_dirs) != set(counts):
        raise ValueError(
            "Expected exactly the 16 directories 0-1 ... 7-2; "
            f"missing={sorted(set(counts) - set(observed_dirs))}"
        )

    records: list[SampleRecord] = []
    manifest_digest = hashlib.sha256()
    csv_digest = hashlib.sha256()
    mat_digest = hashlib.sha256()
    data_digest = hashlib.sha256()

    for (label, replicate), cube_dir in sorted(observed_dirs.items()):
        paired = _paired_paths(cube_dir)
        expected_count = counts[(label, replicate)]
        if len(paired) != expected_count:
            raise ValueError(
                f"Expected {expected_count} CSV/MAT pairs in {cube_dir.name}; observed {len(paired)}"
            )

        rng_seed = int(base_seed) + label * 101 + replicate * 1009
        randomized_order = np.random.default_rng(rng_seed).permutation(len(paired))
        # ``randomized_order`` contains source indices in shuffled order.  Batch
        # IDs come from each sample's rank in that order, so the permutation must
        # be inverted before indexing by the numeric-stem sorted position.
        assigned_batches = np.empty(len(paired), dtype=np.int64)
        assigned_batches[randomized_order] = np.arange(len(paired)) % len(
            CONSTRUCTED_BATCHES
        )
        for sorted_position, (seed_id, csv_path, mat_path) in enumerate(paired):
            batch = int(assigned_batches[sorted_position])
            split: SplitName = "development" if batch in DEVELOPMENT_BATCHES else "locked"
            relative_csv = csv_path.relative_to(data_root).as_posix()
            relative_mat = mat_path.relative_to(data_root).as_posix()
            csv_path_hash = _sha256_text(relative_csv)
            mat_path_hash = _sha256_text(relative_mat)
            csv_size = csv_path.stat().st_size
            mat_size = mat_path.stat().st_size
            csv_hash = sha256_file(csv_path) if hash_files else ""
            mat_hash = sha256_file(mat_path) if hash_files else ""
            record_payload = (
                f"{label},{CLASS_NAMES[label]},{replicate},{seed_id},{batch},{split},"
                f"{relative_csv},{csv_size},{csv_hash},{relative_mat},{mat_size},{mat_hash}\n"
            )
            record_hash = hashlib.sha256(record_payload.encode("utf-8")).hexdigest()
            manifest_digest.update(record_payload.encode("utf-8"))
            if hash_files:
                csv_digest.update(f"{relative_csv}\0{csv_hash}\n".encode("utf-8"))
                mat_digest.update(f"{relative_mat}\0{mat_hash}\n".encode("utf-8"))
            data_digest.update(record_hash.encode("ascii"))
            records.append(
                SampleRecord(
                    sample_index=len(records),
                    label=label,
                    class_name=CLASS_NAMES[label],
                    replicate=replicate,
                    source_cube=cube_dir.name,
                    seed_id=seed_id,
                    constructed_batch=batch,
                    analysis_split=split,
                    csv_path=csv_path.resolve(),
                    mat_path=mat_path.resolve(),
                    relative_csv_path=relative_csv,
                    relative_mat_path=relative_mat,
                    csv_path_sha256=csv_path_hash,
                    mat_path_sha256=mat_path_hash,
                    csv_size_bytes=csv_size,
                    mat_size_bytes=mat_size,
                    csv_sha256=csv_hash,
                    mat_sha256=mat_hash,
                    record_sha256=record_hash,
                )
            )

    return Manifest(
        data_root=data_root,
        records=tuple(records),
        manifest_sha256=manifest_digest.hexdigest(),
        csv_content_sha256=csv_digest.hexdigest() if hash_files else "",
        mat_content_sha256=mat_digest.hexdigest() if hash_files else "",
        data_fingerprint_sha256=data_digest.hexdigest(),
        hashes_complete=bool(hash_files),
    )


def _read_spectral_records(
    records: Sequence[SampleRecord],
    *,
    split: SplitName,
    expected_bands: int,
    verify_hashes: bool,
) -> SpectralDataset:
    if not records:
        raise ValueError(f"No records supplied for the {split} split")
    wrong_split = [record.sample_id for record in records if record.analysis_split != split]
    if wrong_split:
        error_type = LockedDataAccessError if split == "development" else ValueError
        raise error_type(
            f"{split} loader received {len(wrong_split)} record(s) from another split; "
            f"first={wrong_split[0]}"
        )
    if expected_bands <= 0:
        raise ValueError("expected_bands must be positive")

    # All split checks above intentionally precede the first file access.
    spectra: list[np.ndarray] = []
    labels: list[int] = []
    wavelength_reference: np.ndarray | None = None
    loaded_digest = hashlib.sha256()
    for record in records:
        if verify_hashes and record.csv_sha256:
            observed_hash = sha256_file(record.csv_path)
            if observed_hash != record.csv_sha256:
                raise ValueError(f"CSV SHA-256 mismatch: {record.relative_csv_path}")
        try:
            values = np.loadtxt(record.csv_path, delimiter=",", dtype=np.float64)
        except ValueError as exc:
            raise ValueError(f"CSV is not a numeric two-column spectrum: {record.csv_path}") from exc
        if values.shape != (expected_bands, 2):
            raise ValueError(
                f"Expected ({expected_bands}, 2) in {record.csv_path}; observed {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite wavelength or reflectance in {record.csv_path}")
        wavelengths = values[:, 0]
        if np.any(np.diff(wavelengths) <= 0):
            raise ValueError(f"Wavelengths are not strictly increasing in {record.csv_path}")
        if wavelength_reference is None:
            wavelength_reference = wavelengths.copy()
        elif not np.allclose(wavelengths, wavelength_reference, rtol=0.0, atol=1e-6):
            maximum_error = float(np.max(np.abs(wavelengths - wavelength_reference)))
            raise ValueError(
                f"Wavelength grid mismatch in {record.csv_path}; max error={maximum_error:g} nm"
            )
        spectra.append(values[:, 1])
        labels.append(record.label)
        loaded_digest.update(record.record_sha256.encode("ascii"))

    assert wavelength_reference is not None
    return SpectralDataset(
        X=np.asarray(spectra, dtype=np.float64),
        y=np.asarray(labels, dtype=np.int64),
        wavelengths=wavelength_reference,
        records=tuple(records),
        analysis_split=split,
        loaded_csv_fingerprint_sha256=loaded_digest.hexdigest(),
    )


def load_development_csv(
    records: Sequence[SampleRecord],
    *,
    expected_bands: int = EXPECTED_BANDS,
    verify_hashes: bool = True,
) -> SpectralDataset:
    """Load development spectra, rejecting any locked record before file I/O."""

    return _read_spectral_records(
        tuple(records),
        split="development",
        expected_bands=expected_bands,
        verify_hashes=verify_hashes,
    )


def load_locked_csv(
    records: Sequence[SampleRecord],
    *,
    expected_bands: int = EXPECTED_BANDS,
    verify_hashes: bool = True,
) -> SpectralDataset:
    """Load only records already assigned to the explicitly requested locked split."""

    return _read_spectral_records(
        tuple(records),
        split="locked",
        expected_bands=expected_bands,
        verify_hashes=verify_hashes,
    )


def load_csv_split(
    manifest: Manifest,
    *,
    split: SplitName,
    expected_bands: int = EXPECTED_BANDS,
    verify_hashes: bool = True,
) -> SpectralDataset:
    """Load one explicitly named split; no implicit all-data mode exists."""

    split = _validate_split_name(split)
    records = manifest.records_for_split(split)
    if split == "development":
        return load_development_csv(
            records, expected_bands=expected_bands, verify_hashes=verify_hashes
        )
    return load_locked_csv(records, expected_bands=expected_bands, verify_hashes=verify_hashes)


class SavitzkyGolayTransformer(BaseEstimator, TransformerMixin):
    """Apply sample-wise Savitzky--Golay filtering inside a fitted pipeline."""

    def __init__(
        self,
        window_length: int = 15,
        polyorder: int = 2,
        deriv: int = 1,
        delta: float = 1.0,
    ) -> None:
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
        self.delta = delta

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> SavitzkyGolayTransformer:
        values = check_array(X, dtype=np.float64, ensure_2d=True)
        if self.window_length <= 0 or self.window_length % 2 == 0:
            raise ValueError("window_length must be a positive odd integer")
        if self.window_length > values.shape[1]:
            raise ValueError("window_length cannot exceed the number of bands")
        if self.polyorder < 0 or self.polyorder >= self.window_length:
            raise ValueError("polyorder must satisfy 0 <= polyorder < window_length")
        if self.deriv < 0 or self.deriv > self.polyorder:
            raise ValueError("deriv must satisfy 0 <= deriv <= polyorder")
        if not math.isfinite(self.delta) or self.delta <= 0:
            raise ValueError("delta must be finite and positive")
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "n_features_in_")
        values = check_array(X, dtype=np.float64, ensure_2d=True)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Transform input has a different number of bands")
        return savgol_filter(
            values,
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=self.deriv,
            delta=self.delta,
            axis=1,
            mode="interp",
        )


class StandardNormalVariate(BaseEstimator, TransformerMixin):
    """Center and scale every spectrum independently (sample SD, ddof=1)."""

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> StandardNormalVariate:
        values = check_array(X, dtype=np.float64, ensure_2d=True)
        if values.shape[1] < 2:
            raise ValueError("SNV requires at least two bands")
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "n_features_in_")
        values = check_array(X, dtype=np.float64, ensure_2d=True)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Transform input has a different number of bands")
        centered = values - values.mean(axis=1, keepdims=True)
        scale = values.std(axis=1, ddof=1, keepdims=True)
        safe_scale = np.where(scale <= np.finfo(np.float64).eps, 1.0, scale)
        return centered / safe_scale


class MultiplicativeScatterCorrection(BaseEstimator, TransformerMixin):
    """MSC using only the training-fold mean spectrum as its reference."""

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> MultiplicativeScatterCorrection:
        values = check_array(X, dtype=np.float64, ensure_2d=True)
        self.reference_ = values.mean(axis=0)
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("reference_", "n_features_in_"))
        values = check_array(X, dtype=np.float64, ensure_2d=True)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Transform input has a different number of bands")
        reference = np.asarray(self.reference_, dtype=np.float64)
        design = np.column_stack([np.ones(reference.size), reference])
        coefficients = np.linalg.lstsq(design, values.T, rcond=None)[0]
        intercept = coefficients[0]
        slope = coefficients[1]
        epsilon = np.finfo(np.float64).eps
        safe_slope = np.where(
            np.abs(slope) <= epsilon,
            np.where(slope < 0.0, -epsilon, epsilon),
            slope,
        )
        return (values - intercept[:, None]) / safe_slope[:, None]


def _sg15_steps() -> list[tuple[str, Any]]:
    return [
        ("sg_first_derivative", SavitzkyGolayTransformer(15, 2, 1)),
        ("standardize", StandardScaler()),
    ]


def build_sg15_shrinkage_lda() -> Pipeline:
    """SG(15,2,first derivative) + scaling + analytic shrinkage LDA."""

    return Pipeline(
        _sg15_steps()
        + [("classifier", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))]
    )


def build_sg15_logistic_regression(
    *,
    C: float = 1.0,
    max_iter: int = 5_000,
    tol: float = 1e-4,
    random_state: int = BASE_BATCH_SEED,
) -> Pipeline:
    """SG(15,2,first derivative) + scaling + multinomial logistic model."""

    return Pipeline(
        _sg15_steps()
        + [
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    solver="lbfgs",
                    max_iter=max_iter,
                    tol=tol,
                    random_state=random_state,
                ),
            )
        ]
    )


def build_sg15_rbf_svm() -> Pipeline:
    """Frozen uncalibrated RBF-SVM; calibration uses grouped OOF scores only."""

    return Pipeline(
        _sg15_steps()
        + [
            (
                "classifier",
                SVC(
                    C=10.0,
                    kernel="rbf",
                    gamma="scale",
                    probability=False,
                    decision_function_shape="ovr",
                ),
            )
        ]
    )


def _validate_probability_matrix(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("probabilities must be a non-empty sample-by-class matrix")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    row_sums = values.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("Every probability row must sum to one")
    return values


def _class_indices(y: Sequence[int], classes: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y, dtype=np.int64)
    class_array = np.asarray(classes, dtype=np.int64)
    if labels.ndim != 1 or class_array.ndim != 1 or class_array.size < 2:
        raise ValueError("y and classes must be one-dimensional; at least two classes are required")
    if np.unique(class_array).size != class_array.size:
        raise ValueError("classes must be unique")
    mapping = {int(label): index for index, label in enumerate(class_array)}
    try:
        indices = np.asarray([mapping[int(label)] for label in labels], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Observed label is absent from classes: {exc.args[0]}") from exc
    return labels, indices


def multiclass_metrics(
    y_true: Sequence[int],
    probabilities: np.ndarray,
    *,
    classes: Sequence[int] | None = None,
    ece_bins: int = 10,
) -> dict[str, float | int]:
    """Compute discrimination and calibration metrics from class probabilities."""

    values = _validate_probability_matrix(probabilities)
    class_array = (
        np.arange(values.shape[1], dtype=np.int64)
        if classes is None
        else np.asarray(classes, dtype=np.int64)
    )
    labels, indices = _class_indices(y_true, class_array)
    if labels.size != values.shape[0] or class_array.size != values.shape[1]:
        raise ValueError("y, probabilities, and classes have incompatible shapes")
    if ece_bins <= 0:
        raise ValueError("ece_bins must be positive")

    predicted_indices = values.argmax(axis=1)
    predictions = class_array[predicted_indices]
    clipped = np.clip(values, 1e-15, 1.0)
    one_hot = np.eye(class_array.size, dtype=np.float64)[indices]
    confidences = values[np.arange(values.shape[0]), predicted_indices]
    correct = predictions == labels
    bin_indices = np.minimum((confidences * ece_bins).astype(int), ece_bins - 1)
    ece = 0.0
    for bin_index in range(ece_bins):
        mask = bin_indices == bin_index
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidences[mask].mean()))

    return {
        "n": int(labels.size),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, labels=class_array, average="macro", zero_division=0)
        ),
        "negative_log_likelihood": float(
            -np.log(clipped[np.arange(labels.size), indices]).mean()
        ),
        "multiclass_brier_score": float(np.mean(np.sum((values - one_hot) ** 2, axis=1))),
        "expected_calibration_error": float(ece),
    }


def _validate_grouped_inputs(
    X: np.ndarray,
    y: Sequence[int],
    groups: Sequence[int],
    group_order: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = check_array(X, dtype=np.float64, ensure_2d=True)
    labels = np.asarray(y, dtype=np.int64)
    batch_ids = np.asarray(groups, dtype=np.int64)
    ordered_groups = np.asarray(group_order, dtype=np.int64)
    if labels.ndim != 1 or batch_ids.ndim != 1:
        raise ValueError("y and groups must be one-dimensional")
    if features.shape[0] != labels.size or labels.size != batch_ids.size:
        raise ValueError("X, y, and groups must contain the same number of samples")
    if ordered_groups.ndim != 1 or np.unique(ordered_groups).size != ordered_groups.size:
        raise ValueError("group_order must contain unique one-dimensional group IDs")
    if set(batch_ids.tolist()) != set(ordered_groups.tolist()):
        raise ValueError(
            f"Observed groups {sorted(set(batch_ids.tolist()))} do not equal fixed folds "
            f"{ordered_groups.tolist()}"
        )
    return features, labels, batch_ids, ordered_groups


def _aligned_estimator_output(
    estimator: BaseEstimator,
    values: np.ndarray,
    global_classes: np.ndarray,
) -> np.ndarray:
    estimator_classes = np.asarray(getattr(estimator, "classes_"), dtype=np.int64)
    output = np.asarray(values, dtype=np.float64)
    if output.ndim != 2 or output.shape[1] != estimator_classes.size:
        raise ValueError("Estimator output is not a sample-by-class matrix")
    positions = {int(label): index for index, label in enumerate(estimator_classes)}
    if set(positions) != set(global_classes.tolist()):
        raise ValueError("Estimator classes do not match the global development classes")
    return output[:, [positions[int(label)] for label in global_classes]]


def _validate_each_fold_has_all_classes(
    labels: np.ndarray, batch_ids: np.ndarray, ordered_groups: np.ndarray
) -> np.ndarray:
    classes = np.unique(labels)
    expected = set(classes.tolist())
    for held_out in ordered_groups:
        test_classes = set(labels[batch_ids == held_out].tolist())
        train_classes = set(labels[batch_ids != held_out].tolist())
        if test_classes != expected or train_classes != expected:
            raise ValueError(
                f"Held-out batch {held_out} does not preserve every class in train and validation"
            )
    return classes


def grouped_oof_probabilities(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: Sequence[int],
    groups: Sequence[int],
    *,
    group_order: Sequence[int] = DEVELOPMENT_BATCHES,
) -> OOFProbabilityResult:
    """Generate fixed grouped OOF probabilities with whole batches held out."""

    features, labels, batch_ids, ordered_groups = _validate_grouped_inputs(
        X, y, groups, group_order
    )
    classes = _validate_each_fold_has_all_classes(labels, batch_ids, ordered_groups)
    probabilities = np.full((labels.size, classes.size), np.nan, dtype=np.float64)
    held_out_batch = np.full(labels.size, -1, dtype=np.int64)
    fold_metrics: list[dict[str, float | int]] = []
    for group in ordered_groups:
        test_mask = batch_ids == group
        fitted = clone(estimator).fit(features[~test_mask], labels[~test_mask])
        if not hasattr(fitted, "predict_proba"):
            raise TypeError("Estimator must implement predict_proba for probability OOF")
        fold_probabilities = _aligned_estimator_output(
            fitted, fitted.predict_proba(features[test_mask]), classes
        )
        fold_probabilities = _validate_probability_matrix(fold_probabilities)
        probabilities[test_mask] = fold_probabilities
        held_out_batch[test_mask] = group
        fold_metrics.append(
            {
                "held_out_batch": int(group),
                **multiclass_metrics(
                    labels[test_mask], fold_probabilities, classes=classes
                ),
            }
        )
    if np.any(~np.isfinite(probabilities)) or np.any(held_out_batch < 0):
        raise AssertionError("OOF generation did not cover every development sample exactly once")
    return OOFProbabilityResult(probabilities, classes, held_out_batch, tuple(fold_metrics))


def grouped_oof_decision_scores(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: Sequence[int],
    groups: Sequence[int],
    *,
    group_order: Sequence[int] = DEVELOPMENT_BATCHES,
) -> OOFDecisionResult:
    """Generate fixed grouped OOF multiclass decision scores without Platt fitting."""

    features, labels, batch_ids, ordered_groups = _validate_grouped_inputs(
        X, y, groups, group_order
    )
    classes = _validate_each_fold_has_all_classes(labels, batch_ids, ordered_groups)
    scores = np.full((labels.size, classes.size), np.nan, dtype=np.float64)
    held_out_batch = np.full(labels.size, -1, dtype=np.int64)
    for group in ordered_groups:
        test_mask = batch_ids == group
        fitted = clone(estimator).fit(features[~test_mask], labels[~test_mask])
        if not hasattr(fitted, "decision_function"):
            raise TypeError("Estimator must implement decision_function for score OOF")
        fold_scores = _aligned_estimator_output(
            fitted, fitted.decision_function(features[test_mask]), classes
        )
        if not np.all(np.isfinite(fold_scores)):
            raise ValueError("Estimator returned non-finite decision scores")
        scores[test_mask] = fold_scores
        held_out_batch[test_mask] = group
    if np.any(~np.isfinite(scores)) or np.any(held_out_batch < 0):
        raise AssertionError("OOF generation did not cover every development sample exactly once")
    return OOFDecisionResult(scores, classes, held_out_batch)


def decision_scores_to_probabilities(
    decision_scores: np.ndarray, temperature: float
) -> np.ndarray:
    """Convert multiclass decision scores to softmax probabilities at temperature T."""

    scores = np.asarray(decision_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] < 2:
        raise ValueError("decision_scores must be a non-empty sample-by-class matrix")
    if not np.all(np.isfinite(scores)):
        raise ValueError("decision_scores must be finite")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    scaled = scores / float(temperature)
    scaled -= scaled.max(axis=1, keepdims=True)
    exponentiated = np.exp(scaled)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def fit_decision_temperature(
    decision_scores: np.ndarray,
    y_true: Sequence[int],
    *,
    classes: Sequence[int] | None = None,
    log_temperature_bounds: tuple[float, float] = (-4.0, 4.0),
) -> float:
    """Fit one temperature by multiclass NLL on grouped OOF decision scores."""

    scores = np.asarray(decision_scores, dtype=np.float64)
    if scores.ndim != 2 or not np.all(np.isfinite(scores)):
        raise ValueError("decision_scores must be a finite two-dimensional matrix")
    class_array = (
        np.arange(scores.shape[1], dtype=np.int64)
        if classes is None
        else np.asarray(classes, dtype=np.int64)
    )
    labels, indices = _class_indices(y_true, class_array)
    if labels.size != scores.shape[0] or class_array.size != scores.shape[1]:
        raise ValueError("y, decision_scores, and classes have incompatible shapes")
    lower, upper = map(float, log_temperature_bounds)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError("log_temperature_bounds must be finite and increasing")

    def objective(log_temperature: float) -> float:
        probabilities = decision_scores_to_probabilities(scores, math.exp(log_temperature))
        return float(-np.log(np.clip(probabilities[np.arange(labels.size), indices], 1e-15, 1.0)).mean())

    result = minimize_scalar(
        objective,
        method="bounded",
        bounds=(lower, upper),
        options={"xatol": 1e-10, "maxiter": 500},
    )
    if not result.success or not math.isfinite(float(result.fun)):
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    candidate = float(math.exp(float(result.x)))
    # T=1 is always an admissible, reproducible fallback and cannot be worsened.
    return candidate if objective(float(result.x)) < objective(0.0) else 1.0


def crossfit_oof_decision_temperature(
    decision_scores: np.ndarray,
    y_true: Sequence[int],
    groups: Sequence[int],
    *,
    classes: Sequence[int] | None = None,
    group_order: Sequence[int] = DEVELOPMENT_BATCHES,
) -> TemperatureCalibrationResult:
    """Post-hoc group-held scaling of an already generated OOF score matrix.

    This helper excludes a group's rows while fitting its temperature, but OOF
    models behind the other rows may have trained on that group.  It is therefore
    a calibration diagnostic, not an unbiased outer-fold performance estimate.
    Use :func:`nested_grouped_oof_temperature_probabilities` for model assessment.
    ``final_temperature`` remains valid for deployment when the input scores were
    produced by grouped OOF on the complete development set.
    """

    scores = np.asarray(decision_scores, dtype=np.float64)
    labels = np.asarray(y_true, dtype=np.int64)
    batch_ids = np.asarray(groups, dtype=np.int64)
    ordered_groups = np.asarray(group_order, dtype=np.int64)
    class_array = (
        np.arange(scores.shape[1], dtype=np.int64)
        if classes is None
        else np.asarray(classes, dtype=np.int64)
    )
    if scores.ndim != 2 or scores.shape[0] != labels.size or labels.size != batch_ids.size:
        raise ValueError("decision_scores, y_true, and groups have incompatible shapes")
    if set(batch_ids.tolist()) != set(ordered_groups.tolist()):
        raise ValueError("groups must equal the fixed development group_order")
    probabilities = np.full_like(scores, np.nan, dtype=np.float64)
    fold_temperatures: list[tuple[int, float]] = []
    for group in ordered_groups:
        held_out = batch_ids == group
        temperature = fit_decision_temperature(
            scores[~held_out], labels[~held_out], classes=class_array
        )
        probabilities[held_out] = decision_scores_to_probabilities(scores[held_out], temperature)
        fold_temperatures.append((int(group), temperature))
    if np.any(~np.isfinite(probabilities)):
        raise AssertionError("Cross-fitted temperature scaling left unfilled rows")
    final_temperature = fit_decision_temperature(scores, labels, classes=class_array)
    return TemperatureCalibrationResult(
        probabilities=probabilities,
        final_temperature=final_temperature,
        fold_temperatures=tuple(fold_temperatures),
    )


def nested_grouped_oof_temperature_probabilities(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: Sequence[int],
    groups: Sequence[int],
    *,
    group_order: Sequence[int] = DEVELOPMENT_BATCHES,
) -> TemperatureCalibrationResult:
    """Create leakage-safe outer OOF probabilities with nested score calibration.

    For outer batch ``g``, every temperature-fitting score is generated by inner
    OOF fits using only ``~g``.  The estimator is then fitted on all ``~g`` rows,
    scored on ``g``, and converted with that outer fold's temperature.  Finally,
    a deployable temperature is fitted from an independent eight-fold OOF score
    matrix over all development rows; it can be applied after refitting the
    estimator on the complete development set.
    """

    features, labels, batch_ids, ordered_groups = _validate_grouped_inputs(
        X, y, groups, group_order
    )
    classes = _validate_each_fold_has_all_classes(labels, batch_ids, ordered_groups)
    if ordered_groups.size < 3:
        raise ValueError("Nested grouped calibration requires at least three groups")

    probabilities = np.full((labels.size, classes.size), np.nan, dtype=np.float64)
    fold_temperatures: list[tuple[int, float]] = []
    for outer_group in ordered_groups:
        outer_test = batch_ids == outer_group
        outer_train = ~outer_test
        inner_group_order = tuple(
            int(group) for group in ordered_groups if group != outer_group
        )
        inner_result = grouped_oof_decision_scores(
            estimator,
            features[outer_train],
            labels[outer_train],
            batch_ids[outer_train],
            group_order=inner_group_order,
        )
        temperature = fit_decision_temperature(
            inner_result.decision_scores,
            labels[outer_train],
            classes=classes,
        )
        outer_estimator = clone(estimator).fit(features[outer_train], labels[outer_train])
        if not hasattr(outer_estimator, "decision_function"):
            raise TypeError("Estimator must implement decision_function for nested calibration")
        outer_scores = _aligned_estimator_output(
            outer_estimator,
            outer_estimator.decision_function(features[outer_test]),
            classes,
        )
        probabilities[outer_test] = decision_scores_to_probabilities(
            outer_scores, temperature
        )
        fold_temperatures.append((int(outer_group), temperature))

    if np.any(~np.isfinite(probabilities)):
        raise AssertionError("Nested temperature scaling left unfilled outer-fold rows")

    deployment_oof = grouped_oof_decision_scores(
        estimator,
        features,
        labels,
        batch_ids,
        group_order=tuple(int(group) for group in ordered_groups),
    )
    final_temperature = fit_decision_temperature(
        deployment_oof.decision_scores,
        labels,
        classes=classes,
    )
    return TemperatureCalibrationResult(
        probabilities=probabilities,
        final_temperature=final_temperature,
        fold_temperatures=tuple(fold_temperatures),
    )


def equal_weight_probability_average(
    probability_matrices: Sequence[np.ndarray],
) -> np.ndarray:
    """Average aligned class probabilities with exactly equal model weights."""

    if not probability_matrices:
        raise ValueError("At least one probability matrix is required")
    validated = [_validate_probability_matrix(matrix) for matrix in probability_matrices]
    shape = validated[0].shape
    if any(matrix.shape != shape for matrix in validated[1:]):
        raise ValueError("All probability matrices must have the same shape and class order")
    average = np.mean(np.stack(validated, axis=0), axis=0)
    # Floating-point averaging should already sum to one; normalization removes
    # only machine-scale drift and does not introduce fitted weights.
    return average / average.sum(axis=1, keepdims=True)
