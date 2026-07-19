"""Leakage-safe data preparation for the NASA C-MAPSS turbofan dataset.

The raw trajectory files contain one row per engine cycle.  This module keeps
engine trajectories separate throughout splitting and window construction so
that a window can never contain cycles from two engines.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from .base import (
    DataBundle,
    EvaluationBundle,
    EvaluationSpec,
    RulStageFilter,
    rul_stage_mask,
)


ENGINE_ID_COLUMN = "engine_id"
CYCLE_COLUMN = "cycle"
SETTING_COLUMNS = tuple(f"setting_{index}" for index in range(1, 4))
SENSOR_COLUMNS = tuple(f"sensor_{index}" for index in range(1, 22))
FEATURE_COLUMNS = SETTING_COLUMNS + SENSOR_COLUMNS
CMAPSS_COLUMNS = (ENGINE_ID_COLUMN, CYCLE_COLUMN) + FEATURE_COLUMNS
RUL_COLUMN = "rul"

# Descriptive aliases make the raw schema easy to discover from a CLI or
# notebook without requiring users to know the shorter internal names.
OPERATIONAL_SETTING_COLUMNS = SETTING_COLUMNS
COLUMN_NAMES = CMAPSS_COLUMNS
VALID_SUBSETS = ("FD001", "FD002", "FD003", "FD004")


def normalize_subset(subset: str | int) -> str:
    """Return a canonical subset name such as ``FD001``."""

    if isinstance(subset, (int, np.integer)):
        name = f"FD{int(subset):03d}"
    else:
        value = str(subset).strip().upper()
        if value.isdigit():
            name = f"FD{int(value):03d}"
        else:
            match = re.fullmatch(r"FD(\d{1,3})", value)
            name = f"FD{int(match.group(1)):03d}" if match else value
    if name not in VALID_SUBSETS:
        raise ValueError(
            f"Unknown C-MAPSS subset {subset!r}; expected one of {VALID_SUBSETS}."
        )
    return name


def _require_file(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"C-MAPSS file not found: {resolved}")
    return resolved


def read_cmapss_data(path: str | Path) -> pd.DataFrame:
    """Read a C-MAPSS train or test trajectory file.

    Files are whitespace-delimited and often contain trailing spaces.  Reading
    with ``sep=r"\s+"`` handles both the inter-column and trailing whitespace
    without creating empty columns.
    """

    file_path = _require_file(path)
    frame = pd.read_csv(file_path, sep=r"\s+", header=None)
    if frame.shape[1] != len(CMAPSS_COLUMNS):
        raise ValueError(
            f"Expected {len(CMAPSS_COLUMNS)} columns in {file_path}, "
            f"found {frame.shape[1]}."
        )
    frame.columns = list(CMAPSS_COLUMNS)
    if frame.empty:
        raise ValueError(f"C-MAPSS trajectory file is empty: {file_path}")

    numeric = frame.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-numeric or non-finite value found in {file_path}.")

    for column in (ENGINE_ID_COLUMN, CYCLE_COLUMN):
        column_values = numeric[column].to_numpy(dtype=np.float64)
        if not np.equal(column_values, np.floor(column_values)).all():
            raise ValueError(f"{column} must contain integer values in {file_path}.")
        numeric[column] = column_values.astype(np.int64)

    if (numeric[ENGINE_ID_COLUMN] <= 0).any() or (numeric[CYCLE_COLUMN] <= 0).any():
        raise ValueError("Engine IDs and cycle numbers must be positive integers.")
    if numeric.duplicated([ENGINE_ID_COLUMN, CYCLE_COLUMN]).any():
        raise ValueError(f"Duplicate engine/cycle row found in {file_path}.")

    return numeric.reset_index(drop=True)


def read_cmapss_rul(path: str | Path) -> np.ndarray:
    """Read the official test endpoint RUL file as a one-dimensional array."""

    file_path = _require_file(path)
    frame = pd.read_csv(file_path, sep=r"\s+", header=None)
    if frame.shape[1] != 1 or frame.empty:
        raise ValueError(
            f"Expected one non-empty RUL column in {file_path}, found {frame.shape}."
        )
    values = pd.to_numeric(frame.iloc[:, 0], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"RUL values in {file_path} must be finite and non-negative.")
    return values


def map_test_rul_to_engines(
    test_frame: pd.DataFrame, official_rul: Sequence[float] | np.ndarray
) -> pd.Series:
    """Map official RUL rows to strictly ascending test engine IDs.

    NASA publishes one RUL value per test engine, in engine-number order.  The
    count is checked before constructing an explicitly engine-indexed series;
    callers therefore cannot accidentally align labels to trajectory row order.
    """

    _require_columns(test_frame, (ENGINE_ID_COLUMN,))
    engine_ids = tuple(
        int(value) for value in np.sort(test_frame[ENGINE_ID_COLUMN].unique()).tolist()
    )
    values = np.asarray(official_rul, dtype=np.float64).reshape(-1)
    if len(engine_ids) != values.size:
        raise ValueError(
            "Official test RUL count does not match the number of test engines: "
            f"{values.size} labels for {len(engine_ids)} engines."
        )
    if not engine_ids:
        raise ValueError("Cannot map RUL labels for an empty test frame.")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Official test RUL values must be finite and non-negative.")
    return pd.Series(
        values, index=pd.Index(engine_ids, name=ENGINE_ID_COLUMN), name=RUL_COLUMN
    )


@dataclass(frozen=True)
class CmapssRawData:
    """Raw train/test trajectories and engine-indexed official test labels."""

    subset: str
    train: pd.DataFrame
    test: pd.DataFrame
    test_rul: pd.Series


def load_cmapss_files(
    data_dir: str | Path, subset: str | int = "FD001"
) -> CmapssRawData:
    """Load one complete C-MAPSS subset from ``data_dir``."""

    subset_name = normalize_subset(subset)
    root = Path(data_dir)
    train = read_cmapss_data(root / f"train_{subset_name}.txt")
    test = read_cmapss_data(root / f"test_{subset_name}.txt")
    official_rul = read_cmapss_rul(root / f"RUL_{subset_name}.txt")
    test_rul = map_test_rul_to_engines(test, official_rul)
    return CmapssRawData(subset_name, train, test, test_rul)


def load_cmapss_split(
    data_dir: str | Path, subset: str | int = "FD001", split: str = "train"
) -> pd.DataFrame | pd.Series:
    """Convenience reader for a named ``train``, ``test``, or ``rul`` split."""

    subset_name = normalize_subset(subset)
    root = Path(data_dir)
    name = split.strip().lower()
    if name == "train":
        return read_cmapss_data(root / f"train_{subset_name}.txt")
    if name == "test":
        return read_cmapss_data(root / f"test_{subset_name}.txt")
    if name in {"rul", "test_rul"}:
        test = read_cmapss_data(root / f"test_{subset_name}.txt")
        official_rul = read_cmapss_rul(root / f"RUL_{subset_name}.txt")
        return map_test_rul_to_engines(test, official_rul)
    raise ValueError("split must be 'train', 'test', or 'rul'.")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Data frame is missing required columns: {missing}")


def split_engine_ids(
    frame_or_ids: pd.DataFrame | Sequence[int] | np.ndarray,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Reproducibly split whole engines into disjoint train/validation sets."""

    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in the half-open interval [0, 1).")
    if isinstance(frame_or_ids, pd.DataFrame):
        _require_columns(frame_or_ids, (ENGINE_ID_COLUMN,))
        raw_ids = frame_or_ids[ENGINE_ID_COLUMN].to_numpy()
    else:
        raw_ids = np.asarray(frame_or_ids).reshape(-1)
    if raw_ids.size == 0:
        raise ValueError("At least one engine ID is required for splitting.")
    if not np.isfinite(raw_ids.astype(np.float64)).all():
        raise ValueError("Engine IDs must be finite integers.")
    if not np.equal(raw_ids, np.floor(raw_ids.astype(np.float64))).all():
        raise ValueError("Engine IDs must be integers.")

    engine_ids = np.unique(raw_ids.astype(np.int64))
    if (engine_ids <= 0).any():
        raise ValueError("Engine IDs must be positive integers.")
    if val_fraction == 0.0:
        return tuple(int(value) for value in engine_ids), ()
    if engine_ids.size < 2:
        raise ValueError("At least two engines are required for a validation split.")

    val_count = min(
        engine_ids.size - 1, max(1, int(math.ceil(engine_ids.size * val_fraction)))
    )
    shuffled = np.random.default_rng(seed).permutation(engine_ids)
    val_ids = tuple(int(value) for value in np.sort(shuffled[:val_count]))
    train_ids = tuple(int(value) for value in np.sort(shuffled[val_count:]))
    return train_ids, val_ids


def _normalized_cap(rul_cap: float | None) -> float | None:
    if rul_cap is None or rul_cap <= 0:
        return None
    cap = float(rul_cap)
    if not math.isfinite(cap):
        raise ValueError("rul_cap must be finite, None, or non-positive to disable it.")
    return cap


def add_train_rul_labels(
    frame: pd.DataFrame, rul_cap: float | None = 125.0
) -> pd.DataFrame:
    """Attach piecewise-linear RUL labels to complete run-to-failure engines."""

    _require_columns(frame, (ENGINE_ID_COLUMN, CYCLE_COLUMN))
    result = frame.copy()
    max_cycle = result.groupby(ENGINE_ID_COLUMN)[CYCLE_COLUMN].transform("max")
    rul = (max_cycle - result[CYCLE_COLUMN]).to_numpy(dtype=np.float64)
    cap = _normalized_cap(rul_cap)
    if cap is not None:
        rul = np.minimum(rul, cap)
    result[RUL_COLUMN] = rul
    return result


def add_test_rul_labels(
    frame: pd.DataFrame,
    endpoint_rul: Mapping[int, float] | pd.Series,
    rul_cap: float | None = 125.0,
) -> pd.DataFrame:
    """Attach test RUL labels derived from each engine's official endpoint RUL."""

    _require_columns(frame, (ENGINE_ID_COLUMN, CYCLE_COLUMN))
    mapping = {int(key): float(value) for key, value in dict(endpoint_rul).items()}
    engine_ids = {int(value) for value in frame[ENGINE_ID_COLUMN].unique()}
    if set(mapping) != engine_ids:
        missing = sorted(engine_ids - set(mapping))
        extra = sorted(set(mapping) - engine_ids)
        raise ValueError(
            f"Test endpoint RUL engine IDs do not match trajectories; "
            f"missing={missing}, extra={extra}."
        )
    endpoint_values = np.asarray(list(mapping.values()), dtype=np.float64)
    if not np.isfinite(endpoint_values).all() or (endpoint_values < 0).any():
        raise ValueError("Test endpoint RUL values must be finite and non-negative.")

    result = frame.copy()
    max_cycle = result.groupby(ENGINE_ID_COLUMN)[CYCLE_COLUMN].transform("max")
    endpoint = result[ENGINE_ID_COLUMN].map(mapping).to_numpy(dtype=np.float64)
    rul = endpoint + (max_cycle - result[CYCLE_COLUMN]).to_numpy(dtype=np.float64)
    cap = _normalized_cap(rul_cap)
    if cap is not None:
        rul = np.minimum(rul, cap)
    result[RUL_COLUMN] = rul
    return result


class CmapssPreprocessor:
    """Training-only low-variance filter followed by ``StandardScaler``.

    ``state_dict`` intentionally contains only JSON-safe primitive values.  It
    can therefore be embedded in a model checkpoint without pickling sklearn
    estimator instances.
    """

    STATE_VERSION = 1

    def __init__(
        self,
        variance_threshold: float = 1e-12,
        feature_columns: Sequence[str] | None = None,
    ) -> None:
        if variance_threshold < 0 or not math.isfinite(float(variance_threshold)):
            raise ValueError("variance_threshold must be finite and non-negative.")
        self.variance_threshold = float(variance_threshold)
        self.feature_columns = tuple(feature_columns or FEATURE_COLUMNS)
        if not self.feature_columns or len(set(self.feature_columns)) != len(
            self.feature_columns
        ):
            raise ValueError(
                "feature_columns must be a non-empty list of unique names."
            )

        self.selected_indices: tuple[int, ...] = ()
        self.selected_features: tuple[str, ...] = ()
        self.feature_variances: tuple[float, ...] = ()
        self.fit_engine_ids: tuple[int, ...] = ()
        self.n_samples_seen: int = 0
        self._scaler: StandardScaler | None = None

    @property
    def is_fitted(self) -> bool:
        return self._scaler is not None

    @property
    def output_dim(self) -> int:
        self._check_fitted()
        return len(self.selected_indices)

    def _feature_array(self, frame: pd.DataFrame) -> np.ndarray:
        _require_columns(frame, self.feature_columns)
        values = frame.loc[:, self.feature_columns].to_numpy(dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("Cannot process an empty feature frame.")
        if not np.isfinite(values).all():
            raise ValueError("Feature values must be finite.")
        return values

    def fit(self, frame: pd.DataFrame) -> "CmapssPreprocessor":
        values = self._feature_array(frame)
        selector = VarianceThreshold(threshold=self.variance_threshold)
        try:
            selector.fit(values)
        except ValueError as error:
            raise ValueError(
                "No features remain after low-variance filtering; lower "
                "variance_threshold."
            ) from error

        selected = tuple(int(value) for value in selector.get_support(indices=True))
        scaler = StandardScaler()
        scaler.fit(values[:, selected])

        self.selected_indices = selected
        self.selected_features = tuple(
            self.feature_columns[index] for index in selected
        )
        self.feature_variances = tuple(float(value) for value in selector.variances_)
        self.n_samples_seen = int(values.shape[0])
        if ENGINE_ID_COLUMN in frame:
            self.fit_engine_ids = tuple(
                int(value) for value in np.sort(frame[ENGINE_ID_COLUMN].unique())
            )
        else:
            self.fit_engine_ids = ()
        self._scaler = scaler
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Return selected and standardized features as ``float32``."""

        self._check_fitted()
        values = self._feature_array(frame)
        assert self._scaler is not None
        transformed = self._scaler.transform(values[:, self.selected_indices])
        return transformed.astype(np.float32, copy=False)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)

    def _check_fitted(self) -> None:
        if self._scaler is None:
            raise RuntimeError("CmapssPreprocessor must be fitted before use.")

    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-safe preprocessing state."""

        self._check_fitted()
        assert self._scaler is not None
        return {
            "version": self.STATE_VERSION,
            "variance_threshold": self.variance_threshold,
            "feature_columns": list(self.feature_columns),
            "selected_indices": list(self.selected_indices),
            "selected_features": list(self.selected_features),
            "feature_variances": list(self.feature_variances),
            "mean": [float(value) for value in self._scaler.mean_],
            "scale": [float(value) for value in self._scaler.scale_],
            "variance": [float(value) for value in self._scaler.var_],
            "n_samples_seen": self.n_samples_seen,
            "fit_engine_ids": list(self.fit_engine_ids),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> "CmapssPreprocessor":
        """Restore a state produced by :meth:`state_dict`."""

        if int(state.get("version", -1)) != self.STATE_VERSION:
            raise ValueError(
                f"Unsupported preprocessing state version: {state.get('version')}"
            )
        feature_columns = tuple(str(value) for value in state["feature_columns"])
        selected_indices = tuple(int(value) for value in state["selected_indices"])
        selected_features = tuple(str(value) for value in state["selected_features"])
        feature_variances = tuple(float(value) for value in state["feature_variances"])
        mean = np.asarray(state["mean"], dtype=np.float64)
        scale = np.asarray(state["scale"], dtype=np.float64)
        variance = np.asarray(state["variance"], dtype=np.float64)

        if not feature_columns or len(set(feature_columns)) != len(feature_columns):
            raise ValueError("Invalid feature_columns in preprocessing state.")
        if not selected_indices or any(
            index < 0 or index >= len(feature_columns) for index in selected_indices
        ):
            raise ValueError("Invalid selected_indices in preprocessing state.")
        expected_names = tuple(feature_columns[index] for index in selected_indices)
        if selected_features != expected_names:
            raise ValueError("selected_features do not match selected_indices.")
        if len(feature_variances) != len(feature_columns):
            raise ValueError("feature_variances length does not match feature_columns.")
        if not (mean.size == scale.size == variance.size == len(selected_indices)):
            raise ValueError("Scaler state dimensions do not match selected features.")
        if (
            not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or not np.isfinite(variance).all()
            or (scale <= 0).any()
            or (variance < 0).any()
        ):
            raise ValueError("Scaler state contains invalid values.")

        variance_threshold = float(state["variance_threshold"])
        if variance_threshold < 0 or not math.isfinite(variance_threshold):
            raise ValueError("Invalid variance_threshold in preprocessing state.")
        n_samples_seen = int(state["n_samples_seen"])
        if n_samples_seen <= 0:
            raise ValueError("n_samples_seen must be positive.")

        scaler = StandardScaler()
        scaler.mean_ = mean
        scaler.scale_ = scale
        scaler.var_ = variance
        scaler.n_features_in_ = mean.size
        scaler.n_samples_seen_ = n_samples_seen

        self.variance_threshold = variance_threshold
        self.feature_columns = feature_columns
        self.selected_indices = selected_indices
        self.selected_features = selected_features
        self.feature_variances = feature_variances
        self.n_samples_seen = n_samples_seen
        self.fit_engine_ids = tuple(int(value) for value in state["fit_engine_ids"])
        self._scaler = scaler
        return self

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "CmapssPreprocessor":
        instance = cls(
            variance_threshold=float(state.get("variance_threshold", 0.0)),
            feature_columns=state.get("feature_columns", FEATURE_COLUMNS),
        )
        return instance.load_state_dict(state)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.state_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "CmapssPreprocessor":
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            raise ValueError("Preprocessing state file must contain a JSON object.")
        return cls.from_state_dict(state)


@dataclass(frozen=True)
class _EngineSequence:
    engine_id: int
    features: np.ndarray
    targets: np.ndarray
    cycles: np.ndarray


class WindowedCmapssDataset(Dataset):
    """Memory-efficient, engine-isolated C-MAPSS sliding windows.

    Normal engines contribute full-length sliding windows.  If an entire engine
    trajectory is shorter than ``window_size``, it contributes one final window
    with zero-valued left padding and a ``True`` padding mask.  ``last_only`` is
    used for official testing and produces exactly one final window per engine.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        preprocessor: CmapssPreprocessor,
        window_size: int = 30,
        stride: int = 1,
        last_only: bool = False,
        include_partial: bool = False,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        if stride <= 0:
            raise ValueError("stride must be positive.")
        _require_columns(frame, (ENGINE_ID_COLUMN, CYCLE_COLUMN, RUL_COLUMN))
        preprocessor._check_fitted()

        self.window_size = int(window_size)
        self.stride = int(stride)
        self.last_only = bool(last_only)
        self.include_partial = bool(include_partial)
        self.feature_names = preprocessor.selected_features
        self.feature_dim = preprocessor.output_dim
        self._sequences: list[_EngineSequence] = []
        self._sample_index: list[tuple[int, int]] = []
        engine_to_indices: dict[int, list[int]] = {}

        for engine_id, engine_frame in frame.groupby(ENGINE_ID_COLUMN, sort=True):
            ordered = engine_frame.sort_values(CYCLE_COLUMN, kind="stable")
            if ordered[CYCLE_COLUMN].duplicated().any():
                raise ValueError(
                    f"Engine {engine_id} contains duplicate cycle numbers."
                )
            features = np.ascontiguousarray(preprocessor.transform(ordered))
            targets = ordered[RUL_COLUMN].to_numpy(dtype=np.float32, copy=True)
            cycles = ordered[CYCLE_COLUMN].to_numpy(dtype=np.int64, copy=True)
            if not np.isfinite(targets).all():
                raise ValueError(
                    f"Engine {engine_id} contains a non-finite RUL target."
                )

            sequence_index = len(self._sequences)
            numeric_engine_id = int(engine_id)
            self._sequences.append(
                _EngineSequence(numeric_engine_id, features, targets, cycles)
            )
            if self.last_only:
                endpoints = (len(ordered) - 1,)
            elif self.include_partial:
                endpoint_values = list(range(0, len(ordered), self.stride))
                if endpoint_values[-1] != len(ordered) - 1:
                    endpoint_values.append(len(ordered) - 1)
                endpoints = endpoint_values
            elif len(ordered) < self.window_size:
                endpoints = (len(ordered) - 1,)
            else:
                endpoints = range(self.window_size - 1, len(ordered), self.stride)
            for endpoint in endpoints:
                sample_position = len(self._sample_index)
                self._sample_index.append((sequence_index, int(endpoint)))
                engine_to_indices.setdefault(numeric_engine_id, []).append(
                    sample_position
                )

        self.engine_ids = tuple(sequence.engine_id for sequence in self._sequences)
        self._engine_to_indices = {
            engine_id: tuple(indices)
            for engine_id, indices in engine_to_indices.items()
        }

    def __len__(self) -> int:
        return len(self._sample_index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sequence_index, endpoint = self._sample_index[index]
        sequence = self._sequences[sequence_index]
        if index > 0 and self._sample_index[index - 1][0] == sequence_index:
            previous_endpoint = self._sample_index[index - 1][1]
            state_new_tokens = endpoint - previous_endpoint
        else:
            state_new_tokens = min(self.window_size, endpoint + 1)
        start = max(0, endpoint - self.window_size + 1)
        valid = sequence.features[start : endpoint + 1]
        pad_length = self.window_size - valid.shape[0]

        if pad_length:
            window = np.zeros((self.window_size, self.feature_dim), dtype=np.float32)
            window[pad_length:] = valid
        else:
            window = valid
        padding_mask = np.zeros(self.window_size, dtype=np.bool_)
        padding_mask[:pad_length] = True

        return {
            "features": torch.from_numpy(np.ascontiguousarray(window)),
            "padding_mask": torch.from_numpy(padding_mask),
            "target": torch.tensor(sequence.targets[endpoint], dtype=torch.float32),
            "entity_id": torch.tensor(sequence.engine_id, dtype=torch.long),
            "time_index": torch.tensor(sequence.cycles[endpoint], dtype=torch.long),
            "state_new_tokens": torch.tensor(state_new_tokens, dtype=torch.long),
            "engine_id": torch.tensor(sequence.engine_id, dtype=torch.long),
            "cycle": torch.tensor(sequence.cycles[endpoint], dtype=torch.long),
        }

    def indices_for_engine(self, engine_id: int) -> tuple[int, ...]:
        """Return dataset positions belonging to one engine."""

        return self._engine_to_indices.get(int(engine_id), ())

    def continuous_evaluation_view(self) -> "EngineTrajectoryCmapssDataset":
        """Expose each complete observed trajectory for stateful endpoint RUL."""
        return EngineTrajectoryCmapssDataset(
            self._sequences,
            feature_names=self.feature_names,
            feature_dim=self.feature_dim,
        )

    @property
    def sample_engine_ids(self) -> tuple[int, ...]:
        return tuple(
            self._sequences[sequence_index].engine_id
            for sequence_index, _ in self._sample_index
        )


class EngineTrajectoryCmapssDataset(Dataset):
    """One complete observed sequence per engine for continuous-state testing."""

    requires_batch_size_one = True

    def __init__(
        self,
        sequences: Sequence[_EngineSequence],
        *,
        feature_names: Sequence[str],
        feature_dim: int,
    ) -> None:
        if not sequences:
            raise ValueError("continuous C-MAPSS evaluation requires trajectories")
        self._sequences = tuple(sequences)
        self.feature_names = tuple(feature_names)
        self.feature_dim = int(feature_dim)
        self.engine_ids = tuple(sequence.engine_id for sequence in self._sequences)

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sequence = self._sequences[index]
        length = sequence.features.shape[0]
        return {
            "features": torch.from_numpy(
                np.ascontiguousarray(sequence.features)
            ),
            "padding_mask": torch.zeros(length, dtype=torch.bool),
            "target": torch.tensor(sequence.targets[-1], dtype=torch.float32),
            "entity_id": torch.tensor(sequence.engine_id, dtype=torch.long),
            "time_index": torch.tensor(sequence.cycles[-1], dtype=torch.long),
            "state_new_tokens": torch.tensor(length, dtype=torch.long),
            "engine_id": torch.tensor(sequence.engine_id, dtype=torch.long),
            "cycle": torch.tensor(sequence.cycles[-1], dtype=torch.long),
        }

    def indices_for_engine(self, engine_id: int) -> tuple[int, ...]:
        try:
            return (self.engine_ids.index(int(engine_id)),)
        except ValueError:
            return ()

    @property
    def sample_engine_ids(self) -> tuple[int, ...]:
        return self.engine_ids


# A shorter alias is convenient in training code while retaining the explicit
# class name in documentation.
CmapssWindowDataset = WindowedCmapssDataset


@dataclass(frozen=True)
class CmapssDataBundle:
    subset: str
    train_dataset: WindowedCmapssDataset
    val_dataset: WindowedCmapssDataset
    test_dataset: WindowedCmapssDataset
    preprocessor: CmapssPreprocessor
    train_engine_ids: tuple[int, ...]
    val_engine_ids: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def train(self) -> WindowedCmapssDataset:
        return self.train_dataset

    @property
    def val(self) -> WindowedCmapssDataset:
        return self.val_dataset

    @property
    def test(self) -> WindowedCmapssDataset:
        return self.test_dataset


def prepare_cmapss_test_dataset(
    data_dir: str | Path,
    subset: str | int,
    preprocessor: CmapssPreprocessor,
    window_size: int = 30,
    rul_cap: float | None = 125.0,
) -> WindowedCmapssDataset:
    """Build official endpoint test windows without reading training data.

    ``preprocessor`` must already contain training-fitted feature selection and
    scaling state, normally restored from a checkpoint.  Only the subset's
    ``test_*.txt`` and ``RUL_*.txt`` files are accessed.
    """

    if not isinstance(preprocessor, CmapssPreprocessor):
        raise TypeError("preprocessor must be a CmapssPreprocessor instance.")
    preprocessor._check_fitted()
    subset_name = normalize_subset(subset)
    root = Path(data_dir)
    test_frame = read_cmapss_data(root / f"test_{subset_name}.txt")
    official_rul = read_cmapss_rul(root / f"RUL_{subset_name}.txt")
    test_rul = map_test_rul_to_engines(test_frame, official_rul)
    labelled_test = add_test_rul_labels(test_frame, test_rul, rul_cap=rul_cap)
    return WindowedCmapssDataset(
        labelled_test,
        preprocessor,
        window_size=window_size,
        last_only=True,
    )


def prepare_cmapss_datasets(
    data_dir: str | Path,
    subset: str | int = "FD001",
    window_size: int = 30,
    stride: int = 1,
    val_fraction: float = 0.2,
    seed: int = 42,
    rul_cap: float | None = 125.0,
    variance_threshold: float = 1e-12,
    preprocessor: CmapssPreprocessor | None = None,
    train_rul_filter: RulStageFilter | None = None,
) -> CmapssDataBundle:
    """Build leakage-safe train, validation, and official test datasets.

    If ``preprocessor`` is omitted, feature selection and scaling are fitted on
    the selected training engines.  A restored preprocessor may instead be
    injected for resume/evaluation; its recorded fitting engines must exactly
    match the deterministic split, and it is never refitted.
    """

    raw = load_cmapss_files(data_dir, subset)
    labelled_train = add_train_rul_labels(raw.train, rul_cap=rul_cap)
    train_ids, val_ids = split_engine_ids(
        labelled_train, val_fraction=val_fraction, seed=seed
    )
    train_frame = labelled_train[
        labelled_train[ENGINE_ID_COLUMN].isin(train_ids)
    ].reset_index(drop=True)
    val_frame = labelled_train[
        labelled_train[ENGINE_ID_COLUMN].isin(val_ids)
    ].reset_index(drop=True)

    active_filter = train_rul_filter or RulStageFilter()
    train_mask, filter_metadata = rul_stage_mask(
        train_frame[ENGINE_ID_COLUMN].to_numpy(dtype=np.int64),
        train_frame[RUL_COLUMN].to_numpy(dtype=np.float64),
        active_filter,
    )
    train_frame = train_frame.loc[train_mask].reset_index(drop=True)

    if preprocessor is None:
        active_preprocessor = CmapssPreprocessor(
            variance_threshold=variance_threshold
        ).fit(train_frame)
    else:
        if not isinstance(preprocessor, CmapssPreprocessor):
            raise TypeError("preprocessor must be a CmapssPreprocessor instance.")
        preprocessor._check_fitted()
        if preprocessor.fit_engine_ids != train_ids:
            raise ValueError(
                "Injected preprocessor fit_engine_ids do not match the "
                "deterministic training split: "
                f"state={preprocessor.fit_engine_ids}, split={train_ids}."
            )
        active_preprocessor = preprocessor
    labelled_test = add_test_rul_labels(raw.test, raw.test_rul, rul_cap=rul_cap)

    train_dataset = WindowedCmapssDataset(
        train_frame,
        active_preprocessor,
        window_size=window_size,
        stride=stride,
        include_partial=active_filter.enabled,
    )
    val_dataset = WindowedCmapssDataset(
        val_frame, active_preprocessor, window_size=window_size, stride=stride
    )
    test_dataset = WindowedCmapssDataset(
        labelled_test,
        active_preprocessor,
        window_size=window_size,
        stride=stride,
        last_only=True,
    )
    return CmapssDataBundle(
        subset=raw.subset,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        preprocessor=active_preprocessor,
        train_engine_ids=train_ids,
        val_engine_ids=val_ids,
        metadata={"train_rul_filter": filter_metadata},
    )


# Backwards-friendly singular spelling for small scripts.
prepare_cmapss_data = prepare_cmapss_datasets


class CmapssAdapter:
    """Expose the existing C-MAPSS pipeline through the shared data contract."""

    name = "cmapss"

    @staticmethod
    def _validate_settings(settings: Any) -> None:
        normalize_subset(settings.subset)
        if settings.options:
            raise ValueError(
                f"C-MAPSS does not define data.options; got {sorted(settings.options)}."
            )

    def checkpoint_config(self, settings: Any) -> dict[str, Any]:
        self._validate_settings(settings)
        values = settings.checkpoint_values()
        values["name"] = self.name
        values["subset"] = normalize_subset(settings.subset)
        return values

    def validate_checkpoint(self, settings: Any, checkpoint: Mapping[str, Any]) -> None:
        expected = self.checkpoint_config(settings)
        actual = checkpoint.get("data_config")
        if not isinstance(actual, Mapping):
            raise ValueError("Checkpoint is missing data_config.")
        actual_name = str(
            checkpoint.get("dataset_name", actual.get("name", "cmapss"))
        ).lower()
        if actual_name != self.name:
            raise ValueError(
                f"Configured dataset {self.name!r} does not match checkpoint "
                f"dataset {actual_name!r}."
            )
        defaults = {
            "evaluation_stride": 1,
            "train_rul_filter": RulStageFilter().to_dict(),
            "options": {},
        }
        for key in (
            "subset",
            "window_size",
            "stride",
            "evaluation_stride",
            "val_fraction",
            "seed",
            "rul_cap",
            "variance_threshold",
            "train_rul_filter",
            "options",
        ):
            found = actual.get(key, defaults.get(key))
            if found != expected[key]:
                raise ValueError(
                    f"Configured data.{key} does not match checkpoint: "
                    f"{expected[key]!r} != {found!r}."
                )

    @staticmethod
    def _label_policy(rul_cap: float | None) -> dict[str, Any]:
        return {
            "train": "piecewise_linear_to_failure",
            "test": "official_endpoint_rul",
            "cap_applied_to": "train_and_test_targets"
            if rul_cap is not None
            else "none",
        }

    def prepare_training(
        self,
        settings: Any,
        preprocessor_state: Mapping[str, Any] | None = None,
    ) -> DataBundle:
        data_config = self.checkpoint_config(settings)
        restored = (
            CmapssPreprocessor.from_state_dict(preprocessor_state)
            if preprocessor_state is not None
            else None
        )
        bundle = prepare_cmapss_datasets(
            data_dir=settings.data_dir,
            subset=settings.subset,
            window_size=settings.window_size,
            stride=settings.stride,
            val_fraction=settings.val_fraction,
            seed=settings.split_seed,
            rul_cap=settings.rul_cap,
            variance_threshold=settings.variance_threshold,
            preprocessor=restored,
            train_rul_filter=settings.train_rul_filter,
        )
        splits = {
            "train": bundle.train_engine_ids,
            "val": bundle.val_engine_ids,
            "test": bundle.test_dataset.engine_ids,
        }
        return DataBundle(
            dataset_name=self.name,
            subset=bundle.subset,
            train_dataset=bundle.train_dataset,
            val_dataset=bundle.val_dataset,
            test_dataset=bundle.test_dataset,
            preprocessor=bundle.preprocessor,
            split_entity_ids=splits,
            evaluation_spec=EvaluationSpec(
                protocol="endpoint_per_entity",
                batch_size=1,
                reset_each_batch=True,
                require_single_item=True,
            ),
            data_config=data_config,
            label_policy=self._label_policy(settings.rul_cap),
            metadata=bundle.metadata,
        )

    def prepare_evaluation(
        self,
        settings: Any,
        preprocessor_state: Mapping[str, Any],
    ) -> EvaluationBundle:
        data_config = self.checkpoint_config(settings)
        preprocessor = CmapssPreprocessor.from_state_dict(preprocessor_state)
        dataset = prepare_cmapss_test_dataset(
            data_dir=settings.data_dir,
            subset=settings.subset,
            preprocessor=preprocessor,
            window_size=settings.window_size,
            rul_cap=settings.rul_cap,
        )
        return EvaluationBundle(
            dataset_name=self.name,
            subset=normalize_subset(settings.subset),
            test_dataset=dataset,
            preprocessor=preprocessor,
            evaluation_spec=EvaluationSpec(
                protocol="endpoint_per_entity",
                batch_size=1,
                reset_each_batch=True,
                require_single_item=True,
            ),
            data_config=data_config,
            label_policy=self._label_policy(settings.rul_cap),
        )


__all__ = [
    "ENGINE_ID_COLUMN",
    "CYCLE_COLUMN",
    "SETTING_COLUMNS",
    "OPERATIONAL_SETTING_COLUMNS",
    "SENSOR_COLUMNS",
    "FEATURE_COLUMNS",
    "CMAPSS_COLUMNS",
    "COLUMN_NAMES",
    "RUL_COLUMN",
    "VALID_SUBSETS",
    "CmapssRawData",
    "CmapssPreprocessor",
    "EngineTrajectoryCmapssDataset",
    "WindowedCmapssDataset",
    "CmapssWindowDataset",
    "CmapssDataBundle",
    "CmapssAdapter",
    "normalize_subset",
    "read_cmapss_data",
    "read_cmapss_rul",
    "map_test_rul_to_engines",
    "load_cmapss_files",
    "load_cmapss_split",
    "split_engine_ids",
    "add_train_rul_labels",
    "add_test_rul_labels",
    "prepare_cmapss_test_dataset",
    "prepare_cmapss_datasets",
    "prepare_cmapss_data",
]
