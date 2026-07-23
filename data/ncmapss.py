from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .base import DatasetBundle
from .preprocessing import FeatureScaler, RunningMoments


ALLOWED_FEATURE_GROUPS = {"W", "X_s", "X_v"}


@dataclass(frozen=True)
class EntitySpan:
    entity_id: int
    start: int
    stop: int


def _decode_names(values: np.ndarray) -> list[str]:
    return [
        value.decode() if isinstance(value, bytes) else str(value) for value in values
    ]


def _resolve_h5(root: Path, subset: str) -> Path:
    candidate = root / subset
    if candidate.suffix.lower() != ".h5":
        name = subset if subset.startswith("N-CMAPSS_") else f"N-CMAPSS_{subset}"
        candidate = root / f"{name}.h5"
    if not candidate.is_file():
        raise FileNotFoundError(f"Missing N-CMAPSS file: {candidate}")
    return candidate


def _validate_schema(path: Path, groups: list[str]) -> dict[str, list[str]]:
    try:
        with h5py.File(path, "r") as handle:
            names: dict[str, list[str]] = {}
            if "A_var" not in handle:
                raise ValueError("Missing HDF5 variable names: A_var")
            auxiliary_names = _decode_names(handle["A_var"][:])
            if len(auxiliary_names) < 2 or auxiliary_names[1].lower() != "cycle":
                raise ValueError("N-CMAPSS A arrays must contain cycle in column 1")
            for split in ("dev", "test"):
                required = [
                    f"A_{split}",
                    f"Y_{split}",
                    *[f"{g}_{split}" for g in groups],
                ]
                missing = [key for key in required if key not in handle]
                if missing:
                    raise ValueError(f"Missing HDF5 datasets: {missing}")
                lengths = {handle[key].shape[0] for key in required}
                if len(lengths) != 1:
                    raise ValueError(f"Unaligned N-CMAPSS {split} arrays")
                auxiliary = handle[f"A_{split}"]
                if auxiliary.ndim != 2 or auxiliary.shape[1] != len(auxiliary_names):
                    raise ValueError(f"Malformed N-CMAPSS A_{split} array")
            for group in groups:
                variable_key = f"{group}_var"
                if variable_key not in handle:
                    raise ValueError(f"Missing HDF5 variable names: {variable_key}")
                names[group] = _decode_names(handle[variable_key][:])
            return names
    except OSError as error:
        raise ValueError(f"Cannot open N-CMAPSS HDF5 file {path}: {error}") from error


def _scan_entity_spans(path: Path, split: str, chunk_rows: int) -> list[EntitySpan]:
    spans: list[EntitySpan] = []
    with h5py.File(path, "r") as handle:
        dataset = handle[f"A_{split}"]
        total = dataset.shape[0]
        current_id: int | None = None
        span_start = 0
        for start in range(0, total, chunk_rows):
            stop = min(start + chunk_rows, total)
            unit_ids = dataset[start:stop, 0].astype(np.int64)
            if unit_ids.size == 0:
                continue
            if current_id is None:
                current_id = int(unit_ids[0])
                span_start = start
            elif int(unit_ids[0]) != current_id:
                spans.append(EntitySpan(current_id, span_start, start))
                current_id = int(unit_ids[0])
                span_start = start
            changes = np.flatnonzero(unit_ids[1:] != unit_ids[:-1]) + 1
            for offset in changes:
                boundary = start + int(offset)
                assert current_id is not None
                spans.append(EntitySpan(current_id, span_start, boundary))
                current_id = int(unit_ids[offset])
                span_start = boundary
        if current_id is not None:
            spans.append(EntitySpan(current_id, span_start, total))
    ids = [span.entity_id for span in spans]
    if len(ids) != len(set(ids)):
        raise ValueError("N-CMAPSS rows for an entity must be contiguous")
    return spans


def _split_spans(
    spans: list[EntitySpan], validation_fraction: float, seed: int
) -> tuple[list[EntitySpan], list[EntitySpan]]:
    if len(spans) < 2:
        raise ValueError("At least two N-CMAPSS development units are required")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(spans))
    validation_count = max(1, int(round(len(spans) * validation_fraction)))
    validation_count = min(validation_count, len(spans) - 1)
    validation_indices = set(order[:validation_count].tolist())
    train = [
        span for index, span in enumerate(spans) if index not in validation_indices
    ]
    validation = [
        span for index, span in enumerate(spans) if index in validation_indices
    ]
    return train, validation


def _read_feature_chunk(
    handle: h5py.File,
    split: str,
    groups: list[str],
    start: int,
    stop: int,
    step: int = 1,
    include_cycle: bool = False,
) -> np.ndarray:
    arrays = [handle[f"{group}_{split}"][start:stop:step] for group in groups]
    if include_cycle:
        arrays.insert(0, handle[f"A_{split}"][start:stop:step, 1:2])
    return np.concatenate(arrays, axis=1).astype(np.float32, copy=False)


def _fit_scaler(
    path: Path,
    spans: Iterable[EntitySpan],
    groups: list[str],
    names: list[str],
    chunk_rows: int,
    variance_threshold: float,
    include_cycle: bool,
) -> FeatureScaler:
    moments = RunningMoments(len(names))
    with h5py.File(path, "r") as handle:
        for span in spans:
            for start in range(span.start, span.stop, chunk_rows):
                stop = min(start + chunk_rows, span.stop)
                moments.update(
                    _read_feature_chunk(
                        handle,
                        "dev",
                        groups,
                        start,
                        stop,
                        include_cycle=include_cycle,
                    )
                )
    return FeatureScaler.from_moments(names, moments, variance_threshold)


class NcmapssWindowDataset(Dataset):
    def __init__(
        self,
        path: Path,
        split: str,
        spans: list[EntitySpan],
        feature_groups: list[str],
        scaler: FeatureScaler,
        window_size: int,
        stride: int,
        downsample: int,
        rul_cap: float | None,
        include_cycle: bool,
        include_partial: bool = False,
    ) -> None:
        self.path = path
        self.split = split
        self.spans = spans
        self.feature_groups = feature_groups
        self.scaler = scaler
        self.window_size = window_size
        self.downsample = downsample
        self.rul_cap = rul_cap
        self.include_cycle = include_cycle
        self._handle: h5py.File | None = None
        self.samples: list[tuple[int, int]] = []
        required_history = (window_size - 1) * downsample
        for span in spans:
            first = span.start if include_partial else span.start + required_history
            endpoints = list(range(first, span.stop, stride))
            if endpoints and endpoints[-1] != span.stop - 1:
                endpoints.append(span.stop - 1)
            self.samples.extend((span.entity_id, endpoint) for endpoint in endpoints)
        self._span_by_id = {span.entity_id: span for span in spans}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def _h5(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        entity_id, endpoint = self.samples[index]
        span = self._span_by_id[entity_id]
        start = max(span.start, endpoint - (self.window_size - 1) * self.downsample)
        raw = _read_feature_chunk(
            self._h5(),
            self.split,
            self.feature_groups,
            start,
            endpoint + 1,
            self.downsample,
            include_cycle=self.include_cycle,
        )
        window = self.scaler.transform(raw).astype(np.float32)
        valid_length = window.shape[0]
        padded = np.zeros((self.window_size, window.shape[1]), dtype=np.float32)
        padded[-valid_length:] = window
        cycles = (
            raw[:, 0].astype(np.int64)
            if self.include_cycle
            else self._h5()[f"A_{self.split}"][
                start : endpoint + 1 : self.downsample, 1
            ].astype(np.int64)
        )
        padded_cycles = np.full(self.window_size, -1, dtype=np.int64)
        padded_cycles[-valid_length:] = cycles
        mask = np.zeros(self.window_size, dtype=np.bool_)
        mask[-valid_length:] = True
        target = float(self._h5()[f"Y_{self.split}"][endpoint, 0])
        if self.rul_cap is not None:
            target = min(target, self.rul_cap) / self.rul_cap
        return {
            "features": torch.from_numpy(padded),
            "mask": torch.from_numpy(mask),
            "cycle_ids": torch.from_numpy(padded_cycles),
            "target": torch.tensor(target, dtype=torch.float32),
            "entity_id": torch.tensor(entity_id, dtype=torch.int64),
            "endpoint": torch.tensor(endpoint - span.start, dtype=torch.int64),
        }

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:
        self.close()


def build_ncmapss_bundle(
    config: dict[str, Any],
    preprocessing_state: dict[str, Any] | None = None,
    split_state: dict[str, Any] | None = None,
) -> DatasetBundle:
    root = Path(config["root"])
    path = _resolve_h5(root, str(config["subset"]))
    groups = list(config["options"].get("feature_groups", ["W", "X_s"]))
    if not groups or len(groups) != len(set(groups)):
        raise ValueError("N-CMAPSS feature_groups must be non-empty and unique")
    invalid = set(groups) - ALLOWED_FEATURE_GROUPS
    if invalid:
        raise ValueError(
            f"Invalid N-CMAPSS feature groups {sorted(invalid)}; T is forbidden"
        )
    names_by_group = _validate_schema(path, groups)
    include_cycle = bool(config["options"].get("include_cycle", False))
    feature_names = [name for group in groups for name in names_by_group[group]]
    if include_cycle:
        feature_names.insert(0, "cycle")
    chunk_rows = int(config["options"].get("chunk_rows", 262_144))
    downsample = int(config["options"].get("downsample", 1))
    if chunk_rows < 1 or downsample < 1:
        raise ValueError("chunk_rows and downsample must be positive")

    development_spans = _scan_entity_spans(path, "dev", chunk_rows)
    test_spans = _scan_entity_spans(path, "test", chunk_rows)
    if split_state is None:
        train_spans, validation_spans = _split_spans(
            development_spans, config["validation_fraction"], config["split_seed"]
        )
    else:
        train_ids = {int(value) for value in split_state["train_entities"]}
        validation_ids = {int(value) for value in split_state["validation_entities"]}
        train_spans = [
            span for span in development_spans if span.entity_id in train_ids
        ]
        validation_spans = [
            span for span in development_spans if span.entity_id in validation_ids
        ]
        if len(train_spans) != len(train_ids) or len(validation_spans) != len(
            validation_ids
        ):
            raise ValueError("Checkpoint N-CMAPSS units do not match the dataset")
    if preprocessing_state is None:
        scaler = _fit_scaler(
            path,
            train_spans,
            groups,
            feature_names,
            chunk_rows,
            config["variance_threshold"],
            include_cycle,
        )
    else:
        scaler = FeatureScaler.from_state_dict(preprocessing_state)
        if list(scaler.source_names) != feature_names:
            raise ValueError("Checkpoint feature schema does not match N-CMAPSS config")
    common = {
        "path": path,
        "feature_groups": groups,
        "scaler": scaler,
        "window_size": config["window_size"],
        "downsample": downsample,
        "rul_cap": config["rul_cap"],
        "include_cycle": include_cycle,
        "include_partial": bool(
            config["options"].get("include_partial_windows", False)
        ),
    }
    return DatasetBundle(
        train=NcmapssWindowDataset(
            split="dev", spans=train_spans, stride=config["stride"], **common
        ),
        validation=NcmapssWindowDataset(
            split="dev", spans=validation_spans, stride=config["stride"], **common
        ),
        test=NcmapssWindowDataset(
            split="test",
            spans=test_spans,
            stride=config["evaluation_stride"],
            **common,
        ),
        input_dim=len(scaler.feature_names),
        feature_names=scaler.feature_names,
        preprocessing_state=scaler.state_dict(),
        split_state={
            "train_entities": [span.entity_id for span in train_spans],
            "validation_entities": [span.entity_id for span in validation_spans],
            "test_entities": [span.entity_id for span in test_spans],
        },
        dataset_name="ncmapss",
        test_protocol=(
            "trajectory_with_partial_windows"
            if config["options"].get("include_partial_windows", False)
            else "full_window_trajectory"
        ),
    )
