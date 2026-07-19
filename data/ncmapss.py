"""Lazy, leakage-safe N-CMAPSS HDF5 data pipeline."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .base import (
    DataBundle,
    EvaluationBundle,
    EvaluationSpec,
    RulStageFilter,
    rul_stage_mask,
    split_entity_ids,
)
from .preprocessing import StreamingStandardizer


_REQUIRED_ARRAYS = tuple(
    f"{name}_{split}"
    for split in ("dev", "test")
    for name in ("W", "X_s", "X_v", "T", "Y", "A")
)
_VARIABLE_ARRAYS = ("W_var", "X_s_var", "X_v_var", "T_var", "A_var")
_ALLOWED_FEATURE_GROUPS = ("W", "X_s", "X_v")
_SCAN_CHUNK_ROWS = 250_000


@dataclass(frozen=True)
class NCmapssOptions:
    feature_groups: tuple[str, ...] = ("W", "X_s")
    downsample_factor: int = 1
    window_boundary: str = "unit"
    validation_units: tuple[int, ...] | None = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "NCmapssOptions":
        allowed = {
            "feature_groups",
            "downsample_factor",
            "window_boundary",
            "validation_units",
        }
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown N-CMAPSS data.options keys: {unknown}")
        groups = tuple(
            str(value) for value in values.get("feature_groups", ("W", "X_s"))
        )
        if not groups or len(set(groups)) != len(groups):
            raise ValueError("N-CMAPSS feature_groups must be non-empty and unique.")
        invalid = [value for value in groups if value not in _ALLOWED_FEATURE_GROUPS]
        if invalid:
            raise ValueError(
                f"Unsupported or leakage-prone N-CMAPSS feature groups: {invalid}; "
                f"allowed={list(_ALLOWED_FEATURE_GROUPS)}."
            )
        downsample = values.get("downsample_factor", 1)
        if (
            isinstance(downsample, bool)
            or not isinstance(downsample, int)
            or downsample <= 0
        ):
            raise ValueError("N-CMAPSS downsample_factor must be a positive integer.")
        boundary = str(values.get("window_boundary", "unit")).lower()
        if boundary not in {"unit", "flight"}:
            raise ValueError("N-CMAPSS window_boundary must be 'unit' or 'flight'.")
        raw_validation = values.get("validation_units")
        validation = None
        if raw_validation is not None:
            if not isinstance(raw_validation, (list, tuple)) or not raw_validation:
                raise ValueError("validation_units must be a non-empty list or null.")
            validation = tuple(sorted(int(value) for value in raw_validation))
            if len(set(validation)) != len(validation) or any(
                value <= 0 for value in validation
            ):
                raise ValueError(
                    "validation_units must contain unique positive integers."
                )
        return cls(groups, downsample, boundary, validation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_groups": list(self.feature_groups),
            "downsample_factor": self.downsample_factor,
            "window_boundary": self.window_boundary,
            "validation_units": (
                list(self.validation_units)
                if self.validation_units is not None
                else None
            ),
        }


@dataclass(frozen=True)
class _UnitSpan:
    entity_id: int
    start: int
    stop: int
    entity_start: int

    @property
    def rows(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class _Schema:
    feature_names: tuple[str, ...]
    group_names: Mapping[str, tuple[str, ...]]
    unit_column: int
    cycle_column: int
    fingerprint: Mapping[str, Any]


def _decode_names(dataset: h5py.Dataset) -> tuple[str, ...]:
    return tuple(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in dataset[...]
    )


def normalize_ncmapss_subset(subset: str) -> str:
    value = str(subset).strip().upper()
    if value.endswith(".H5"):
        value = value[:-3]
    if value.startswith("N-CMAPSS_"):
        value = value[len("N-CMAPSS_") :]
    if not value.startswith("DS"):
        raise ValueError(f"Invalid N-CMAPSS subset {subset!r}; expected a DSxx name.")
    return value


def resolve_ncmapss_file(data_dir: str | Path, subset: str) -> Path:
    normalized = normalize_ncmapss_subset(subset)
    path = Path(data_dir) / f"N-CMAPSS_{normalized}.h5"
    if not path.is_file():
        raise FileNotFoundError(f"N-CMAPSS file not found: {path}")
    return path


def _open_hdf5(path: Path) -> h5py.File:
    try:
        return h5py.File(path, "r")
    except OSError as error:
        raise OSError(f"Cannot open N-CMAPSS HDF5 file {path}: {error}") from error


def _inspect_schema(path: Path, feature_groups: Sequence[str]) -> _Schema:
    with _open_hdf5(path) as hdf:
        missing = [
            key for key in (*_REQUIRED_ARRAYS, *_VARIABLE_ARRAYS) if key not in hdf
        ]
        if missing:
            raise ValueError(f"N-CMAPSS file is missing datasets: {missing}")
        group_names = {
            group: _decode_names(hdf[f"{group}_var"])
            for group in ("W", "X_s", "X_v", "T", "A")
        }
        for split in ("dev", "test"):
            row_count = int(hdf[f"A_{split}"].shape[0])
            for group in ("W", "X_s", "X_v", "T"):
                dataset = hdf[f"{group}_{split}"]
                if dataset.ndim != 2 or dataset.shape != (
                    row_count,
                    len(group_names[group]),
                ):
                    raise ValueError(
                        f"N-CMAPSS dataset {group}_{split} has invalid shape."
                    )
            if hdf[f"Y_{split}"].shape != (row_count, 1):
                raise ValueError(f"N-CMAPSS dataset Y_{split} must have shape [N, 1].")
            if hdf[f"A_{split}"].shape != (row_count, len(group_names["A"])):
                raise ValueError(f"N-CMAPSS dataset A_{split} has invalid shape.")
        try:
            unit_column = group_names["A"].index("unit")
            cycle_column = group_names["A"].index("cycle")
        except ValueError as error:
            raise ValueError("N-CMAPSS A_var must contain unit and cycle.") from error
        feature_names = tuple(
            name for group in feature_groups for name in group_names[group]
        )
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("Selected N-CMAPSS feature names are not unique.")
        fingerprint = {
            "file": path.name,
            "file_size": path.stat().st_size,
            "variables": {key: list(value) for key, value in group_names.items()},
            "shapes": {key: list(hdf[key].shape) for key in (*_REQUIRED_ARRAYS,)},
        }
    return _Schema(feature_names, group_names, unit_column, cycle_column, fingerprint)


def _scan_unit_spans(path: Path, split: str, unit_column: int) -> tuple[_UnitSpan, ...]:
    spans: list[_UnitSpan] = []
    seen: set[int] = set()
    current_id: int | None = None
    current_start = 0
    with _open_hdf5(path) as hdf:
        auxiliary = hdf[f"A_{split}"]
        total = int(auxiliary.shape[0])
        for chunk_start in range(0, total, _SCAN_CHUNK_ROWS):
            chunk_stop = min(total, chunk_start + _SCAN_CHUNK_ROWS)
            raw = np.asarray(auxiliary[chunk_start:chunk_stop, unit_column])
            if not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
                raise ValueError(f"A_{split}.unit must contain finite integers.")
            values = raw.astype(np.int64)
            if (values <= 0).any():
                raise ValueError(f"A_{split}.unit must contain positive integers.")
            boundaries = np.flatnonzero(values[1:] != values[:-1]) + 1
            starts = np.concatenate(([0], boundaries))
            stops = np.concatenate((boundaries, [values.size]))
            for local_start, local_stop in zip(starts, stops):
                entity_id = int(values[local_start])
                global_start = chunk_start + int(local_start)
                global_stop = chunk_start + int(local_stop)
                if current_id is None:
                    current_id = entity_id
                    current_start = global_start
                elif entity_id != current_id:
                    spans.append(
                        _UnitSpan(
                            current_id, current_start, global_start, current_start
                        )
                    )
                    seen.add(current_id)
                    if entity_id in seen:
                        raise ValueError(
                            f"Unit {entity_id} is not stored contiguously in A_{split}."
                        )
                    current_id = entity_id
                    current_start = global_start
                if global_stop == total:
                    spans.append(
                        _UnitSpan(entity_id, current_start, total, current_start)
                    )
        if not spans:
            raise ValueError(f"N-CMAPSS split {split!r} is empty.")
    return tuple(spans)


def _split_flights(
    path: Path, split: str, spans: Sequence[_UnitSpan], cycle_column: int
) -> tuple[_UnitSpan, ...]:
    result: list[_UnitSpan] = []
    with _open_hdf5(path) as hdf:
        auxiliary = hdf[f"A_{split}"]
        for span in spans:
            cycles = np.asarray(auxiliary[span.start : span.stop, cycle_column])
            if (
                not np.isfinite(cycles).all()
                or not np.equal(cycles, np.floor(cycles)).all()
            ):
                raise ValueError(f"A_{split}.cycle must contain finite integers.")
            boundaries = np.flatnonzero(cycles[1:] != cycles[:-1]) + 1
            starts = np.concatenate(([0], boundaries))
            stops = np.concatenate((boundaries, [cycles.size]))
            result.extend(
                _UnitSpan(
                    span.entity_id,
                    span.start + int(start),
                    span.start + int(stop),
                    span.entity_start,
                )
                for start, stop in zip(starts, stops)
            )
    return tuple(result)


def _cap_targets(values: np.ndarray, rul_cap: float | None) -> np.ndarray:
    targets = np.asarray(values, dtype=np.float64)
    if rul_cap is not None:
        targets = np.minimum(targets, float(rul_cap))
    return targets


def _filter_training_spans(
    path: Path,
    spans: Sequence[_UnitSpan],
    stage_filter: RulStageFilter,
    rul_cap: float | None,
) -> tuple[tuple[_UnitSpan, ...], dict[str, Any]]:
    if not stage_filter.enabled:
        metadata = {
            "filter": stage_filter.to_dict(),
            "entities": [
                {
                    "entity_id": span.entity_id,
                    "rows_before": span.rows,
                    "rows_after": span.rows,
                }
                for span in spans
            ],
        }
        return tuple(spans), metadata

    filtered: list[_UnitSpan] = []
    entities: list[dict[str, Any]] = []
    with _open_hdf5(path) as hdf:
        targets_dataset = hdf["Y_dev"]
        for span in spans:
            targets = _cap_targets(targets_dataset[span.start : span.stop, 0], rul_cap)
            mask, metadata = rul_stage_mask(
                np.full(targets.size, span.entity_id, dtype=np.int64),
                targets,
                stage_filter,
            )
            positions = np.flatnonzero(mask)
            boundaries = np.flatnonzero(positions[1:] != positions[:-1] + 1) + 1
            for part in np.split(positions, boundaries):
                filtered.append(
                    _UnitSpan(
                        span.entity_id,
                        span.start + int(part[0]),
                        span.start + int(part[-1]) + 1,
                        span.entity_start,
                    )
                )
            entities.extend(metadata["entities"])
    return tuple(filtered), {"filter": stage_filter.to_dict(), "entities": entities}


def _read_feature_slice(
    hdf: h5py.File,
    split: str,
    groups: Sequence[str],
    row_slice: slice,
) -> np.ndarray:
    arrays = [np.asarray(hdf[f"{group}_{split}"][row_slice]) for group in groups]
    return np.concatenate(arrays, axis=1)


def _feature_batches(
    path: Path,
    split: str,
    spans: Sequence[_UnitSpan],
    groups: Sequence[str],
    downsample_factor: int,
) -> Iterator[np.ndarray]:
    raw_chunk = _SCAN_CHUNK_ROWS * downsample_factor
    with _open_hdf5(path) as hdf:
        for span in spans:
            for start in range(span.start, span.stop, raw_chunk):
                stop = min(span.stop, start + raw_chunk)
                yield _read_feature_slice(
                    hdf, split, groups, slice(start, stop, downsample_factor)
                )


class NCmapssWindowDataset(Dataset):
    """Compact window index over lazily read N-CMAPSS HDF5 arrays."""

    def __init__(
        self,
        path: Path,
        split: str,
        spans: Sequence[_UnitSpan],
        feature_groups: Sequence[str],
        preprocessor: StreamingStandardizer,
        *,
        window_size: int,
        stride: int,
        downsample_factor: int,
        rul_cap: float | None,
        include_partial: bool = False,
        last_only: bool = False,
        cycle_column: int = 1,
    ) -> None:
        if split not in {"dev", "test"}:
            raise ValueError("N-CMAPSS split must be 'dev' or 'test'.")
        if not spans:
            raise ValueError("N-CMAPSS dataset requires at least one sequence span.")
        self.path = Path(path)
        self.split = split
        self.spans = tuple(spans)
        self.feature_groups = tuple(feature_groups)
        self.preprocessor = preprocessor
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.downsample_factor = int(downsample_factor)
        self.rul_cap = rul_cap
        self.include_partial = bool(include_partial)
        self.last_only = bool(last_only)
        self.cycle_column = int(cycle_column)
        self.feature_names = preprocessor.selected_features
        self.feature_dim = preprocessor.output_dim
        self.entity_ids = tuple(sorted({span.entity_id for span in self.spans}))
        self.engine_ids = self.entity_ids
        self._lengths: list[int] = []
        cumulative: list[int] = []
        total = 0
        for span in self.spans:
            length = math.ceil(span.rows / self.downsample_factor)
            self._lengths.append(length)
            if self.last_only:
                count = 1
            elif self.include_partial:
                count = (length - 1) // self.stride + 1
                if (count - 1) * self.stride != length - 1:
                    count += 1
            elif length < self.window_size:
                count = 1
            else:
                count = (length - self.window_size) // self.stride + 1
            total += count
            cumulative.append(total)
        self._cumulative = tuple(cumulative)
        self._hdf: h5py.File | None = None

    def __len__(self) -> int:
        return self._cumulative[-1]

    def _endpoint(self, span_index: int, local_index: int) -> int:
        length = self._lengths[span_index]
        previous = self._cumulative[span_index - 1] if span_index else 0
        span_count = self._cumulative[span_index] - previous
        if self.last_only:
            return length - 1
        if self.include_partial:
            if local_index == span_count - 1:
                return length - 1
            return min(local_index * self.stride, length - 1)
        if length < self.window_size:
            return length - 1
        return self.window_size - 1 + local_index * self.stride

    def _handle(self) -> h5py.File:
        if self._hdf is None:
            self._hdf = _open_hdf5(self.path)
        return self._hdf

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_hdf"] = None
        return state

    def close(self) -> None:
        if self._hdf is not None:
            self._hdf.close()
            self._hdf = None

    def __del__(self) -> None:
        if getattr(self, "_hdf", None) is not None:
            try:
                self.close()
            except Exception:
                pass

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        span_index = bisect.bisect_right(self._cumulative, index)
        previous = self._cumulative[span_index - 1] if span_index else 0
        local_index = index - previous
        span = self.spans[span_index]
        endpoint = self._endpoint(span_index, local_index)
        state_new_tokens = (
            min(self.window_size, endpoint + 1)
            if local_index == 0
            else endpoint - self._endpoint(span_index, local_index - 1)
        )

        logical_start = max(0, endpoint - self.window_size + 1)
        raw_start = span.start + logical_start * self.downsample_factor
        raw_endpoint = min(
            span.start + endpoint * self.downsample_factor, span.stop - 1
        )
        raw_stop = raw_endpoint + 1
        hdf = self._handle()
        raw_features = _read_feature_slice(
            hdf,
            self.split,
            self.feature_groups,
            slice(raw_start, raw_stop, self.downsample_factor),
        )
        valid = self.preprocessor.transform(raw_features)
        pad_length = self.window_size - valid.shape[0]
        window = np.zeros((self.window_size, self.feature_dim), dtype=np.float32)
        window[pad_length:] = valid
        padding_mask = np.zeros(self.window_size, dtype=np.bool_)
        padding_mask[:pad_length] = True
        target = float(hdf[f"Y_{self.split}"][raw_endpoint, 0])
        if self.rul_cap is not None:
            target = min(target, float(self.rul_cap))
        cycle = int(hdf[f"A_{self.split}"][raw_endpoint, self.cycle_column])
        time_index = raw_endpoint - span.entity_start
        return {
            "features": torch.from_numpy(window),
            "padding_mask": torch.from_numpy(padding_mask),
            "target": torch.tensor(target, dtype=torch.float32),
            "entity_id": torch.tensor(span.entity_id, dtype=torch.long),
            "time_index": torch.tensor(time_index, dtype=torch.long),
            "state_new_tokens": torch.tensor(state_new_tokens, dtype=torch.long),
            "unit_id": torch.tensor(span.entity_id, dtype=torch.long),
            "cycle": torch.tensor(cycle, dtype=torch.long),
        }


class NCmapssAdapter:
    name = "ncmapss"

    def _context(self, settings: Any) -> tuple[Path, NCmapssOptions, _Schema]:
        options = NCmapssOptions.from_mapping(settings.options)
        path = resolve_ncmapss_file(settings.data_dir, settings.subset)
        schema = _inspect_schema(path, options.feature_groups)
        return path, options, schema

    def checkpoint_config(self, settings: Any) -> dict[str, Any]:
        path, options, _ = self._context(settings)
        values = settings.checkpoint_values()
        values.update(
            name=self.name,
            subset=normalize_ncmapss_subset(settings.subset),
            data_dir=str(path.parent.resolve()),
            options=options.to_dict(),
        )
        return values

    def validate_checkpoint(self, settings: Any, checkpoint: Mapping[str, Any]) -> None:
        expected = self.checkpoint_config(settings)
        _, _, schema = self._context(settings)
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
            if actual.get(key) != expected[key]:
                raise ValueError(
                    f"Configured data.{key} does not match checkpoint: "
                    f"{expected[key]!r} != {actual.get(key)!r}."
                )
        metadata = checkpoint.get("data_metadata")
        saved_schema = (
            metadata.get("dataset_schema") if isinstance(metadata, Mapping) else None
        )
        if not isinstance(saved_schema, Mapping):
            raise ValueError("N-CMAPSS checkpoint is missing dataset schema metadata.")
        if dict(saved_schema) != dict(schema.fingerprint):
            raise RuntimeError("N-CMAPSS HDF5 schema differs from the checkpoint.")

    @staticmethod
    def _label_policy(rul_cap: float | None) -> dict[str, Any]:
        return {
            "train": "provided_per_sample_rul",
            "test": "provided_full_trajectory_rul",
            "cap_applied_to": "train_and_test_targets"
            if rul_cap is not None
            else "none",
        }

    @staticmethod
    def _partition(
        spans: Sequence[_UnitSpan], options: NCmapssOptions, settings: Any
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        entity_ids = tuple(span.entity_id for span in spans)
        if options.validation_units is None:
            return split_entity_ids(
                entity_ids, settings.val_fraction, settings.split_seed
            )
        available = set(entity_ids)
        val_ids = options.validation_units
        if not set(val_ids).issubset(available):
            raise ValueError(
                f"validation_units are not present in dev split: "
                f"{sorted(set(val_ids) - available)}"
            )
        train_ids = tuple(sorted(available - set(val_ids)))
        if not train_ids:
            raise ValueError("validation_units cannot contain every development unit.")
        return train_ids, val_ids

    def prepare_training(
        self,
        settings: Any,
        preprocessor_state: Mapping[str, Any] | None = None,
    ) -> DataBundle:
        path, options, schema = self._context(settings)
        unit_spans = _scan_unit_spans(path, "dev", schema.unit_column)
        train_ids, val_ids = self._partition(unit_spans, options, settings)
        train_spans = tuple(span for span in unit_spans if span.entity_id in train_ids)
        val_spans = tuple(span for span in unit_spans if span.entity_id in val_ids)
        train_spans, filter_metadata = _filter_training_spans(
            path, train_spans, settings.train_rul_filter, settings.rul_cap
        )
        if options.window_boundary == "flight":
            train_spans = _split_flights(path, "dev", train_spans, schema.cycle_column)
            val_spans = _split_flights(path, "dev", val_spans, schema.cycle_column)
        test_spans = _scan_unit_spans(path, "test", schema.unit_column)
        if options.window_boundary == "flight":
            test_spans = _split_flights(path, "test", test_spans, schema.cycle_column)

        if preprocessor_state is None:
            preprocessor = StreamingStandardizer(
                schema.feature_names, settings.variance_threshold
            ).fit_batches(
                _feature_batches(
                    path,
                    "dev",
                    train_spans,
                    options.feature_groups,
                    options.downsample_factor,
                ),
                train_ids,
            )
        else:
            preprocessor = StreamingStandardizer.from_state_dict(preprocessor_state)
            if preprocessor.feature_names != schema.feature_names:
                raise RuntimeError(
                    "Checkpoint feature names do not match N-CMAPSS schema."
                )
            if preprocessor.fit_entity_ids != train_ids:
                raise RuntimeError(
                    "Checkpoint preprocessor fitting units do not match split."
                )

        common = dict(
            path=path,
            feature_groups=options.feature_groups,
            preprocessor=preprocessor,
            window_size=settings.window_size,
            downsample_factor=options.downsample_factor,
            rul_cap=settings.rul_cap,
            cycle_column=schema.cycle_column,
        )
        train_dataset = NCmapssWindowDataset(
            split="dev",
            spans=train_spans,
            stride=settings.stride,
            include_partial=settings.train_rul_filter.enabled,
            **common,
        )
        val_dataset = NCmapssWindowDataset(
            split="dev", spans=val_spans, stride=settings.stride, **common
        )
        test_dataset = NCmapssWindowDataset(
            split="test",
            spans=test_spans,
            stride=settings.evaluation_stride,
            include_partial=True,
            **common,
        )
        data_config = self.checkpoint_config(settings)
        split_ids = {
            "train": train_ids,
            "val": val_ids,
            "test": tuple(sorted({span.entity_id for span in test_spans})),
        }
        return DataBundle(
            dataset_name=self.name,
            subset=normalize_ncmapss_subset(settings.subset),
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            preprocessor=preprocessor,
            split_entity_ids=split_ids,
            evaluation_spec=EvaluationSpec("all_windows", batch_size=64),
            data_config=data_config,
            label_policy=self._label_policy(settings.rul_cap),
            metadata={
                "dataset_schema": schema.fingerprint,
                "train_rul_filter": filter_metadata,
            },
        )

    def prepare_evaluation(
        self,
        settings: Any,
        preprocessor_state: Mapping[str, Any],
    ) -> EvaluationBundle:
        path, options, schema = self._context(settings)
        preprocessor = StreamingStandardizer.from_state_dict(preprocessor_state)
        if preprocessor.feature_names != schema.feature_names:
            raise RuntimeError("Checkpoint feature names do not match N-CMAPSS schema.")
        spans = _scan_unit_spans(path, "test", schema.unit_column)
        if options.window_boundary == "flight":
            spans = _split_flights(path, "test", spans, schema.cycle_column)
        dataset = NCmapssWindowDataset(
            path,
            "test",
            spans,
            options.feature_groups,
            preprocessor,
            window_size=settings.window_size,
            stride=settings.evaluation_stride,
            downsample_factor=options.downsample_factor,
            rul_cap=settings.rul_cap,
            include_partial=True,
            cycle_column=schema.cycle_column,
        )
        return EvaluationBundle(
            dataset_name=self.name,
            subset=normalize_ncmapss_subset(settings.subset),
            test_dataset=dataset,
            preprocessor=preprocessor,
            evaluation_spec=EvaluationSpec("all_windows", batch_size=64),
            data_config=self.checkpoint_config(settings),
            label_policy=self._label_policy(settings.rul_cap),
            metadata={"dataset_schema": schema.fingerprint},
        )


__all__ = [
    "NCmapssAdapter",
    "NCmapssOptions",
    "NCmapssWindowDataset",
    "normalize_ncmapss_subset",
    "resolve_ncmapss_file",
]
