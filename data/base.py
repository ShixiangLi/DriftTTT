"""Shared contracts for RUL datasets."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np
from torch.utils.data import Dataset


@dataclass(frozen=True)
class RulStageFilter:
    """Select a label-space RUL interval for training trajectories only."""

    enabled: bool = False
    normalized_range: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("train_rul_filter.enabled must be true or false.")
        values = tuple(float(value) for value in self.normalized_range)
        if len(values) != 2 or not all(math.isfinite(value) for value in values):
            raise ValueError(
                "train_rul_filter.normalized_range must contain two finite values."
            )
        lower, upper = values
        if not 0.0 <= lower < upper <= 1.0:
            raise ValueError(
                "train_rul_filter.normalized_range must satisfy "
                "0 <= lower < upper <= 1."
            )
        object.__setattr__(self, "normalized_range", values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "normalized_range": list(
                self.normalized_range if self.enabled else (0.0, 1.0)
            ),
        }


@dataclass(frozen=True)
class EvaluationSpec:
    """Dataset-specific evaluation semantics consumed by the shared engine."""

    protocol: str
    batch_size: int
    reset_each_batch: bool = True
    require_single_item: bool = False

    def __post_init__(self) -> None:
        if self.protocol not in {"endpoint_per_entity", "all_windows"}:
            raise ValueError(f"Unsupported evaluation protocol: {self.protocol!r}")
        if self.batch_size <= 0:
            raise ValueError("Evaluation batch_size must be positive.")


@dataclass(frozen=True)
class DataBundle:
    dataset_name: str
    subset: str
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    preprocessor: Any
    split_entity_ids: Mapping[str, tuple[int, ...]]
    evaluation_spec: EvaluationSpec
    data_config: Mapping[str, Any]
    label_policy: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationBundle:
    dataset_name: str
    subset: str
    test_dataset: Dataset
    preprocessor: Any
    evaluation_spec: EvaluationSpec
    data_config: Mapping[str, Any]
    label_policy: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)


class DatasetAdapter(Protocol):
    name: str

    def checkpoint_config(self, settings: Any) -> dict[str, Any]: ...

    def validate_checkpoint(
        self, settings: Any, checkpoint: Mapping[str, Any]
    ) -> None: ...

    def prepare_training(
        self,
        settings: Any,
        preprocessor_state: Mapping[str, Any] | None = None,
    ) -> DataBundle: ...

    def prepare_evaluation(
        self,
        settings: Any,
        preprocessor_state: Mapping[str, Any],
    ) -> EvaluationBundle: ...


def split_entity_ids(
    entity_ids: np.ndarray | list[int] | tuple[int, ...],
    val_fraction: float,
    seed: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split complete entities without leaking rows between train and validation."""
    values = np.asarray(entity_ids, dtype=np.int64).reshape(-1)
    unique = np.unique(values)
    if unique.size < 2:
        raise ValueError("At least two entities are required for a validation split.")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1).")
    val_count = min(unique.size - 1, max(1, math.ceil(unique.size * val_fraction)))
    shuffled = np.random.default_rng(seed).permutation(unique)
    val_ids = tuple(int(value) for value in np.sort(shuffled[:val_count]))
    train_ids = tuple(int(value) for value in np.sort(shuffled[val_count:]))
    return train_ids, val_ids


def rul_stage_mask(
    entity_ids: np.ndarray,
    effective_rul: np.ndarray,
    stage_filter: RulStageFilter,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a per-entity mask from normalized RUL values, never row fractions."""
    ids = np.asarray(entity_ids, dtype=np.int64).reshape(-1)
    targets = np.asarray(effective_rul, dtype=np.float64).reshape(-1)
    if ids.shape != targets.shape or ids.size == 0:
        raise ValueError(
            "entity_ids and effective_rul must be equally sized and non-empty."
        )
    if not np.isfinite(targets).all() or (targets < 0).any():
        raise ValueError("RUL targets must be finite and non-negative.")

    selected = (
        np.ones(ids.size, dtype=np.bool_)
        if not stage_filter.enabled
        else np.zeros(ids.size, dtype=np.bool_)
    )
    lower, upper = stage_filter.normalized_range
    entities: list[dict[str, Any]] = []
    for entity_id in np.unique(ids):
        entity_positions = np.flatnonzero(ids == entity_id)
        entity_targets = targets[entity_positions]
        maximum = float(entity_targets.max())
        normalized = (
            entity_targets / maximum if maximum > 0.0 else np.zeros_like(entity_targets)
        )
        if stage_filter.enabled:
            keep = (normalized >= lower) & (normalized <= upper)
            selected[entity_positions] = keep
        kept = int(selected[entity_positions].sum())
        if kept == 0:
            raise ValueError(
                f"RUL stage filter removed every row for entity {int(entity_id)}."
            )
        kept_targets = entity_targets[selected[entity_positions]]
        entities.append(
            {
                "entity_id": int(entity_id),
                "rows_before": int(entity_positions.size),
                "rows_after": kept,
                "max_effective_rul": maximum,
                "kept_rul_min": float(kept_targets.min()),
                "kept_rul_max": float(kept_targets.max()),
            }
        )
    return selected, {"filter": stage_filter.to_dict(), "entities": entities}


__all__ = [
    "DataBundle",
    "DatasetAdapter",
    "EvaluationBundle",
    "EvaluationSpec",
    "RulStageFilter",
    "rul_stage_mask",
    "split_entity_ids",
]
