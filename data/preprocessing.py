from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


class RunningMoments:
    """Numerically stable feature-wise moments accumulated in chunks."""

    def __init__(self, n_features: int) -> None:
        self.count = 0
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.m2 = np.zeros(n_features, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        if values.ndim != 2 or values.shape[1] != self.mean.size:
            raise ValueError("Expected a two-dimensional array with matching features")
        if values.shape[0] == 0:
            return
        batch = np.asarray(values, dtype=np.float64)
        if not np.isfinite(batch).all():
            raise ValueError("Input features contain NaN or infinite values")
        batch_count = batch.shape[0]
        batch_mean = batch.mean(axis=0)
        batch_m2 = np.square(batch - batch_mean).sum(axis=0)
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * (batch_count / total)
        self.m2 += batch_m2 + delta * delta * self.count * batch_count / total
        self.count = total

    @property
    def variance(self) -> np.ndarray:
        if self.count == 0:
            raise ValueError("Cannot compute statistics without observations")
        return self.m2 / self.count


@dataclass(frozen=True)
class FeatureScaler:
    """Variance selection and standardization fitted on training entities only."""

    source_names: tuple[str, ...]
    selected_indices: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def from_moments(
        cls,
        names: Sequence[str],
        moments: RunningMoments,
        variance_threshold: float,
    ) -> "FeatureScaler":
        variance = moments.variance
        selected = np.flatnonzero(variance > variance_threshold)
        if selected.size == 0:
            raise ValueError("Variance filtering removed every input feature")
        scale = np.sqrt(variance[selected])
        scale[scale == 0.0] = 1.0
        return cls(
            source_names=tuple(names),
            selected_indices=selected.astype(np.int64),
            mean=moments.mean[selected].astype(np.float32),
            scale=scale.astype(np.float32),
        )

    @property
    def feature_names(self) -> list[str]:
        return [self.source_names[index] for index in self.selected_indices]

    def transform(self, values: np.ndarray) -> np.ndarray:
        selected = np.asarray(values, dtype=np.float32)[:, self.selected_indices]
        return (selected - self.mean) / self.scale

    def state_dict(self) -> dict[str, Any]:
        return {
            "source_names": list(self.source_names),
            "selected_indices": self.selected_indices.tolist(),
            "selected_names": self.feature_names,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "FeatureScaler":
        return cls(
            source_names=tuple(state["source_names"]),
            selected_indices=np.asarray(state["selected_indices"], dtype=np.int64),
            mean=np.asarray(state["mean"], dtype=np.float32),
            scale=np.asarray(state["scale"], dtype=np.float32),
        )
