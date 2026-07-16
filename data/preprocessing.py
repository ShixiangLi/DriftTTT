"""Streaming feature standardization for datasets that do not fit in memory."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler


class StreamingStandardizer:
    """Variance filtering and standardization fitted from sequential chunks."""

    STATE_VERSION = 1

    def __init__(self, feature_names: Sequence[str], variance_threshold: float = 1e-12):
        names = tuple(str(value) for value in feature_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("feature_names must be non-empty and unique.")
        threshold = float(variance_threshold)
        if threshold < 0 or not math.isfinite(threshold):
            raise ValueError("variance_threshold must be finite and non-negative.")
        self.feature_names = names
        self.variance_threshold = threshold
        self.selected_indices: tuple[int, ...] = ()
        self.selected_features: tuple[str, ...] = ()
        self.feature_variances: tuple[float, ...] = ()
        self.fit_entity_ids: tuple[int, ...] = ()
        self.n_samples_seen = 0
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._variance: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self._mean is not None

    @property
    def output_dim(self) -> int:
        self._check_fitted()
        return len(self.selected_indices)

    def fit_batches(
        self,
        batches: Iterable[np.ndarray],
        fit_entity_ids: Sequence[int],
    ) -> "StreamingStandardizer":
        scaler = StandardScaler()
        seen = 0
        for batch in batches:
            values = np.asarray(batch, dtype=np.float64)
            if values.ndim != 2 or values.shape[1] != len(self.feature_names):
                raise ValueError("Feature batch shape does not match feature_names.")
            if values.shape[0] == 0:
                continue
            if not np.isfinite(values).all():
                raise ValueError("Feature batches must contain only finite values.")
            scaler.partial_fit(values)
            seen += int(values.shape[0])
        if seen == 0:
            raise ValueError("No feature rows were provided to fit the preprocessor.")

        variances = np.asarray(scaler.var_, dtype=np.float64)
        selected = tuple(
            int(index) for index in np.flatnonzero(variances > self.variance_threshold)
        )
        if not selected:
            raise ValueError("No features remain after low-variance filtering.")
        indices = np.asarray(selected, dtype=np.int64)
        self.selected_indices = selected
        self.selected_features = tuple(self.feature_names[index] for index in selected)
        self.feature_variances = tuple(float(value) for value in variances)
        self.fit_entity_ids = tuple(sorted(int(value) for value in fit_entity_ids))
        self.n_samples_seen = seen
        self._mean = np.asarray(scaler.mean_, dtype=np.float64)[indices]
        selected_variance = variances[indices]
        self._variance = selected_variance
        self._scale = np.sqrt(selected_variance)
        self._scale[self._scale == 0.0] = 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        self._check_fitted()
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != len(self.feature_names):
            raise ValueError("Feature array shape does not match fitted feature_names.")
        if not np.isfinite(array).all():
            raise ValueError("Feature values must be finite.")
        assert self._mean is not None and self._scale is not None
        result = (array[:, self.selected_indices] - self._mean) / self._scale
        return result.astype(np.float32, copy=False)

    def _check_fitted(self) -> None:
        if self._mean is None or self._scale is None or self._variance is None:
            raise RuntimeError("StreamingStandardizer must be fitted before use.")

    def state_dict(self) -> dict[str, Any]:
        self._check_fitted()
        assert (
            self._mean is not None
            and self._scale is not None
            and self._variance is not None
        )
        return {
            "type": "streaming_standardizer",
            "version": self.STATE_VERSION,
            "variance_threshold": self.variance_threshold,
            "feature_names": list(self.feature_names),
            "selected_indices": list(self.selected_indices),
            "selected_features": list(self.selected_features),
            "feature_variances": list(self.feature_variances),
            "mean": self._mean.tolist(),
            "scale": self._scale.tolist(),
            "variance": self._variance.tolist(),
            "n_samples_seen": self.n_samples_seen,
            "fit_entity_ids": list(self.fit_entity_ids),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "StreamingStandardizer":
        if int(state.get("version", -1)) != cls.STATE_VERSION:
            raise ValueError(
                f"Unsupported preprocessing state version: {state.get('version')}"
            )
        instance = cls(
            state["feature_names"],
            variance_threshold=float(state["variance_threshold"]),
        )
        selected = tuple(int(value) for value in state["selected_indices"])
        selected_features = tuple(str(value) for value in state["selected_features"])
        if selected_features != tuple(
            instance.feature_names[index] for index in selected
        ):
            raise ValueError("Preprocessor selected feature metadata is inconsistent.")
        variances = np.asarray(state["feature_variances"], dtype=np.float64)
        mean = np.asarray(state["mean"], dtype=np.float64)
        scale = np.asarray(state["scale"], dtype=np.float64)
        variance = np.asarray(state["variance"], dtype=np.float64)
        if len(variances) != len(instance.feature_names) or not (
            len(mean) == len(scale) == len(variance) == len(selected)
        ):
            raise ValueError("Preprocessor state dimensions are inconsistent.")
        if not all(
            np.isfinite(values).all() for values in (variances, mean, scale, variance)
        ):
            raise ValueError("Preprocessor state contains non-finite values.")
        if (scale <= 0).any() or (variance < 0).any():
            raise ValueError("Preprocessor scale or variance is invalid.")
        instance.selected_indices = selected
        instance.selected_features = selected_features
        instance.feature_variances = tuple(float(value) for value in variances)
        instance.fit_entity_ids = tuple(int(value) for value in state["fit_entity_ids"])
        instance.n_samples_seen = int(state["n_samples_seen"])
        instance._mean = mean
        instance._scale = scale
        instance._variance = variance
        return instance


__all__ = ["StreamingStandardizer"]
