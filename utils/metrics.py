"""Regression metrics used by the C-MAPSS RUL benchmark."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


ArrayLike = Sequence[float] | np.ndarray | Any


class RegressionMetricAccumulator:
    """Exact streaming RMSE, MAE, and NASA Score accumulation."""

    def __init__(self) -> None:
        self.count = 0
        self.squared_error = 0.0
        self.absolute_error = 0.0
        self.score = 0.0

    def update(self, y_true: ArrayLike, y_pred: ArrayLike) -> None:
        true, pred = _validated_pair(y_true, y_pred)
        difference = pred - true
        exponent = np.where(difference < 0.0, -difference / 13.0, difference / 10.0)
        penalties = np.expm1(exponent)
        if not np.isfinite(penalties).all():
            raise ValueError("NASA Score overflowed for the supplied predictions.")
        self.count += int(difference.size)
        self.squared_error += float(np.square(difference).sum(dtype=np.float64))
        self.absolute_error += float(np.abs(difference).sum(dtype=np.float64))
        self.score += float(penalties.sum(dtype=np.float64))

    def compute(self) -> dict[str, float]:
        if self.count == 0:
            raise ValueError("Cannot compute metrics without samples.")
        return {
            "rmse": float(np.sqrt(self.squared_error / self.count)),
            "mae": self.absolute_error / self.count,
            "nasa_score": self.score,
        }


def _to_1d_finite(values: ArrayLike, name: str) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        values = values.numpy()
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validated_pair(
    y_true: ArrayLike, y_pred: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    true = _to_1d_finite(y_true, "y_true")
    pred = _to_1d_finite(y_pred, "y_pred")
    if true.shape != pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same number of values; "
            f"got {true.size} and {pred.size}."
        )
    return true, pred


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error."""

    true, pred = _validated_pair(y_true, y_pred)
    return float(np.sqrt(np.mean(np.square(pred - true))))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error."""

    true, pred = _validated_pair(y_true, y_pred)
    return float(np.mean(np.abs(pred - true)))


def nasa_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """NASA's asymmetric, summed C-MAPSS scoring function.

    With ``d = prediction - target``, early predictions (``d < 0``) incur
    ``exp(-d / 13) - 1`` and late predictions (``d >= 0``) incur the steeper
    ``exp(d / 10) - 1`` penalty.
    """

    true, pred = _validated_pair(y_true, y_pred)
    difference = pred - true
    penalties = np.where(
        difference < 0.0,
        np.expm1(-difference / 13.0),
        np.expm1(difference / 10.0),
    )
    return float(np.sum(penalties))


def compute_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Compute all benchmark metrics after one shared input validation pass."""

    true, pred = _validated_pair(y_true, y_pred)
    difference = pred - true
    penalties = np.where(
        difference < 0.0,
        np.expm1(-difference / 13.0),
        np.expm1(difference / 10.0),
    )
    return {
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "mae": float(np.mean(np.abs(difference))),
        "nasa_score": float(np.sum(penalties)),
    }


evaluate_metrics = compute_metrics


__all__ = [
    "RegressionMetricAccumulator",
    "rmse",
    "mae",
    "nasa_score",
    "compute_metrics",
    "evaluate_metrics",
]
