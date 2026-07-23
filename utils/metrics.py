from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RegressionAccumulator:
    target_scale: float = 1.0
    count: int = 0
    squared_error: float = 0.0
    absolute_error: float = 0.0
    nasa_score: float = 0.0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        prediction_array = predictions.detach().float().cpu().numpy().astype(np.float64)
        target_array = targets.detach().float().cpu().numpy().astype(np.float64)
        difference = (prediction_array - target_array) * self.target_scale
        self.count += difference.size
        self.squared_error += float(np.square(difference).sum())
        self.absolute_error += float(np.abs(difference).sum())
        clipped = np.clip(difference, -1_000.0, 1_000.0)
        contributions = np.where(
            clipped < 0.0,
            np.exp(-clipped / 13.0) - 1.0,
            np.exp(clipped / 10.0) - 1.0,
        )
        self.nasa_score += float(contributions.sum())

    def compute(self) -> dict[str, float | int]:
        if self.count == 0:
            raise ValueError("No predictions were accumulated")
        mse = self.squared_error / self.count
        return {
            "mse": mse,
            "rmse": math.sqrt(mse),
            "mae": self.absolute_error / self.count,
            "nasa_score": self.nasa_score,
            "count": self.count,
        }
