from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetBundle:
    """Datasets and fitted preprocessing state for one experiment."""

    train: Dataset
    validation: Dataset
    test: Dataset
    input_dim: int
    feature_names: list[str]
    preprocessing_state: dict[str, Any]
    split_state: dict[str, Any]
    dataset_name: str
    test_protocol: str
