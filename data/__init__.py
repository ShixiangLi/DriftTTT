"""Dataset adapters for remaining-useful-life prediction."""

from .base import DatasetBundle
from .registry import build_dataset_bundle

__all__ = ["DatasetBundle", "build_dataset_bundle"]
