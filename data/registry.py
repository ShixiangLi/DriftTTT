"""Explicit registry for supported RUL datasets."""

from __future__ import annotations

from .base import DatasetAdapter
from .cmapss import CmapssAdapter
from .ncmapss import NCmapssAdapter


_ADAPTERS: dict[str, DatasetAdapter] = {
    "cmapss": CmapssAdapter(),
    "ncmapss": NCmapssAdapter(),
}


def get_dataset_adapter(name: str) -> DatasetAdapter:
    normalized = str(name).strip().lower()
    try:
        return _ADAPTERS[normalized]
    except KeyError as error:
        raise ValueError(
            f"Unknown dataset {name!r}; expected one of {sorted(_ADAPTERS)}."
        ) from error


def supported_datasets() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))


__all__ = ["get_dataset_adapter", "supported_datasets"]
