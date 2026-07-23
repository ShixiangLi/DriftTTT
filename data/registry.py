from __future__ import annotations

from typing import Any, Callable

from .base import DatasetBundle
from .cmapss import build_cmapss_bundle
from .ncmapss import build_ncmapss_bundle


BUILDERS: dict[str, Callable[..., DatasetBundle]] = {
    "cmapss": build_cmapss_bundle,
    "ncmapss": build_ncmapss_bundle,
}


def build_dataset_bundle(
    config: dict[str, Any],
    preprocessing_state: dict[str, Any] | None = None,
    split_state: dict[str, Any] | None = None,
) -> DatasetBundle:
    name = str(config["name"]).lower()
    try:
        builder = BUILDERS[name]
    except KeyError as error:
        raise ValueError(f"Unknown dataset adapter: {name}") from error
    return builder(config, preprocessing_state, split_state)
