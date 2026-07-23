from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .base import DatasetBundle
from .preprocessing import FeatureScaler, RunningMoments


SETTING_NAMES = [f"setting_{index}" for index in range(1, 4)]
SENSOR_NAMES = [f"sensor_{index}" for index in range(1, 22)]
COLUMNS = ["engine_id", "cycle", *SETTING_NAMES, *SENSOR_NAMES]


def _read_trajectory(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing C-MAPSS file: {path}")
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    if frame.shape[1] != len(COLUMNS) or frame.isna().any().any():
        raise ValueError(f"Malformed C-MAPSS trajectory file: {path}")
    return frame


def _split_entities(
    entity_ids: Iterable[int], validation_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    entities = np.asarray(sorted(entity_ids), dtype=np.int64)
    if entities.size < 2:
        raise ValueError("At least two training engines are required")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(entities)
    validation_count = max(1, int(round(entities.size * validation_fraction)))
    validation_count = min(validation_count, entities.size - 1)
    return (
        sorted(shuffled[validation_count:].tolist()),
        sorted(shuffled[:validation_count].tolist()),
    )


class CmapssWindowDataset(Dataset):
    def __init__(
        self,
        features: dict[int, np.ndarray],
        targets: dict[int, np.ndarray],
        cycle_ids: dict[int, np.ndarray],
        window_size: int,
        stride: int,
        endpoint_only: bool,
    ) -> None:
        self.features = features
        self.targets = targets
        self.cycle_ids = cycle_ids
        self.window_size = window_size
        self.samples: list[tuple[int, int]] = []
        for entity_id in sorted(features):
            length = features[entity_id].shape[0]
            if endpoint_only:
                endpoints = [length - 1]
            else:
                first = min(window_size - 1, length - 1)
                endpoints = list(range(first, length, stride))
                if endpoints[-1] != length - 1:
                    endpoints.append(length - 1)
            self.samples.extend((entity_id, endpoint) for endpoint in endpoints)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        entity_id, endpoint = self.samples[index]
        values = self.features[entity_id]
        start = max(0, endpoint - self.window_size + 1)
        window = values[start : endpoint + 1]
        valid_length = window.shape[0]
        padded = np.zeros((self.window_size, values.shape[1]), dtype=np.float32)
        padded[-valid_length:] = window
        padded_cycles = np.full(self.window_size, -1, dtype=np.int64)
        padded_cycles[-valid_length:] = self.cycle_ids[entity_id][start : endpoint + 1]
        mask = np.zeros(self.window_size, dtype=np.bool_)
        mask[-valid_length:] = True
        return {
            "features": torch.from_numpy(padded),
            "mask": torch.from_numpy(mask),
            "cycle_ids": torch.from_numpy(padded_cycles),
            "target": torch.tensor(
                self.targets[entity_id][endpoint], dtype=torch.float32
            ),
            "entity_id": torch.tensor(entity_id, dtype=torch.int64),
            "endpoint": torch.tensor(endpoint, dtype=torch.int64),
        }


def _group_arrays(
    frame: pd.DataFrame,
    entity_ids: Iterable[int],
    feature_names: list[str],
    scaler: FeatureScaler,
    rul_cap: float | None,
    test_rul: dict[int, float] | None = None,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    feature_groups: dict[int, np.ndarray] = {}
    target_groups: dict[int, np.ndarray] = {}
    cycle_groups: dict[int, np.ndarray] = {}
    by_entity = frame.groupby("engine_id", sort=True)
    for entity_id in entity_ids:
        entity = by_entity.get_group(entity_id).sort_values("cycle")
        cycles = entity["cycle"].to_numpy(dtype=np.float32)
        if test_rul is None:
            targets = cycles.max() - cycles
        else:
            targets = cycles.max() - cycles + test_rul[entity_id]
        if rul_cap is not None:
            targets = np.minimum(targets, rul_cap) / np.float32(rul_cap)
        raw = entity[feature_names].to_numpy(dtype=np.float32)
        feature_groups[int(entity_id)] = scaler.transform(raw).astype(np.float32)
        target_groups[int(entity_id)] = targets.astype(np.float32)
        cycle_groups[int(entity_id)] = entity["cycle"].to_numpy(dtype=np.int64)
    return feature_groups, target_groups, cycle_groups


def build_cmapss_bundle(
    config: dict[str, Any],
    preprocessing_state: dict[str, Any] | None = None,
    split_state: dict[str, Any] | None = None,
) -> DatasetBundle:
    root = Path(config["root"])
    subset = str(config["subset"]).upper()
    if subset not in {"FD001", "FD002", "FD003", "FD004"}:
        raise ValueError(f"Unknown C-MAPSS subset: {subset}")
    train_frame = _read_trajectory(root / f"train_{subset}.txt")
    test_frame = _read_trajectory(root / f"test_{subset}.txt")
    rul_path = root / f"RUL_{subset}.txt"
    rul_values = pd.read_csv(rul_path, sep=r"\s+", header=None).iloc[:, 0].to_numpy()
    test_ids = sorted(test_frame["engine_id"].astype(int).unique().tolist())
    if len(rul_values) != len(test_ids):
        raise ValueError("C-MAPSS test engines and RUL labels have different lengths")
    test_rul = {
        entity_id: float(value) for entity_id, value in zip(test_ids, rul_values)
    }

    if split_state is None:
        train_ids, validation_ids = _split_entities(
            train_frame["engine_id"].astype(int).unique(),
            config["validation_fraction"],
            config["split_seed"],
        )
    else:
        train_ids = [int(value) for value in split_state["train_entities"]]
        validation_ids = [int(value) for value in split_state["validation_entities"]]
    include_settings = bool(config["options"].get("include_settings", True))
    include_cycle = bool(config["options"].get("include_cycle", False))
    source_names = (
        [*SETTING_NAMES, *SENSOR_NAMES] if include_settings else [*SENSOR_NAMES]
    )
    if include_cycle:
        source_names.insert(0, "cycle")
    if preprocessing_state is None:
        fit_rows = train_frame[train_frame["engine_id"].isin(train_ids)][source_names]
        moments = RunningMoments(len(source_names))
        moments.update(fit_rows.to_numpy(dtype=np.float32))
        scaler = FeatureScaler.from_moments(
            source_names, moments, config["variance_threshold"]
        )
    else:
        scaler = FeatureScaler.from_state_dict(preprocessing_state)
        if list(scaler.source_names) != source_names:
            raise ValueError("Checkpoint feature schema does not match C-MAPSS config")
    rul_cap = config["rul_cap"]

    train_features, train_targets, train_cycles = _group_arrays(
        train_frame, train_ids, source_names, scaler, rul_cap
    )
    validation_features, validation_targets, validation_cycles = _group_arrays(
        train_frame, validation_ids, source_names, scaler, rul_cap
    )
    test_features, test_targets, test_cycles = _group_arrays(
        test_frame, test_ids, source_names, scaler, rul_cap, test_rul
    )
    return DatasetBundle(
        train=CmapssWindowDataset(
            train_features,
            train_targets,
            train_cycles,
            config["window_size"],
            config["stride"],
            endpoint_only=False,
        ),
        validation=CmapssWindowDataset(
            validation_features,
            validation_targets,
            validation_cycles,
            config["window_size"],
            config["stride"],
            endpoint_only=False,
        ),
        test=CmapssWindowDataset(
            test_features,
            test_targets,
            test_cycles,
            config["window_size"],
            config["evaluation_stride"],
            endpoint_only=True,
        ),
        input_dim=len(scaler.feature_names),
        feature_names=scaler.feature_names,
        preprocessing_state=scaler.state_dict(),
        split_state={
            "train_entities": train_ids,
            "validation_entities": validation_ids,
        },
        dataset_name="cmapss",
        test_protocol="endpoint",
    )
