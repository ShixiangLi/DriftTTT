from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULTS: dict[str, Any] = {
    "experiment": {"name": None, "output_dir": None},
    "data": {
        "name": None,
        "root": None,
        "subset": None,
        "window_size": 64,
        "stride": 1,
        "evaluation_stride": 1,
        "validation_fraction": 0.2,
        "split_seed": 42,
        "rul_cap": 125.0,
        "variance_threshold": 1.0e-12,
        "batch_size": 128,
        "num_workers": 0,
        "pin_memory": True,
        "options": {},
    },
    "model": {
        "type": "transformer",
        "sequence_mixer": "attention",
        "d_model": 128,
        "nhead": 4,
        "num_layers": 3,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "activation": "gelu",
        "norm_first": True,
        "ttt": {
            "hidden_multiplier": 2.0,
            "inner_learning_rate": 0.1,
            "chunk_size": 16,
            "inner_gradient_clip": 1.0,
            "activation": "silu",
            "qkv_bias": True,
            "multiscale": {
                "short_rank_ratio": 0.5,
                "long_ema_decay": 0.9,
                "long_update_interval": 4,
                "long_inner_learning_rate": 0.025,
                "center_long_residual": True,
            },
        },
    },
    "training": {
        "epochs": 30,
        "learning_rate": 1.0e-3,
        "weight_decay": 1.0e-4,
        "device": "auto",
        "precision": "auto",
        "gradient_clip": 1.0,
        "seed": 42,
        "deterministic": False,
        "early_stopping_patience": 8,
        "resume": None,
        "max_train_batches": None,
        "max_validation_batches": None,
        "plots": True,
    },
    "evaluation": {
        "checkpoint": None,
        "device": "auto",
        "max_test_batches": None,
        "metrics_file": "test_metrics.json",
        "predictions_file": None,
        "plots": True,
    },
}

ALLOWED_OPTIONS = {
    "include_settings",
    "include_cycle",
    "include_partial_windows",
    "feature_groups",
    "downsample",
    "chunk_rows",
}


def _merge_and_validate(
    supplied: dict[str, Any], defaults: dict[str, Any], location: str = "config"
) -> dict[str, Any]:
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown keys under {location}: {sorted(unknown)}")
    result = deepcopy(defaults)
    for key, value in supplied.items():
        if isinstance(defaults[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{location}.{key} must be a mapping")
            if key == "options":
                invalid = set(value) - ALLOWED_OPTIONS
                if invalid:
                    raise ValueError(
                        f"Unknown keys under {location}.{key}: {sorted(invalid)}"
                    )
                result[key] = deepcopy(value)
            else:
                result[key] = _merge_and_validate(
                    value, defaults[key], f"{location}.{key}"
                )
        else:
            result[key] = value
    return result


def normalize_config(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")
    config = _merge_and_validate(raw, DEFAULTS)
    required = {
        "experiment.name": config["experiment"]["name"],
        "experiment.output_dir": config["experiment"]["output_dir"],
        "data.name": config["data"]["name"],
        "data.root": config["data"]["root"],
        "data.subset": config["data"]["subset"],
    }
    missing = [key for key, value in required.items() if value in (None, "")]
    if missing:
        raise ValueError(f"Missing required configuration values: {missing}")

    data = config["data"]
    data["name"] = str(data["name"]).lower()
    if data["name"] not in {"cmapss", "ncmapss"}:
        raise ValueError("data.name must be 'cmapss' or 'ncmapss'")
    dataset_options = {
        "cmapss": {"include_settings", "include_cycle"},
        "ncmapss": {
            "feature_groups",
            "downsample",
            "chunk_rows",
            "include_cycle",
            "include_partial_windows",
        },
    }
    invalid_options = set(data["options"]) - dataset_options[data["name"]]
    if invalid_options:
        raise ValueError(
            f"Invalid options for {data['name']}: {sorted(invalid_options)}"
        )
    for key in ("window_size", "stride", "evaluation_stride", "batch_size"):
        if not isinstance(data[key], int) or data[key] < 1:
            raise ValueError(f"data.{key} must be a positive integer")
    if not isinstance(data["num_workers"], int) or data["num_workers"] < 0:
        raise ValueError("data.num_workers must be a non-negative integer")
    include_partial = data["options"].get("include_partial_windows", False)
    if not isinstance(include_partial, bool):
        raise ValueError("data.options.include_partial_windows must be a boolean")
    if not 0.0 < float(data["validation_fraction"]) < 1.0:
        raise ValueError("data.validation_fraction must lie strictly between 0 and 1")
    if float(data["variance_threshold"]) < 0.0:
        raise ValueError("data.variance_threshold must be non-negative")
    if data["rul_cap"] is not None:
        cap = float(data["rul_cap"])
        data["rul_cap"] = None if cap <= 0 else cap

    model = config["model"]
    if model["type"] != "transformer":
        raise ValueError(
            "The current implementation supports only model.type=transformer"
        )
    valid_mixers = {"attention", "ttt_mlp", "ttt_multiscale_moe"}
    if model["sequence_mixer"] not in valid_mixers:
        raise ValueError(
            "model.sequence_mixer must be attention, ttt_mlp, or ttt_multiscale_moe"
        )
    for key in ("d_model", "nhead", "num_layers", "dim_feedforward"):
        if not isinstance(model[key], int) or model[key] < 1:
            raise ValueError(f"model.{key} must be a positive integer")
    if model["d_model"] % model["nhead"] != 0:
        raise ValueError("model.d_model must be divisible by model.nhead")
    if not 0.0 <= float(model["dropout"]) < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")
    if model["activation"] not in {"relu", "gelu"}:
        raise ValueError("model.activation must be relu or gelu")
    ttt = model["ttt"]
    if float(ttt["hidden_multiplier"]) <= 0.0:
        raise ValueError("model.ttt.hidden_multiplier must be positive")
    if float(ttt["inner_learning_rate"]) <= 0.0:
        raise ValueError("model.ttt.inner_learning_rate must be positive")
    if not isinstance(ttt["chunk_size"], int) or ttt["chunk_size"] < 1:
        raise ValueError("model.ttt.chunk_size must be a positive integer")
    inner_clip = ttt["inner_gradient_clip"]
    if inner_clip is not None and float(inner_clip) <= 0.0:
        raise ValueError("model.ttt.inner_gradient_clip must be null or positive")
    if ttt["activation"] not in {"silu", "gelu"}:
        raise ValueError("model.ttt.activation must be silu or gelu")
    multiscale = ttt["multiscale"]
    if not 0.0 < float(multiscale["short_rank_ratio"]) < 1.0:
        raise ValueError("model.ttt.multiscale.short_rank_ratio must be in (0, 1)")
    if not 0.0 <= float(multiscale["long_ema_decay"]) < 1.0:
        raise ValueError("model.ttt.multiscale.long_ema_decay must be in [0, 1)")
    if (
        not isinstance(multiscale["long_update_interval"], int)
        or multiscale["long_update_interval"] < 1
    ):
        raise ValueError(
            "model.ttt.multiscale.long_update_interval must be a positive integer"
        )
    if float(multiscale["long_inner_learning_rate"]) <= 0.0:
        raise ValueError(
            "model.ttt.multiscale.long_inner_learning_rate must be positive"
        )
    if not isinstance(multiscale["center_long_residual"], bool):
        raise ValueError("model.ttt.multiscale.center_long_residual must be a boolean")
    if model["sequence_mixer"] == "ttt_multiscale_moe":
        head_dim = model["d_model"] // model["nhead"]
        hidden_dim = max(1, round(head_dim * float(ttt["hidden_multiplier"])))
        if hidden_dim < 2:
            raise ValueError(
                "ttt_multiscale_moe requires at least two hidden channels per head"
            )

    training = config["training"]
    for key in ("epochs", "early_stopping_patience"):
        if not isinstance(training[key], int) or training[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    if float(training["learning_rate"]) <= 0.0:
        raise ValueError("training.learning_rate must be positive")
    if float(training["weight_decay"]) < 0.0:
        raise ValueError("training.weight_decay must be non-negative")
    for section_name in ("training", "evaluation"):
        device = config[section_name]["device"]
        if device not in {"auto", "cpu", "cuda"} and not str(device).startswith(
            "cuda:"
        ):
            raise ValueError(f"{section_name}.device is invalid: {device}")
    if training["precision"] not in {"auto", "fp32", "bf16"}:
        raise ValueError("training.precision must be auto, fp32, or bf16")
    for section_name, key in (
        ("training", "max_train_batches"),
        ("training", "max_validation_batches"),
        ("evaluation", "max_test_batches"),
    ):
        value = config[section_name][key]
        if value is not None and (not isinstance(value, int) or value < 1):
            raise ValueError(f"{section_name}.{key} must be null or positive")
    predictions = config["evaluation"]["predictions_file"]
    if predictions is None:
        config["evaluation"]["predictions_file"] = (
            "test_predictions.jsonl"
            if data["name"] == "ncmapss"
            else "test_predictions.json"
        )
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return normalize_config(raw)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
