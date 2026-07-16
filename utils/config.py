"""Typed YAML configuration for training and evaluating RUL experiments."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml


_SUBSETS = {"FD001", "FD002", "FD003", "FD004"}
_MODEL_TYPES = {"ttt", "transformer"}
_T = TypeVar("_T")


def _finite(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite.")
    return result


def _integer(value: Any, path: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{path} must be >= {minimum}.")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be true or false.")
    return value


def _path(value: str | Path | None, path: str, optional: bool = False) -> Path | None:
    if value is None and optional:
        return None
    if not isinstance(value, (str, Path)) or not str(value).strip():
        suffix = " or null" if optional else ""
        raise ValueError(f"{path} must be a non-empty path{suffix}.")
    return Path(value)


def _optional_positive(value: Any, path: str) -> int | None:
    return None if value is None else _integer(value, path, minimum=1)


def _build(cls: type[_T], value: Any, path: str) -> _T:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping.")
    try:
        return cls(**value)
    except TypeError as error:
        raise ValueError(f"Invalid fields in {path}: {error}") from error


@dataclass
class ExperimentSettings:
    name: str
    output_dir: str | Path

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("experiment.name must be a non-empty string.")
        self.name = self.name.strip()
        self.output_dir = _path(self.output_dir, "experiment.output_dir")  # type: ignore[assignment]


@dataclass
class DataSettings:
    data_dir: str | Path
    subset: str
    window_size: int = 30
    stride: int = 1
    rul_cap: float | None = 125.0
    val_fraction: float = 0.2
    variance_threshold: float = 1e-12
    split_seed: int = 42

    def __post_init__(self) -> None:
        self.data_dir = _path(self.data_dir, "data.data_dir")  # type: ignore[assignment]
        self.subset = str(self.subset).upper()
        if self.subset not in _SUBSETS:
            raise ValueError(f"data.subset must be one of {sorted(_SUBSETS)}.")
        self.window_size = _integer(self.window_size, "data.window_size", minimum=1)
        self.stride = _integer(self.stride, "data.stride", minimum=1)
        if self.rul_cap is not None:
            self.rul_cap = _finite(self.rul_cap, "data.rul_cap")
            if self.rul_cap <= 0:
                self.rul_cap = None
        self.val_fraction = _finite(self.val_fraction, "data.val_fraction")
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError("data.val_fraction must be in (0, 1).")
        self.variance_threshold = _finite(
            self.variance_threshold, "data.variance_threshold"
        )
        if self.variance_threshold < 0:
            raise ValueError("data.variance_threshold cannot be negative.")
        self.split_seed = _integer(self.split_seed, "data.split_seed")

    def checkpoint_values(self) -> dict[str, Any]:
        return {
            "data_dir": str(Path(self.data_dir).resolve()),
            "subset": self.subset,
            "window_size": self.window_size,
            "stride": self.stride,
            "val_fraction": self.val_fraction,
            "seed": self.split_seed,
            "rul_cap": self.rul_cap,
            "variance_threshold": self.variance_threshold,
        }


@dataclass
class TTTSettings:
    qkv_bias: bool = True
    inner_lr: float = 1.0
    inner_scale: float = 1.0 / 3.0
    cpe_kernel_size: int = 3

    def __post_init__(self) -> None:
        self.qkv_bias = _boolean(self.qkv_bias, "model.ttt.qkv_bias")
        self.inner_lr = _finite(self.inner_lr, "model.ttt.inner_lr")
        self.inner_scale = _finite(self.inner_scale, "model.ttt.inner_scale")
        if self.inner_lr < 0 or self.inner_scale <= 0:
            raise ValueError("model.ttt inner_lr/inner_scale values are invalid.")
        self.cpe_kernel_size = _integer(
            self.cpe_kernel_size, "model.ttt.cpe_kernel_size", minimum=1
        )
        if self.cpe_kernel_size % 2 == 0:
            raise ValueError("model.ttt.cpe_kernel_size must be odd.")


@dataclass
class ModelSettings:
    type: str
    d_model: int = 64
    num_layers: int = 2
    num_heads: int = 4
    ffn_ratio: float = 4.0
    dropout: float = 0.1
    ttt: TTTSettings | dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.type = str(self.type).lower()
        if self.type not in _MODEL_TYPES:
            raise ValueError(f"model.type must be one of {sorted(_MODEL_TYPES)}.")
        self.d_model = _integer(self.d_model, "model.d_model", minimum=1)
        self.num_layers = _integer(self.num_layers, "model.num_layers", minimum=1)
        self.num_heads = _integer(self.num_heads, "model.num_heads", minimum=1)
        if self.d_model % self.num_heads != 0:
            raise ValueError("model.d_model must be divisible by model.num_heads.")
        self.ffn_ratio = _finite(self.ffn_ratio, "model.ffn_ratio")
        self.dropout = _finite(self.dropout, "model.dropout")
        if self.ffn_ratio <= 0 or not 0.0 <= self.dropout < 1.0:
            raise ValueError("model ffn_ratio/dropout values are invalid.")
        if self.type == "ttt":
            if self.ttt is None:
                raise ValueError("model.ttt is required when model.type is 'ttt'.")
            if isinstance(self.ttt, dict):
                self.ttt = _build(TTTSettings, self.ttt, "model.ttt")
            elif not isinstance(self.ttt, TTTSettings):
                raise ValueError("model.ttt must be a mapping.")
        elif self.ttt is not None:
            raise ValueError("model.ttt is only valid when model.type is 'ttt'.")

    def constructor_values(self, input_dim: int) -> dict[str, Any]:
        values: dict[str, Any] = {
            "input_dim": input_dim,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "ffn_ratio": self.ffn_ratio,
            "dropout": self.dropout,
        }
        if self.type == "ttt":
            assert isinstance(self.ttt, TTTSettings)
            values.update(
                qkv_bias=self.ttt.qkv_bias,
                inner_lr=self.ttt.inner_lr,
                inner_scale=self.ttt.inner_scale,
                cpe_kernel_size=self.ttt.cpe_kernel_size,
            )
        return values


@dataclass
class TrainingSettings:
    batch_size: int = 64
    epochs: int = 50
    patience: int = 10
    min_delta: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float | None = 1.0
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0
    deterministic: bool = True
    plots: bool = True
    resume: str | Path | None = None
    max_train_batches: int | None = None
    max_val_batches: int | None = None

    def __post_init__(self) -> None:
        self.batch_size = _integer(self.batch_size, "training.batch_size", minimum=1)
        self.epochs = _integer(self.epochs, "training.epochs", minimum=1)
        self.patience = _integer(self.patience, "training.patience", minimum=0)
        self.min_delta = _finite(self.min_delta, "training.min_delta")
        self.learning_rate = _finite(self.learning_rate, "training.learning_rate")
        self.weight_decay = _finite(self.weight_decay, "training.weight_decay")
        if self.min_delta < 0 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("training optimizer/early-stop values are invalid.")
        if self.grad_clip is not None:
            self.grad_clip = _finite(self.grad_clip, "training.grad_clip")
            if self.grad_clip <= 0:
                self.grad_clip = None
        self.seed = _integer(self.seed, "training.seed")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("training.device must be a non-empty string.")
        self.num_workers = _integer(
            self.num_workers, "training.num_workers", minimum=0
        )
        self.deterministic = _boolean(self.deterministic, "training.deterministic")
        self.plots = _boolean(self.plots, "training.plots")
        self.resume = _path(self.resume, "training.resume", optional=True)
        self.max_train_batches = _optional_positive(
            self.max_train_batches, "training.max_train_batches"
        )
        self.max_val_batches = _optional_positive(
            self.max_val_batches, "training.max_val_batches"
        )


@dataclass
class EvaluationSettings:
    checkpoint: str | Path | None = None
    device: str = "auto"
    num_workers: int = 0
    max_test_engines: int | None = None
    output: str | Path | None = None
    predictions_output: str | Path | None = None
    plot_output: str | Path | None = None
    plots: bool = True

    def __post_init__(self) -> None:
        self.checkpoint = _path(
            self.checkpoint, "evaluation.checkpoint", optional=True
        )
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("evaluation.device must be a non-empty string.")
        self.num_workers = _integer(
            self.num_workers, "evaluation.num_workers", minimum=0
        )
        self.max_test_engines = _optional_positive(
            self.max_test_engines, "evaluation.max_test_engines"
        )
        self.output = _path(self.output, "evaluation.output", optional=True)
        self.predictions_output = _path(
            self.predictions_output, "evaluation.predictions_output", optional=True
        )
        self.plot_output = _path(
            self.plot_output, "evaluation.plot_output", optional=True
        )
        self.plots = _boolean(self.plots, "evaluation.plots")


@dataclass
class ExperimentConfig:
    experiment: ExperimentSettings
    data: DataSettings
    model: ModelSettings
    training: TrainingSettings
    evaluation: EvaluationSettings
    source: Path

    @property
    def evaluation_checkpoint(self) -> Path:
        checkpoint = self.evaluation.checkpoint
        return Path(checkpoint) if checkpoint else Path(self.experiment.output_dir) / "best.pt"

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values.pop("source", None)

        def native(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {key: native(item) for key, item in value.items()}
            if isinstance(value, list):
                return [native(item) for item in value]
            return value

        return native(values)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8"
        )

    def verify_checkpoint_identity(self, checkpoint: dict[str, Any]) -> None:
        actual_model = str(checkpoint.get("model_name", "ttt"))
        if actual_model != self.model.type:
            raise ValueError(
                f"Configured model {self.model.type!r} does not match "
                f"checkpoint model {actual_model!r}."
            )
        actual_model_config = checkpoint.get("model_config")
        if not isinstance(actual_model_config, dict):
            raise ValueError("Checkpoint is missing model_config.")
        input_dim = _integer(
            actual_model_config.get("input_dim"), "checkpoint.model_config.input_dim", 1
        )
        expected_model_config = self.model.constructor_values(input_dim)
        for key, expected in expected_model_config.items():
            if key not in actual_model_config:
                if key == "inner_scale" and expected == 1.0 / 3.0:
                    continue  # Compatibility with checkpoints before this option existed.
                raise ValueError(f"Checkpoint model_config is missing {key!r}.")
            if actual_model_config[key] != expected:
                raise ValueError(
                    f"Configured model.{key} does not match checkpoint: "
                    f"{expected!r} != {actual_model_config[key]!r}."
                )
        actual_data = checkpoint.get("data_config")
        if not isinstance(actual_data, dict):
            raise ValueError("Checkpoint is missing data_config.")
        expected_data = self.data.checkpoint_values()
        for key in (
            "subset",
            "window_size",
            "stride",
            "val_fraction",
            "seed",
            "rul_cap",
            "variance_threshold",
        ):
            if actual_data.get(key) != expected_data[key]:
                raise ValueError(
                    f"Configured data.{key} does not match checkpoint: "
                    f"{expected_data[key]!r} != {actual_data.get(key)!r}."
                )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Configuration file not found: {source}")
    root = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(root, dict):
        raise ValueError("Configuration root must be a mapping.")
    allowed = {"experiment", "data", "model", "training", "evaluation"}
    unknown = sorted(set(root) - allowed)
    if unknown:
        raise ValueError(f"Unknown keys in config: {unknown}")
    try:
        return ExperimentConfig(
            experiment=_build(ExperimentSettings, root.get("experiment"), "experiment"),
            data=_build(DataSettings, root.get("data"), "data"),
            model=_build(ModelSettings, root.get("model"), "model"),
            training=_build(TrainingSettings, root.get("training"), "training"),
            evaluation=_build(EvaluationSettings, root.get("evaluation"), "evaluation"),
            source=source,
        )
    except KeyError as error:
        raise ValueError(f"Missing configuration section: {error}") from error


__all__ = [
    "DataSettings",
    "EvaluationSettings",
    "ExperimentConfig",
    "ExperimentSettings",
    "ModelSettings",
    "TTTSettings",
    "TrainingSettings",
    "load_experiment_config",
]
