"""Typed YAML configuration for training and evaluating RUL experiments."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

from data.base import RulStageFilter

_MODEL_TYPES = {"ttt", "transformer"}
_MODEL_ARCHITECTURES = {"encoder", "decoder"}
_PRECISIONS = {"auto", "fp32", "bf16"}
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
    name: str = "cmapss"
    window_size: int = 30
    stride: int = 1
    evaluation_stride: int = 1
    rul_cap: float | None = 125.0
    val_fraction: float = 0.2
    variance_threshold: float = 1e-12
    split_seed: int = 42
    train_rul_filter: RulStageFilter | dict[str, Any] | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data_dir = _path(self.data_dir, "data.data_dir")  # type: ignore[assignment]
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("data.name must be a non-empty string.")
        self.name = self.name.strip().lower()
        if not isinstance(self.subset, str) or not self.subset.strip():
            raise ValueError("data.subset must be a non-empty string.")
        self.subset = self.subset.strip().upper()
        self.window_size = _integer(self.window_size, "data.window_size", minimum=1)
        self.stride = _integer(self.stride, "data.stride", minimum=1)
        self.evaluation_stride = _integer(
            self.evaluation_stride, "data.evaluation_stride", minimum=1
        )
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
        if self.train_rul_filter is None:
            self.train_rul_filter = RulStageFilter()
        elif isinstance(self.train_rul_filter, dict):
            self.train_rul_filter = _build(
                RulStageFilter, self.train_rul_filter, "data.train_rul_filter"
            )
        elif not isinstance(self.train_rul_filter, RulStageFilter):
            raise ValueError("data.train_rul_filter must be a mapping.")
        if not isinstance(self.options, dict):
            raise ValueError("data.options must be a mapping.")

    def checkpoint_values(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "data_dir": str(Path(self.data_dir).resolve()),
            "subset": self.subset,
            "window_size": self.window_size,
            "stride": self.stride,
            "evaluation_stride": self.evaluation_stride,
            "val_fraction": self.val_fraction,
            "seed": self.split_seed,
            "rul_cap": self.rul_cap,
            "variance_threshold": self.variance_threshold,
            "train_rul_filter": self.train_rul_filter.to_dict(),
            "options": dict(self.options),
        }


@dataclass
class TTTSettings:
    qkv_bias: bool = True
    inner_lr: float = 1.0
    inner_scale: float = 1.0 / 3.0
    cpe_kernel_size: int = 3
    chunk_size: int = 16
    continuous_state: bool = False

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
        self.chunk_size = _integer(
            self.chunk_size, "model.ttt.chunk_size", minimum=1
        )
        self.continuous_state = _boolean(
            self.continuous_state, "model.ttt.continuous_state"
        )


@dataclass
class AutoregressiveSettings:
    """Continuous next-step feature objective used by decoder models."""

    enabled: bool = True
    loss: str = "smooth_l1"
    weight: float = 0.2

    def __post_init__(self) -> None:
        self.enabled = _boolean(self.enabled, "model.autoregressive.enabled")
        self.loss = str(self.loss).strip().lower()
        if self.loss not in {"mse", "smooth_l1"}:
            raise ValueError(
                "model.autoregressive.loss must be 'mse' or 'smooth_l1'."
            )
        self.weight = _finite(self.weight, "model.autoregressive.weight")
        if self.weight <= 0:
            raise ValueError("model.autoregressive.weight must be positive.")


@dataclass
class ModelSettings:
    type: str
    architecture: str = "encoder"
    d_model: int = 64
    num_layers: int = 2
    num_heads: int = 4
    ffn_ratio: float = 4.0
    dropout: float = 0.1
    ttt: TTTSettings | dict[str, Any] | None = None
    autoregressive: AutoregressiveSettings | dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.type = str(self.type).lower()
        if self.type not in _MODEL_TYPES:
            raise ValueError(f"model.type must be one of {sorted(_MODEL_TYPES)}.")
        self.architecture = str(self.architecture).strip().lower()
        if self.architecture not in _MODEL_ARCHITECTURES:
            raise ValueError(
                "model.architecture must be 'encoder' or 'decoder'."
            )
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
        if self.autoregressive is None:
            self.autoregressive = AutoregressiveSettings(
                enabled=self.architecture == "decoder"
            )
        elif isinstance(self.autoregressive, dict):
            self.autoregressive = _build(
                AutoregressiveSettings,
                self.autoregressive,
                "model.autoregressive",
            )
        elif not isinstance(self.autoregressive, AutoregressiveSettings):
            raise ValueError("model.autoregressive must be a mapping.")
        if self.architecture == "encoder" and self.autoregressive.enabled:
            raise ValueError(
                "model.autoregressive.enabled requires model.architecture='decoder'."
            )
        if (
            isinstance(self.ttt, TTTSettings)
            and self.ttt.continuous_state
            and self.architecture != "decoder"
        ):
            raise ValueError(
                "model.ttt.continuous_state requires model.architecture='decoder'."
            )

    def constructor_values(self, input_dim: int) -> dict[str, Any]:
        values: dict[str, Any] = {
            "input_dim": input_dim,
            "architecture": self.architecture,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "ffn_ratio": self.ffn_ratio,
            "dropout": self.dropout,
        }
        assert isinstance(self.autoregressive, AutoregressiveSettings)
        if self.autoregressive.enabled:
            values.update(
                autoregressive=True,
                autoregressive_loss=self.autoregressive.loss,
                autoregressive_weight=self.autoregressive.weight,
            )
        if self.type == "ttt":
            assert isinstance(self.ttt, TTTSettings)
            values.update(
                qkv_bias=self.ttt.qkv_bias,
                inner_lr=self.ttt.inner_lr,
                inner_scale=self.ttt.inner_scale,
                cpe_kernel_size=self.ttt.cpe_kernel_size,
            )
            if self.architecture == "decoder":
                values["chunk_size"] = self.ttt.chunk_size
            if self.ttt.continuous_state:
                values["continuous_state"] = True
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
    precision: str = "fp32"
    compile: bool = False
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
        self.num_workers = _integer(self.num_workers, "training.num_workers", minimum=0)
        self.deterministic = _boolean(self.deterministic, "training.deterministic")
        self.precision = str(self.precision).strip().lower()
        if self.precision not in _PRECISIONS:
            raise ValueError(
                f"training.precision must be one of {sorted(_PRECISIONS)}."
            )
        self.compile = _boolean(self.compile, "training.compile")
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
    batch_size: int | None = None
    max_test_engines: int | None = None
    max_test_batches: int | None = None
    output: str | Path | None = None
    predictions_output: str | Path | None = None
    plot_output: str | Path | None = None
    plots: bool = True

    def __post_init__(self) -> None:
        self.checkpoint = _path(self.checkpoint, "evaluation.checkpoint", optional=True)
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("evaluation.device must be a non-empty string.")
        self.num_workers = _integer(
            self.num_workers, "evaluation.num_workers", minimum=0
        )
        self.batch_size = _optional_positive(self.batch_size, "evaluation.batch_size")
        self.max_test_engines = _optional_positive(
            self.max_test_engines, "evaluation.max_test_engines"
        )
        self.max_test_batches = _optional_positive(
            self.max_test_batches, "evaluation.max_test_batches"
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

    def __post_init__(self) -> None:
        assert isinstance(self.model.autoregressive, AutoregressiveSettings)
        if self.model.autoregressive.enabled and self.data.window_size < 2:
            raise ValueError(
                "data.window_size must be at least 2 for autoregressive training."
            )

    @property
    def evaluation_checkpoint(self) -> Path:
        checkpoint = self.evaluation.checkpoint
        return (
            Path(checkpoint)
            if checkpoint
            else Path(self.experiment.output_dir) / "best.pt"
        )

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values.pop("source", None)

        def native(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {key: native(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
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
                if key == "architecture" and expected == "encoder":
                    continue  # Compatibility with encoder checkpoints from format v1.
                raise ValueError(f"Checkpoint model_config is missing {key!r}.")
            if actual_model_config[key] != expected:
                raise ValueError(
                    f"Configured model.{key} does not match checkpoint: "
                    f"{expected!r} != {actual_model_config[key]!r}."
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
    "AutoregressiveSettings",
    "DataSettings",
    "EvaluationSettings",
    "ExperimentConfig",
    "ExperimentSettings",
    "ModelSettings",
    "TTTSettings",
    "TrainingSettings",
    "load_experiment_config",
]
