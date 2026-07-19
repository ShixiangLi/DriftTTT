"""Training, evaluation, and checkpoint utilities for C-MAPSS RUL models."""

from __future__ import annotations

from contextlib import nullcontext
import json
import math
import os
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


MetricFn = Callable[[Sequence[float], Sequence[float]], Mapping[str, float]]
_PRECISIONS = {"auto", "fp32", "bf16"}


def rul_label_policy(rul_cap: float | None) -> dict[str, Any]:
    """Describe the train/test RUL labels represented by reported metrics."""
    return {
        "train": "piecewise_linear_to_failure",
        "test": "official_endpoint_rul",
        "cap_applied_to": "train_and_test_targets" if rul_cap is not None else "none",
    }


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch without assuming CUDA is available."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker from PyTorch's worker-specific initial seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve ``auto`` to CUDA, MPS, or CPU in that order."""
    if isinstance(device, torch.device):
        return device
    value = device.lower()
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    resolved = torch.device(value)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if resolved.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
    return resolved


def resolve_precision(precision: str, device: torch.device) -> str:
    """Resolve runtime precision and reject unsupported BF16 execution."""
    normalized = str(precision).strip().lower()
    if normalized not in _PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(_PRECISIONS)}")
    if normalized == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return "bf16"
        return "fp32"
    if normalized == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Configured CUDA device does not support BF16")
        if device.type not in {"cpu", "cuda"}:
            raise RuntimeError(f"BF16 autocast is not supported on {device.type}")
    return normalized


def compile_model(model: nn.Module, enabled: bool) -> nn.Module:
    """Compile a module in place so checkpoint keys and public methods stay stable."""
    if not enabled:
        return model
    compile_method = getattr(model, "compile", None)
    if not callable(compile_method):
        raise RuntimeError("training.compile requires torch.nn.Module.compile()")
    compile_method()
    return model


def _autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def reset_model_state(model: nn.Module) -> None:
    """Reset TTT fast state, including through a DDP-style ``module`` wrapper."""
    target = getattr(model, "module", model)
    reset = getattr(target, "reset_ttt_state", None)
    if callable(reset):
        reset()
        return

    return


def uses_continuous_state(model: nn.Module) -> bool:
    """Whether the model expects one chronological entity stream at a time."""
    target = getattr(model, "module", model)
    return bool(getattr(target, "continuous_state", False))


def build_model(model_name: str, model_config: Mapping[str, Any]) -> nn.Module:
    """Build a supported model from primitive checkpoint constructor metadata."""
    from models.rul_transformer import StandardRULTransformer, TTTRULTransformer

    normalized = model_name.strip().lower()
    model_types = {
        "ttt": TTTRULTransformer,
        "ttt_transformer": TTTRULTransformer,
        "transformer": StandardRULTransformer,
    }
    try:
        model_type = model_types[normalized]
    except KeyError as error:
        raise ValueError(
            f"Unknown model {model_name!r}; expected one of {sorted(model_types)}"
        ) from error
    return model_type(**dict(model_config))


def verify_data_provenance(bundle: Any, checkpoint: Mapping[str, Any]) -> None:
    """Reject restored evaluation data that does not match checkpoint fitting state."""
    expected_splits = checkpoint.get(
        "split_entity_ids", checkpoint.get("split_engine_ids")
    )
    if not isinstance(expected_splits, Mapping):
        raise ValueError("Checkpoint is missing split entity metadata")
    if hasattr(bundle, "split_entity_ids"):
        actual_splits = {
            name: [int(value) for value in bundle.split_entity_ids[name]]
            for name in ("train", "val", "test")
        }
    else:
        actual_splits = {
            "train": [int(value) for value in bundle.train_engine_ids],
            "val": [int(value) for value in bundle.val_engine_ids],
            "test": [int(value) for value in bundle.test_dataset.engine_ids],
        }
    normalized_expected = {
        name: [int(value) for value in expected_splits.get(name, [])]
        for name in actual_splits
    }
    if normalized_expected != actual_splits:
        raise RuntimeError(
            "Prepared entity splits differ from the checkpoint; refusing a "
            f"potentially leaked evaluation (expected={normalized_expected}, "
            f"actual={actual_splits})"
        )

    fitted_values = getattr(
        bundle.preprocessor,
        "fit_entity_ids",
        getattr(bundle.preprocessor, "fit_engine_ids", ()),
    )
    fitted_ids = [int(value) for value in fitted_values]
    if fitted_ids != actual_splits["train"]:
        raise RuntimeError(
            "Preprocessor fitting IDs do not match the training entity split"
        )

    expected_preprocessor = checkpoint.get("preprocessor")
    if not isinstance(expected_preprocessor, Mapping):
        raise ValueError("Checkpoint is missing preprocessor metadata")
    actual_preprocessor = _native_value(
        bundle.preprocessor.state_dict(), "preprocessor"
    )
    if _native_value(expected_preprocessor, "preprocessor") != actual_preprocessor:
        raise RuntimeError(
            "Prepared feature selection/scaling state differs from the checkpoint"
        )


def verify_test_provenance(
    test_dataset: Any,
    preprocessor: Any,
    checkpoint: Mapping[str, Any],
) -> None:
    """Validate restored fitting IDs and actual official test engine IDs."""
    expected_splits = checkpoint.get(
        "split_entity_ids", checkpoint.get("split_engine_ids")
    )
    if not isinstance(expected_splits, Mapping):
        raise ValueError("Checkpoint is missing split entity metadata")
    train_ids = [int(value) for value in expected_splits.get("train", [])]
    val_ids = [int(value) for value in expected_splits.get("val", [])]
    expected_test_ids = [int(value) for value in expected_splits.get("test", [])]
    if not train_ids or not val_ids or not expected_test_ids:
        raise ValueError(
            "Checkpoint contains an empty train, val, or test entity split"
        )
    if set(train_ids).intersection(val_ids):
        raise RuntimeError("Checkpoint train and validation entity IDs overlap")
    fitted_values = getattr(
        preprocessor, "fit_entity_ids", getattr(preprocessor, "fit_engine_ids", ())
    )
    fitted_ids = [int(value) for value in fitted_values]
    if fitted_ids != train_ids:
        raise RuntimeError(
            "Restored preprocessor fitting IDs do not match saved training IDs"
        )
    actual_values = getattr(test_dataset, "entity_ids", test_dataset.engine_ids)
    actual_test_ids = [int(value) for value in actual_values]
    if actual_test_ids != expected_test_ids:
        raise RuntimeError(
            "Official test entity IDs differ from the checkpoint: "
            f"expected={expected_test_ids}, actual={actual_test_ids}"
        )


def _native_value(value: Any, path: str = "metadata") -> Any:
    """Convert checkpoint metadata to JSON-compatible, weights-only-safe values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _native_value(value.item(), path)
    if isinstance(value, np.ndarray):
        return _native_value(value.tolist(), path)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, (str, int)):
                raise TypeError(f"{path} has unsupported key type {type(key).__name__}")
            key_string = str(key)
            result[key_string] = _native_value(item, f"{path}.{key_string}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _native_value(item, f"{path}[{index}]") for index, item in enumerate(value)
        ]
    raise TypeError(f"{path} has unsupported value type {type(value).__name__}")


def save_json(path: str | Path, value: Mapping[str, Any] | Sequence[Any]) -> None:
    """Atomically write strict JSON containing only native metadata values."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    native = _native_value(value)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(native, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


class JsonlPredictionWriter:
    """Atomically stream prediction records without retaining them in memory."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.temporary = self.path.with_name(f"{self.path.name}.tmp")
        self._handle: Any = None
        self.count = 0

    def __enter__(self) -> "JsonlPredictionWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.temporary.open("w", encoding="utf-8", newline="\n")
        return self

    def write(self, records: Sequence[Mapping[str, Any]]) -> None:
        if self._handle is None:
            raise RuntimeError(
                "JsonlPredictionWriter must be used as a context manager."
            )
        for record in records:
            json.dump(
                _native_value(record, "prediction"),
                self._handle,
                sort_keys=True,
                allow_nan=False,
            )
            self._handle.write("\n")
            self.count += 1

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc, traceback
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        if exc_type is None:
            os.replace(self.temporary, self.path)
        elif self.temporary.exists():
            self.temporary.unlink()


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    *,
    epoch: int,
    best_val_loss: float,
    metadata: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]] = (),
    epochs_without_improvement: int = 0,
) -> None:
    """Atomically save tensor state dictionaries and primitive metadata only."""
    reserved = {
        "format_version",
        "model_state_dict",
        "optimizer_state_dict",
        "epoch",
        "best_val_loss",
        "history",
        "epochs_without_improvement",
    }
    collisions = reserved.intersection(metadata)
    if collisions:
        names = ", ".join(sorted(collisions))
        raise ValueError(f"Checkpoint metadata uses reserved keys: {names}")

    native_metadata = _native_value(metadata)
    payload: dict[str, Any] = {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
        if optimizer is not None
        else None,
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "history": _native_value(history, "history"),
        "epochs_without_improvement": int(epochs_without_improvement),
        **native_metadata,
    }

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint through PyTorch's restricted weights-only unpickler."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError as error:
        raise RuntimeError(
            "Safe checkpoint loading requires a PyTorch version that supports "
            "torch.load(..., weights_only=True)"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint root must be a dictionary")
    if payload.get("format_version") != 1:
        raise ValueError(
            f"Unsupported checkpoint format: {payload.get('format_version')!r}"
        )
    if not isinstance(payload.get("model_state_dict"), Mapping):
        raise ValueError("Checkpoint is missing model_state_dict")
    return payload


def restore_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load model and optional optimizer state, returning the full safe payload."""
    payload = load_checkpoint(path, device=device)
    model.load_state_dict(payload["model_state_dict"], strict=strict)
    optimizer_state = payload.get("optimizer_state_dict")
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    return payload


def _unpack_batch(
    batch: Mapping[str, Any] | Sequence[Any],
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    dict[str, torch.Tensor],
]:
    metadata: dict[str, Any] = {}
    if isinstance(batch, Mapping):
        try:
            features = batch["features"]
            target = batch["target"]
        except KeyError as error:
            raise KeyError(
                "Batch dictionaries require 'features' and 'target'"
            ) from error
        padding_mask = batch.get("padding_mask")
        entity_id = batch.get("entity_id", batch.get("engine_id"))
        time_index = batch.get("time_index", batch.get("cycle"))
        metadata = {
            key: batch[key]
            for key in (
                "engine_id",
                "unit_id",
                "cycle",
                "sample_index",
                "state_new_tokens",
            )
            if key in batch
        }
    elif isinstance(batch, Sequence) and len(batch) >= 2:
        features, target = batch[0], batch[1]
        padding_mask = batch[2] if len(batch) >= 3 else None
        entity_id = batch[3] if len(batch) >= 4 else None
        time_index = batch[4] if len(batch) >= 5 else None
    else:
        raise TypeError(
            "A batch must be a mapping or a sequence with at least two values"
        )

    non_blocking = device.type == "cuda"
    features_tensor = torch.as_tensor(features).to(
        device=device, dtype=torch.float32, non_blocking=non_blocking
    )
    target_tensor = (
        torch.as_tensor(target)
        .to(device=device, dtype=torch.float32, non_blocking=non_blocking)
        .reshape(-1)
    )
    mask_tensor = None
    if padding_mask is not None:
        mask_tensor = torch.as_tensor(padding_mask).to(
            device=device, dtype=torch.bool, non_blocking=non_blocking
        )
    entity_tensor = (
        torch.as_tensor(entity_id).reshape(-1) if entity_id is not None else None
    )
    time_tensor = (
        torch.as_tensor(time_index).reshape(-1) if time_index is not None else None
    )
    metadata_tensors = {
        key: torch.as_tensor(value).reshape(-1) for key, value in metadata.items()
    }
    return (
        features_tensor,
        mask_tensor,
        target_tensor,
        entity_tensor,
        time_tensor,
        metadata_tensors,
    )


def _forward_model(
    model: nn.Module,
    features: torch.Tensor,
    padding_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    if padding_mask is None:
        prediction = model(features)
    else:
        prediction = model(features, padding_mask=padding_mask)
    output: Mapping[str, Any] = {}
    if isinstance(prediction, Mapping):
        output = prediction
        if "prediction" not in output:
            raise KeyError("Model output dictionaries require a 'prediction' value")
        prediction = output["prediction"]
    if not isinstance(prediction, torch.Tensor):
        raise TypeError("Model output must be a tensor")
    return prediction.reshape(-1), output


def _autoregressive_terms(
    model: nn.Module,
    output: Mapping[str, Any],
    features: torch.Tensor,
    padding_mask: torch.Tensor | None,
) -> dict[str, Any] | None:
    predicted = output.get("next_features")
    if predicted is None:
        return None
    if not isinstance(predicted, torch.Tensor):
        raise TypeError("next_features must be a tensor")
    if predicted.shape != features.shape:
        raise ValueError(
            "next_features must match input shape [batch, length, features]; "
            f"got {tuple(predicted.shape)} and {tuple(features.shape)}"
        )
    if features.shape[1] < 2:
        return None

    transition_mask = torch.ones(
        features.shape[:2], dtype=torch.bool, device=features.device
    )
    transition_mask = transition_mask[:, 1:]
    if padding_mask is not None:
        transition_mask = (~padding_mask[:, :-1]) & (~padding_mask[:, 1:])
    element_mask = transition_mask.unsqueeze(-1).expand(
        -1, -1, features.shape[-1]
    )
    count = int(element_mask.sum().item())
    if count == 0:
        return None

    error = predicted[:, :-1] - features[:, 1:]
    mask_values = element_mask.to(dtype=error.dtype)
    target_model = getattr(model, "module", model)
    loss_name = str(getattr(target_model, "autoregressive_loss", "smooth_l1"))
    weight = float(getattr(target_model, "autoregressive_weight", 0.0))
    if weight <= 0:
        raise ValueError("Autoregressive model weight must be positive")
    if loss_name == "mse":
        element_loss = error.square()
    elif loss_name == "smooth_l1":
        element_loss = F.smooth_l1_loss(
            predicted[:, :-1], features[:, 1:], reduction="none"
        )
    else:
        raise ValueError(f"Unsupported autoregressive loss: {loss_name!r}")
    loss_sum = (element_loss * mask_values).sum()
    return {
        "loss": loss_sum / count,
        "loss_sum": loss_sum.detach(),
        "squared_error": (error.square() * mask_values).sum().detach(),
        "absolute_error": (error.abs() * mask_values).sum().detach(),
        "count": count,
        "weight": weight,
        "name": loss_name,
    }


class _EntityStreamTracker:
    """Validate entity-contiguous chronological batches and reset boundaries."""

    def __init__(self) -> None:
        self.active_entity: int | None = None
        self.last_time: int | None = None
        self.completed_entities: set[int] = set()

    def advance(self, model: nn.Module, entity_id: int, time_index: int) -> None:
        if entity_id != self.active_entity:
            if entity_id in self.completed_entities:
                raise ValueError(
                    "continuous TTT state requires entity-contiguous sampling; "
                    f"entity {entity_id} appeared again after its stream ended"
                )
            if self.active_entity is not None:
                self.completed_entities.add(self.active_entity)
            reset_model_state(model)
            self.active_entity = entity_id
            self.last_time = None
        if self.last_time is not None and time_index <= self.last_time:
            raise ValueError(
                "continuous TTT state requires strictly increasing time_index "
                f"within entity {entity_id}; got {time_index} after {self.last_time}"
            )
        self.last_time = time_index


def _combine_autoregressive_terms(
    terms: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if not terms:
        return None
    count = sum(int(term["count"]) for term in terms)
    first = terms[0]
    weight = float(first["weight"])
    name = str(first["name"])
    if any(
        float(term["weight"]) != weight or str(term["name"]) != name
        for term in terms[1:]
    ):
        raise RuntimeError("autoregressive settings changed within one batch")
    loss_numerator = sum(
        term["loss"] * int(term["count"]) for term in terms
    )
    return {
        "loss": loss_numerator / count,
        "loss_sum": sum(term["loss_sum"] for term in terms),
        "squared_error": sum(term["squared_error"] for term in terms),
        "absolute_error": sum(term["absolute_error"] for term in terms),
        "count": count,
        "weight": weight,
        "name": name,
    }


def _forward_continuous_batch(
    model: nn.Module,
    features: torch.Tensor,
    padding_mask: torch.Tensor | None,
    entity_ids: torch.Tensor | None,
    time_indices: torch.Tensor | None,
    batch_metadata: Mapping[str, torch.Tensor],
    tracker: _EntityStreamTracker,
) -> tuple[torch.Tensor, dict[str, Any] | None]:
    if entity_ids is None or time_indices is None:
        raise ValueError(
            "continuous TTT state requires entity_id and time_index metadata"
        )
    new_token_values = batch_metadata.get("state_new_tokens")
    if new_token_values is None:
        raise ValueError(
            "continuous TTT state requires state_new_tokens from the dataset"
        )
    batch_size = features.shape[0]
    if any(
        values.numel() != batch_size
        for values in (entity_ids, time_indices, new_token_values)
    ):
        raise ValueError("continuous-state metadata must have one value per sample")

    predictions: list[torch.Tensor] = []
    feature_terms: list[Mapping[str, Any]] = []
    index = 0
    while index < batch_size:
        entity_id = int(entity_ids[index])
        stream_segments: list[torch.Tensor] = []
        endpoint_positions: list[int] = []
        stream_length = 0
        while index < batch_size and int(entity_ids[index]) == entity_id:
            tracker.advance(model, entity_id, int(time_indices[index]))
            sample = features[index]
            valid = (
                sample
                if padding_mask is None
                else sample[~padding_mask[index]]
            )
            new_tokens = int(new_token_values[index])
            if new_tokens <= 0 or new_tokens > valid.shape[0]:
                raise ValueError(
                    "state_new_tokens must select a non-empty suffix of the "
                    f"valid window; got {new_tokens} for {valid.shape[0]} "
                    "valid tokens"
                )
            segment = valid[-new_tokens:]
            stream_segments.append(segment)
            stream_length += new_tokens
            endpoint_positions.append(stream_length - 1)
            index += 1

        incremental = torch.cat(stream_segments, dim=0).unsqueeze(0)
        raw_output = model(
            incremental,
            return_sequence_predictions=True,
        )
        if not isinstance(raw_output, Mapping):
            raise TypeError(
                "continuous TTT model must return sequence prediction metadata"
            )
        sequence_predictions = raw_output.get("sequence_predictions")
        if not isinstance(sequence_predictions, torch.Tensor):
            raise KeyError(
                "continuous TTT model output requires sequence_predictions"
            )
        if sequence_predictions.shape != incremental.shape[:2]:
            raise ValueError(
                "sequence_predictions must have shape [1, stream_length]"
            )
        positions = torch.tensor(
            endpoint_positions,
            dtype=torch.long,
            device=sequence_predictions.device,
        )
        predictions.extend(sequence_predictions[0].index_select(0, positions))
        terms = _autoregressive_terms(model, raw_output, incremental, None)
        if terms is not None:
            feature_terms.append(terms)
    return torch.stack(predictions), _combine_autoregressive_terms(feature_terms)


def _fallback_metrics(
    y_true: Sequence[float], y_pred: Sequence[float]
) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    error = prediction - truth
    rmse = float(np.sqrt(np.mean(np.square(error))))
    mae = float(np.mean(np.abs(error)))
    exponent = np.where(error < 0.0, -error / 13.0, error / 10.0)
    nasa_score = float(np.sum(np.exp(np.clip(exponent, None, 80.0)) - 1.0))
    return {"rmse": rmse, "mae": mae, "nasa_score": nasa_score}


def _compute_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    metric_fn: MetricFn | None,
) -> dict[str, float]:
    if metric_fn is None:
        try:
            from .metrics import compute_metrics
        except ImportError:
            metric_fn = _fallback_metrics
        else:
            metric_fn = compute_metrics
    raw_metrics = metric_fn(y_true, y_pred)
    return {str(name): float(value) for name, value in raw_metrics.items()}


def _run_loader(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any] | Sequence[Any]],
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    grad_clip: float | None,
    max_batches: int | None,
    metric_fn: MetricFn | None,
    reset_each_batch: bool,
    require_single_item: bool,
    include_predictions: bool,
    prediction_sink: Callable[[Sequence[Mapping[str, Any]]], None] | None = None,
    precision: str = "fp32",
) -> dict[str, Any]:
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when provided")
    training = optimizer is not None
    resolved_precision = resolve_precision(precision, device)
    model.train(training)
    continuous_state = uses_continuous_state(model)
    stream_tracker = _EntityStreamTracker() if continuous_state else None
    if continuous_state or not reset_each_batch:
        reset_model_state(model)

    total_loss = 0.0
    sample_count = 0
    gradient_norm_sum = 0.0
    gradient_steps = 0
    feature_loss_sum = 0.0
    feature_squared_error = 0.0
    feature_absolute_error = 0.0
    feature_count = 0
    autoregressive_weight: float | None = None
    autoregressive_loss_name: str | None = None
    y_true: list[float] = []
    y_pred: list[float] = []
    accumulator: Any = None
    if metric_fn is None:
        from .metrics import RegressionMetricAccumulator

        accumulator = RegressionMetricAccumulator()
    records: list[dict[str, Any]] = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            if reset_each_batch and not continuous_state:
                reset_model_state(model)
            (
                features,
                padding_mask,
                target,
                entity_ids,
                time_indices,
                batch_metadata,
            ) = _unpack_batch(batch, device)
            if require_single_item and target.numel() != 1:
                raise ValueError(
                    "Per-engine evaluation requires DataLoader batch_size=1"
                )

            if training:
                optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, resolved_precision):
                if stream_tracker is None:
                    prediction, model_output = _forward_model(
                        model, features, padding_mask
                    )
                    feature_terms = _autoregressive_terms(
                        model, model_output, features, padding_mask
                    )
                else:
                    prediction, feature_terms = _forward_continuous_batch(
                        model,
                        features,
                        padding_mask,
                        entity_ids,
                        time_indices,
                        batch_metadata,
                        stream_tracker,
                    )
                if prediction.numel() != target.numel():
                    raise ValueError(
                        "Model returned "
                        f"{prediction.numel()} values for {target.numel()} targets"
                    )
                loss = criterion(prediction, target)
                objective = loss
                if feature_terms is not None:
                    objective = (
                        objective
                        + feature_terms["weight"] * feature_terms["loss"]
                    )
            if feature_terms is not None:
                feature_loss_sum += float(feature_terms["loss_sum"])
                feature_squared_error += float(feature_terms["squared_error"])
                feature_absolute_error += float(feature_terms["absolute_error"])
                feature_count += int(feature_terms["count"])
                autoregressive_weight = float(feature_terms["weight"])
                autoregressive_loss_name = str(feature_terms["name"])
            if training:
                objective.backward()
                if grad_clip is not None and grad_clip > 0.0:
                    norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    gradient_norm_sum += float(norm.detach())
                    gradient_steps += 1
                optimizer.step()

            batch_size = target.numel()
            total_loss += float(loss.detach()) * batch_size
            sample_count += batch_size
            target_values = target.detach().cpu().tolist()
            prediction_values = prediction.detach().cpu().tolist()
            if accumulator is not None:
                accumulator.update(target_values, prediction_values)
            else:
                y_true.extend(float(value) for value in target_values)
                y_pred.extend(float(value) for value in prediction_values)

            if include_predictions or prediction_sink is not None:
                entity_values = (
                    entity_ids.tolist()
                    if entity_ids is not None
                    else [None] * batch_size
                )
                time_values = (
                    time_indices.tolist()
                    if time_indices is not None
                    else [None] * batch_size
                )
                metadata_values = {
                    key: value.tolist()
                    for key, value in batch_metadata.items()
                    if key != "state_new_tokens"
                }
                batch_records: list[dict[str, Any]] = []
                for index in range(batch_size):
                    record = {
                        "entity_id": (
                            int(entity_values[index])
                            if entity_values[index] is not None
                            else None
                        ),
                        "time_index": (
                            int(time_values[index])
                            if time_values[index] is not None
                            else None
                        ),
                        "target": float(target_values[index]),
                        "prediction": float(prediction_values[index]),
                    }
                    for key, values in metadata_values.items():
                        record[key] = int(values[index])
                    batch_records.append(record)
                if include_predictions:
                    records.extend(batch_records)
                if prediction_sink is not None:
                    prediction_sink(batch_records)

    if sample_count == 0:
        raise ValueError("DataLoader produced no batches")
    result: dict[str, Any] = {
        "loss": total_loss / sample_count,
        "num_samples": sample_count,
    }
    if accumulator is not None:
        result.update(accumulator.compute())
    else:
        result.update(_compute_metrics(y_true, y_pred, metric_fn))
    if gradient_steps:
        result["gradient_norm"] = gradient_norm_sum / gradient_steps
    if feature_count:
        feature_loss = feature_loss_sum / feature_count
        assert autoregressive_weight is not None
        result.update(
            objective_loss=result["loss"] + autoregressive_weight * feature_loss,
            feature_loss=feature_loss,
            feature_rmse=math.sqrt(feature_squared_error / feature_count),
            feature_mae=feature_absolute_error / feature_count,
            feature_elements=feature_count,
            autoregressive_weight=autoregressive_weight,
            autoregressive_loss=autoregressive_loss_name,
        )
    if include_predictions:
        result["predictions"] = records
    return result


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any] | Sequence[Any]],
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    *,
    criterion: nn.Module | None = None,
    grad_clip: float | None = 1.0,
    max_batches: int | None = None,
    metric_fn: MetricFn | None = None,
    precision: str = "fp32",
) -> dict[str, float]:
    """Train for one epoch with MSE loss and optional global gradient clipping."""
    resolved_device = resolve_device(device)
    loss_fn = criterion if criterion is not None else nn.MSELoss()
    return _run_loader(
        model,
        loader,
        resolved_device,
        optimizer=optimizer,
        criterion=loss_fn,
        grad_clip=grad_clip,
        max_batches=max_batches,
        metric_fn=metric_fn,
        reset_each_batch=False,
        require_single_item=False,
        include_predictions=False,
        precision=precision,
    )


def evaluate_loader(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any] | Sequence[Any]],
    device: str | torch.device,
    *,
    criterion: nn.Module | None = None,
    max_batches: int | None = None,
    metric_fn: MetricFn | None = None,
    include_predictions: bool = False,
    prediction_sink: Callable[[Sequence[Mapping[str, Any]]], None] | None = None,
    reset_each_batch: bool = False,
    require_single_item: bool = False,
    precision: str = "fp32",
) -> dict[str, Any]:
    """Evaluate an ordinary validation loader without mutating model parameters."""
    resolved_device = resolve_device(device)
    loss_fn = criterion if criterion is not None else nn.MSELoss()
    return _run_loader(
        model,
        loader,
        resolved_device,
        optimizer=None,
        criterion=loss_fn,
        grad_clip=None,
        max_batches=max_batches,
        metric_fn=metric_fn,
        reset_each_batch=reset_each_batch,
        require_single_item=require_single_item,
        include_predictions=include_predictions,
        prediction_sink=prediction_sink,
        precision=precision,
    )


def evaluate_by_engine(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any] | Sequence[Any]],
    device: str | torch.device,
    *,
    criterion: nn.Module | None = None,
    max_engines: int | None = None,
    metric_fn: MetricFn | None = None,
    include_predictions: bool = True,
    precision: str = "fp32",
) -> dict[str, Any]:
    """Evaluate one engine per batch, resetting TTT state before every engine."""
    resolved_device = resolve_device(device)
    loss_fn = criterion if criterion is not None else nn.MSELoss()
    return _run_loader(
        model,
        loader,
        resolved_device,
        optimizer=None,
        criterion=loss_fn,
        grad_clip=None,
        max_batches=max_engines,
        metric_fn=metric_fn,
        reset_each_batch=True,
        require_single_item=True,
        include_predictions=include_predictions,
        precision=precision,
    )


def evaluate_dataset(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any] | Sequence[Any]],
    device: str | torch.device,
    *,
    max_batches: int | None = None,
    reset_each_batch: bool = True,
    require_single_item: bool = False,
    include_predictions: bool = False,
    prediction_sink: Callable[[Sequence[Mapping[str, Any]]], None] | None = None,
    precision: str = "fp32",
) -> dict[str, Any]:
    """Evaluate a dataset according to its adapter-provided lifecycle policy."""
    return evaluate_loader(
        model,
        loader,
        device,
        max_batches=max_batches,
        include_predictions=include_predictions,
        prediction_sink=prediction_sink,
        reset_each_batch=reset_each_batch,
        require_single_item=require_single_item,
        precision=precision,
    )


def fit(
    model: nn.Module,
    train_loader: Iterable[Mapping[str, Any] | Sequence[Any]],
    val_loader: Iterable[Mapping[str, Any] | Sequence[Any]],
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    *,
    epochs: int,
    output_dir: str | Path,
    checkpoint_metadata: Mapping[str, Any],
    patience: int = 10,
    min_delta: float = 0.0,
    grad_clip: float | None = 1.0,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    start_epoch: int = 0,
    best_val_loss: float = math.inf,
    epochs_without_improvement: int = 0,
    history: Sequence[Mapping[str, Any]] = (),
    metric_fn: MetricFn | None = None,
    verbose: bool = True,
    precision: str = "fp32",
) -> dict[str, Any]:
    """Fit with validation early stopping and atomic best/last checkpoints."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if start_epoch < 0 or start_epoch >= epochs:
        raise ValueError("start_epoch must be in [0, epochs)")
    if patience < 0:
        raise ValueError("patience cannot be negative")
    if min_delta < 0:
        raise ValueError("min_delta cannot be negative")

    resolved_device = resolve_device(device)
    model.to(resolved_device)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    best_path = destination / "best.pt"
    last_path = destination / "last.pt"
    records = [_native_value(record, "history") for record in history]
    best_epoch = -1
    for record in records:
        val = record.get("val", {})
        if val.get("loss") == best_val_loss:
            best_epoch = int(record["epoch"])

    stopped_early = False
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            resolved_device,
            grad_clip=grad_clip,
            max_batches=max_train_batches,
            metric_fn=metric_fn,
            precision=precision,
        )
        val_metrics = evaluate_loader(
            model,
            val_loader,
            resolved_device,
            max_batches=max_val_batches,
            metric_fn=metric_fn,
            precision=precision,
        )
        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        records.append(epoch_record)
        last_epoch = epoch

        improved = float(val_metrics["loss"]) < best_val_loss - min_delta
        if improved:
            best_val_loss = float(val_metrics["loss"])
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                metadata=checkpoint_metadata,
                history=records,
                epochs_without_improvement=epochs_without_improvement,
            )
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            last_path,
            model,
            optimizer,
            epoch=epoch,
            best_val_loss=best_val_loss,
            metadata=checkpoint_metadata,
            history=records,
            epochs_without_improvement=epochs_without_improvement,
        )

        if verbose:
            auxiliary = ""
            if "feature_rmse" in train_metrics:
                auxiliary = (
                    f" train_objective={train_metrics['objective_loss']:.6f} "
                    f"val_feature_rmse={val_metrics['feature_rmse']:.6f}"
                )
            print(
                f"epoch={epoch + 1}/{epochs} "
                f"train_mse={train_metrics['loss']:.6f} "
                f"val_mse={val_metrics['loss']:.6f} "
                f"val_rmse={val_metrics.get('rmse', math.nan):.6f}"
                f"{auxiliary}"
            )
        if patience > 0 and epochs_without_improvement >= patience:
            stopped_early = True
            break

    return {
        "history": records,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "last_epoch": last_epoch,
        "epochs_without_improvement": epochs_without_improvement,
        "stopped_early": stopped_early,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
    }
