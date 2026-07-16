"""Training, evaluation, and checkpoint utilities for C-MAPSS RUL models."""

from __future__ import annotations

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


MetricFn = Callable[[Sequence[float], Sequence[float]], Mapping[str, float]]


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
    if deterministic:
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False


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


def reset_model_state(model: nn.Module) -> None:
    """Reset TTT fast state, including through a DDP-style ``module`` wrapper."""
    target = getattr(model, "module", model)
    reset = getattr(target, "reset_ttt_state", None)
    if callable(reset):
        reset()
        return

    return


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
            for key in ("engine_id", "unit_id", "cycle", "sample_index")
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

    features_tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    target_tensor = torch.as_tensor(target, dtype=torch.float32, device=device).reshape(
        -1
    )
    mask_tensor = None
    if padding_mask is not None:
        mask_tensor = torch.as_tensor(padding_mask, dtype=torch.bool, device=device)
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
) -> torch.Tensor:
    if padding_mask is None:
        prediction = model(features)
    else:
        prediction = model(features, padding_mask=padding_mask)
    if isinstance(prediction, Mapping):
        if "prediction" not in prediction:
            raise KeyError("Model output dictionaries require a 'prediction' value")
        prediction = prediction["prediction"]
    if not isinstance(prediction, torch.Tensor):
        raise TypeError("Model output must be a tensor")
    return prediction.reshape(-1)


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
) -> dict[str, Any]:
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when provided")
    training = optimizer is not None
    model.train(training)
    if not reset_each_batch:
        reset_model_state(model)

    total_loss = 0.0
    sample_count = 0
    gradient_norm_sum = 0.0
    gradient_steps = 0
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
            if reset_each_batch:
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
            prediction = _forward_model(model, features, padding_mask)
            if prediction.numel() != target.numel():
                raise ValueError(
                    f"Model returned {prediction.numel()} values for {target.numel()} targets"
                )
            loss = criterion(prediction, target)
            if training:
                loss.backward()
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
                    key: value.tolist() for key, value in batch_metadata.items()
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
        )
        val_metrics = evaluate_loader(
            model,
            val_loader,
            resolved_device,
            max_batches=max_val_batches,
            metric_fn=metric_fn,
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
            print(
                f"epoch={epoch + 1}/{epochs} "
                f"train_mse={train_metrics['loss']:.6f} "
                f"val_mse={val_metrics['loss']:.6f} "
                f"val_rmse={val_metrics.get('rmse', math.nan):.6f}"
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
