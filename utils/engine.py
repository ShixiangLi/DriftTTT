from __future__ import annotations

import json
import os
import random
import time
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import DatasetBundle, build_dataset_bundle
from models import RULTransformer
from utils.complexity import model_complexity
from utils.config import normalize_config, save_config
from utils.metrics import RegressionAccumulator
from utils.visualization import plot_history, plot_predictions


def _write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def _save_checkpoint(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    required = {
        "model_state",
        "config",
        "input_dim",
        "preprocessing_state",
        "split_state",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing fields: {sorted(missing)}")
    return checkpoint


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _worker_seed(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def _loader(
    dataset: torch.utils.data.Dataset,
    data_config: dict[str, Any],
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    workers = int(data_config["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(data_config["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=bool(data_config["pin_memory"]) and device.type == "cuda",
        persistent_workers=workers > 0,
        worker_init_fn=_worker_seed if workers > 0 else None,
        generator=generator,
    )


def _autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def _use_bfloat16(precision: str, device: torch.device) -> bool:
    if precision == "fp32":
        return False
    supported = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if precision == "bf16" and not supported:
        raise RuntimeError(
            "BF16 was requested but is unsupported on the selected device"
        )
    return supported


def _observed_cycle_counts(cycle_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Count contiguous observed cycles in each window."""
    valid_pairs = mask[:, 1:] & mask[:, :-1]
    transitions = valid_pairs & (cycle_ids[:, 1:] != cycle_ids[:, :-1])
    return mask.any(dim=1).long() + transitions.sum(dim=1)


def _run_epoch(
    model: RULTransformer,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip: float | None,
    use_bfloat16: bool,
    max_batches: int | None,
    target_scale: float,
) -> dict[str, float | int]:
    training = optimizer is not None
    model.train(training)
    metrics = RegressionAccumulator(target_scale=target_scale)
    cycle_count_sum = torch.zeros((), device=device, dtype=torch.int64)
    multi_cycle_count = torch.zeros((), device=device, dtype=torch.int64)
    cycle_window_count = 0
    objective_sum = torch.zeros((), device=device)
    sample_count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        features = batch["features"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        cycle_ids = batch.get("cycle_ids")
        observed_cycles = None
        if cycle_ids is not None:
            cycle_ids = cycle_ids.to(device, non_blocking=True)
            observed_cycles = _observed_cycle_counts(cycle_ids, mask)
            cycle_count_sum += observed_cycles.sum()
            multi_cycle_count += (observed_cycles > 1).sum()
            cycle_window_count += observed_cycles.numel()
        targets = batch["target"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with _autocast_context(device, use_bfloat16):
                predictions = model(features, mask, cycle_ids)
            # Regression loss stays in FP32 so BF16 predictions and FP32 labels
            # cannot create an unsupported mixed-dtype MSE backward graph.
            loss = nn.functional.mse_loss(
                predictions.float(), targets.float()
            )
            if training:
                loss.backward()
                if gradient_clip is not None and gradient_clip > 0.0:
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
        batch_size = targets.numel()
        objective_sum += loss.detach() * batch_size
        sample_count += batch_size
        metrics.update(predictions, targets)
    result = metrics.compute()
    if training and sample_count:
        result["objective"] = (objective_sum / sample_count).item()
    if cycle_window_count:
        result["mean_cycles_per_window"] = (
            cycle_count_sum.float() / cycle_window_count
        ).item()
        result["multi_cycle_fraction"] = (
            multi_cycle_count.float() / cycle_window_count
        ).item()
    return result


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    bundle: DatasetBundle,
    complexity: dict[str, int],
    epoch: int,
    best_validation_mse: float,
    epochs_without_improvement: int,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_validation_mse": best_validation_mse,
        "epochs_without_improvement": epochs_without_improvement,
        "config": deepcopy(config),
        "input_dim": bundle.input_dim,
        "feature_names": bundle.feature_names,
        "preprocessing_state": bundle.preprocessing_state,
        "split_state": bundle.split_state,
        "complexity": complexity,
    }


def _prediction_record(
    entity_id: int, endpoint: int, target: float, prediction: float
) -> dict[str, int | float]:
    return {
        "entity_id": entity_id,
        "endpoint": endpoint,
        "target": target,
        "prediction": prediction,
        "error": prediction - target,
    }


@torch.inference_mode()
def evaluate_model(
    model: RULTransformer,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    config: dict[str, Any],
    complexity: dict[str, int],
    dataset_name: str,
    test_protocol: str,
) -> dict[str, Any]:
    model.eval()
    evaluation = config["evaluation"]
    prediction_path = output_dir / evaluation["predictions_file"]
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    json_lines = prediction_path.suffix.lower() == ".jsonl"
    target_scale = float(config["data"]["rul_cap"] or 1.0)
    metrics = RegressionAccumulator(target_scale=target_scale)
    cycle_count_sum = torch.zeros((), device=device, dtype=torch.int64)
    multi_cycle_count = torch.zeros((), device=device, dtype=torch.int64)
    cycle_window_count = 0
    records: list[dict[str, int | float]] = []
    plot_records: list[dict[str, int | float]] = []
    line_handle = prediction_path.open("w", encoding="utf-8") if json_lines else None
    try:
        for batch_index, batch in enumerate(loader):
            maximum = evaluation["max_test_batches"]
            if maximum is not None and batch_index >= maximum:
                break
            features = batch["features"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            cycle_ids = batch.get("cycle_ids")
            if cycle_ids is not None:
                cycle_ids = cycle_ids.to(device, non_blocking=True)
                observed_cycles = _observed_cycle_counts(cycle_ids, mask)
                cycle_count_sum += observed_cycles.sum()
                multi_cycle_count += (observed_cycles > 1).sum()
                cycle_window_count += observed_cycles.numel()
            targets = batch["target"].to(device, non_blocking=True)
            predictions = model(features, mask, cycle_ids)
            metrics.update(predictions, targets)
            values = zip(
                batch["entity_id"].tolist(),
                batch["endpoint"].tolist(),
                (targets * target_scale).cpu().tolist(),
                (predictions.float() * target_scale).cpu().tolist(),
            )
            for entity_id, endpoint, target, prediction in values:
                record = _prediction_record(entity_id, endpoint, target, prediction)
                if line_handle is not None:
                    line_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                else:
                    records.append(record)
                if len(plot_records) < 5_000:
                    plot_records.append(record)
    finally:
        if line_handle is not None:
            line_handle.close()
    if not json_lines:
        _write_json(records, prediction_path)

    result: dict[str, Any] = {
        **metrics.compute(),
        "dataset": dataset_name,
        "protocol": test_protocol,
        "partial": evaluation["max_test_batches"] is not None,
        "rul_cap": config["data"]["rul_cap"],
        "complexity": complexity,
    }
    if cycle_window_count:
        result["mean_cycles_per_window"] = (
            cycle_count_sum.float() / cycle_window_count
        ).item()
        result["multi_cycle_fraction"] = (
            multi_cycle_count.float() / cycle_window_count
        ).item()
    _write_json(result, output_dir / evaluation["metrics_file"])
    if evaluation["plots"]:
        plot_predictions(plot_records, output_dir / "test_predictions.png")
    return result


def train_experiment(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")
    training = config["training"]
    device = resolve_device(training["device"])
    seed_everything(int(training["seed"]), bool(training["deterministic"]))

    resume_checkpoint: dict[str, Any] | None = None
    if training["resume"]:
        resume_checkpoint = _load_checkpoint(Path(training["resume"]), device)
        saved_config = normalize_config(resume_checkpoint["config"])
        if saved_config["model"] != config["model"]:
            raise ValueError("Resume checkpoint model configuration does not match")
        if saved_config["data"]["name"] != config["data"]["name"]:
            raise ValueError("Resume checkpoint dataset type does not match")

    bundle = build_dataset_bundle(
        config["data"],
        None if resume_checkpoint is None else resume_checkpoint["preprocessing_state"],
        None if resume_checkpoint is None else resume_checkpoint["split_state"],
    )
    model = RULTransformer(bundle.input_dim, config["model"]).to(device)
    complexity = model_complexity(
        model, bundle.input_dim, config["data"]["window_size"], config["model"]
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    start_epoch = 1
    best_validation_mse = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    history_path = output_dir / "history.json"
    if resume_checkpoint is not None:
        if int(resume_checkpoint["input_dim"]) != bundle.input_dim:
            raise ValueError("Resume checkpoint input dimension does not match data")
        model.load_state_dict(resume_checkpoint["model_state"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_validation_mse = float(resume_checkpoint["best_validation_mse"])
        epochs_without_improvement = int(
            resume_checkpoint.get("epochs_without_improvement", 0)
        )
        if history_path.is_file():
            with history_path.open("r", encoding="utf-8") as handle:
                history = json.load(handle)
    if start_epoch > int(training["epochs"]):
        raise ValueError("training.epochs must exceed the resumed checkpoint epoch")

    train_loader = _loader(
        bundle.train, config["data"], True, int(training["seed"]), device
    )
    validation_loader = _loader(
        bundle.validation, config["data"], False, int(training["seed"]), device
    )
    use_bfloat16 = _use_bfloat16(training["precision"], device)
    print(
        f"device={device} precision={'bf16' if use_bfloat16 else 'fp32'} "
        f"mixer={config['model']['sequence_mixer']} "
        f"train_samples={len(bundle.train)} validation_samples={len(bundle.validation)} "
        f"input_dim={bundle.input_dim} parameters={complexity['parameters']:,}"
    )
    best_path = output_dir / "best.pt"
    target_scale = float(config["data"]["rul_cap"] or 1.0)
    started = time.perf_counter()
    for epoch in range(start_epoch, int(training["epochs"]) + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            training["gradient_clip"],
            use_bfloat16,
            training["max_train_batches"],
            target_scale,
        )
        validation_metrics = _run_epoch(
            model,
            validation_loader,
            device,
            None,
            None,
            use_bfloat16,
            training["max_validation_batches"],
            target_scale,
        )
        row = {
            "epoch": epoch,
            "train_objective": train_metrics.get(
                "objective", train_metrics["mse"]
            ),
            "train_mse": train_metrics["mse"],
            "train_rmse": train_metrics["rmse"],
            "train_mae": train_metrics["mae"],
            "train_mean_cycles_per_window": train_metrics.get(
                "mean_cycles_per_window", 0.0
            ),
            "train_multi_cycle_fraction": train_metrics.get(
                "multi_cycle_fraction", 0.0
            ),
            "validation_mse": validation_metrics["mse"],
            "validation_rmse": validation_metrics["rmse"],
            "validation_mae": validation_metrics["mae"],
            "validation_mean_cycles_per_window": validation_metrics.get(
                "mean_cycles_per_window", 0.0
            ),
            "validation_multi_cycle_fraction": validation_metrics.get(
                "multi_cycle_fraction", 0.0
            ),
        }
        history.append(row)
        improved = float(validation_metrics["mse"]) < best_validation_mse
        if improved:
            best_validation_mse = float(validation_metrics["mse"])
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        payload = _checkpoint_payload(
            model,
            optimizer,
            config,
            bundle,
            complexity,
            epoch,
            best_validation_mse,
            epochs_without_improvement,
        )
        _save_checkpoint(payload, output_dir / "last.pt")
        if improved:
            _save_checkpoint(payload, best_path)
        _write_json(history, history_path)
        print(
            f"epoch={epoch:03d} train_rmse={train_metrics['rmse']:.4f} "
            f"validation_rmse={validation_metrics['rmse']:.4f} "
            f"best_mse={best_validation_mse:.4f}"
        )
        if epochs_without_improvement >= int(training["early_stopping_patience"]):
            print(f"early_stopping epoch={epoch}")
            break
    if training["plots"]:
        plot_history(history, output_dir / "training_history.png")

    best_checkpoint = _load_checkpoint(best_path, device)
    model.load_state_dict(best_checkpoint["model_state"])
    test_loader = _loader(
        bundle.test, config["data"], False, int(training["seed"]), device
    )
    test_metrics = evaluate_model(
        model,
        test_loader,
        device,
        output_dir,
        config,
        complexity,
        bundle.dataset_name,
        bundle.test_protocol,
    )
    elapsed = time.perf_counter() - started
    print(
        f"test_rmse={test_metrics['rmse']:.4f} test_mae={test_metrics['mae']:.4f} "
        f"nasa_score={test_metrics['nasa_score']:.4f} elapsed_seconds={elapsed:.1f}"
    )
    return test_metrics


def evaluate_experiment(requested_config: dict[str, Any]) -> dict[str, Any]:
    requested_output = Path(requested_config["experiment"]["output_dir"])
    checkpoint_value = requested_config["evaluation"]["checkpoint"]
    checkpoint_path = (
        Path(checkpoint_value) if checkpoint_value else requested_output / "best.pt"
    )
    device = resolve_device(requested_config["evaluation"]["device"])
    checkpoint = _load_checkpoint(checkpoint_path, device)
    config = normalize_config(checkpoint["config"])
    config["data"]["root"] = requested_config["data"]["root"]
    config["evaluation"] = deepcopy(requested_config["evaluation"])
    config["experiment"]["output_dir"] = str(requested_output)
    bundle = build_dataset_bundle(
        config["data"],
        checkpoint["preprocessing_state"],
        checkpoint["split_state"],
    )
    if bundle.input_dim != int(checkpoint["input_dim"]):
        raise ValueError("Checkpoint input dimension does not match evaluation data")
    model = RULTransformer(bundle.input_dim, config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    complexity = checkpoint.get(
        "complexity",
        model_complexity(
            model, bundle.input_dim, config["data"]["window_size"], config["model"]
        ),
    )
    loader = _loader(
        bundle.test,
        config["data"],
        False,
        int(config["training"]["seed"]),
        device,
    )
    metrics = evaluate_model(
        model,
        loader,
        device,
        requested_output,
        config,
        complexity,
        bundle.dataset_name,
        bundle.test_protocol,
    )
    print(json.dumps(metrics, indent=2))
    return metrics
