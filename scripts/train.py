"""Train and test a configured RUL Transformer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from data.registry import get_dataset_adapter
from utils.complexity import estimate_model_complexity, format_model_complexity
from utils.config import ExperimentConfig, load_experiment_config
from utils.engine import (
    JsonlPredictionWriter,
    build_model,
    compile_model,
    evaluate_dataset,
    fit,
    load_checkpoint,
    resolve_device,
    resolve_precision,
    restore_checkpoint,
    save_json,
    seed_worker,
    set_seed,
    uses_continuous_state,
    verify_data_provenance,
)
from utils.visualization import plot_rul_predictions, plot_training_history


def parse_config() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return load_experiment_config(parser.parse_args().config)


def _make_loader(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    worker_options = {"prefetch_factor": 4} if num_workers else {}
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_worker if num_workers else None,
        generator=generator,
        persistent_workers=num_workers > 0,
        **worker_options,
    )


def _resolve_output_dir(requested: Path, resume: Path | None) -> Path:
    if resume is None:
        return requested
    checkpoint_dir = resume.parent.resolve()
    if requested.resolve() != checkpoint_dir:
        raise ValueError(
            "Resumed training must write to the checkpoint directory so the "
            "historical best.pt remains available."
        )
    return checkpoint_dir


def _evaluate_test(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    bundle: Any,
    output_dir: Path,
    max_batches: int | None,
    plots: bool,
    precision: str,
) -> tuple[dict[str, Any], Path, int]:
    spec = bundle.evaluation_spec
    if spec.protocol == "endpoint_per_entity":
        result = evaluate_dataset(
            model,
            loader,
            device,
            max_batches=max_batches,
            reset_each_batch=spec.reset_each_batch,
            require_single_item=spec.require_single_item,
            include_predictions=True,
            precision=precision,
        )
        predictions = result.pop("predictions")
        path = output_dir / "test_predictions.json"
        save_json(path, predictions)
        count = len(predictions)
    else:
        path = output_dir / "test_predictions.jsonl"
        with JsonlPredictionWriter(path) as writer:
            result = evaluate_dataset(
                model,
                loader,
                device,
                max_batches=max_batches,
                reset_each_batch=spec.reset_each_batch,
                require_single_item=spec.require_single_item,
                prediction_sink=writer.write,
                precision=precision,
            )
        count = writer.count
    if plots:
        plot_rul_predictions(path, output_dir / "test_predictions.png")
    return result, path, count


def main() -> None:
    config = parse_config()
    data_settings = config.data
    training = config.training
    evaluation = config.evaluation
    adapter = get_dataset_adapter(data_settings.name)
    device = resolve_device(training.device)
    precision = resolve_precision(training.precision, device)
    resume_payload = (
        load_checkpoint(training.resume, "cpu") if training.resume else None
    )

    if resume_payload is not None:
        config.verify_checkpoint_identity(resume_payload)
        adapter.validate_checkpoint(data_settings, resume_payload)
    set_seed(training.seed, deterministic=training.deterministic)
    preprocessor_state = None
    if resume_payload is not None:
        value = resume_payload.get("preprocessor")
        if not isinstance(value, dict):
            raise ValueError("Resume checkpoint is missing preprocessor metadata.")
        preprocessor_state = value
    bundle = adapter.prepare_training(data_settings, preprocessor_state)

    if resume_payload is None:
        model_name = config.model.type
        requested_model_config = config.model.constructor_values(
            bundle.preprocessor.output_dim
        )
    else:
        verify_data_provenance(bundle, resume_payload)
        model_name = str(resume_payload.get("model_name", "ttt"))
        requested_model_config = dict(resume_payload.get("model_config", {}))
        if not requested_model_config:
            raise ValueError("Resume checkpoint is missing model_config.")
        if int(requested_model_config["input_dim"]) != bundle.preprocessor.output_dim:
            raise RuntimeError("Checkpoint input_dim does not match prepared features.")

    model = build_model(model_name, requested_model_config).to(device)
    model_config = (
        model.get_config() if hasattr(model, "get_config") else requested_model_config
    )
    model_complexity = estimate_model_complexity(model, data_settings.window_size)
    architecture = str(getattr(model, "architecture", "encoder"))
    continuous_state = uses_continuous_state(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )

    start_epoch = 0
    best_val_loss = float("inf")
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state_dict"], strict=True)
        optimizer_state = resume_payload.get("optimizer_state_dict")
        if optimizer_state is None:
            raise ValueError("Resume checkpoint is missing optimizer_state_dict.")
        optimizer.load_state_dict(optimizer_state)
        start_epoch = int(resume_payload["epoch"]) + 1
        best_val_loss = float(resume_payload["best_val_loss"])
        stale_epochs = int(resume_payload.get("epochs_without_improvement", 0))
        history = list(resume_payload.get("history", []))
        if start_epoch >= training.epochs:
            raise ValueError(
                f"Checkpoint already reached epoch {start_epoch}; "
                "training.epochs must be larger."
            )

    compile_model(model, training.compile)

    output_dir = _resolve_output_dir(
        Path(config.experiment.output_dir), training.resume
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.yaml")
    pin_memory = device.type == "cuda"
    test_dataset = bundle.test_dataset
    if continuous_state:
        continuous_view = getattr(test_dataset, "continuous_evaluation_view", None)
        if callable(continuous_view):
            test_dataset = continuous_view()
    train_loader = _make_loader(
        bundle.train_dataset,
        batch_size=training.batch_size,
        shuffle=not continuous_state,
        num_workers=training.num_workers,
        pin_memory=pin_memory,
        seed=training.seed,
    )
    val_loader = _make_loader(
        bundle.val_dataset,
        batch_size=training.batch_size,
        shuffle=False,
        num_workers=training.num_workers,
        pin_memory=pin_memory,
        seed=training.seed,
    )
    test_batch_size = evaluation.batch_size or bundle.evaluation_spec.batch_size
    if (
        getattr(test_dataset, "requires_batch_size_one", False)
        and test_batch_size != 1
    ):
        raise ValueError(
            "continuous C-MAPSS trajectory evaluation requires batch_size=1"
        )
    test_loader = _make_loader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=evaluation.num_workers,
        pin_memory=pin_memory,
        seed=training.seed,
    )

    split_ids = {name: list(values) for name, values in bundle.split_entity_ids.items()}
    checkpoint_metadata = {
        "dataset_name": bundle.dataset_name,
        "model_name": model_name,
        "model_architecture": architecture,
        "continuous_state": continuous_state,
        "experiment_name": config.experiment.name,
        "experiment_config": config.to_dict(),
        "model_config": model_config,
        "data_config": dict(bundle.data_config),
        "preprocessor": bundle.preprocessor.state_dict(),
        "split_entity_ids": split_ids,
        "subset": bundle.subset,
        "window_size": data_settings.window_size,
        "rul_cap": data_settings.rul_cap,
        "label_policy": dict(bundle.label_policy),
        "evaluation_protocol": bundle.evaluation_spec.protocol,
        "model_complexity": model_complexity,
        "data_metadata": dict(bundle.metadata),
        "training_config": {
            "optimizer": "AdamW",
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "weight_decay": float(optimizer.param_groups[0]["weight_decay"]),
            "batch_size": training.batch_size,
            "grad_clip": training.grad_clip,
            "patience": training.patience,
            "min_delta": training.min_delta,
            "seed": training.seed,
            "split_seed": data_settings.split_seed,
            "precision": precision,
            "compile": training.compile,
        },
    }
    if bundle.dataset_name == "cmapss":
        checkpoint_metadata["split_engine_ids"] = split_ids

    print(
        f"device={device} dataset={bundle.dataset_name} model={model_name} "
        f"architecture={architecture} "
        f"continuous_state={continuous_state} "
        f"precision={precision} compile={training.compile} "
        f"subset={bundle.subset} features={bundle.preprocessor.output_dim} "
        f"train_windows={len(bundle.train_dataset)} "
        f"val_windows={len(bundle.val_dataset)} test_windows={len(test_dataset)}"
    )
    print(format_model_complexity(model_complexity))
    fit_result = fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        epochs=training.epochs,
        output_dir=output_dir,
        checkpoint_metadata=checkpoint_metadata,
        patience=training.patience,
        min_delta=training.min_delta,
        grad_clip=training.grad_clip,
        max_train_batches=training.max_train_batches,
        max_val_batches=training.max_val_batches,
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
        epochs_without_improvement=stale_epochs,
        history=history,
        precision=precision,
    )
    save_json(
        output_dir / "history.json",
        {
            "experiment_name": config.experiment.name,
            "dataset_name": bundle.dataset_name,
            "model_name": model_name,
            "model_architecture": architecture,
            "continuous_state": continuous_state,
            "precision": precision,
            "compile": training.compile,
            "best_epoch": fit_result["best_epoch"],
            "best_val_loss": fit_result["best_val_loss"],
            "stopped_early": fit_result["stopped_early"],
            "history": fit_result["history"],
        },
    )

    best_path = Path(fit_result["best_checkpoint"])
    if not best_path.is_file():
        resumed_best = training.resume.parent / "best.pt" if training.resume else None
        best_path = (
            resumed_best
            if resumed_best is not None and resumed_best.is_file()
            else Path(fit_result["last_checkpoint"])
        )
    restore_checkpoint(best_path, model, device=device)
    evaluation_limit = (
        evaluation.max_test_engines
        if bundle.evaluation_spec.protocol == "endpoint_per_entity"
        else evaluation.max_test_batches
    )
    test_result, predictions_path, prediction_count = _evaluate_test(
        model,
        test_loader,
        device,
        bundle,
        output_dir,
        evaluation_limit,
        training.plots,
        precision,
    )
    metrics_document = {
        "checkpoint": str(best_path.resolve()),
        "experiment_name": config.experiment.name,
        "dataset_name": bundle.dataset_name,
        "model_name": model_name,
        "model_architecture": architecture,
        "continuous_state": continuous_state,
        "precision": precision,
        "compile": training.compile,
        "subset": bundle.subset,
        "num_predictions": prediction_count,
        "predictions": str(predictions_path.resolve()),
        "rul_cap": data_settings.rul_cap,
        "label_policy": dict(bundle.label_policy),
        "evaluation_protocol": bundle.evaluation_spec.protocol,
        "model_complexity": model_complexity,
        "metrics": test_result,
    }
    save_json(output_dir / "test_metrics.json", metrics_document)
    if training.plots:
        plot_training_history(
            output_dir / "history.json", output_dir / "training_history.png"
        )
    print(json.dumps(metrics_document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
