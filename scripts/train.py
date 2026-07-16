"""Train and test a configured RUL Transformer on one C-MAPSS subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from data.cmapss import CmapssPreprocessor, prepare_cmapss_datasets
from utils.complexity import estimate_model_complexity, format_model_complexity
from utils.config import ExperimentConfig, load_experiment_config
from utils.engine import (
    build_model,
    evaluate_by_engine,
    fit,
    load_checkpoint,
    resolve_device,
    restore_checkpoint,
    rul_label_policy,
    save_json,
    seed_worker,
    set_seed,
    verify_data_provenance,
)
from utils.visualization import plot_rul_predictions, plot_training_history


def parse_config() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return load_experiment_config(parser.parse_args().config)


def _checkpoint_data_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    config = checkpoint.get("data_config")
    if not isinstance(config, dict):
        raise ValueError("Resume checkpoint is missing data_config")
    required = {
        "subset",
        "window_size",
        "stride",
        "val_fraction",
        "seed",
        "rul_cap",
        "variance_threshold",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Resume checkpoint data_config is missing {sorted(missing)}")
    return config


def _make_loader(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
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
    )


def _resolve_output_dir(
    requested: Path | None,
    resume: Path | None,
    subset: str,
) -> Path:
    if resume is not None:
        checkpoint_dir = resume.parent.resolve()
        if requested is not None and requested.resolve() != checkpoint_dir:
            raise ValueError(
                "Resumed training must write to the checkpoint directory so the "
                "historical best.pt remains available"
            )
        return checkpoint_dir
    return requested if requested is not None else Path("outputs") / subset


def _resolve_data_dir(requested: Path | None, checkpoint_value: Any = None) -> str:
    if requested is not None:
        return str(requested.resolve())
    if checkpoint_value is not None:
        return str(checkpoint_value)
    return str(Path("dataset/cmapss").resolve())


def main() -> None:
    config = parse_config()
    data_settings = config.data
    model_settings = config.model
    training = config.training
    evaluation = config.evaluation

    device = resolve_device(training.device)
    resume_payload = load_checkpoint(training.resume, "cpu") if training.resume else None
    if training.resume is not None:
        _resolve_output_dir(
            config.experiment.output_dir, training.resume, data_settings.subset
        )

    if resume_payload is None:
        data_config: dict[str, Any] = {
            **data_settings.checkpoint_values(),
        }
        model_name = model_settings.type
    else:
        config.verify_checkpoint_identity(resume_payload)
        data_config = dict(_checkpoint_data_config(resume_payload))
        # The dataset may be relocated without changing its preprocessing provenance.
        data_config["data_dir"] = _resolve_data_dir(
            data_settings.data_dir, data_config.get("data_dir")
        )
        model_name = str(resume_payload.get("model_name", "ttt"))

    set_seed(training.seed, deterministic=training.deterministic)
    if not 0.0 < float(data_config["val_fraction"]) < 1.0:
        raise ValueError("Checkpoint val_fraction must be in (0, 1)")
    restored_preprocessor = None
    if resume_payload is not None:
        preprocessor_state = resume_payload.get("preprocessor")
        if not isinstance(preprocessor_state, dict):
            raise ValueError("Resume checkpoint is missing preprocessor metadata")
        restored_preprocessor = CmapssPreprocessor.from_state_dict(preprocessor_state)
    bundle = prepare_cmapss_datasets(
        data_dir=data_config["data_dir"],
        subset=data_config["subset"],
        window_size=int(data_config["window_size"]),
        stride=int(data_config["stride"]),
        val_fraction=float(data_config["val_fraction"]),
        seed=int(data_config["seed"]),
        rul_cap=data_config["rul_cap"],
        variance_threshold=float(data_config["variance_threshold"]),
        preprocessor=restored_preprocessor,
    )

    if resume_payload is None:
        requested_model_config = model_settings.constructor_values(
            bundle.preprocessor.output_dim
        )
    else:
        verify_data_provenance(bundle, resume_payload)
        requested_model_config = dict(resume_payload.get("model_config", {}))
        if not requested_model_config:
            raise ValueError("Resume checkpoint is missing model_config")
        if int(requested_model_config["input_dim"]) != bundle.preprocessor.output_dim:
            raise RuntimeError("Checkpoint input_dim does not match prepared features")

    model = build_model(model_name, requested_model_config).to(device)
    model_config = model.get_config() if hasattr(model, "get_config") else requested_model_config
    model_complexity = estimate_model_complexity(
        model, int(data_config["window_size"])
    )
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
            raise ValueError("Resume checkpoint is missing optimizer_state_dict")
        optimizer.load_state_dict(optimizer_state)
        start_epoch = int(resume_payload["epoch"]) + 1
        best_val_loss = float(resume_payload["best_val_loss"])
        stale_epochs = int(resume_payload.get("epochs_without_improvement", 0))
        history = list(resume_payload.get("history", []))
        if start_epoch >= training.epochs:
            raise ValueError(
                f"Checkpoint already reached epoch {start_epoch}; "
                "training.epochs must be larger"
            )

    output_dir = _resolve_output_dir(
        config.experiment.output_dir, training.resume, bundle.subset
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.yaml")

    pin_memory = device.type == "cuda"
    train_loader = _make_loader(
        bundle.train_dataset,
        batch_size=training.batch_size,
        shuffle=True,
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
    test_loader = _make_loader(
        bundle.test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=training.num_workers,
        pin_memory=pin_memory,
        seed=training.seed,
    )

    split_engine_ids = {
        "train": list(bundle.train_engine_ids),
        "val": list(bundle.val_engine_ids),
        "test": list(bundle.test_dataset.engine_ids),
    }
    checkpoint_metadata = {
        "model_name": model_name,
        "experiment_name": config.experiment.name,
        "experiment_config": config.to_dict(),
        "model_config": model_config,
        "data_config": data_config,
        "preprocessor": bundle.preprocessor.state_dict(),
        "split_engine_ids": split_engine_ids,
        "subset": bundle.subset,
        "window_size": int(data_config["window_size"]),
        "rul_cap": data_config["rul_cap"],
        "label_policy": rul_label_policy(data_config["rul_cap"]),
        "model_complexity": model_complexity,
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
        },
    }
    print(
        f"device={device} model={model_name} subset={bundle.subset} "
        f"features={bundle.preprocessor.output_dim} "
        f"train_windows={len(bundle.train_dataset)} val_windows={len(bundle.val_dataset)} "
        f"test_engines={len(bundle.test_dataset)}"
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
    )
    save_json(
        output_dir / "history.json",
        {
            "experiment_name": config.experiment.name,
            "model_name": model_name,
            "best_epoch": fit_result["best_epoch"],
            "best_val_loss": fit_result["best_val_loss"],
            "stopped_early": fit_result["stopped_early"],
            "history": fit_result["history"],
        },
    )

    best_path = Path(fit_result["best_checkpoint"])
    if not best_path.is_file():
        resumed_best = (
            training.resume.parent / "best.pt" if training.resume else None
        )
        if resumed_best is not None and resumed_best.is_file():
            best_path = resumed_best
        else:
            best_path = Path(fit_result["last_checkpoint"])
    restore_checkpoint(best_path, model, device=device)
    test_result = evaluate_by_engine(
        model,
        test_loader,
        device,
        max_engines=evaluation.max_test_engines,
        include_predictions=True,
    )
    predictions = test_result.pop("predictions")
    metrics_document = {
        "checkpoint": str(best_path.resolve()),
        "experiment_name": config.experiment.name,
        "model_name": model_name,
        "subset": bundle.subset,
        "num_engines": len(predictions),
        "rul_cap": data_config["rul_cap"],
        "label_policy": rul_label_policy(data_config["rul_cap"]),
        "model_complexity": model_complexity,
        "metrics": test_result,
    }
    save_json(output_dir / "test_metrics.json", metrics_document)
    save_json(output_dir / "test_predictions.json", predictions)
    if training.plots:
        plot_training_history(
            output_dir / "history.json", output_dir / "training_history.png"
        )
        plot_rul_predictions(
            output_dir / "test_predictions.json", output_dir / "test_predictions.png"
        )
    print(json.dumps(metrics_document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
