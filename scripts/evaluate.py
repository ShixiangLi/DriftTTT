"""Restore a checkpoint and evaluate its configured RUL test set."""

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
    load_checkpoint,
    resolve_device,
    resolve_precision,
    save_json,
    seed_worker,
    set_seed,
    verify_test_provenance,
)
from utils.visualization import plot_rul_predictions


def parse_config() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return load_experiment_config(parser.parse_args().config)


def _required_mapping(checkpoint: dict[str, Any], key: str) -> dict[str, Any]:
    value = checkpoint.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint is missing {key}.")
    return value


def main() -> None:
    config = parse_config()
    evaluation = config.evaluation
    adapter = get_dataset_adapter(config.data.name)
    checkpoint_path = config.evaluation_checkpoint
    checkpoint = load_checkpoint(checkpoint_path, "cpu")
    config.verify_checkpoint_identity(checkpoint)
    adapter.validate_checkpoint(config.data, checkpoint)
    model_config = _required_mapping(checkpoint, "model_config")
    preprocessor_state = _required_mapping(checkpoint, "preprocessor")
    model_name = str(checkpoint.get("model_name", "ttt"))

    set_seed(config.training.seed, deterministic=config.training.deterministic)
    device = resolve_device(evaluation.device)
    precision = resolve_precision(config.training.precision, device)
    bundle = adapter.prepare_evaluation(config.data, preprocessor_state)
    verify_test_provenance(bundle.test_dataset, bundle.preprocessor, checkpoint)
    if int(model_config.get("input_dim", -1)) != bundle.preprocessor.output_dim:
        raise RuntimeError("Checkpoint input_dim does not match prepared features.")

    model = build_model(model_name, model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    compile_model(model, config.training.compile)
    architecture = str(getattr(model, "architecture", "encoder"))
    continuous_state = bool(getattr(model, "continuous_state", False))
    test_dataset = bundle.test_dataset
    if continuous_state:
        continuous_view = getattr(test_dataset, "continuous_evaluation_view", None)
        if callable(continuous_view):
            test_dataset = continuous_view()
    model_complexity = estimate_model_complexity(model, config.data.window_size)
    print(format_model_complexity(model_complexity))
    generator = torch.Generator().manual_seed(config.training.seed)
    spec = bundle.evaluation_spec
    batch_size = evaluation.batch_size or spec.batch_size
    if (
        getattr(test_dataset, "requires_batch_size_one", False)
        and batch_size != 1
    ):
        raise ValueError(
            "continuous C-MAPSS trajectory evaluation requires batch_size=1"
        )
    worker_options = {"prefetch_factor": 4} if evaluation.num_workers else {}
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=evaluation.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        worker_init_fn=seed_worker if evaluation.num_workers else None,
        generator=generator,
        persistent_workers=evaluation.num_workers > 0,
        **worker_options,
    )
    print(f"device={device} precision={precision} compile={config.training.compile}")

    output = evaluation.output or checkpoint_path.parent / "evaluation.json"
    if spec.protocol == "endpoint_per_entity":
        result = evaluate_dataset(
            model,
            test_loader,
            device,
            max_batches=evaluation.max_test_engines,
            reset_each_batch=spec.reset_each_batch,
            require_single_item=spec.require_single_item,
            include_predictions=True,
            precision=precision,
        )
        predictions = result.pop("predictions")
        predictions_output = evaluation.predictions_output or output.with_name(
            f"{output.stem}_predictions.json"
        )
        save_json(predictions_output, predictions)
        prediction_count = len(predictions)
    else:
        predictions_output = evaluation.predictions_output or output.with_name(
            f"{output.stem}_predictions.jsonl"
        )
        with JsonlPredictionWriter(predictions_output) as writer:
            result = evaluate_dataset(
                model,
                test_loader,
                device,
                max_batches=evaluation.max_test_batches,
                reset_each_batch=spec.reset_each_batch,
                require_single_item=spec.require_single_item,
                prediction_sink=writer.write,
                precision=precision,
            )
        prediction_count = writer.count

    document = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "experiment_name": config.experiment.name,
        "dataset_name": bundle.dataset_name,
        "model_name": model_name,
        "model_architecture": architecture,
        "continuous_state": continuous_state,
        "precision": precision,
        "compile": config.training.compile,
        "subset": bundle.subset,
        "num_predictions": prediction_count,
        "predictions": str(Path(predictions_output).resolve()),
        "rul_cap": config.data.rul_cap,
        "label_policy": dict(bundle.label_policy),
        "evaluation_protocol": spec.protocol,
        "model_complexity": model_complexity,
        "metrics": result,
    }
    save_json(output, document)
    if evaluation.plots:
        plot_rul_predictions(
            predictions_output,
            evaluation.plot_output or Path(predictions_output).with_suffix(".png"),
        )
    print(json.dumps(document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
