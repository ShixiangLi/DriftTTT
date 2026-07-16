"""Restore a C-MAPSS checkpoint and evaluate the official per-engine test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from data.cmapss import CmapssPreprocessor, prepare_cmapss_test_dataset
from utils.complexity import estimate_model_complexity, format_model_complexity
from utils.config import ExperimentConfig, load_experiment_config
from utils.engine import (
    build_model,
    evaluate_by_engine,
    load_checkpoint,
    resolve_device,
    rul_label_policy,
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
        raise ValueError(f"Checkpoint is missing {key}")
    return value


def main() -> None:
    config = parse_config()
    evaluation = config.evaluation
    checkpoint_path = config.evaluation_checkpoint
    checkpoint = load_checkpoint(checkpoint_path, "cpu")
    config.verify_checkpoint_identity(checkpoint)
    data_config = _required_mapping(checkpoint, "data_config")
    model_config = _required_mapping(checkpoint, "model_config")
    model_name = str(checkpoint.get("model_name", "ttt"))

    data_dir_value = config.data.data_dir
    if data_dir_value is None:
        raise ValueError("No data directory configured; set data.data_dir")
    required_data_keys = {
        "subset",
        "window_size",
        "stride",
        "val_fraction",
        "seed",
        "rul_cap",
        "variance_threshold",
    }
    missing = required_data_keys.difference(data_config)
    if missing:
        raise ValueError(f"Checkpoint data_config is missing {sorted(missing)}")

    set_seed(config.training.seed, deterministic=config.training.deterministic)
    device = resolve_device(evaluation.device)
    preprocessor_state = _required_mapping(checkpoint, "preprocessor")
    preprocessor = CmapssPreprocessor.from_state_dict(preprocessor_state)
    test_dataset = prepare_cmapss_test_dataset(
        data_dir=data_dir_value,
        subset=data_config["subset"],
        preprocessor=preprocessor,
        window_size=int(data_config["window_size"]),
        rul_cap=data_config["rul_cap"],
    )
    verify_test_provenance(test_dataset, preprocessor, checkpoint)
    if int(model_config.get("input_dim", -1)) != preprocessor.output_dim:
        raise RuntimeError("Checkpoint input_dim does not match prepared features")

    model = build_model(model_name, model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model_complexity = estimate_model_complexity(
        model, int(data_config["window_size"])
    )
    print(format_model_complexity(model_complexity))
    generator = torch.Generator().manual_seed(config.training.seed)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=evaluation.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        worker_init_fn=seed_worker if evaluation.num_workers else None,
        generator=generator,
        persistent_workers=evaluation.num_workers > 0,
    )
    result = evaluate_by_engine(
        model,
        test_loader,
        device,
        max_engines=evaluation.max_test_engines,
        include_predictions=True,
    )
    predictions = result.pop("predictions")
    document = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "experiment_name": config.experiment.name,
        "model_name": model_name,
        "subset": str(data_config["subset"]),
        "num_engines": len(predictions),
        "rul_cap": data_config["rul_cap"],
        "label_policy": rul_label_policy(data_config["rul_cap"]),
        "model_complexity": model_complexity,
        "metrics": result,
    }

    output = evaluation.output or checkpoint_path.parent / "evaluation.json"
    predictions_output = (
        evaluation.predictions_output
        or output.with_name(f"{output.stem}_predictions.json")
    )
    save_json(output, document)
    save_json(predictions_output, predictions)
    if evaluation.plots:
        plot_rul_predictions(
            predictions_output,
            evaluation.plot_output or predictions_output.with_suffix(".png"),
        )
    print(json.dumps(document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
