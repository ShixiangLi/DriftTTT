"""Matplotlib visualizations for C-MAPSS training and endpoint evaluation."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


JsonSource = str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]]


def _load_json(source: JsonSource) -> Any:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Visualization input not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))
    return source


def _load_prediction_records(source: JsonSource, limit: int = 5000) -> Any:
    if not isinstance(source, (str, Path)) or Path(source).suffix.lower() != ".jsonl":
        return _load_json(source)
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Visualization input not found: {path}")
    rng = random.Random(0)
    reservoir: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            record = json.loads(line)
            if index < limit:
                reservoir.append(record)
            else:
                replacement = rng.randint(0, index)
                if replacement < limit:
                    reservoir[replacement] = record
    return reservoir


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _output_path(path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def plot_training_history(
    history: JsonSource,
    output_path: str | Path,
) -> Path:
    """Plot training/validation MSE and RMSE from ``history.json``."""
    document = _load_json(history)
    records = document.get("history") if isinstance(document, Mapping) else document
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("Training history must contain a sequence under 'history'.")
    if not records:
        raise ValueError("Training history is empty.")

    epochs: list[int] = []
    train_loss: list[float] = []
    val_loss: list[float] = []
    train_rmse: list[float] = []
    val_rmse: list[float] = []
    train_feature_rmse: list[float] = []
    val_feature_rmse: list[float] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError("Every history record must be an object.")
        train = record.get("train")
        val = record.get("val")
        if not isinstance(train, Mapping) or not isinstance(val, Mapping):
            raise ValueError("Every history record requires train and val metrics.")
        epochs.append(int(record.get("epoch", index)) + 1)
        train_loss.append(float(train["loss"]))
        val_loss.append(float(val["loss"]))
        train_rmse.append(float(train["rmse"]))
        val_rmse.append(float(val["rmse"]))
        if "feature_rmse" in train or "feature_rmse" in val:
            if "feature_rmse" not in train or "feature_rmse" not in val:
                raise ValueError(
                    "Autoregressive history requires train and val feature_rmse."
                )
            train_feature_rmse.append(float(train["feature_rmse"]))
            val_feature_rmse.append(float(val["feature_rmse"]))

    arrays = (
        train_loss,
        val_loss,
        train_rmse,
        val_rmse,
        train_feature_rmse,
        val_feature_rmse,
    )
    if not all(np.isfinite(values).all() for values in arrays):
        raise ValueError("Training history contains non-finite metrics.")

    plt = _pyplot()
    autoregressive = bool(train_feature_rmse)
    columns = 3 if autoregressive else 2
    fig, axes = plt.subplots(
        1, columns, figsize=(5.5 * columns, 4.2), constrained_layout=True
    )
    axes[0].plot(
        epochs, train_loss, label="Train", linewidth=2, marker="o", markersize=4
    )
    axes[0].plot(
        epochs, val_loss, label="Validation", linewidth=2, marker="o", markersize=4
    )
    axes[0].set(title="MSE Loss", xlabel="Epoch", ylabel="MSE")
    axes[1].plot(
        epochs, train_rmse, label="Train", linewidth=2, marker="o", markersize=4
    )
    axes[1].plot(
        epochs, val_rmse, label="Validation", linewidth=2, marker="o", markersize=4
    )
    axes[1].set(title="RMSE", xlabel="Epoch", ylabel="Cycles")
    if autoregressive:
        axes[2].plot(
            epochs,
            train_feature_rmse,
            label="Train",
            linewidth=2,
            marker="o",
            markersize=4,
        )
        axes[2].plot(
            epochs,
            val_feature_rmse,
            label="Validation",
            linewidth=2,
            marker="o",
            markersize=4,
        )
        axes[2].set(
            title="Next-feature RMSE",
            xlabel="Epoch",
            ylabel="Standardized feature value",
        )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    destination = _output_path(output_path)
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return destination


def plot_rul_predictions(
    predictions: JsonSource,
    output_path: str | Path,
) -> Path:
    """Plot endpoint predictions by engine and as a target/prediction parity plot."""
    records = _load_prediction_records(predictions)
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("Predictions must be a sequence of records.")
    if not records:
        raise ValueError("Predictions are empty.")

    entity_key = "engine_id" if "engine_id" in records[0] else "entity_id"
    time_key = "cycle" if "cycle" in records[0] else "time_index"
    ordered = sorted(
        records,
        key=lambda row: (int(row[entity_key]), int(row.get(time_key, 0))),
    )
    entity_ids = np.asarray([int(row[entity_key]) for row in ordered])
    targets = np.asarray([float(row["target"]) for row in ordered], dtype=np.float64)
    predicted = np.asarray(
        [float(row["prediction"]) for row in ordered], dtype=np.float64
    )
    if not np.isfinite(targets).all() or not np.isfinite(predicted).all():
        raise ValueError("Predictions contain non-finite target or prediction values.")

    rmse = float(np.sqrt(np.mean(np.square(predicted - targets))))
    mae = float(np.mean(np.abs(predicted - targets)))
    lower = float(min(targets.min(), predicted.min()))
    upper = float(max(targets.max(), predicted.max()))
    if lower == upper:
        lower -= 1.0
        upper += 1.0

    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    x_values = (
        entity_ids
        if np.unique(entity_ids).size == len(entity_ids)
        else np.arange(len(ordered))
    )
    axes[0].plot(x_values, targets, label="Target", linewidth=1.3)
    axes[0].plot(x_values, predicted, label="Prediction", linewidth=1.1)
    if np.unique(entity_ids).size == len(entity_ids):
        title, xlabel = "Endpoint RUL by Entity", "Entity ID"
    else:
        title, xlabel = "RUL Prediction Samples", "Sample"
    axes[0].set(title=title, xlabel=xlabel, ylabel="RUL")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].scatter(targets, predicted, s=24, alpha=0.7)
    axes[1].plot([lower, upper], [lower, upper], linestyle="--", color="black")
    axes[1].set(
        title=f"Prediction Parity  RMSE={rmse:.2f}  MAE={mae:.2f}",
        xlabel="Target RUL",
        ylabel="Predicted RUL",
        xlim=(lower, upper),
        ylim=(lower, upper),
    )
    axes[1].grid(alpha=0.25)
    destination = _output_path(output_path)
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return destination


def create_run_visualizations(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
) -> list[Path]:
    """Generate every visualization supported by files in a training run."""
    run_path = Path(run_dir)
    destination = Path(output_dir) if output_dir is not None else run_path
    outputs: list[Path] = []

    history_path = run_path / "history.json"
    if history_path.is_file():
        outputs.append(
            plot_training_history(history_path, destination / "training_history.png")
        )
    predictions_path = run_path / "test_predictions.json"
    if not predictions_path.is_file():
        predictions_path = run_path / "test_predictions.jsonl"
    if predictions_path.is_file():
        outputs.append(
            plot_rul_predictions(predictions_path, destination / "test_predictions.png")
        )
    if not outputs:
        raise FileNotFoundError(
            f"No history.json or test_predictions.json found in {run_path}"
        )
    return outputs


__all__ = [
    "create_run_visualizations",
    "plot_rul_predictions",
    "plot_training_history",
]
