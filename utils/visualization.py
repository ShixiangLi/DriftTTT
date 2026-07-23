from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_history(history: list[dict[str, Any]], destination: str | Path) -> None:
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_mse"] for row in history], label="train")
    axes[0].plot(epochs, [row["validation_mse"] for row in history], label="validation")
    axes[0].set(title="MSE", xlabel="Epoch", ylabel="MSE")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, [row["train_rmse"] for row in history], label="train")
    axes[1].plot(
        epochs, [row["validation_rmse"] for row in history], label="validation"
    )
    axes[1].set(title="RMSE", xlabel="Epoch", ylabel="RUL")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_predictions(
    records: Iterable[dict[str, Any]], destination: str | Path
) -> None:
    rows = list(records)
    if not rows:
        return
    targets = np.asarray([row["target"] for row in rows], dtype=np.float64)
    predictions = np.asarray([row["prediction"] for row in rows], dtype=np.float64)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(targets, label="target", linewidth=1.2)
    axes[0].plot(predictions, label="prediction", linewidth=1.0)
    axes[0].set(title="Prediction sequence", xlabel="Sample", ylabel="RUL")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    lower = float(min(targets.min(), predictions.min()))
    upper = float(max(targets.max(), predictions.max()))
    axes[1].scatter(targets, predictions, s=10, alpha=0.55)
    axes[1].plot([lower, upper], [lower, upper], "--", color="black", linewidth=1)
    axes[1].set(title="Prediction parity", xlabel="Target RUL", ylabel="Predicted RUL")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def read_prediction_records(
    path: str | Path, limit: int = 5_000
) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix == ".jsonl":
        rows = []
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
                    if len(rows) >= limit:
                        break
        return rows
    with source.open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    return values[:limit]


def visualize_run(run_directory: str | Path) -> None:
    run_dir = Path(run_directory)
    history_path = run_dir / "history.json"
    if history_path.is_file():
        with history_path.open("r", encoding="utf-8") as handle:
            plot_history(json.load(handle), run_dir / "training_history.png")
    candidates = [
        run_dir / "test_predictions.json",
        run_dir / "test_predictions.jsonl",
    ]
    prediction_path = next((path for path in candidates if path.is_file()), None)
    if prediction_path is not None:
        plot_predictions(
            read_prediction_records(prediction_path), run_dir / "test_predictions.png"
        )
