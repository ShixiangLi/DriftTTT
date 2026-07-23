from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import Any

import h5py

from models.rul_transformer import MIXER_BUILDERS
from utils.config import load_config, normalize_config, save_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIGS = {
    "cmapss": PROJECT_ROOT / "configs" / "cmapss_transformer.yaml",
    "ncmapss": PROJECT_ROOT / "configs" / "ncmapss_transformer.yaml",
}


def _comma_separated(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("Selection cannot be empty")
    return values


def _available_cmapss(config: dict[str, Any]) -> dict[str, Path]:
    root = PROJECT_ROOT / config["data"]["root"]
    available: dict[str, Path] = {}
    for train_path in sorted(root.glob("train_FD*.txt")):
        subset = train_path.stem.removeprefix("train_").upper()
        if (root / f"test_{subset}.txt").is_file() and (
            root / f"RUL_{subset}.txt"
        ).is_file():
            available[subset.lower()] = train_path
    return available


def _available_ncmapss(config: dict[str, Any]) -> dict[str, Path]:
    root = PROJECT_ROOT / config["data"]["root"]
    available: dict[str, Path] = {}
    for path in sorted(root.glob("N-CMAPSS_*.h5")):
        subset = path.stem.removeprefix("N-CMAPSS_")
        available[subset.lower()] = path
    return available


def _check_ncmapss(path: Path) -> str | None:
    try:
        with h5py.File(path, "r") as handle:
            required = {
                "A_dev",
                "A_test",
                "W_dev",
                "W_test",
                "X_s_dev",
                "X_s_test",
                "Y_dev",
                "Y_test",
            }
            missing = required - set(handle)
            if missing:
                return f"missing arrays: {sorted(missing)}"
    except (OSError, ValueError) as error:
        return str(error)
    return None


def _canonical_subset(dataset: str, path: Path) -> str:
    if dataset == "cmapss":
        return path.stem.removeprefix("train_").upper()
    return path.stem.removeprefix("N-CMAPSS_")


def _resolve_selections(
    dataset_selection: str, subset_selection: str, mixer_selection: str
) -> list[tuple[str, str, str]]:
    datasets = (
        ["cmapss", "ncmapss"] if dataset_selection == "all" else [dataset_selection]
    )
    configs = {name: load_config(BASE_CONFIGS[name]) for name in datasets}
    available = {
        name: (
            _available_cmapss(configs[name])
            if name == "cmapss"
            else _available_ncmapss(configs[name])
        )
        for name in datasets
    }
    if any(not values for values in available.values()):
        empty = [name for name, values in available.items() if not values]
        raise ValueError(f"No complete dataset files found for: {empty}")

    requested_subsets = _comma_separated(subset_selection)
    select_all_subsets = (
        len(requested_subsets) == 1 and requested_subsets[0].lower() == "all"
    )
    if not select_all_subsets:
        known = {key for values in available.values() for key in values}
        unknown = [value for value in requested_subsets if value.lower() not in known]
        if unknown:
            choices = sorted(
                _canonical_subset(name, path)
                for name, values in available.items()
                for path in values.values()
            )
            raise ValueError(f"Unknown subsets {unknown}; available: {choices}")

    requested_mixers = _comma_separated(mixer_selection)
    if len(requested_mixers) == 1 and requested_mixers[0].lower() == "all":
        mixers = list(MIXER_BUILDERS)
    else:
        mixers = [value.lower() for value in requested_mixers]
        unknown_mixers = sorted(set(mixers) - set(MIXER_BUILDERS))
        if unknown_mixers:
            raise ValueError(
                f"Unknown mixers {unknown_mixers}; available: {list(MIXER_BUILDERS)}"
            )

    experiments: list[tuple[str, str, str]] = []
    for dataset in datasets:
        if select_all_subsets:
            selected_paths = list(available[dataset].values())
        else:
            keys = {value.lower() for value in requested_subsets}
            selected_paths = [
                path for key, path in available[dataset].items() if key in keys
            ]
        for path in selected_paths:
            subset = _canonical_subset(dataset, path)
            if dataset == "ncmapss":
                problem = _check_ncmapss(path)
                if problem is not None:
                    if select_all_subsets:
                        print(f"Skipping unusable N-CMAPSS subset {subset}: {problem}")
                        continue
                    raise ValueError(f"N-CMAPSS subset {subset} is unusable: {problem}")
            experiments.extend((dataset, subset, mixer) for mixer in mixers)
    if not experiments:
        raise ValueError("The selections produced no runnable experiments")
    return experiments


def _new_batch_directory() -> Path:
    root = PROJECT_ROOT / "outputs" / "batches"
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root / timestamp
    suffix = 1
    while candidate.exists():
        candidate = root / f"{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _experiment_config(
    dataset: str, subset: str, mixer: str, batch_directory: Path
) -> dict[str, Any]:
    config = deepcopy(load_config(BASE_CONFIGS[dataset]))
    subset_slug = subset.lower().replace("-", "_")
    run_name = f"{dataset}_{subset_slug}_{mixer}"
    config["experiment"]["name"] = run_name
    relative_output = batch_directory.relative_to(PROJECT_ROOT) / run_name
    config["experiment"]["output_dir"] = relative_output.as_posix()
    config["data"]["subset"] = subset
    config["model"]["sequence_mixer"] = mixer
    config["training"]["resume"] = None
    config["evaluation"]["checkpoint"] = None
    return normalize_config(config)


def _write_summary_csv(
    completed: list[tuple[str, str, str, dict[str, Any]]],
    destination: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for dataset, subset, mixer, config in completed:
        output_dir = PROJECT_ROOT / config["experiment"]["output_dir"]
        metrics_path = output_dir / config["evaluation"]["metrics_file"]
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing experiment metrics: {metrics_path}")
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        complexity = metrics["complexity"]
        rows.append(
            {
                "dataset": dataset,
                "subset": subset,
                "mixer": mixer,
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "mse": metrics["mse"],
                "nasa_score": metrics["nasa_score"],
                "count": metrics["count"],
                "partial": metrics["partial"],
                "mean_cycles_per_window": metrics.get("mean_cycles_per_window", ""),
                "multi_cycle_fraction": metrics.get("multi_cycle_fraction", ""),
                "parameters": complexity["parameters"],
                "trainable_parameters": complexity["trainable_parameters"],
                "forward_macs_per_sample": complexity["forward_macs_per_sample"],
                "forward_flops_per_sample": complexity["forward_flops_per_sample"],
                "rmse_delta_vs_attention": "",
                "rmse_change_percent_vs_attention": "",
                "parameter_ratio_vs_attention": "",
                "mac_ratio_vs_attention": "",
                "output_dir": config["experiment"]["output_dir"],
            }
        )

    attention_rows = {
        (row["dataset"], row["subset"]): row
        for row in rows
        if row["mixer"] == "attention"
    }
    for row in rows:
        baseline = attention_rows.get((row["dataset"], row["subset"]))
        if baseline is None:
            continue
        rmse_delta = float(row["rmse"]) - float(baseline["rmse"])
        row["rmse_delta_vs_attention"] = rmse_delta
        row["rmse_change_percent_vs_attention"] = (
            100.0 * rmse_delta / float(baseline["rmse"])
        )
        row["parameter_ratio_vs_attention"] = float(row["parameters"]) / float(
            baseline["parameters"]
        )
        row["mac_ratio_vs_attention"] = float(row["forward_macs_per_sample"]) / float(
            baseline["forward_macs_per_sample"]
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_gpus(value: str) -> list[str]:
    gpus = _comma_separated(value)
    if len(gpus) != len(set(gpus)):
        raise ValueError("GPU identifiers must be unique")
    return gpus


def _jobs_per_gpu(value: str, experiment_count: int, gpu_count: int) -> int:
    if value.lower() == "all":
        return ceil(experiment_count / gpu_count)
    try:
        jobs = int(value)
    except ValueError as error:
        raise ValueError("jobs-per-gpu must be a positive integer or all") from error
    if jobs < 1:
        raise ValueError("jobs-per-gpu must be a positive integer or all")
    return jobs


def _training_command(config_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "scripts.train",
        "--config",
        str(config_path),
    ]


def _run_parallel(
    planned: list[tuple[str, str, str, dict[str, Any], Path]],
    gpus: list[str],
    jobs_per_gpu: int,
) -> None:
    available_gpus: queue.Queue[str] = queue.Queue()
    for gpu in gpus:
        for _ in range(jobs_per_gpu):
            available_gpus.put(gpu)

    def run_job(job: tuple[str, str, str, dict[str, Any], Path]) -> None:
        _, _, _, config, config_path = job
        run_name = config["experiment"]["name"]
        output_dir = PROJECT_ROOT / config["experiment"]["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "train.log"
        gpu = available_gpus.get()
        try:
            print(f"Starting {run_name} on GPU {gpu}; log={log_path}", flush=True)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            with log_path.open("w", encoding="utf-8") as log_handle:
                log_handle.write(f"CUDA_VISIBLE_DEVICES={gpu}\n")
                log_handle.flush()
                result = subprocess.run(
                    _training_command(config_path),
                    cwd=PROJECT_ROOT,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if result.returncode != 0:
                raise RuntimeError(
                    f"{run_name} failed with exit code {result.returncode}; "
                    f"see {log_path}"
                )
            print(f"Completed {run_name} on GPU {gpu}", flush=True)
        finally:
            available_gpus.put(gpu)

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=len(gpus) * jobs_per_gpu) as executor:
        futures = [executor.submit(run_job, job) for job in planned]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                failures.append(str(error))
    if failures:
        raise RuntimeError("Parallel experiments failed:\n" + "\n".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a matrix of dataset subsets and sequence mixers"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("cmapss", "ncmapss", "all"),
        help="Dataset adapter to run",
    )
    parser.add_argument(
        "--subsets",
        default="all",
        help="Comma-separated subsets, or all",
    )
    parser.add_argument(
        "--mixers",
        default="all",
        help="Comma-separated sequence mixers, or all",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate configurations without starting training",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated GPU IDs for parallel runs; omitted means serial",
    )
    parser.add_argument(
        "--jobs-per-gpu",
        default="1",
        help="Concurrent experiments per GPU, or all; default: 1",
    )
    arguments = parser.parse_args()
    try:
        experiments = _resolve_selections(
            arguments.dataset, arguments.subsets, arguments.mixers
        )
        gpus = None if arguments.gpus is None else _parse_gpus(arguments.gpus)
        if gpus is None:
            if arguments.jobs_per_gpu != "1":
                raise ValueError("--jobs-per-gpu requires --gpus")
            jobs_per_gpu = 1
        else:
            jobs_per_gpu = _jobs_per_gpu(
                arguments.jobs_per_gpu, len(experiments), len(gpus)
            )
    except ValueError as error:
        parser.error(str(error))

    batch_directory = _new_batch_directory()
    generated_root = batch_directory / "configs"
    planned: list[tuple[str, str, str, dict[str, Any], Path]] = []
    print(f"Selected {len(experiments)} experiment(s)")
    print(f"Batch directory: {batch_directory.relative_to(PROJECT_ROOT)}")
    for index, (dataset, subset, mixer) in enumerate(experiments, start=1):
        config = _experiment_config(dataset, subset, mixer, batch_directory)
        run_name = config["experiment"]["name"]
        config_path = generated_root / f"{run_name}.yaml"
        planned.append((dataset, subset, mixer, config, config_path))
        save_config(config, config_path)
        print(
            f"[{index}/{len(experiments)}] dataset={dataset} subset={subset} "
            f"mixer={mixer} output={config['experiment']['output_dir']}"
        )
    if not arguments.dry_run:
        if gpus is None:
            for _, _, _, _, config_path in planned:
                subprocess.run(
                    _training_command(config_path), cwd=PROJECT_ROOT, check=True
                )
        else:
            print(
                f"Parallel GPUs: {','.join(gpus)}; "
                f"jobs_per_gpu={jobs_per_gpu}; "
                f"max_parallel={len(gpus) * jobs_per_gpu}"
            )
            _run_parallel(planned, gpus, jobs_per_gpu)
    if not arguments.dry_run:
        summary_path = batch_directory / "summary.csv"
        completed = [
            (dataset, subset, mixer, config)
            for dataset, subset, mixer, config, _ in planned
        ]
        _write_summary_csv(completed, summary_path)
        print(f"Batch summary: {summary_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
