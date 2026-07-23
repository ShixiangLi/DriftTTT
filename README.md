# DriftTTT

DriftTTT is a PyTorch project for remaining-useful-life (RUL) prediction on
NASA C-MAPSS and N-CMAPSS. It uses one Transformer encoder backbone with a
configuration-selected sequence mixer. The current mixers are standard
self-attention, a test-time-training MLP (`ttt_mlp`), and a fixed-rank
multiscale TTT mixture of experts (`ttt_multiscale_moe`).

## Current model

The model maps input windows from `[B,L,F]` to `[B,L,d_model]`, adds dynamic
sinusoidal positions, and applies shared Transformer blocks. Every block has
the same normalization, residual, feed-forward, and masking path; only its
sequence mixer changes. The last valid timestep is normalized and passed to a
scalar RUL regression head.

Select the mixer in either dataset configuration:

```yaml
model:
  sequence_mixer: ttt_multiscale_moe
```

`attention` uses standard multi-head self-attention. `ttt_mlp` projects
queries, keys, and values per head and treats a two-layer MLP as sample-local
fast state. Its label-free inner objective always reconstructs `value - key`
from the same-time key. For each configured chunk, it takes one differentiable
inner gradient step and applies the updated MLP to queries. Fast weights start
from learned outer parameters, remain separate for every sample, and are
discarded after each forward call. They are never written back or shared
across batches. TTT inner updates stay in FP32 under mixed precision.

`ttt_multiscale_moe` partitions the same fast-MLP hidden rank between a short
expert and a long expert instead of constructing two complete networks. The
short expert receives the base-token high-frequency residual and adapts at the
observation clock. Independent integer cycle metadata groups observations into
physical cycle states for the long expert; it is never taken from the
normalized cycle feature. Cycle means can be smoothed using the actual cycle
gap, and each long result is mapped back to its source observations. A
lightweight per-head gate fuses both outputs. QKV, output projections, and the
total fast-MLP rank remain shared and fixed. Both experts use the same stable
reconstruction objective at their respective clocks. The final gated long
contribution can be centered over valid observations, so it expresses relative
degradation shape without replacing the absolute-health query path. A sample
with fewer than two observed cycles automatically bypasses the long correction.
Short- and long-expert fast states are sample-local and discarded after every
window.

All TTT-MLP settings live in the same YAML:

```yaml
model:
  ttt:
    hidden_multiplier: 2.0
    inner_learning_rate: 0.1
    chunk_size: 16
    inner_gradient_clip: 1.0
    activation: silu
    qkv_bias: true
    multiscale:
      short_rank_ratio: 0.5
      long_ema_decay: 0.9
      long_update_interval: 3
      long_inner_learning_rate: 0.025
      center_long_residual: true
```

The common `inner_learning_rate` and `chunk_size` configure short-expert
updates. `long_update_interval` is the number of cycle states in one long
update. `long_ema_decay` is applied per lifecycle-cycle gap rather than per raw
observation; zero uses observed cycle means directly. N-CMAPSS uses zero
because cycle aggregation is already a strong low-pass operation, while
C-MAPSS retains cross-cycle EMA smoothing. `center_long_residual=false`
provides the absolute-offset ablation. These multiscale values are ignored by
`attention` and `ttt_mlp`.

All TTT mixers use only the label-free same-coordinate reconstruction task
during their inner update. RUL labels are used exclusively by the outer
regression loss. Evaluation reports RMSE, MAE, MSE, and the asymmetric NASA
score. Checkpoints contain model and optimizer states, the normalized
configuration, feature schema, fitted preprocessing statistics, and entity
split IDs.

## Data processing

### C-MAPSS

Place the official files under `dataset/cmapss/`. All four subsets are
supported: FD001, FD002, FD003, and FD004.

- Official training engines are split into disjoint train/validation engines.
- Variance selection and standardization are fitted only on training engines.
- Windows never cross engine boundaries.
- Training and validation labels are `max_cycle - current_cycle`.
- The configured piecewise RUL cap is applied consistently to every split;
  targets are divided by the cap for optimization and restored for reporting.
- `options.include_cycle` adds the observed lifecycle cycle as a feature; its
  normalization is fitted only on training engines.
- Integer cycle IDs are also returned as non-feature metadata for the
  cycle-aware long expert, regardless of `include_cycle`.
- Official testing uses the final observed window and supplied `RUL_FD*.txt`
  target for each test engine.

Trajectory columns are interpreted as engine ID, cycle, three operating
settings, and 21 sensor measurements. Engine ID is never a model feature.

### N-CMAPSS

Place the official HDF5 files under `dataset/n-cmapss/`. A subset is loaded
lazily, so feature windows are read on demand rather than materializing a
multi-gigabyte file in memory.

- Development units are split into disjoint train/validation units.
- Official test units never contribute preprocessing statistics.
- Statistics are fitted incrementally from chunks of training-unit rows.
- The default observed inputs are `W` and `X_s`; `X_v` may be enabled.
- Health parameters in `T` are rejected as inputs to avoid target leakage.
- Windows stay inside a unit and can downsample the original 1 Hz stream.
- `options.include_cycle` adds the observed flight cycle from `A` as a
  train-statistics-normalized lifecycle-position feature.
- The same raw cycle column is returned separately as integer metadata for
  cycle grouping; it is never normalized or exposed as an extra feature.
- `data.options.include_partial_windows` controls one shared train/validation/test
  policy and defaults to `false`, so all splits use complete windows.
- Test predictions cover each unit trajectory at `evaluation_stride`, plus its
  final row, and are streamed to JSON Lines.

The adapter checks required datasets, aligned row counts, feature names, and
contiguous unit spans before training.

RMSE, MAE, MSE, NASA Score, prediction files, and plots are all reported in
the original RUL unit even though capped labels are normalized during training.

The current local `N-CMAPSS_DS08d-010.h5` file has inconsistent HDF5 end-of-file
metadata and is not usable. An `all` batch skips unreadable files with a
diagnostic. The reference configuration uses DS02-006 and is unaffected.

## Setup

The existing virtual environment can be used directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Python 3.10 or newer and the direct dependencies in `pyproject.toml` are
supported.

## Training

C-MAPSS FD004 experiment:

```powershell
.\.venv\Scripts\python.exe -m scripts.train --config configs\cmapss_transformer.yaml
```

N-CMAPSS DS02 experiment:

```powershell
.\.venv\Scripts\python.exe -m scripts.train --config configs\ncmapss_transformer.yaml
```

### Batch experiments

The repository-root launchers build a Cartesian product of selected subsets
and sequence mixers. Run either launcher without arguments for interactive
prompts:

```powershell
.\run_experiments.bat
```

```bash
bash run_experiments.sh
```

Selections can also be passed directly. For example, compare the original and
multiscale TTT mixers on two C-MAPSS subsets:

```powershell
.\run_experiments.bat --dataset cmapss --subsets FD001,FD002 --mixers ttt_mlp,ttt_multiscale_moe
```

```bash
bash run_experiments.sh --dataset cmapss --subsets FD001,FD002 --mixers ttt_mlp,ttt_multiscale_moe
```

Run TTT-MLP on every usable N-CMAPSS file:

```powershell
.\run_experiments.bat --dataset ncmapss --subsets all --mixers ttt_mlp
```

On a four-GPU server, add `--gpus 0,1,2,3` to run at most four independent
experiments concurrently, with one process isolated to each GPU:

```bash
bash run_experiments.sh --dataset cmapss --subsets all --mixers attention,ttt_mlp --gpus 0,1,2,3
```

If memory permits multiple experiments on each GPU, set a numeric concurrency
or use `all`. For the eight C-MAPSS combinations below, `all` resolves to two
jobs per GPU and starts the complete matrix concurrently:

```bash
bash run_experiments.sh --dataset cmapss --subsets all --mixers attention,ttt_mlp --gpus 0,1,2,3 --jobs-per-gpu all
```

Use `--jobs-per-gpu 2` for an explicit two jobs per GPU. This is a trust-based
capacity setting rather than dynamic memory reservation; choose it only after
confirming peak memory usage. Running many N-CMAPSS jobs together can also be
limited by HDF5 storage bandwidth and host memory even when GPU memory is free.

The same option works with the Windows launcher. Without `--gpus`, execution
remains serial, and `--gpus` alone defaults to one job per GPU. Parallel console
output is written to `train.log` inside each experiment directory so messages
from different processes do not interleave.
Keep `training.device` and `evaluation.device` set to `auto` (the reference
default); inside each isolated process the assigned physical GPU appears as
`cuda:0`.

Use `--dataset all --subsets all --mixers all` for the complete matrix, or add
`--dry-run` to inspect generated combinations without training. Each batch is
grouped under one timestamp directory:

```text
outputs/batches/YYYYMMDD_HHMMSS/
  configs/                        generated launch configurations
  cmapss_fd001_attention/         complete attention run
  cmapss_fd001_ttt_mlp/           complete TTT-MLP run
  cmapss_fd001_ttt_multiscale_moe/ complete multiscale TTT run
  ...
  summary.csv                     accuracy and complexity comparison
```

`summary.csv` records RMSE, MAE, MSE, NASA Score, observed cycles per window,
multi-cycle coverage, parameter counts, analytical per-sample MACs/FLOPs, and
differences or ratios relative to attention for the same dataset subset. The
cycle fields make long-expert activation coverage explicit. The MoE estimate
uses the full sequence length as a conservative upper bound for the
data-dependent slow sequence; it does not model Python-loop, synchronization,
or kernel-launch overhead. An unreadable N-CMAPSS
HDF5 file is rejected when selected explicitly and skipped with a diagnostic
message during an `all` run.

Change `model.sequence_mixer` in the same file to compare `attention`,
`ttt_mlp`, and `ttt_multiscale_moe` while retaining the identical data split,
model backbone, optimizer, metrics, and evaluation protocol. Also change
`experiment.name` and `experiment.output_dir` so runs do not overwrite one
another. The command trains the model, restores the best validation checkpoint,
and evaluates the official test split.

`training.precision: auto` selects BF16 on a compatible CUDA device and FP32
otherwise. Model parameters, optimization, and TTT fast-weight updates remain
FP32. Batch limits are available for integration smoke runs; metrics from
partial runs are not benchmark-comparable.

To resume, set `training.resume` to the prior `last.pt`, retain the same output
directory, and increase `training.epochs` to the desired total epoch count.

## Evaluation and visualization

Evaluate the configured checkpoint, or `best.pt` in the output directory when
`evaluation.checkpoint` is null:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate --config configs\cmapss_transformer.yaml
```

Regenerate plots from saved JSON/JSONL output:

```powershell
.\.venv\Scripts\python.exe -m scripts.visualize --run-dir outputs\cmapss_fd004_ttt_multiscale_moe
```

Each completed run contains:

```text
best.pt                 best validation-MSE checkpoint
last.pt                 latest checkpoint and optimizer state
config.yaml             normalized run configuration
history.json            epoch-level training and validation metrics
test_metrics.json       RUL metrics and complexity summary
test_predictions.json   C-MAPSS endpoint predictions
test_predictions.jsonl  N-CMAPSS streamed trajectory predictions
training_history.png    loss/RMSE curves
test_predictions.png    prediction sequence and parity plots
```

## Project layout

```text
configs/
  cmapss_transformer.yaml
  ncmapss_transformer.yaml
data/
  base.py               shared dataset bundle contract
  preprocessing.py      streaming statistics and feature scaler
  cmapss.py             C-MAPSS parsing, split, labels, and windows
  ncmapss.py            lazy HDF5 schema, split, scaling, and windows
  registry.py           explicit adapter selection
models/
  rul_transformer.py    shared Transformer blocks and mixer registry
  ttt_layer.py          shared TTT core, standard MLP, and multiscale MoE
utils/
  config.py             strict YAML loading and validation
  complexity.py         parameter and analytical operation counts
  metrics.py            regression and NASA metrics
  engine.py             training, evaluation, and checkpoints
  visualization.py      training and prediction figures
scripts/
  train.py
  evaluate.py
  visualize.py
```
