# DriftTTT

An independent PyTorch project for remaining useful life (RUL) prediction on
NASA C-MAPSS and N-CMAPSS. It supports both the temporal TTT layer adapted from
the local `ViTTT/` reference source and a standard self-attention Transformer,
without importing that project at runtime. Both backbones support the original
encoder-only RUL architecture and a causal decoder with continuous next-feature
prediction. Dataset adapters share the model, training, checkpoint, metrics,
complexity, and visualization pipeline.

## Upstream TTT analysis

The reusable implementation is `ViTTT/ttt_block.py`; its copies under
`ViTTT/vittt/models/` and `ViTTT/dittt/` are byte-identical. Its interface is
`TTT(dim, num_heads).forward(x, h, w)`, with input and output shape `[B,N,C]`,
`N=h*w`, and head dimension `D=C/num_heads`.

The joint projection emits `3C+3D` values and feeds two inner models:

- simplified SwiGLU branch: q/k/v are `[B,H,N,D]`, with base weights w1/w2
  shaped `[1,H,D,D]`;
- depthwise-convolution branch: q/k/v are `[B,D,h,w]`, with base weights w3
  shaped `[D,1,3,3]`.

Each forward call derives one set of fast weights per sample from k/v using
the source's closed-form gradients, `g/(norm+1)` stabilization, and one inner
step. Fast weights are local tensors: they are not written back to parameters
and are not shared by batch items or later calls. The outer RUL loss still
backpropagates through the update formula into qkv, w1/w2/w3, and projection
parameters. ViTTT uses the layer as a pre-norm attention replacement:
`x = x + TTT(LayerNorm(x))`, followed by an FFN residual.

For time series, `models/ttt_layer.py` maps the effective `h=1,w=L`
operation to `Conv1d(kernel_size=3)`. This is numerically equivalent to the
only active center row of the source 3x3 kernel, while preserving the source
scale `9**-0.5 == 1/3`. Padding-aware inner updates are the only material API
extension. `torch.nn.init.trunc_normal_` replaces the sole `timm` dependency.

Decoder mode makes every temporal operation causal. TTT inner-gradient
statistics are accumulated from completed chunks, so queries in the current
chunk are adapted only by earlier chunks. Its temporal branch and positional
convolution also use only current and earlier values. `chunk_size: 1` gives
token-wise causal adaptation; larger chunks allow parallel computation inside
each chunk. With `continuous_state: true`, completed-chunk statistics and the
short convolution histories continue across calls. Partial chunks also continue
across window boundaries, so chunk semantics do not depend on evaluation stride.

The detection and segmentation copies were deliberately not used: their old
`TTTAttention` mutates `self.scale` in `forward`, so first and later calls can
behave differently.

## C-MAPSS inventory

`dataset/cmapss/` contains 12 ASCII, LF-terminated, headerless files. Trajectory
rows have exactly 26 whitespace-delimited values and trailing spaces; use
`sep=r"\s+"`, not a literal single-space separator. RUL files contain one
integer per test engine in ascending engine-ID order.

| Subset | Conditions | Fault modes | Train rows/engines | Test rows/engines |
| --- | ---: | ---: | ---: | ---: |
| FD001 | 1 | 1 | 20,631 / 100 | 13,096 / 100 |
| FD002 | 6 | 1 | 53,759 / 260 | 33,991 / 259 |
| FD003 | 1 | 2 | 24,720 / 100 | 16,596 / 100 |
| FD004 | 6 | 2 | 61,249 / 249 | 41,214 / 248 |

The single fault mode is HPC degradation; the two-mode subsets additionally
contain fan degradation. Each trajectory row is:

```text
engine_id cycle setting_1 setting_2 setting_3 sensor_1 ... sensor_21
```

`engine_id` and `cycle` are used for grouping, ordering, and labels, not as
model features. The 21 standard sensor fields are:

| Field | Operational variable |
| --- | --- |
| setting_1 | Altitude |
| setting_2 | Mach number |
| setting_3 | Throttle resolver angle (TRA) |

| Field | Standard name | Measurement |
| --- | --- | --- |
| sensor_1 | T2 | Fan inlet total temperature |
| sensor_2 | T24 | LPC outlet total temperature |
| sensor_3 | T30 | HPC outlet total temperature |
| sensor_4 | T50 | LPT outlet total temperature |
| sensor_5 | P2 | Fan inlet pressure |
| sensor_6 | P15 | Bypass-duct total pressure |
| sensor_7 | P30 | HPC outlet total pressure |
| sensor_8 | Nf | Physical fan speed |
| sensor_9 | Nc | Physical core speed |
| sensor_10 | epr | Engine pressure ratio, P50/P2 |
| sensor_11 | Ps30 | HPC outlet static pressure |
| sensor_12 | phi | Fuel-flow/Ps30 ratio |
| sensor_13 | NRf | Corrected fan speed |
| sensor_14 | NRc | Corrected core speed |
| sensor_15 | BPR | Bypass ratio |
| sensor_16 | farB | Burner fuel-air ratio |
| sensor_17 | htBleed | Bleed enthalpy |
| sensor_18 | Nf_dmd | Demanded fan speed |
| sensor_19 | PCNfR_dmd | Demanded corrected fan speed |
| sensor_20 | W31 | HPT coolant bleed |
| sensor_21 | W32 | LPT coolant bleed |

## Leakage controls

1. Official training engines are split by whole `engine_id` before windows are
   built. No engine can occur in both training and validation.
2. Variance filtering and `StandardScaler` are fitted only on rows from the
   training-engine split. Their complete state and fitting IDs are checkpointed.
3. Windows are indexed inside one engine trajectory and can never cross an
   engine boundary. Short test trajectories receive zero left padding plus a
   boolean mask.
4. Train/validation RUL at cycle `t` is `max_cycle(engine)-t`. The default
   piecewise target applies `min(RUL,125)` consistently to train, validation,
   and official test endpoint labels. Use `--rul-cap 0` for raw linear RUL.
5. Official testing uses exactly the last window of each test engine. RMSE,
   MAE, and NASA Score are therefore computed over one prediction per engine.
6. Test loading restores the checkpoint scaler and needs only `test_*.txt` and
   `RUL_*.txt`; it never fits on validation or test data.
7. Continuous TTT execution preserves chronological engine order, consumes only
   the non-overlapping suffix of each new window, and resets state exactly when
   `entity_id` changes. A finished engine cannot reappear later in the loader.

NASA Score uses `d=prediction-target`: under-prediction contributes
`exp(-d/13)-1`, over-prediction contributes `exp(d/10)-1`, summed over evaluated
predictions.

## N-CMAPSS

N-CMAPSS adapters read one `N-CMAPSS_DS*.h5` file lazily. The default observed
features are the four scenario descriptors in `W` plus the 14 measurements in
`X_s`. `X_v` can be enabled explicitly. The unobservable health parameters in
`T` are rejected as model features to prevent degradation-state leakage.

The official `dev` units are split into disjoint train/validation units and the
official `test` units are never used for preprocessing. Feature variances and
normalization statistics are fitted sequentially from training-unit HDF5
chunks. Window indices contain only compact unit spans and cumulative counts;
individual feature windows are loaded on demand. HDF5 files are schema-checked
before training, including aligned row counts and required variable names.

Unlike classic C-MAPSS endpoint testing, N-CMAPSS supplies full test
run-to-failure trajectories and one RUL label per 1 Hz sample. Its default
evaluation protocol therefore scores all test windows and streams predictions
to JSON Lines instead of retaining millions of records in memory.

Training-only degradation-stage filtering is label based, never row-count
based. For each training entity, `effective_rul / max_effective_rul` is used:

```yaml
data:
  train_rul_filter:
    enabled: true
    normalized_range: [0.0, 0.7]
```

This retains the RUL interval closest to failure through 70% of the entity's
effective label range. With a C-MAPSS cap of 125, it retains RUL 0 through 87.5
and excludes the capped 125 plateau. Validation and test trajectories always
remain complete. Set `[0.3, 1.0]` for the earlier-life label range or disable
the filter to preserve the original behavior.

## Setup

From this repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For an editable install from the direct dependencies in `pyproject.toml`:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

Or use any Python 3.10+ environment with the declared dependencies.

## Train

Training and evaluation accept one required YAML configuration and no individual
hyperparameter flags. Reference configurations are provided for both models:

```powershell
.\.venv\Scripts\python.exe -m scripts.train --config configs\cmapss_ttt.yaml
```

```powershell
.\.venv\Scripts\python.exe -m scripts.train --config configs\cmapss_transformer.yaml
```

Decoder-only TTT with autoregressive continuous-feature training:

```powershell
.\.venv\Scripts\python.exe -m scripts.train --config configs\cmapss_ttt_decoder.yaml
```

N-CMAPSS reference runs:

```powershell
.\.venv\Scripts\python.exe -m scripts.train --config configs\ncmapss_ttt.yaml
.\.venv\Scripts\python.exe -m scripts.train --config configs\ncmapss_transformer.yaml
```

The YAML sections are:

| Section | Contents |
| --- | --- |
| `experiment` | Run name and output directory |
| `data` | Adapter name, subset, window, RUL cap/filter, validation and sampling |
| `data.options` | Dataset-specific feature groups, downsampling and boundaries |
| `model` | Backbone type, encoder/decoder architecture, width, depth, heads |
| `model.autoregressive` | Next-feature loss type and objective weight |
| `model.ttt` | qkv bias, inner step, scale, CPE kernel, chunk and state mode |
| `training` | Optimizer, runtime precision/compilation, early stop, seed and limits |
| `evaluation` | Checkpoint, device, output files, engine limit, plots |

`data.split_seed` controls only the engine split. `training.seed` controls model
initialization, batch shuffling, and runtime randomness, so initialization seeds
can be varied while keeping exactly the same train/validation engines.

`training.precision` accepts `fp32`, `bf16`, or `auto`; `auto` selects BF16 on a
compatible CUDA GPU and otherwise keeps FP32. Model parameters and TTT inner
gradient/state statistics remain FP32 under autocast. `training.compile` enables
in-place `torch.compile` without changing checkpoint keys. It is disabled in the
portable reference configs because compilation support depends on the PyTorch
platform; enable it on a supported Linux CUDA server after a short smoke test.
Pinned-memory loaders use asynchronous CUDA transfers, and worker loaders
prefetch batches automatically. The N-CMAPSS references use four workers and
non-deterministic kernels for throughput; set `deterministic: true` when exact
repeatability is more important than speed.

Both variants share input projection, padding behavior, last-valid-token pooling,
final normalization, RUL head, data split, optimizer, metrics, checkpoint format,
and visualizations. The standard variant uses dynamic sinusoidal positions and
PyTorch pre-norm `TransformerEncoderLayer`; TTT uses CPE plus the ViTTT-derived
layer. TTT-only settings live under `model.ttt` and are rejected for a standard
Transformer configuration.

`model.architecture` defaults to `encoder`, preserving existing configurations
and checkpoints. Set it to `decoder` for causal processing. Decoder mode can
enable `model.autoregressive`: hidden state `t` predicts standardized feature
vector `t+1`, while the last valid hidden state predicts endpoint RUL. The
shared loop optimizes `RUL_MSE + weight * next_feature_loss`; early stopping
remains based on RUL MSE, while feature loss, RMSE, and MAE are also reported.
For a fair causal attention baseline, use `model.type: transformer`, keep the
decoder/autoregressive settings, and omit `model.ttt`.

The TTT decoder configurations enable engine-level state with:

```yaml
model:
  architecture: decoder
  ttt:
    chunk_size: 32
    continuous_state: true
```

In this mode, training shuffling is disabled automatically. Consecutive windows
from the same engine are collapsed to their unique new-token stream inside each
batch, RUL predictions are gathered at the original window endpoints, state is
detached at batch boundaries, and a new engine starts from an empty state. The
default `continuous_state: false` retains independent-window behavior and old
checkpoint compatibility.

For official C-MAPSS endpoint testing, the continuous decoder consumes the
complete observed test trajectory once and reports only its final RUL; this is
numerically the same causal stream as incremental windows without repeatedly
feeding their overlap. N-CMAPSS retains its all-window protocol and carries the
same state across successive windows and flights of one unit.

`data.name` selects `cmapss` or `ncmapss`; older C-MAPSS YAML files without this
field default to `cmapss`. `data.stride` controls training/validation endpoints,
while `data.evaluation_stride` independently controls full-trajectory testing.
For N-CMAPSS, `evaluation.max_test_batches` is available for smoke tests;
partial metrics are not benchmark-comparable.

Copy a reference YAML and change `data.subset` and `experiment.output_dir` for
FD002, FD003, or FD004. With `evaluation.checkpoint: null`, evaluation uses
`experiment.output_dir/best.pt` automatically.

For resume, set `training.resume` to `outputs/.../last.pt`, keep
`experiment.output_dir` at the checkpoint directory, and set `training.epochs`
to the new total epoch count. Resume uses the checkpoint model, split, and
preprocessing state.

Resume re-seeds from `training.seed`, but checkpoints do not preserve the
exact Python/NumPy/PyTorch/DataLoader RNG stream positions; it is reproducible
as a resumed run, not bitwise-identical to an uninterrupted run.

## Evaluate

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate --config configs\cmapss_ttt.yaml
.\.venv\Scripts\python.exe -m scripts.evaluate --config configs\ncmapss_ttt.yaml
```

The training and evaluation commands also print parameter count and analytical
forward MACs/FLOPs for one configured input window. Evaluation writes the same
complexity summary alongside RMSE, MAE, NASA Score, and MSE loss. By default,
metrics use the checkpoint's RUL-cap policy. `evaluation.max_test_engines` is
intended only for smoke tests because a partial NASA Score is not
benchmark-comparable. Set `training.plots` or `evaluation.plots` to `false` to
disable automatic PNG generation.

Autoregressive decoders additionally report teacher-forced next-feature loss,
RMSE, and MAE on valid transitions. These values use standardized feature space;
the benchmark RUL metrics and prediction files retain their existing semantics.

Plots can also be regenerated from saved JSON files:

```powershell
.\.venv\Scripts\python.exe -m scripts.visualize --run-dir outputs\fd002_ttt
```

Each training output directory contains:

```text
best.pt                 best validation-MSE checkpoint
last.pt                 latest checkpoint, including optimizer state
config.yaml             normalized configuration used by the run
history.json            per-epoch train/validation metrics
test_metrics.json       endpoint test metrics and label policy
test_predictions.json   endpoint predictions for C-MAPSS
test_predictions.jsonl  streamed full-trajectory predictions for N-CMAPSS
training_history.png    training and validation loss/RMSE curves
test_predictions.png    endpoint trend and prediction parity plot
```

## Model shapes

For input feature count `F`, window length `L`, model width `C`, `H` heads, and
head width `D=C/H`:

| Model | Sequence encoder |
| --- | --- |
| `ttt` | CPE + sample-local closed-form TTT update + FFN |
| `transformer` | Dynamic sinusoidal positions + multi-head self-attention + FFN |

| Stage | Shape |
| --- | --- |
| Dataset batch | `[B,L,F]`, mask `[B,L]` |
| Input projection / blocks | `[B,L,C]` |
| Standard attention q/k/v | `[B,H,L,D]` |
| TTT SwiGLU q/k/v | `[B,H,L,D]` |
| TTT temporal q/k/v | `[B,D,L]` |
| TTT per-sample fast w1/w2 | `[B,H,D,D]` |
| TTT per-sample fast w3 | `[B*D,1,3]` |
| Regression output | `[B]` |
| Decoder next-feature output | `[B,L,F]` |

During teacher forcing, positions `0..L-2` are compared with input features
`1..L-1`; transitions touching padding are excluded. Continuous measurements
are predicted directly rather than quantized into artificial token bins.

## Project layout

```text
data/
  base.py               shared adapter, bundle, evaluation, and RUL-filter contracts
  registry.py           explicit dataset adapter registry
  preprocessing.py      streaming feature variance and normalization state
  cmapss.py             parsing, split, scaling, RUL, windows
  ncmapss.py            lazy HDF5 schema, preprocessing, splits, and windows
configs/
  cmapss_ttt.yaml       C-MAPSS TTT experiment
  cmapss_ttt_decoder.yaml C-MAPSS causal TTT decoder experiment
  cmapss_transformer.yaml C-MAPSS standard Transformer experiment
  ncmapss_ttt.yaml      N-CMAPSS TTT experiment
  ncmapss_transformer.yaml N-CMAPSS standard Transformer experiment
models/
  ttt_layer.py          ViTTT-derived temporal TTT layer
  rul_transformer.py    shared RUL backbone, TTT and standard Transformer blocks
utils/
  config.py             typed YAML loading and strict validation
  complexity.py         parameter and analytical MAC/FLOP estimates
  metrics.py            RMSE, MAE, NASA Score
  engine.py             training, evaluation, checkpoints, reset lifecycle
  visualization.py      history and endpoint prediction plots
scripts/
  train.py              training/resume/test CLI
  evaluate.py           checkpoint-only official test CLI
  visualize.py          regenerate plots from saved JSON
THIRD_PARTY_NOTICES.md
requirements.txt
```
