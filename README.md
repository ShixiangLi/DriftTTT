# DriftTTT 执行说明

本文档只说明项目的安装、训练、评估和批量运行方法。算法说明见 [docs/ncmapss_method_and_results_zh.md](docs/ncmapss_method_and_results_zh.md)。

## 1. 环境准备

项目要求 Python 3.10 及以上版本。

Linux 服务器：

```bash
cd /home/lsx/workspace/DriftTTT
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows：

```powershell
cd F:\workspace\py\llm\DriftTTT
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

根目录下的 `run_experiments.sh` 和 `run_experiments.bat` 会自动使用项目虚拟环境，批量运行时不必提前激活环境。

## 2. 数据目录

将数据放在以下位置：

```text
dataset/
├─ cmapss/       C-MAPSS 数据文件
└─ n-cmapss/     N-CMAPSS HDF5 数据文件
```

默认配置文件：

```text
configs/cmapss_transformer.yaml
configs/ncmapss_transformer.yaml
```

## 3. 单次训练

训练 C-MAPSS：

```bash
.venv/bin/python -m scripts.train --config configs/cmapss_transformer.yaml
```

训练 N-CMAPSS：

```bash
.venv/bin/python -m scripts.train --config configs/ncmapss_transformer.yaml
```

Windows 将 `.venv/bin/python` 替换为 `.\.venv\Scripts\python.exe`：

```powershell
.\.venv\Scripts\python.exe -m scripts.train --config configs\ncmapss_transformer.yaml
```

单次训练使用 YAML 中的 `model.sequence_mixer`：

```yaml
model:
  sequence_mixer: ttt_multiscale_moe
```

可选值：

- `attention`：标准 Attention；
- `ttt_mlp`：标准 TTT Layer；
- `ttt_multiscale_moe`：TTT-MoE。

CB-DTS 建议通过批量入口中的 `--mixers cb_dts` 启动，脚本会自动选择 TTT-MoE 并开启 CB-DTS 训练损失。

## 4. 单次评估与绘图

评估配置对应的检查点：

```bash
.venv/bin/python -m scripts.evaluate --config configs/ncmapss_transformer.yaml
```

根据已有实验目录重新绘图：

```bash
.venv/bin/python -m scripts.visualize --run-dir outputs/实验目录名
```

Windows 示例：

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate --config configs\ncmapss_transformer.yaml
.\.venv\Scripts\python.exe -m scripts.visualize --run-dir outputs\实验目录名
```

## 5. Linux 服务器批量实验

服务器上的批量实验统一从 shell 脚本启动：

```bash
cd /home/lsx/workspace/DriftTTT
bash run_experiments.sh [参数]
```

不带参数运行时，脚本会交互式询问数据集、子集、方法、seed 和 GPU：

```bash
bash run_experiments.sh
```

常用参数：

| 参数 | 说明 | 示例 |
|---|---|---|
| `--dataset` | `cmapss`、`ncmapss` 或 `all` | `--dataset ncmapss` |
| `--subsets` | 一个、多个或全部子集 | `--subsets DS02-006,DS05` |
| `--mixers` | 一个、多个或全部方法 | `--mixers attention,ttt_multiscale_moe` |
| `--seeds` | 一个或多个训练 seed | `--seeds 7,42,123,202,3407` |
| `--gpus` | 并行使用的 GPU 编号 | `--gpus 0,1,2,3` |
| `--jobs-per-gpu` | 每张 GPU 同时运行的任务数，或 `all` | `--jobs-per-gpu 2` |
| `--dry-run` | 只生成配置，不开始训练 | `--dry-run` |

`--mixers` 可选：`attention`、`ttt_mlp`、`ttt_multiscale_moe`、`cb_dts`、`all`。其中 `cb_dts` 只用于 N-CMAPSS。

### 5.1 N-CMAPSS 全子集、全方法、五个 seed

建议先限制每张 GPU 的并发数：

```bash
bash run_experiments.sh \
  --dataset ncmapss \
  --subsets all \
  --mixers all \
  --seeds 7,42,123,202,3407 \
  --gpus 0,1,2,3,4,5,6,7 \
  --jobs-per-gpu 2
```

如果已经确认 GPU 显存、主机内存和 HDF5 读取带宽足够，可以让脚本自动把全部任务尽量并发分配：

```bash
bash run_experiments.sh \
  --dataset ncmapss \
  --subsets all \
  --mixers all \
  --seeds 7,42,123,202,3407 \
  --gpus 0,1,2,3,4,5,6,7 \
  --jobs-per-gpu all
```

### 5.2 指定 N-CMAPSS 子集和方法

```bash
bash run_experiments.sh \
  --dataset ncmapss \
  --subsets DS02-006,DS05 \
  --mixers attention,ttt_mlp,ttt_multiscale_moe,cb_dts \
  --seeds 42,202 \
  --gpus 0,1,2,3 \
  --jobs-per-gpu 1
```

### 5.3 C-MAPSS 全子集对比

```bash
bash run_experiments.sh \
  --dataset cmapss \
  --subsets all \
  --mixers attention,ttt_mlp,ttt_multiscale_moe \
  --seeds 7,42,123,202,3407 \
  --gpus 0,1,2,3 \
  --jobs-per-gpu 2
```

### 5.4 运行前检查任务矩阵

下面的命令只生成本轮配置并打印任务，不执行训练：

```bash
bash run_experiments.sh \
  --dataset ncmapss \
  --subsets all \
  --mixers all \
  --seeds 7,42,123,202,3407 \
  --gpus 0,1,2,3,4,5,6,7 \
  --jobs-per-gpu 2 \
  --dry-run
```

## 6. Windows 批量实验

交互式运行：

```powershell
.\run_experiments.bat
```

直接传入参数：

```powershell
.\run_experiments.bat `
  --dataset ncmapss `
  --subsets all `
  --mixers all `
  --seeds 7,42,123,202,3407 `
  --gpus 0,1,2,3 `
  --jobs-per-gpu 1
```

## 7. 输出目录

批量实验统一保存在时间戳目录下：

```text
outputs/batches/YYYYMMDD_HHMMSS/
├─ configs/       本批次自动生成的配置
├─ 各实验目录/    检查点、指标、预测和图片
└─ summary.csv    本批次汇总结果
```

单个完整实验通常包含：

```text
best.pt
last.pt
config.yaml
history.json
test_metrics.json
test_predictions.json 或 test_predictions.jsonl
training_history.png
test_predictions.png
train.log
```

如需断点续训，在对应 YAML 中设置 `training.resume` 为已有的 `last.pt`，保持原输出目录不变，并把 `training.epochs` 调整为期望的总 epoch 数。
