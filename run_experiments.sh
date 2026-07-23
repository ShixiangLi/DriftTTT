#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
elif [[ -x ".venv/Scripts/python.exe" ]]; then
    PYTHON=".venv/Scripts/python.exe"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
else
    echo "Python 3 was not found." >&2
    exit 1
fi

if [[ $# -gt 0 ]]; then
    exec "$PYTHON" -m scripts.run_experiments "$@"
fi

echo "Available datasets: cmapss, ncmapss, all"
read -r -p "Dataset [cmapss]: " DATASET
DATASET="${DATASET:-cmapss}"

echo "Enter comma-separated subsets or all."
echo "Examples: FD001,FD002  or  DS01-005,DS02-006"
read -r -p "Subsets [all]: " SUBSETS
SUBSETS="${SUBSETS:-all}"

echo "Available mixers: attention, ttt_mlp, ttt_multiscale_moe, all"
read -r -p "Mixers [all]: " MIXERS
MIXERS="${MIXERS:-all}"

read -r -p "Parallel GPU IDs, comma-separated [serial]: " GPUS

ARGS=(
    --dataset "$DATASET"
    --subsets "$SUBSETS"
    --mixers "$MIXERS"
)
if [[ -n "$GPUS" ]]; then
    ARGS+=(--gpus "$GPUS")
    read -r -p "Concurrent jobs per GPU [1]: " JOBS_PER_GPU
    ARGS+=(--jobs-per-gpu "${JOBS_PER_GPU:-1}")
fi

exec "$PYTHON" -m scripts.run_experiments "${ARGS[@]}"
