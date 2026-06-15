#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
fi
export ACTIVATION_BENCHMARK_DISABLE_AFFINITY="${ACTIVATION_BENCHMARK_DISABLE_AFFINITY:-1}"

exec "$PYTHON" -m activation_benchmark.audio_benchmark \
    --config configs/benchmark_audio_activations.yaml "$@"
