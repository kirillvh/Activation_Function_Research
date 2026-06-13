#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
fi

exec "$PYTHON" -m activation_benchmark.plot_activations \
    --activations peuaf sine_triangle gelu gelu_sine_triangle \
    --w 1.0 --blend 0.5 "$@"
