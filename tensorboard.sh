#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
fi

echo "TensorBoard will be available at http://localhost:6006"
exec "$PYTHON" -m tensorboard.main --logdir runs --port 6006 "$@"
