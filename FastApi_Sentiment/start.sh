#!/bin/bash
set -euo pipefail

PORT=${PORT:-10000}

MODEL_FILE="bilstm_att_correct_mask_final.keras"
if [ ! -f "$MODEL_FILE" ] && [ -n "${MODEL_URL:-}" ]; then
  echo "Model not found — downloading..."
  curl -L --fail -o "$MODEL_FILE" "$MODEL_URL"
fi

python -m pip install --upgrade pip

exec uvicorn main:app --host 0.0.0.0 --port "$PORT" --workers 1
