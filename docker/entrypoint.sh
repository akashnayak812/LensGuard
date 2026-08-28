#!/bin/bash
# ── LensGuard Backend Entrypoint ─────────────────────────────────────────────
# 1. Create model stub if no trained model exists (first-run only)
# 2. Start uvicorn
set -e

echo "=== LensGuard Backend Starting ==="

# Create directories
mkdir -p "${UPLOAD_DIR:-/app/data/uploads}" "${HEATMAP_DIR:-/app/data/heatmaps}"

# Create model stub if trained model doesn't exist
MODEL_DIR="${MODEL_PATH:-/app/ml/models}/${MODEL_VERSION:-v1}"
if [ ! -f "$MODEL_DIR/model.keras" ] && [ ! -f "$MODEL_DIR/model.h5" ]; then
    echo "No trained model found at $MODEL_DIR. Creating stub model..."
    python /app/ml/create_model_stub.py
else
    echo "Model found at $MODEL_DIR — skipping stub creation."
fi

echo "Starting uvicorn..."
exec python -m uvicorn backend.app.main:app \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8000}" \
    --workers 1 \
    --log-level "${LOG_LEVEL:-info}"
