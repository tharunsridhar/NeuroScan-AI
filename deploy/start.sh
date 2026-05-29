#!/usr/bin/env bash
set -euo pipefail

HOST="${NEUROSCAN_HOST:-0.0.0.0}"
PORT="${NEUROSCAN_PORT:-8000}"

cd "$(dirname "$0")/.."
python -m uvicorn app.main:app --host "$HOST" --port "$PORT"
