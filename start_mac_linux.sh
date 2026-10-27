#!/usr/bin/env bash
# Latent Space Explorer - one-click start for macOS / Linux
set -e
cd "$(dirname "$0")"
echo "============================================================"
echo "  Latent Space Explorer  -  StyleGAN2 latent interpolation"
echo "============================================================"

PY=""
for c in python3.12 python3.11 python3.10 python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info>=(3,9) else 1)' 2>/dev/null; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "[ERROR] Python 3.9+ not found. Install it from https://www.python.org/downloads/ (or: brew install python) and run again."
  exit 1
fi
echo "Using: $($PY --version)"

if [ ! -x ".venv/bin/python" ]; then
  echo; echo "[1/3] Creating virtual environment .venv ..."
  "$PY" -m venv .venv
fi
VPY=".venv/bin/python"

if [ ! -f ".venv/.installed" ]; then
  echo; echo "[2/3] Installing packages - first run only, takes a few minutes ..."
  "$VPY" -m pip install --upgrade pip
  if [ "$(uname)" = "Darwin" ]; then
    "$VPY" -m pip install torch
  else
    "$VPY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu || "$VPY" -m pip install torch
  fi
  "$VPY" -m pip install -r requirements.txt
  touch .venv/.installed
fi

echo; echo "[3/3] Checking pretrained StyleGAN2 weights ..."
"$VPY" download_model.py --no-verify || echo "[WARNING] weights not downloaded - see README > Troubleshooting"

echo; echo "Starting the app - your browser will open at http://127.0.0.1:8765"
echo "Keep this terminal open while using the app. Press Ctrl+C to stop."; echo
exec "$VPY" app.py "$@"
