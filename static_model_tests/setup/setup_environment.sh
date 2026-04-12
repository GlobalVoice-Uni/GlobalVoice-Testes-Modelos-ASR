#!/usr/bin/env bash
set -e

SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SETUP_DIR/../.." && pwd)"
STATIC_ROOT="$(cd "$SETUP_DIR/.." && pwd)"

# Create a virtual environment at repository root
python3 -m venv "$REPO_ROOT/.venv"

# Activate the virtual environment
source "$REPO_ROOT/.venv/bin/activate"

# Install dependencies
pip install -r "$STATIC_ROOT/requirements.txt"

# Best-effort: upgrade WhisperX Lightning checkpoint once to avoid runtime auto-upgrade overhead
CKPT_PATH="$REPO_ROOT/.venv/lib/python$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/whisperx/assets/pytorch_model.bin"
if [ -f "$CKPT_PATH" ]; then
	export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
	"$REPO_ROOT/.venv/bin/python" -m lightning.pytorch.utilities.upgrade_checkpoint "$CKPT_PATH" || \
		echo "[WARN] Could not upgrade WhisperX checkpoint now. Continuing setup."
else
	echo "[INFO] WhisperX checkpoint not found yet. Skipping optional upgrade."
fi

# Create static benchmark output directory
mkdir -p "$STATIC_ROOT/resultados"

# Print message
echo "Environment setup is complete."
echo "venv: $REPO_ROOT/.venv"
echo "outputs: $STATIC_ROOT/resultados"