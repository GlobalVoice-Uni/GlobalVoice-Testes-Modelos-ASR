# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Best-effort: upgrade WhisperX Lightning checkpoint once to avoid runtime auto-upgrade overhead
CKPT_PATH=".venv/lib/python$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/whisperx/assets/pytorch_model.bin"
if [ -f "$CKPT_PATH" ]; then
	export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
	.venv/bin/python -m lightning.pytorch.utilities.upgrade_checkpoint "$CKPT_PATH" || \
		echo "[WARN] Could not upgrade WhisperX checkpoint now. Continuing setup."
else
	echo "[INFO] WhisperX checkpoint not found yet. Skipping optional upgrade."
fi

# Create results directory
mkdir -p results

# Print message
echo "Environment setup is complete."