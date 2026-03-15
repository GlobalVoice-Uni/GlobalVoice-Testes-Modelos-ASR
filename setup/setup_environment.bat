@echo off
:: Create a virtual environment
python -m venv .venv

:: Activate the virtual environment
call .venv\Scripts\activate.bat

:: Install dependencies
pip install -r requirements.txt

:: Best-effort: upgrade WhisperX Lightning checkpoint once to avoid runtime auto-upgrade overhead
set "CKPT_PATH=.venv\Lib\site-packages\whisperx\assets\pytorch_model.bin"
if exist "%CKPT_PATH%" (
	set "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1"
	.venv\Scripts\python.exe -m lightning.pytorch.utilities.upgrade_checkpoint "%CKPT_PATH%"
	if errorlevel 1 (
		echo [WARN] Could not upgrade WhisperX checkpoint now. Continuing setup.
	) else (
		echo [OK] WhisperX checkpoint upgraded successfully.
	)
) else (
	echo [INFO] WhisperX checkpoint not found yet. Skipping optional upgrade.
)

:: Create resultados directory
mkdir resultados

:: Print message
echo Environment setup is complete.