@echo off
setlocal

set "SETUP_DIR=%~dp0"
for %%I in ("%SETUP_DIR%..\..") do set "REPO_ROOT=%%~fI"
for %%I in ("%SETUP_DIR%..") do set "STATIC_ROOT=%%~fI"

:: Create a virtual environment at repository root
python -m venv "%REPO_ROOT%\.venv"

:: Activate the virtual environment
call "%REPO_ROOT%\.venv\Scripts\activate.bat"

:: Install dependencies
pip install -r "%STATIC_ROOT%\requirements.txt"

:: Best-effort: upgrade WhisperX Lightning checkpoint once to avoid runtime auto-upgrade overhead
set "CKPT_PATH=%REPO_ROOT%\.venv\Lib\site-packages\whisperx\assets\pytorch_model.bin"
if exist "%CKPT_PATH%" (
	set "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1"
	"%REPO_ROOT%\.venv\Scripts\python.exe" -m lightning.pytorch.utilities.upgrade_checkpoint "%CKPT_PATH%"
	if errorlevel 1 (
		echo [WARN] Could not upgrade WhisperX checkpoint now. Continuing setup.
	) else (
		echo [OK] WhisperX checkpoint upgraded successfully.
	)
) else (
	echo [INFO] WhisperX checkpoint not found yet. Skipping optional upgrade.
)

:: Create static benchmark output directory
if not exist "%STATIC_ROOT%\resultados" mkdir "%STATIC_ROOT%\resultados"

:: Print message
echo Environment setup is complete.
echo venv: %REPO_ROOT%\.venv
echo outputs: %STATIC_ROOT%\resultados