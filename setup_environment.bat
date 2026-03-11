@echo off
:: Create a virtual environment
python -m venv .venv

:: Activate the virtual environment
call .venv\Scripts\activate.bat

:: Install dependencies
pip install -r requirements.txt

:: Create results directory
mkdir results

:: Print message
echo Environment setup is complete.