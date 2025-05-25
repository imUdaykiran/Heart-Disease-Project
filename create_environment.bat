@echo off
SETLOCAL

REM Step 1: Create virtual environment named "python_env"
python -m venv python_env

REM Step 2: Activate the virtual environment
CALL python_env\Scripts\activate.bat

REM Step 3: Upgrade pip
python -m pip install --upgrade pip

REM Step 4: Install packages from requirements.txt
pip install -r requirements.txt

echo.
echo Environment setup complete.
ENDLOCAL
pause
