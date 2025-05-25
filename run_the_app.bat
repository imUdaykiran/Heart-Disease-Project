@echo off
REM Activate virtual environment
call python_env\Scripts\activate.bat

REM go to proj directory
cd D:\project_v2

REM Run the app
streamlit run main.py
pause
