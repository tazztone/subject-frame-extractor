@echo off
setlocal

REM Check if virtual environment exists
if not exist venv (
    echo Virtual environment not found. Please run windows_install.bat first.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if activation was successful
if "%VIRTUAL_ENV%" == "" (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)

REM Open browser to Gradio interface first
echo Opening browser to http://localhost:7860
start http://localhost:7860

REM Run the app_small.py application
echo.
echo Starting Frame Extractor and Analyzer (Small Version)...
echo The browser window should open automatically.
echo If the app loads successfully, you'll see the interface in your browser.
echo.
python app_small.py

REM Keep the window open if there's an error
if errorlevel 1 (
    echo.
    echo Application exited with error code %errorlevel%.
    pause
)

deactivate