@echo off
REM Relaunch under a persistent shell so the window stays open
if "%~1"=="-persistent" goto :main
start "Standalone Install" cmd /k "%~f0" -persistent
exit /b

:main
shift
setlocal EnableExtensions EnableDelayedExpansion

echo --- Starting Project Setup ---

REM 1) Clone repo (branch choice)
echo [1/6] Cloning the main repository...
echo.
echo Select the branch to clone:
echo [1] Main (stable)
echo [2] Dev (latest features)
echo.
CHOICE /C 12 /M "Enter your choice:"
if ERRORLEVEL 2 (
  echo Cloning dev branch...
  git clone -b dev https://github.com/tazztone/subject-frame-extractor.git || goto :end
) else (
  echo Cloning main branch...
  git clone https://github.com/tazztone/subject-frame-extractor.git || goto :end
)

if not exist subject-frame-extractor (
  echo ERROR: Failed to clone the repository.
  goto :end
)

pushd subject-frame-extractor

REM 2) Submodules (SAM3_repo)
echo [2/6] Initializing git submodules...
git submodule update --init --recursive || goto :end

REM 3) Virtual env
echo [3/6] Creating Python virtual environment...
python -m venv venv || goto :end
call venv\Scripts\activate.bat || goto :end

REM 4) Base deps (install PyTorch first for correct CUDA build)
echo [4/6] Installing Python dependencies...
python -m pip install -U pip || goto :end
pip install -U uv || goto :end
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 || goto :end

REM Install project requirements (includes groundingdino-py)
if exist requirements.txt (
  uv pip install -r requirements.txt || goto :end
)



echo.
echo --- Installation Complete! ---

:end
echo.
echo Finished. The shell stays open; review messages above.
