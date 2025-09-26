@echo off
setlocal

echo --- Starting Project Setup ---

REM 1) Clone main repo
echo [1/8] Cloning the main repository...
git clone https://github.com/tazztone/subject-frame-extractor.git
if not exist subject-frame-extractor (
  echo ERROR: Failed to clone the repository.
  goto end
)

REM 2) Enter repo and init submodules
pushd subject-frame-extractor
echo [2/8] Initializing submodules...
git submodule update --init --recursive

REM 3) Create and activate venv (Python 3.10 recommended by GSAM2)
echo [3/8] Creating Python virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

REM 4) Install/upgrade pip and uv
echo [4/8] Installing Python tooling...
python -m pip install -U pip
pip install -U uv

REM 5) Install PyTorch with CUDA (use cu121 for CUDA 12.1 per GSAM2 readme)
REM    Adjust TORCH_INDEX_URL to cu126/cu128 if that matches the local toolkit/driver.
set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128"
echo [5/8] Installing PyTorch from %TORCH_INDEX_URL% ...
uv pip install torch torchvision --index-url %TORCH_INDEX_URL%

REM 6) Install main app requirements
if exist requirements.txt (
  echo [6/8] Installing app requirements...
  uv pip install -r requirements.txt
)

REM 7) Install submodules
echo [7/8] Installing submodule dependencies...

REM 7a) DAM4SAM deps + editable install
if exist DAM4SAM\requirements.txt (
  uv pip install -r DAM4SAM\requirements.txt
)
pushd DAM4SAM
pip install -e .
popd

REM 7b) Grounded-SAM-2 deps + editable installs
if exist Grounded-SAM-2\requirements.txt (
  uv pip install -r Grounded-SAM-2\requirements.txt
)

REM Install Segment Anything 2 in editable mode (per readme)
pushd Grounded-SAM-2
pip install -e .
popd

REM Set CUDA_HOME for building GroundingDINO if CUDA default path exists
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8" (
  set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
  set "PATH=%CUDA_HOME%\bin;%CUDA_HOME%\libnvvp;%PATH%"
  echo Detected CUDA at %CUDA_HOME%
) else (
  echo WARNING: CUDA not found at default path. Set CUDA_HOME before installing GroundingDINO if building with GPU.
)

REM Build and install GroundingDINO with no build isolation (per readme)
uv pip install --no-build-isolation -e Grounded-SAM-2\grounding_dino

REM 8) Add submodules to Python path via .pth for robustness
echo [8/8] Configuring Python path...
> venv\Lib\site-packages\dam4sam.pth echo %cd%\DAM4SAM
> venv\Lib\site-packages\groundedsam2.pth echo %cd%\Grounded-SAM-2

echo.
echo --- Installation Complete! ---
echo Reminder:
echo  - Ensure ffmpeg is installed and in PATH (the app checks for it at startup).
echo  - If GroundingDINO build failed, verify CUDA_HOME and MSVC/CMake toolchain, then re-run the GroundingDINO install line.
echo The project is set up in the 'subject-frame-extractor' folder.

:end
popd
echo Press any key to exit.
pause > nul
