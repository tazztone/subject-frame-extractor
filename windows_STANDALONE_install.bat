@echo off
setlocal enableextensions

echo --- Starting Project Setup ---

rem 1) Clone main repo
echo [1/8] Cloning the main repository...
git clone https://github.com/tazztone/subject-frame-extractor.git || goto :fail

if not exist subject-frame-extractor (
  echo ERROR: Failed to clone the repository.
  goto :fail
)

rem 2) Enter repo and init submodules
pushd subject-frame-extractor || goto :fail
echo [2/8] Initializing submodules...
git submodule update --init --recursive || goto :fail

rem 3) Create and activate venv
echo [3/8] Creating Python virtual environment...
python -m venv venv || goto :fail
call venv\Scripts\activate.bat || goto :fail

rem 4) Install/upgrade pip and uv
echo [4/8] Installing Python tooling...
python -m pip install -U pip || goto :fail
pip install -U uv || goto :fail

rem 6) Install main app requirements (light/fast deps)
if exist requirements.txt (
  echo [6/8] Installing app requirements...
  uv pip install -r requirements.txt || goto :fail
)

rem 7) Submodules (light parts only)
echo [7/8] Installing submodule dependencies (light)...
if exist DAM4SAM\requirements.txt (
  uv pip install -r DAM4SAM\requirements.txt || goto :fail
  pushd DAM4SAM || goto :fail
  pip install -e . || goto :fail
  popd
)
if exist Grounded-SAM-2\requirements.txt (
  uv pip install -r Grounded-SAM-2\requirements.txt || goto :fail
  pushd Grounded-SAM-2 || goto :fail
  pip install -e . || goto :fail
  popd
)

rem 8) Configure Python path
echo [8/8] Configuring Python path...
> venv\Lib\site-packages\dam4sam.pth echo %cd%\DAM4SAM
> venv\Lib\site-packages\groundedsam2.pth echo %cd%\Grounded-SAM-2

rem Only reach heavy section if all prior steps succeeded
goto :heavy

:fail
echo Installation failed before heavy dependencies; skipping heavyweight installs.
goto :end

:heavy
rem Heavy installs last
set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128"
echo Installing PyTorch from %TORCH_INDEX_URL% ...
uv pip install torch torchvision --index-url %TORCH_INDEX_URL% || goto :end

rem Optional CUDA_HOME detection
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8" (
  set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
  set "PATH=%CUDA_HOME%\bin;%CUDA_HOME%\libnvvp;%PATH%"
  echo Detected CUDA at %CUDA_HOME%
) else (
  echo WARNING: CUDA not found at default path. Set CUDA_HOME before installing GroundingDINO if building with GPU.
)

rem Build and install GroundingDINO last
uv pip install --no-build-isolation -e Grounded-SAM-2\grounding_dino

:end
popd
echo Done.
pause > nul
