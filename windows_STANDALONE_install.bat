REM Manual installation steps (run each command manually in sequence):
REM 1. Clone repo: git clone https://github.com/tazztone/subject-frame-extractor.git (or -b dev for dev branch)
REM 2. cd subject-frame-extractor && git submodule update --init --recursive
REM 3. python -m venv venv && call venv\Scripts\activate.bat
REM 4. python -m pip install -U pip && pip install -U uv && uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 && (if exist requirements.txt uv pip install -r requirements.txt)
REM 5a. cd DAM4SAM && (if exist requirements.txt uv pip install -r requirements.txt || echo Warning: Some dependencies failed to install, continuing...) && set SAM2_BUILD_CUDA=0 && pip install -e .
REM 5b. cd .. && cd Grounded-SAM-2 && (if exist requirements.txt uv pip install -r requirements.txt) && set SETUPTOOLS_ENABLE_FEATURES=legacy-editable && uv pip install --no-build-isolation -e grounding_dino
REM 6. Create venv\Lib\site-packages\00_dam4sam.pth with path to DAM4SAM dir && create venv\Lib\site-packages\10_grounded_sam2.pth with path to Grounded-SAM-2 dir
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
echo   [1] Main (stable)
echo   [2] Dev (latest features)
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

REM 2) Submodules (DAM4SAM, Grounded-SAM-2 fork)
echo [2/6] Initializing git submodules...
git submodule update --init --recursive || goto :end

REM 3) Virtual env
echo [3/6] Creating Python virtual environment...
python -m venv venv || goto :end
call venv\Scripts\activate.bat || goto :end

REM 4) Base deps
echo [4/6] Installing Python dependencies...
python -m pip install -U pip || goto :end
pip install -U uv || goto :end

REM Install PyTorch (adjust CUDA wheel index if needed)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 || goto :end

REM Install project requirements (optional)
if exist requirements.txt (
  uv pip install -r requirements.txt || goto :end
)

REM 5a) DAM4SAM (SAM2 provider)
echo [5/6] Installing DAM4SAM (preferred sam2)...
pushd DAM4SAM
if exist requirements.txt (
  uv pip install -r requirements.txt || echo Warning: Some dependencies failed to install, continuing...
)
REM Skip optional CUDA extension on Windows (pure Python path is fine)
set "SAM2_BUILD_CUDA=0"
pip install -e . || goto :end
popd

REM 5b) Grounding DINO only from fork (no SAM2 in repo)
echo [5/6] Installing Grounding DINO only (skip repo-level SAM2)...
pushd Grounded-SAM-2

if exist requirements.txt ( uv pip install -r requirements.txt || goto :end )

uv pip install --no-build-isolation -e "%CD%\grounding_dino" || (
  echo Editable install failed, falling back to non-editable...
  uv pip install --no-build-isolation "%CD%\grounding_dino" || goto :end
)

popd.

REM 6) Path precedence via .pth (alphabetical order -> DAM4SAM first)
echo [6/6] Configuring Python path precedence...
set "ROOT=%CD%"
> "venv\Lib\site-packages\00_dam4sam.pth" echo %ROOT%\DAM4SAM
> "venv\Lib\site-packages\10_grounded_sam2.pth" echo %ROOT%\Grounded-SAM-2

echo.
echo --- Installation Complete! ---
echo - DAM4SAM's sam2 will be imported due to .pth order.
echo - Grounding DINO is installed without registering another sam2.
echo - To verify: python -c "import sam2; print(sam2.__file__)"

:end
echo.
echo Finished. The shell stays open; review messages above.
