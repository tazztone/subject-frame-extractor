@echo off
REM Keep CMD open after finishing
if "%~1"=="-persistent" goto :main
start "Standalone Install" cmd /k "%~f0" -persistent
exit /b

:main
shift
setlocal EnableExtensions EnableDelayedExpansion

echo --- Starting Project Setup ---

REM 1) Clone repo (choose branch)
echo [1/5] Cloning repository...
echo.
echo Select branch: [1] Main (stable)  [2] Dev (latest)
CHOICE /C 12 /M "Enter your choice:"
if ERRORLEVEL 2 (
  git clone -b dev https://github.com/tazztone/subject-frame-extractor.git || goto :end
) else (
  git clone https://github.com/tazztone/subject-frame-extractor.git || goto :end
)

if not exist subject-frame-extractor (
  echo ERROR: Failed to clone repo.
  goto :end
)

pushd subject-frame-extractor

REM 2) Submodules (DAM4SAM + your Grounded-SAM-2 fork without sam2)
echo [2/5] Initializing submodules...
git submodule update --init --recursive || goto :end

REM 3) Virtualenv + base deps
echo [3/5] Creating venv and installing base deps...
python -m venv venv || goto :end
call venv\Scripts\activate.bat || goto :end
python -m pip install -U pip || goto :end
pip install -U uv || goto :end
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 || goto :end
if exist requirements.txt ( uv pip install -r requirements.txt || goto :end )

REM 4) Install DAM4SAM (the ONLY SAM2 provider here)
echo [4/5] Installing DAM4SAM (SAM2 provider)...
pushd DAM4SAM
if exist requirements.txt ( uv pip install -r requirements.txt || goto :end )
set "SAM2_BUILD_CUDA=0"
pip install -e . || goto :end
popd

REM 5) Install ONLY Grounding DINO from your fork (no repo-level SAM2)
echo [5/5] Installing Grounding DINO (no SAM2)...
pushd Grounded-SAM-2
if exist requirements.txt ( uv pip install -r requirements.txt || goto :end )
set "SETUPTOOLS_ENABLE_FEATURES=legacy-editable"
uv pip install --no-build-isolation -e "grounding_dino" || (
  echo Editable failed, trying non-editable...
  uv pip install --no-build-isolation "grounding_dino" || goto :end
)
popd

REM Path precedence: ensure DAM4SAM comes first for `import sam2`
set "ROOT=%CD%"
> "venv\Lib\site-packages\00_dam4sam.pth" echo %ROOT%\DAM4SAM
> "venv\Lib\site-packages\10_grounded_sam2.pth" echo %ROOT%\Grounded-SAM-2

echo.
echo --- Installation Complete! ---
echo - SAM2 comes from DAM4SAM due to .pth order.
echo - Grounding DINO installed from your fork subdir.
echo - Verify: python -c "import sam2; print(sam2.__file__)"

:end
echo.
echo Finished. The shell stays open; review messages above.
