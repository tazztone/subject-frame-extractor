@echo off
REM Minimal, robust Windows setup that keeps the window open

REM Relaunch under cmd /k so the window stays open on finish or errors
if "%~1"=="-k" goto :main
start "Setup" cmd /k "%~f0 -k"
exit /b

:main
setlocal EnableExtensions

echo --- Starting Setup ---

REM 1) Clone the repo (edit the next line to add -b dev if you want the dev branch)
if not exist subject-frame-extractor (
  git clone -b dev https://github.com/tazztone/subject-frame-extractor.git
)
cd subject-frame-extractor

REM 2) Initialize submodules (DAM4SAM and your Grounded-SAM-2 fork without sam2)
git submodule update --init --recursive

REM 3) Create and activate venv
python -m venv venv
call venv\Scripts\activate.bat

REM 4) Base deps (PyTorch CUDA 12.8)
python -m pip install -U pip
pip install -U uv
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

REM 5) Install DAM4SAM (editable). Do NOT install its requirements.txt on Windows.
REM    Skip the optional SAM2 CUDA extension build to avoid NVCC/MSVC issues.
set "SAM2_BUILD_CUDA=0"
pip install -e DAM4SAM

REM 6) Install ONLY Grounding DINO from local folder (non-editable = fewer Windows issues)
uv pip install wheel
uv pip install --no-build-isolation "%CD%\Grounded-SAM-2\grounding_dino"

REM 7) Enforce import order so sam2 resolves to DAM4SAM first
> "venv\Lib\site-packages\00_dam4sam.pth" echo %CD%\DAM4SAM
> "venv\Lib\site-packages\10_grounded_sam2.pth" echo %CD%\Grounded-SAM-2

echo.
echo --- Installation Complete ---
echo To verify:
echo   call venv\Scripts\activate
echo   python -c "import sam2; print(sam2.__file__)"
