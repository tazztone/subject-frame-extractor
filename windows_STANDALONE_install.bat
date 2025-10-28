@echo off
setlocal

REM ##################################################################
REM ##                  STANDALONE PROJECT SETUP                  ##
REM ##################################################################

echo --- Starting Project Setup ---

REM 1. Clone the main repository from GitHub
echo [1/6] Cloning the main repository...
git clone https://github.com/tazztone/subject-frame-extractor.git

REM Check if cloning was successful
if not exist subject-frame-extractor (
    echo ERROR: Failed to clone the repository.
    goto end
)

REM Change directory into the newly cloned repository
pushd subject-frame-extractor

REM 2. Initialize and download all submodules (DAM4SAM, Grounded-SAM-2, etc.)
echo [2/6] Initializing git submodules...
git submodule update --init --recursive

REM 3. Create and activate the Python virtual environment
echo [3/6] Creating Python virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

REM 4. Install/Upgrade Python packages
echo [4/6] Installing Python dependencies...
python -m pip install -U pip
pip install -U uv

REM Install PyTorch (adjust CUDA wheel index if needed)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

REM Install the main app's requirements
if exist requirements.txt (
  uv pip install -r requirements.txt
)

REM 5a. Install DAM4SAM and its dependencies (this provides the desired sam2)
echo [5/6] Installing DAM4SAM (preferred sam2)...
pushd DAM4SAM
if exist requirements.txt (
  uv pip install -r requirements.txt
)
pip install -e .

REM If DAM4SAM or its bundled SAM2 needs C++ extensions, build them
if exist setup.py (
  echo Building C++ extensions (if defined)...
  python setup.py build_ext --inplace
)
popd

REM 5b. Install ONLY Grounding DINO from Grounded-SAM-2; do NOT install repo-level SAM2
echo [5/6] Installing Grounding DINO only (skip SAM2)...
pushd Grounded-SAM-2
if exist requirements.txt (
    uv pip install -r requirements.txt
)
REM IMPORTANT: do NOT run "pip install -e ." here; that would install SAM2 from this repo
uv pip install --no-build-isolation -e grounding_dino
popd

REM 6. Force DAM4SAM to be first on sys.path via .pth files (alphabetical order)
echo [6/6] Configuring Python path precedence...
for /f "delims=" %%i in ('cd') do set ROOT=%%i

REM Write early .pth for DAM4SAM so its sam2 is found first
echo %ROOT%\DAM4SAM> venv\Lib\site-packages\00_dam4sam.pth

REM Optional: expose Grounded-SAM-2 tools without installing its SAM2
echo %ROOT%\Grounded-SAM-2> venv\Lib\site-packages\10_grounded_sam2.pth

echo.
echo --- Installation Complete! ---
echo DAM4SAM's sam2 will be imported due to .pth order, and Grounded-DINO is installed without registering another sam2.
echo The project is set up in the 'subject-frame-extractor' folder.

:end
popd
echo Press any key to exit.
pause > nul
