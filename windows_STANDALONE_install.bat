@echo off
setlocal

REM ##################################################################
REM ##                  STANDALONE PROJECT SETUP                  ##
REM ##################################################################

echo --- Starting Project Setup ---

REM 1. Clone the main repository from GitHub
echo [1/5] Cloning the main repository...
git clone https://github.com/tazztone/subject-frame-extractor.git

REM Check if cloning was successful
if not exist subject-frame-extractor (
    echo ERROR: Failed to clone the repository.
    goto end
)

REM Change directory into the newly cloned repository
pushd subject-frame-extractor

REM 2. Initialize and download the DAM4SAM submodule
echo [2/5] Setting up the DAM4SAM submodule...
REM This single command correctly initializes and clones the code for any submodules
REM listed in the main project's .gitmodules file. This is the key fix.
git submodule update --init --recursive

REM 3. Create and activate the Python virtual environment
echo [3/5] Creating Python virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

REM 4. Install/Upgrade Python packages
echo [4/5] Installing Python dependencies...
python -m pip install -U pip
pip install -U uv

REM Install PyTorch (This may show a network error, but should be installed by requirements.txt)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

REM Install the main app's requirements
if exist requirements.txt (
  uv pip install -r requirements.txt
)

REM Install DAM4SAM dependencies and the SAM2 package itself
pushd DAM4SAM
if exist requirements.txt (
  uv pip install -r requirements.txt
)
pip install -e .

REM Build SAM2 C++ extensions
echo Building SAM2 C++ extensions...
python setup.py build_ext --inplace

popd

REM Install Grounded-SAM-2 dependencies
pushd Grounded-SAM-2
if exist requirements.txt (
  uv pip install -r requirements.txt
)
pip install -e .
popd

REM 5. Add DAM4SAM to the Python path for the venv
echo [5/5] Configuring Python path...
echo %cd%\DAM4SAM > venv\Lib\site-packages\dam4sam.pth
echo %cd%\Grounded-SAM-2 > venv\Lib\site-packages\grounded_sam2.pth

echo.
echo --- Installation Complete! ---
echo Please check the output above for any errors.
echo The project is set up in the 'subject-frame-extractor' folder.

:end
popd
echo Press any key to exit.
pause > nul