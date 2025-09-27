@echo off
setlocal enableextensions

:: --- Configuration ---
set "REPO_URL=https://github.com/tazztone/subject-frame-extractor.git"
set "VENV_DIR=venv"

:: Automatically determine the repo directory name from the URL
for %%A in ("%REPO_URL%") do set "REPO_DIR=%%~nA"

echo --- Starting Project Setup ---
echo Repository: %REPO_URL%
echo Local Directory: %REPO_DIR%
echo.

rem Check for Git
git --version >nul 2>&1
if %errorlevel% neq 0 (
  echo ERROR: Git is not installed or not found in your PATH.
  goto :fail_early
)

rem 1) Clone main repo
[cite_start]echo [1/8] Cloning the main repository... [cite: 1]
if exist "%REPO_DIR%" (
  echo Repository directory '%REPO_DIR%' already exists. Skipping clone.
) else (
  [cite_start]git clone "%REPO_URL%" || goto :fail [cite: 1]
)

rem 2) Enter repo and init submodules
pushd "%REPO_DIR%" || (
  echo ERROR: Failed to enter the repository directory '%REPO_DIR%'.
  goto :fail_early
)
[cite_start]echo [2/8] Initializing submodules... [cite: 3]
[cite_start]git submodule update --init --recursive || goto :fail [cite: 4]

rem 3) Create and activate venv
[cite_start]echo [3/8] Creating Python virtual environment... [cite: 5]
if not exist "%VENV_DIR%" (
  python -m venv %VENV_DIR% || goto :fail
)
[cite_start]call "%VENV_DIR%\Scripts\activate.bat" || goto :fail [cite: 5]

rem 4) Install/upgrade pip and uv
[cite_start]echo [4/8] Installing/upgrading Python tooling (pip, uv)... [cite: 6]
[cite_start]python -m pip install -U pip || goto :fail [cite: 6]
pip install -U uv || goto :fail

rem 5) Install main app requirements (light/fast deps)
echo [5/8] Installing app requirements...
if exist requirements.txt (
  uv pip install -r requirements.txt || goto :fail
)

rem 6) Install submodule requirements (light parts only)
echo [6/8] Installing submodule dependencies (light)...
if exist DAM4SAM\requirements.txt (
  uv pip install -r DAM4SAM\requirements.txt || goto :fail
)
if exist Grounded-SAM-2\requirements.txt (
  uv pip install -r Grounded-SAM-2\requirements.txt || goto :fail
)

rem 7) Perform editable installs for submodules
echo [7/8] Setting up editable installs for submodules...
pip install -e DAM4SAM || goto :fail
pip install -e Grounded-SAM-2 || goto :fail

rem 8) Configure Python path
[cite_start]echo [8/8] Configuring Python path for submodules... [cite: 7]
[cite_start]echo %cd%\DAM4SAM > "%VENV_DIR%\Lib\site-packages\dam4sam.pth" [cite: 7]
[cite_start]echo %cd%\Grounded-SAM-2 > "%VENV_DIR%\Lib\site-packages\groundedsam2.pth" [cite: 7]

rem --- Heavyweight Installs ---
echo.
[cite_start]echo --- Starting Heavyweight Dependency Installation --- [cite: 8]

rem Install PyTorch
[cite_start]echo Installing PyTorch... [cite: 9]
[cite_start]uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || goto :fail [cite: 9]

rem Optional CUDA_HOME detection
if defined CUDA_HOME (
  echo CUDA_HOME is already set to: %CUDA_HOME%
) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1" (
  set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
  echo Found and set CUDA_HOME to default: %CUDA_HOME%
) else (
  echo WARNING: CUDA_HOME is not set and was not found at the default path.
  echo          Set CUDA_HOME manually if you encounter build issues with GroundingDINO.
)

rem Build and install GroundingDINO last
echo Building and installing GroundingDINO...
uv pip install --no-build-isolation -e Grounded-SAM-2\grounding_dino || goto :fail

echo.
echo --- Installation Complete ---
goto :end

:fail
echo.
echo !!! INSTALLATION FAILED - Please review the errors above. !!!
popd

:fail_early
echo Installation cannot continue.

:end
echo.
echo Press any key to close this window...
[cite_start]pause > nul [cite: 10]