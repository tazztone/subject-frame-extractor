@echo off
setlocal enableextensions

:: --- Configuration ---
set "REPO_URL=https://github.com/tazztone/subject-frame-extractor.git"

:: Automatically determine the repo directory name from the URL
for %%A in ("%REPO_URL%") do set "REPO_DIR=%%~nA"

echo --- Starting Comprehensive Update in '%REPO_DIR%' ---

rem Check if directory exists before trying to enter it
if not exist "%REPO_DIR%" (
    echo ERROR: Cannot find the '%REPO_DIR%' directory. Run the install script first.
    goto :fail_early
)

rem --- Enter Repo, Git Pull and Submodules ---
pushd "%REPO_DIR%" || goto :fail_early

echo [1/4] Pulling main repository updates...
git pull || goto :fail

echo [2/4] Updating submodules...
git submodule update --init --recursive || goto :fail

rem --- Activate venv ---
if not exist venv\Scripts\activate.bat (
    echo ERROR: Python virtual environment not found. Run the install script first.
    goto :fail
)
call venv\Scripts\activate.bat || goto :fail

rem --- Reinstall Requirements ---
echo [3/4] Updating Python dependencies...
uv sync || goto :fail

rem --- Update submodules (SAM3_repo) ---
echo [4/4] Ensuring submodules are up-to-date...
git submodule update --remote --merge || echo Warning: Submodule update failed, continuing...

echo.
echo --- Update Complete ---
goto :end

:fail
popd

:fail_early
echo.
echo !!! UPDATE FAILED - See errors above !!!

:end
if "%cd%"=="%~dp0%REPO_DIR%" popd
echo.
echo Press any key to close this window...
pause > nul