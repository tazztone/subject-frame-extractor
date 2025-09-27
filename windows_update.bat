@echo off
setlocal enableextensions

:: --- Configuration ---
set "REPO_URL=https://github.com/tazztone/subject-frame-extractor.git"

:: Automatically determine the repo directory name from the URL
for %%A in ("%REPO_URL%") do set "REPO_DIR=%%~nA"

[cite_start]echo --- Starting Comprehensive Update in '%REPO_DIR%' --- [cite: 11]

rem Check if directory exists before trying to enter it
if not exist "%REPO_DIR%" (
    [cite_start]echo ERROR: Cannot find the '%REPO_DIR%' directory. Run the install script first. [cite: 14]
    goto :fail_early
)

rem --- Enter Repo, Git Pull and Submodules ---
pushd "%REPO_DIR%" || goto :fail_early

[cite_start]echo [1/4] Pulling main repository updates... [cite: 11]
[cite_start]git pull || goto :fail [cite: 12]

[cite_start]echo [2/4] Updating submodules... [cite: 12]
[cite_start]git submodule update --init --recursive || goto :fail [cite: 13]

rem --- Activate venv ---
if not exist venv\Scripts\activate.bat (
    echo ERROR: Python virtual environment not found. Run the install script first.
    goto :fail
)
[cite_start]call venv\Scripts\activate.bat || goto :fail [cite: 15]

rem --- Reinstall Requirements ---
[cite_start]echo [3/4] Updating Python dependencies... [cite: 15]
uv pip install --upgrade -r requirements.txt || goto :fail
uv pip install --upgrade -r DAM4SAM\requirements.txt || goto :fail
uv pip install --upgrade -r Grounded-SAM-2\requirements.txt || goto :fail

rem --- Re-link and Re-install Editable Packages ---
[cite_start]echo [4/4] Re-linking and re-installing editable packages... [cite: 16]
[cite_start]pip install -e DAM4SAM || goto :fail [cite: 16]
pip install -e Grounded-SAM-2 || goto :fail
[cite_start]uv pip install --no-build-isolation -e Grounded-SAM-2\grounding_dino || goto :fail [cite: 17]

echo.
echo --- Update Complete ---
goto :end

:fail
popd

:fail_early
echo.
echo !!! [cite_start]UPDATE FAILED - See errors above !!! [cite: 18]

:end
if "%cd%"=="%~dp0%REPO_DIR%" popd
echo.
[cite_start]echo Press any key to close this window... [cite: 19]
[cite_start]pause > nul [cite: 19]