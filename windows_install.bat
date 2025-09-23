@echo off
setlocal

REM Clone DAM4SAM if not present
if not exist DAM4SAM (
  git clone https://github.com/tazztone/DAM4SAM.git
)

REM Create and activate venv
python -m venv venv
call venv\Scripts\activate.bat

REM Upgrade pip/uv
python -m pip install -U pip
pip install -U uv

REM Install PyTorch (adjust CUDA/CPU as needed)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

REM Install THIS app's requirements at project root if present
if exist requirements.txt (
  uv pip install -r requirements.txt
)

REM Install DAM4SAM dependencies AND the SAM2 package
pushd DAM4SAM
if exist requirements.txt (
  uv pip install -r requirements.txt
)
REM This is the crucial missing step - install SAM2
pip install -e .
popd

REM Add DAM4SAM to Python path
echo %cd%\DAM4SAM > venv\Lib\site-packages\dam4sam.pth

echo All requirements installed (app + DAM4SAM).
pause
