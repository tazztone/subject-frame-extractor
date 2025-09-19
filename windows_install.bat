@echo off
setlocal

REM Clone DAM4SAM if not present
if not exist DAM4SAM (
  git clone https://github.com/jovanavidenovic/DAM4SAM.git
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

REM Now install DAM4SAM's requirements and the package
pushd DAM4SAM
if exist requirements.txt (
  uv pip install -r requirements.txt
)
pip install -e .
popd

echo All requirements installed (app + DAM4SAM).
pause
