@echo off
REM KVTC 5090 Benchmark — Run this on the 5090 PC
REM Prerequisites: Python 3.10+ with pip
REM This script installs deps, clones the repo, and runs the benchmark

echo ============================================
echo KVTC 5090 GPU Benchmark Setup
echo ============================================

REM Check Python
python --version 2>nul
if errorlevel 1 (
    echo ERROR: Python not found! Install Python 3.10+ first.
    echo Download: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check if CUDA is available via PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')" 2>nul
if errorlevel 1 (
    echo Installing PyTorch with CUDA...
    pip install torch --index-url https://download.pytorch.org/whl/cu124
)

REM Install dependencies
echo.
echo Installing dependencies...
pip install transformers accelerate sentencepiece protobuf

REM Clone or update repo
if exist kvtc-repo (
    echo Updating existing repo...
    cd kvtc-repo
    git pull
) else (
    echo Cloning KVTC repo...
    git clone https://github.com/OnlyTerp/kvtc.git kvtc-repo
    cd kvtc-repo
)

REM Install KVTC
pip install -e .

echo.
echo ============================================
echo Running KVTC Benchmark on GPU
echo ============================================

REM Run TinyLlama first (quick validation)
echo.
echo [TEST 1] TinyLlama-1.1B on GPU (quick validation)
set PYTHONIOENCODING=utf-8
python -m src.benchmark_gpu --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device cuda --seq-len 512 --bit-budget-ratio 0.35

echo.
echo [TEST 2] TinyLlama-1.1B - high quality mode
python -m src.benchmark_gpu --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device cuda --seq-len 1024 --bit-budget-ratio 0.5

echo.
echo [TEST 3] Nemotron-Nano-4B on GPU (closer to TerpBot Pro architecture)
python -m src.benchmark_gpu --model nvidia/NVIDIA-Nemotron-3-Nano-4B-V1 --device cuda --seq-len 512 --bit-budget-ratio 0.35

echo.
echo ============================================
echo All benchmarks complete!
echo ============================================
pause
