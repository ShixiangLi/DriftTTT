@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo Python environment not found: %PYTHON%
    exit /b 1
)

if not "%~1"=="" (
    "%PYTHON%" -m scripts.run_experiments %*
    exit /b
)

echo Available datasets: cmapss, ncmapss, all
set /p "DATASET=Dataset [cmapss]: "
if "%DATASET%"=="" set "DATASET=cmapss"

echo Enter comma-separated subsets or all.
echo Examples: FD001,FD002  or  DS01-005,DS02-006
set /p "SUBSETS=Subsets [all]: "
if "%SUBSETS%"=="" set "SUBSETS=all"

echo Available mixers: attention, ttt_mlp, ttt_multiscale_moe, all
set /p "MIXERS=Mixers [all]: "
if "%MIXERS%"=="" set "MIXERS=all"

set /p "GPUS=Parallel GPU IDs, comma-separated [serial]: "
if "%GPUS%"=="" (
    "%PYTHON%" -m scripts.run_experiments --dataset "%DATASET%" --subsets "%SUBSETS%" --mixers "%MIXERS%"
) else (
    set /p "JOBS_PER_GPU=Concurrent jobs per GPU [1]: "
    if "!JOBS_PER_GPU!"=="" set "JOBS_PER_GPU=1"
    "%PYTHON%" -m scripts.run_experiments --dataset "%DATASET%" --subsets "%SUBSETS%" --mixers "%MIXERS%" --gpus "%GPUS%" --jobs-per-gpu "!JOBS_PER_GPU!"
)
exit /b %ERRORLEVEL%
