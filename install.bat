@echo off
setlocal enabledelayedexpansion

title JPEG Restorer - Inference App Installer
echo ============================================================
echo   JPEG Restorer Inference App - Installer
echo ============================================================
echo.

cd /d "%~dp0"

:: ============================================================
:: Step 1: Find Python 3.11
:: ============================================================
set "PYTHON="
set "PYVER="

py -3.11 --version >nul 2>&1
if !errorlevel! equ 0 (
    set "PYTHON=py -3.11"
    for /f "tokens=2" %%i in ('py -3.11 --version 2^>^&1') do set "PYVER=%%i"
    echo Found Python !PYVER! via py launcher.
    goto :found_python
)

python --version >nul 2>&1
if !errorlevel! equ 0 (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set "PYVER=%%i"
    echo !PYVER! | findstr /r "^3\.11\." >nul
    if !errorlevel! equ 0 (
        set "PYTHON=python"
        echo Found Python !PYVER! on PATH.
        goto :found_python
    )
    echo Found Python !PYVER! but need 3.11.x.
)

for %%D in (
    "%LocalAppData%\Programs\Python\Python311\python.exe"
    "C:\Python311\python.exe"
    "%ProgramFiles%\Python311\python.exe"
    "%LocalAppData%\Programs\Python\Python311-64\python.exe"
) do (
    if exist %%D (
        set "PYTHON=%%D"
        for /f "tokens=2" %%i in ('%%D --version 2^>^&1') do set "PYVER=%%i"
        echo Found Python !PYVER! at %%D
        goto :found_python
    )
)

:: ============================================================
:: Step 2: Install Python 3.11 if not found
:: ============================================================
echo.
echo Python 3.11 not found. Downloading and installing...
echo.

set "INSTALLER=%TEMP%\python-3.11.9-amd64.exe"
set "DL_URL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"

echo Downloading Python 3.11.9...
powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object Net.WebClient).DownloadFile('!DL_URL!', '!INSTALLER!')" 2>nul

if not exist "!INSTALLER!" (
    echo PowerShell download failed, trying curl...
    curl -L -o "!INSTALLER!" "!DL_URL!" 2>nul
)

if not exist "!INSTALLER!" (
    echo.
    echo ERROR: Could not download Python 3.11.9.
    echo Please install Python 3.11 from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Installing Python 3.11.9 silently...
"!INSTALLER!" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0 Include_pip=1

if !errorlevel! neq 0 (
    echo ERROR: Python installation failed.
    pause
    exit /b 1
)

del "!INSTALLER!" 2>nul

for /f "tokens=2*" %%A in ('reg query "HKCU\Environment" /v PATH 2^>nul') do set "USER_PATH=%%B"
for /f "tokens=2*" %%A in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v PATH 2^>nul') do set "SYS_PATH=%%B"
set "PATH=!USER_PATH!;!SYS_PATH!"

py -3.11 --version >nul 2>&1
if !errorlevel! equ 0 (
    set "PYTHON=py -3.11"
    echo Python 3.11 installed successfully.
    goto :found_python
)

set "NEWPY=!LocalAppData!\Programs\Python\Python311\python.exe"
if exist "!NEWPY!" (
    set "PYTHON=!NEWPY!"
    echo Python 3.11 installed.
    goto :found_python
)

echo ERROR: Could not locate Python after install. Close and reopen terminal.
pause
exit /b 1

:: ============================================================
:: Step 3: Create venv
:: ============================================================
:found_python
echo.
echo ============================================================
echo   Setting up virtual environment...
echo ============================================================
echo.

if exist "venv\Scripts\python.exe" (
    echo Virtual environment already exists. Reusing.
) else (
    echo Creating virtual environment...
    !PYTHON! -m venv venv
    if !errorlevel! neq 0 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment created.
)

:: ============================================================
:: Step 4: Ask GPU and install PyTorch
:: ============================================================
echo.
echo ============================================================
echo   Installing dependencies...
echo ============================================================
echo.

call venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet

echo.
echo ============================================================
echo   PyTorch Configuration
echo ============================================================
echo.
echo   Do you have an NVIDIA GPU with CUDA support?
echo.
echo     [1] Yes - CUDA-accelerated PyTorch ^(~2.4 GB^)
echo     [2] No  - CPU-only PyTorch ^(~200 MB^)
echo.
choice /c 12 /n /m "Select [1] or [2] (default 2): "
if !errorlevel! equ 1 goto :install_cuda
if !errorlevel! equ 2 goto :install_cpu
goto :install_cpu

:install_cuda
echo.
echo Installing PyTorch with CUDA 12.1 support...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
if !errorlevel! neq 0 (
    echo CUDA 12.1 failed, trying CUDA 11.8...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    if !errorlevel! neq 0 (
        echo CUDA install failed, trying default PyPI...
        pip install torch torchvision
    )
)
goto :deps_done

:install_cpu
echo.
echo Installing PyTorch CPU build...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if !errorlevel! neq 0 (
    echo CPU index failed, trying default PyPI...
    pip install torch torchvision
)
goto :deps_done

:deps_done
echo.
echo Installing remaining dependencies...
pip install numpy Pillow PyQt6 --quiet

:: ============================================================
:: Step 5: Verify
:: ============================================================
echo.
echo ============================================================
echo   Verifying installation...
echo ============================================================
echo.

python -c "import torch; print('PyTorch', torch.__version__)" 2>nul
if !errorlevel! neq 0 echo WARNING: PyTorch import failed.
python -c "import torchvision; print('torchvision', torchvision.__version__)" 2>nul
python -c "import numpy; print('NumPy', numpy.__version__)" 2>nul
python -c "import PIL; print('Pillow', PIL.__version__)" 2>nul
python -c "from PyQt6.QtCore import QT_VERSION_STR; print('PyQt6 (Qt', QT_VERSION_STR + ')')" 2>nul
python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>nul

echo.
echo ============================================================
echo   Installation complete!
echo ============================================================
echo.
echo To run the app, double-click:  run.bat
echo.
pause
