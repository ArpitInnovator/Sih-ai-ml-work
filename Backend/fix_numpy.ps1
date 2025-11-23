# PowerShell script to fix NumPy MINGW-W64 warning by installing proper Windows build
# Usage: .\fix_numpy.ps1

Write-Host "Fixing NumPy installation to remove MINGW-W64 warning..." -ForegroundColor Green

# Activate virtual environment if not already activated
if (-not $env:VIRTUAL_ENV) {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & .\.venv\Scripts\Activate.ps1
}

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Cyan
$pythonVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Write-Host "Python version: $pythonVersion" -ForegroundColor Cyan

# Uninstall current numpy
Write-Host "Uninstalling current NumPy..." -ForegroundColor Cyan
pip uninstall -y numpy 2>$null

# Clear pip cache to ensure fresh download
Write-Host "Clearing pip cache..." -ForegroundColor Cyan
pip cache purge

# Install numpy with only binary wheels (forces Windows wheel, not MINGW-W64)
# Python 3.13 only has numpy 2.x wheels, so we need to use that
Write-Host "Installing NumPy with Windows binary wheel..." -ForegroundColor Cyan
if ($pythonVersion -ge "3.13") {
    Write-Host "Python 3.13 detected - installing NumPy 2.x (latest stable)..." -ForegroundColor Yellow
    pip install --only-binary :all: --no-cache-dir "numpy>=2.1.0"
} else {
    Write-Host "Installing NumPy 1.26.4 (latest 1.x for Python < 3.13)..." -ForegroundColor Cyan
    pip install --only-binary :all: --no-cache-dir numpy==1.26.4
}

# Verify installation
Write-Host "Verifying NumPy installation..." -ForegroundColor Cyan
python -c "import numpy; print(f'NumPy version: {numpy.__version__}'); print(f'NumPy file: {numpy.__file__}')"

Write-Host "NumPy installation fixed! Please restart your server." -ForegroundColor Green

