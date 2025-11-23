# PowerShell script to install dependencies in correct order
# This ensures numpy is installed before pandas

Write-Host "Installing dependencies..." -ForegroundColor Green

# Activate virtual environment if not already activated
if (-not $env:VIRTUAL_ENV) {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & .\.venv\Scripts\Activate.ps1
}

# Install numpy first (required for pandas)
# Force installation of Windows binary wheels (not MINGW-W64) to avoid experimental build warnings
Write-Host "Installing numpy (Windows binary wheel)..." -ForegroundColor Cyan

# Check Python version to determine which numpy version to install
$pythonVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
pip uninstall -y numpy 2>$null

# Python 3.13 only has numpy 2.x wheels available
if ($pythonVersion -ge "3.13") {
    Write-Host "Python 3.13 detected - installing NumPy 2.x..." -ForegroundColor Yellow
    pip install --only-binary :all: --no-cache-dir "numpy>=2.1.0"
} else {
    Write-Host "Installing NumPy 1.26.4 (latest 1.x)..." -ForegroundColor Cyan
    pip install --only-binary :all: --no-cache-dir numpy==1.26.4
}

# Install pandas
Write-Host "Installing pandas..." -ForegroundColor Cyan
pip install pandas==2.1.3

# Install remaining dependencies
Write-Host "Installing remaining dependencies..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "All dependencies installed successfully!" -ForegroundColor Green

