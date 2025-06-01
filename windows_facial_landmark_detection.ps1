# Exit on error
$ErrorActionPreference = "Stop"

# Configuration
$VENV_DIR = ".venv"
$REQUIREMENTS_FILE = "requirements.txt"
$SCRIPT_FILE = "facial_landmark_detection.py"

# Create virtual environment if it doesn't exist
if (-Not (Test-Path $VENV_DIR)) {
    Write-Host "Creating Python virtual environment..."
    python -m venv $VENV_DIR
}

# Activate virtual environment
$activateScript = Join-Path $VENV_DIR "Scripts\Activate.ps1"
Write-Host "Activating virtual environment..."
& $activateScript

# Install requirements
Write-Host "Installing requirements..."
pip install -r $REQUIREMENTS_FILE

# Run the script
Write-Host "Running facial landmark detection..."
python $SCRIPT_FILE

# Deactivate virtual environment
deactivate
