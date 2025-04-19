#!/bin/bash

# Exit on error
set -e

# Configuration
VENV_DIR="venv"
REQUIREMENTS_FILE="requirements.txt"
SCRIPT_FILE="facial_landmark_detection.py"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install requirements
echo "Installing requirements..."
pip install -r "$REQUIREMENTS_FILE"

# Run the script
echo "Running facial landmark detection..."
python "$SCRIPT_FILE"

# Deactivate virtual environment
deactivate 