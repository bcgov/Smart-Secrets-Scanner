#!/bin/bash
# Runs ML-Env-CUDA13 setup and prepares Python environment
set -e

echo "=== Setting up ML-Env-CUDA13 Environment ==="

# Path to ML-Env-CUDA13 (sibling directory)
ML_ENV_PATH="../ML-Env-CUDA13"

if [ ! -d "$ML_ENV_PATH" ]; then
  echo "ERROR: ML-Env-CUDA13 not found at $ML_ENV_PATH"
  echo "Please clone ML-Env-CUDA13 as a sibling directory:"
  echo "  cd .."
  echo "  git clone https://github.com/bcgov/ML-Env-CUDA13.git"
  exit 1
fi

# Run the WSL setup script (not the Windows PowerShell version)
if [ -f "$ML_ENV_PATH/setup_ml_env_wsl.sh" ]; then
  echo "Running ML-Env-CUDA13 WSL setup script..."
  bash "$ML_ENV_PATH/setup_ml_env_wsl.sh"
else
  echo "ERROR: setup_ml_env_wsl.sh not found in $ML_ENV_PATH"
  exit 1
fi

echo "✅ ML-Env-CUDA13 setup complete!"
