#!/bin/bash
# Downloads the base Llama 3 model
set -e

echo "=== Downloading Base Model ==="

# Activate the ML-Env-CUDA13 Python environment
VENV_PATH="$HOME/ml_env"

if [ ! -d "$VENV_PATH" ]; then
  echo "ERROR: ML-Env-CUDA13 virtual environment not found at $VENV_PATH"
  echo "Please run the ML-Env-CUDA13 setup script first"
  exit 1
fi

echo "Activating virtual environment at $VENV_PATH..."
source "$VENV_PATH/bin/activate"

MODEL_DIR="models/base/Meta-Llama-3.1-8B"

if [ -d "$MODEL_DIR" ] && [ "$(ls -A $MODEL_DIR)" ]; then
  echo "Model already downloaded at $MODEL_DIR"
  echo "To re-download, delete the directory first: rm -rf $MODEL_DIR"
  exit 0
fi

echo "Creating model directory..."
mkdir -p "$MODEL_DIR"

# Load token from .env file
if [ -f ".env" ]; then
  echo "Loading Hugging Face token from .env file..."
  # Extract token and strip whitespace/carriage returns
  export HF_TOKEN=$(grep HUGGING_FACE_TOKEN .env | cut -d '=' -f2 | tr -d '\r\n' | xargs)
  
  if [ -z "$HF_TOKEN" ] || [ "$HF_TOKEN" = "your_token_here" ]; then
    echo "ERROR: HUGGING_FACE_TOKEN not set in .env file"
    echo "Please edit .env and add your token from https://huggingface.co/settings/tokens"
    exit 1
  fi
  
  echo "✅ Token loaded from .env"
else
  echo "ERROR: .env file not found"
  echo "Please create .env file with HUGGING_FACE_TOKEN=your_token_here"
  exit 1
fi

echo ""
echo "Downloading Llama 3.1 8B (~15-30 GB)..."
echo "This may take 15-30 minutes depending on your connection speed..."
echo ""

python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id='meta-llama/Llama-3.1-8B', 
    local_dir='$MODEL_DIR', 
    cache_dir='models/base',
    token=os.environ.get('HF_TOKEN')
)
"

echo ""
echo "✅ Model downloaded successfully to $MODEL_DIR"
echo ""
echo "Next steps:"
echo "  1. Validate your dataset: python scripts/validate_dataset.py data/processed/smart-secrets-scanner-train.jsonl"
echo "  2. Start fine-tuning: python scripts/fine_tune.py"
