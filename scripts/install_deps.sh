#!/bin/bash
# Installs required Python libraries for fine-tuning
set -e

echo "=== Installing Fine-Tuning Dependencies ==="

# Activate the ML-Env-CUDA13 Python environment
# ML-Env-CUDA13 creates the venv in ~/ml_env
VENV_PATH="$HOME/ml_env"

if [ ! -d "$VENV_PATH" ]; then
  echo "ERROR: ML-Env-CUDA13 virtual environment not found at $VENV_PATH"
  echo "Please run the ML-Env-CUDA13 setup script first"
  exit 1
fi

echo "Activating virtual environment at $VENV_PATH..."
source "$VENV_PATH/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing core fine-tuning libraries..."
# Note: torch already installed via ML-Env-CUDA13 with cu121
pip install transformers peft bitsandbytes accelerate sentencepiece \
    datasets trl huggingface_hub tqdm pyyaml

echo "✅ All fine-tuning dependencies installed successfully!"
echo ""
echo "Installed packages:"
echo "  - transformers (Hugging Face models)"
echo "  - peft (LoRA/QLoRA adapters)"
echo "  - bitsandbytes (4-bit quantization)"
echo "  - accelerate (distributed training)"
echo "  - trl (SFTTrainer for fine-tuning)"
echo "  - datasets (data loading)"
echo "  - pyyaml (config file support)"
