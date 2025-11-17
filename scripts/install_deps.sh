#!/bin/bash
# install_deps.sh - Dependency installation script
# This script installs Python dependencies for Smart Secrets Scanner

echo "📦 Installing Smart Secrets Scanner dependencies..."
echo "Note: This script assumes ML-Env-CUDA13 environment is activated."
echo ""

# Check if in virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected. Please activate ML-Env-CUDA13 first:"
    echo "   source ~/ml_env/bin/activate"
    exit 1
fi

# Install dependencies
echo "Installing from requirements.txt..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
    echo ""
    echo "Next steps:"
    echo "1. Test installation: python scripts/test_pytorch.py"
    echo "2. Download model: bash scripts/download_model.sh"
else
    echo "❌ Dependency installation failed"
    exit 1
fi