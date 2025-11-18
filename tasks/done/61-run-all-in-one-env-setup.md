# Task 61: Run All-in-One ML Environment Setup

**Status:** Done  
**Created:** 2025-11-18  
**Completed:** 2025-11-18  

## Prerequisites (Completed)

✅ **Task 60**: Hugging Face authentication configured  
✅ **Task 59**: llama.cpp tools built  
✅ **Task 58**: Repository structure verified  

## Objective

Execute the comprehensive environment setup script that creates the Python virtual environment, installs all ML dependencies, and configures the complete development environment for Smart-Secrets-Scanner.

## Requirements

- Run setup_cuda_env.py with sudo (installs system packages)
- Use --staged --recreate flags for clean environment
- Verify environment creation and basic package installation
- Environment ready for CUDA binaries installation

## Implementation

### 1. Pre-Flight Checks

```bash
# Verify current environment
which python3
python3 --version

# Check available disk space (ML models need ~50GB)
df -h .

# Verify sudo access (required for system packages)
sudo -n true && echo "✅ Sudo access confirmed" || echo "❌ Sudo access required"
```

### 2. Run Setup Script

```bash
# Execute with sudo for system package installation
sudo python3 scripts/setup_cuda_env.py --staged --recreate

# Expected output:
# - System packages installation (python3.11, git-lfs, etc.)
# - Virtual environment creation (~ml_env)
# - Python dependencies installation
# - CUDA toolkit verification
```

### 3. Verify Environment Creation

```bash
# Check virtual environment exists
ls -la ~/ml_env/
ls -la ~/ml_env/bin/

# Verify Python version in environment
~/ml_env/bin/python --version  # Should be 3.11.x

# Check key packages installed
~/ml_env/bin/pip list | grep -E "(torch|transformers|accelerate)"
```

### 4. Environment Activation Test

```bash
# Test environment activation
source ~/ml_env/bin/activate
which python  # Should point to ~/ml_env/bin/python
python -c "import torch; print('PyTorch:', torch.__version__)"
deactivate
```

## What the Script Does

### System Packages Installation
- **python3.11**: Required Python version for ML libraries
- **python3.11-venv**: Virtual environment support
- **git-lfs**: Large file support for model downloads
- **build-essential**: Compilation tools
- **cmake**: Build system for C++ components

### Python Environment Setup
- **Virtual Environment**: `~/ml_env` with Python 3.11
- **Core ML Libraries**:
  - PyTorch 2.9.0+cu126 (CUDA-enabled)
  - transformers, peft, accelerate, bitsandbytes
  - trl, datasets, xformers
- **Model Conversion**: llama-cpp-python (CPU version initially)
- **Development Tools**: jupyter, various utilities

### CUDA Integration
- **CUDA Toolkit**: Verified compatibility (13.x required)
- **GPU Detection**: Confirms NVIDIA GPU available
- **PyTorch CUDA**: Ensures CUDA-enabled PyTorch installation

## Expected Output

```
✅ System packages installed successfully
✅ Virtual environment created at ~/ml_env
✅ Python dependencies installed
✅ CUDA toolkit verified
✅ Environment ready for next steps
```

## Troubleshooting

### Sudo Permission Denied
```bash
# Check sudo configuration
sudo -l

# Contact system administrator if needed
```

### Disk Space Insufficient
```bash
# Check space requirements
df -h
# Need ~10GB free for environment + models
```

### Network Issues
```bash
# Test internet connectivity
ping -c 3 pypi.org

# Use mirror if needed
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## Files Created

- `~/ml_env/` - Python virtual environment
- `~/ml_env/bin/python` - Python 3.11 executable
- `~/ml_env/lib/python3.11/site-packages/` - Installed packages

## Outcome

✅ Complete ML environment created and configured  
✅ All Python dependencies installed with CUDA support  
✅ Ready for surgical strike CUDA binaries installation  
✅ Foundation ready for model training and deployment  

## Next Steps

After this task:
1. **Task 62**: Install critical CUDA binaries (surgical strike)
2. **Task 63**: Build llama-cpp-python bridge
3. **Task 64**: Verify complete environment

## Related Tasks

- Task 60: Setup Hugging Face authentication (prerequisite)
- Task 62: Install CUDA binaries (next)
- Task 28: Create requirements.txt (covered by this script)</content>
<parameter name="filePath">c:\Users\RICHFREM\source\repos\Smart-Secrets-Scanner\tasks\done\61-run-all-in-one-env-setup.md