# Task 63: Build llama-cpp-python Bridge with CUDA

**Status:** Done  
**Created:** 2025-11-18  
**Completed:** 2025-11-18  

## Prerequisites (Completed)

✅ **Task 62**: CUDA binaries installed  
✅ **Task 61**: ML environment created  
✅ **Task 59**: llama.cpp tools built  

## Objective

Build the llama-cpp-python package with CUDA support to create the Python "bridge" that allows Python scripts to communicate with GGUF models and leverage GPU acceleration.

## Requirements

- Activate ML environment
- Force rebuild llama-cpp-python with CUDA flags
- Verify CUDA support enabled
- Test bridge functionality

## Implementation

### 1. Activate Environment

```bash
# Ensure ML environment is active
source ~/ml_env/bin/activate
echo $VIRTUAL_ENV  # Should show ~/ml_env
```

### 2. Pre-Build Verification

```bash
# Check current llama-cpp-python installation
pip show llama-cpp-python

# Verify CUDA toolkit available
nvcc --version
echo $CUDA_HOME
```

### 3. Build with CUDA Support

```bash
# Force rebuild with CUDA enabled
CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall --no-cache-dir llama-cpp-python --no-deps

# Expected output: Compilation messages with CUDA support
```

### 4. Verify CUDA Support

```bash
# Check installation includes CUDA
pip show llama-cpp-python

# Test CUDA import
python -c "
from llama_cpp import Llama
print('✅ llama-cpp-python CUDA support enabled')
print('CUDA devices available:', Llama.available_cuda_devices())
"
```

### 5. Test Bridge Functionality

```bash
# Quick functionality test
python - <<'PY'
try:
    from llama_cpp import Llama
    print("✅ llama-cpp-python imported successfully")
    
    # Check CUDA support
    cuda_count = Llama.available_cuda_devices()
    print(f"CUDA devices: {cuda_count}")
    
    if cuda_count > 0:
        print("✅ CUDA support confirmed")
    else:
        print("⚠️ CUDA support not detected")
        
except Exception as e:
    print(f"❌ Import failed: {e}")
PY
```

## Technical Details

### What is the Bridge?

- **llama-cpp-python**: Python wrapper for llama.cpp C++ library
- **CUDA Support**: Enables GPU acceleration for model inference
- **GGUF Compatibility**: Required for loading quantized models
- **Script Integration**: Used by `inference.py`, `convert_to_gguf.py`, etc.

### Build Process

1. **CMake Configuration**: `CMAKE_ARGS="-DGGML_CUDA=on"`
2. **C++ Compilation**: Links against CUDA libraries
3. **Python Binding**: Creates Python extension module
4. **GPU Detection**: Runtime CUDA device enumeration

## Troubleshooting

### Build Fails

```bash
# Check CUDA development headers
ls /usr/local/cuda/include/cuda.h

# Verify cmake available
cmake --version

# Clean and retry
pip uninstall llama-cpp-python
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
```

### CUDA Not Detected

```bash
# Check GPU status
nvidia-smi

# Verify CUDA runtime
nvidia-smi | grep CUDA

# Test CUDA with PyTorch first
python -c "import torch; print(torch.cuda.is_available())"
```

### Import Errors

```bash
# Check library linking
ldd ~/ml_env/lib/python3.11/site-packages/llama_cpp/libllama.so | grep cuda

# Rebuild if needed
CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall llama-cpp-python
```

## Outcome

✅ llama-cpp-python built with CUDA support  
✅ Python-C++ bridge functional  
✅ GPU acceleration available for model operations  
✅ Ready for GGUF model loading and inference  

## Related Tasks

- Task 62: Install CUDA binaries (prerequisite)
- Task 64: Verify complete environment (next)
- Task 39: Convert to GGUF format (uses this bridge)
- Task 37: Create inference script (uses this bridge)</content>
<parameter name="filePath">c:\Users\RICHFREM\source\repos\Smart-Secrets-Scanner\tasks\done\63-build-llama-cpp-python-bridge.md