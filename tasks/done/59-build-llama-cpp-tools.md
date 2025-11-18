# Task 59: Build llama.cpp Tools with CUDA Support

**Status:** Done  
**Created:** 2025-11-18  
**Completed:** 2025-11-18  

## Prerequisites (Completed)

✅ **Task 58**: Repository structure verified  
✅ **Task 01**: WSL2 Ubuntu setup  
✅ **Task 02**: NVIDIA drivers installed  

## Objective

Compile the llama.cpp C++/CUDA executables required for model conversion and quantization. This builds the core "engine" that Python scripts use for heavy GPU-accelerated computations.

## Requirements

- Navigate to llama.cpp sibling directory
- Configure build with CMake and CUDA support
- Compile executables (llama-cli, quantize, etc.)
- Verify build success
- Return to project directory

## Build Process

### 1. Navigate to llama.cpp Directory

```bash
# From Smart-Secrets-Scanner root
cd ../llama.cpp
pwd  # Should show: /path/to/parent/llama.cpp
```

### 2. Configure Build with CUDA

```bash
# Create build directory and configure with CMake
cmake -B build -DGGML_CUDA=ON

# Expected output: CUDA toolkit found, GGML_CUDA enabled
```

### 3. Compile Executables

```bash
# Build the release version (takes 5-15 minutes)
cmake --build build --config Release

# Expected output: Various compilation messages ending with success
```

### 4. Verify Build

```bash
# Check that executables were created
ls -la build/bin/

# Test llama-cli version
./build/bin/llama-cli --version

# Expected output: Version information
```

### 5. Return to Project Directory

```bash
# Return to Smart-Secrets-Scanner
cd ../Smart-Secrets-Scanner
pwd  # Should show: /path/to/Smart-Secrets-Scanner
```

## Files Created

- `../llama.cpp/build/bin/llama-cli` - Main inference executable
- `../llama.cpp/build/bin/quantize` - Model quantization tool
- `../llama.cpp/build/bin/convert-hf-to-gguf.py` - HF to GGUF converter
- Various other CUDA-accelerated executables

## Troubleshooting

### CMake CUDA Not Found
```bash
# Check CUDA installation
nvcc --version
echo $CUDA_HOME

# Reinstall CUDA if needed
# See ML-Env-CUDA13 setup instructions
```

### Build Fails
```bash
# Clean and retry
rm -rf build/
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```

### GPU Memory Issues
```bash
# Check GPU status
nvidia-smi

# Ensure no other processes using GPU
```

## Outcome

✅ llama.cpp tools compiled with CUDA support  
✅ GPU-accelerated executables available for model conversion  
✅ Ready for Python environment setup and model operations  

## Key Notes

- **One-time setup**: This build is separate from Python environment
- **CUDA required**: GGML_CUDA=ON enables GPU acceleration
- **Sibling directory**: llama.cpp must remain as ../llama.cpp relative to project
- **No rebuild needed**: Unless updating llama.cpp repository

## Related Tasks

- Task 58: Verify repository structure (prerequisite)
- Task 60: Setup Hugging Face authentication (next)
- Task 61: Run all-in-one environment setup (next)</content>
<parameter name="filePath">c:\Users\RICHFREM\source\repos\Smart-Secrets-Scanner\tasks\done\59-build-llama-cpp-tools.md