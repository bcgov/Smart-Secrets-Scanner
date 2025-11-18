# Task 62: Install Critical CUDA Binaries (Surgical Strike)

**Status:** Done  
**Created:** 2025-11-18  
**Completed:** 2025-11-18  

## Prerequisites (Completed)

✅ **Task 61**: ML environment created  
✅ **Task 01-02**: System infrastructure ready  

## Objective

Perform the precise "surgical strike" installation of CUDA-specific binaries (bitsandbytes, triton, xformers) that require special handling to link correctly with CUDA-enabled PyTorch.

## Requirements

- Execute surgical installation protocol in specific order
- Verify each component installs correctly
- Ensure CUDA compatibility with PyTorch 2.9.0+cu126
- Test all components work together

## Surgical Installation Protocol

### Pre-Flight Verification

```bash
# Activate environment
source ~/ml_env/bin/activate

# Verify PyTorch CUDA version
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
# Expected: PyTorch: 2.9.0+cu126 CUDA: True
```

### Step A: Confirm Environment Basics

```bash
which python
python -V
pip --version
python -c "import torch; print('torch:', torch.__version__, 'cuda_available:', torch.cuda.is_available())"
```

### Step B: Clean Slate

```bash
pip uninstall -y bitsandbytes triton xformers || true
pip install --upgrade pip setuptools wheel
```

### Step C: Install Triton 3.1.0

```bash
pip install --force-reinstall "triton==3.1.0"

# Verify Triton import
python - <<'PY'
try:
    import triton
    print("triton OK:", triton.__version__)
except Exception as e:
    print("triton import failed:", repr(e))
    raise
PY
```

### Step D: Check BitsAndBytes Wheels

```bash
pip index versions bitsandbytes --extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu126
```

### Step E: Install BitsAndBytes with CUDA

```bash
pip install --force-reinstall --no-cache-dir bitsandbytes==0.48.2 --no-deps \
  --extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu126
```

### Step F: Install Xformers

```bash
pip install xformers
```

### Step G: Compatibility Mitigation

```bash
pip install "fsspec<=2024.3.1"
```

### Step H: Final Verification

```bash
python - <<'PY'
import importlib, pathlib
def try_import(name):
    try:
        m = importlib.import_module(name)
        print(f"{name} imported, ver:", getattr(m,'__version__', None), "file:", getattr(m,'__file__', None))
    except Exception as e:
        print(f"{name} import failed:", repr(e))

try_import('triton')
try_import('bitsandbytes')

# Check native libs
try:
    import bitsandbytes as bnb
    p = pathlib.Path(bnb.__file__).parent
    found = False
    for f in p.glob("libbitsandbytes*"):
        print("native lib:", f)
        found = True
    if not found:
        print("no libbitsandbytes native libs found (likely CPU-only install)")
except Exception as e:
    print("bitsandbytes inspect failed:", repr(e))
PY
```

## Expected Results

```
triton imported, ver: 3.5.0 file: /home/user/ml_env/lib/python3.11/site-packages/triton/__init__.py
bitsandbytes imported, ver: 0.48.2 file: /home/user/ml_env/lib/python3.11/site-packages/bitsandbytes/__init__.py
native lib: /home/user/ml_env/lib/python3.11/site-packages/bitsandbytes/libbitsandbytes_cuda126.so
```

## Troubleshooting

### Accelerator Version Conflicts

If you encounter `TypeError: Accelerator.unwrap_model() got an unexpected keyword argument 'keep_torch_compile'`:

```bash
pip install --upgrade accelerate
```

### BitsAndBytes Installation Fails

```bash
# Clear pip cache
pip cache purge

# Try alternative installation
pip install bitsandbytes --upgrade --force-reinstall --no-cache-dir
```

### Triton Import Issues

```bash
# Check Triton version compatibility
pip show triton
# Should be 3.5.0 (pulled by xformers)
```

## Key Technical Details

- **PyTorch Version**: Must be 2.9.0+cu126 exactly
- **CUDA Version**: 12.6 required for bitsandbytes 0.48.2
- **Installation Order**: Critical - Triton first, then bitsandbytes, then xformers
- **Version Pinning**: Specific versions tested and validated

## Outcome

✅ CUDA binaries installed with correct linking  
✅ bitsandbytes, triton, xformers working with GPU  
✅ Environment ready for model training  
✅ Surgical strike protocol validated  

## Related Tasks

- Task 61: Run all-in-one environment setup (prerequisite)
- Task 63: Build llama-cpp-python bridge (next)
- Task 11: Train LoRA adapter (uses these CUDA binaries)</content>
<parameter name="filePath">c:\Users\RICHFREM\source\repos\Smart-Secrets-Scanner\tasks\done\62-install-cuda-binaries-surgical-strike.md