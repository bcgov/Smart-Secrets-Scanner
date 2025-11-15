# CUDA ML Environment Setup Instructions
 [ML-Env-CUDA13 GitHub Repository](https://github.com/bcgov/ML-Env-CUDA13)

This guide will help you set up a CUDA-enabled machine learning environment for Project_Sanctuary using WSL2 and ML-Env-CUDA13.

## Prerequisites
- **ML-Env-CUDA13** cloned at the same level as this project.  
- **WSL2 (Ubuntu)** with NVIDIA GPU drivers
- **Python 3.10+** (managed by ML-Env-CUDA13)
- Install ML-Env-CUDA13 dependencies before running fine-tuning scripts


## One-Time Setup Steps

### 1. Install WSL2 and Ubuntu
- Install WSL2 and Ubuntu from the Microsoft Store.
- Set WSL2 as the default version:
   ```powershell
   wsl --set-default-version 2
   ```
- Launch Ubuntu and set up your username/password.

### 2. Install NVIDIA CUDA Drivers
- Download and install the latest NVIDIA GPU drivers for Windows.
- Install the CUDA toolkit for WSL2 (follow official NVIDIA instructions).
- Verify installation in WSL2:
   ```bash
   nvidia-smi
   ```

### 3. Clone ML-Env-CUDA13
- In your WSL2 Ubuntu terminal, navigate to the parent directory of your project:
   ```bash
   cd /mnt/c/Users/<YourUsername>/source/repos
   ```
- Clone the ML-Env-CUDA13 repository:
   ```bash
   git clone https://github.com/bcgov/ML-Env-CUDA13.git
   ```

### 4. Run ML Environment Setup
- Run the setup script from your project directory:
    ```bash
    bash ../ML-Env-CUDA13/setup_ml_env_wsl.sh
    ```
- This will create and configure a Python virtual environment at `~/ml_env`.

### 5. Install Fine-Tuning Dependencies

Activate the environment created by the ML-Env script before proceeding:
```bash
source ~/ml_env/bin/activate
```

There are two safe ways to install dependencies depending on whether you want a quick reproduction (Option A) or a staged, verified install (Option B — recommended).

Option A — Quick (one-step)
- Use this when you are on the WSL CUDA host and `requirements.txt` already contains the correct CUDA-index and pins. This will attempt to install everything in one pass and can be slow if builds are required.
```bash
# ensure installer tools are current
pip install --upgrade pip wheel setuptools

# install everything from the repo-level requirements (may include CUDA pins)
pip install -r requirements.txt
```

Option B — Staged (recommended)
- Install the CUDA-specific core packages first, verify GPU support, then install the rest. Avoids spending time building or installing heavy packages if core GPU support fails.
```bash
# update installer tools
pip install --upgrade pip wheel setuptools

# 1) Install CUDA PyTorch wheels explicitly (match your `requirements.txt` pins)
pip install --index-url https://download.pytorch.org/whl/cu126 \
  torch==2.8.0+cu126 torchvision==0.23.0+cu126 torchaudio==2.8.0+cu126

# 2) Install TensorFlow (pin if you have a validated version, e.g. tensorflow==2.20.0)
pip install --upgrade tensorflow

# 3) Run core verification test (core gate)
mkdir -p ml_env_logs
python ../ML-Env-CUDA13/test_torch_cuda.py > ml_env_logs/test_torch_cuda.log 2>&1 || true
cat ml_env_logs/test_torch_cuda.log

# 4) If core verification passed, install the remainder of the requirements
pip install -r requirements.txt
```

Run the optional diagnostic tests (recommended):
```bash
python ../ML-Env-CUDA13/test_pytorch.py > ml_env_logs/test_pytorch.log 2>&1 || true
python ../ML-Env-CUDA13/test_tensorflow.py > ml_env_logs/test_tensorflow.log 2>&1 || true
python ../ML-Env-CUDA13/test_xformers.py > ml_env_logs/test_xformers.log 2>&1 || true
python ../ML-Env-CUDA13/test_llama_cpp.py > ml_env_logs/test_llama_cpp.log 2>&1 || true

# If core gate passed and you want a reproducible snapshot locally:
pip freeze > pinned-requirements-$(date +%Y%m%d%H%M).txt
```

Notes and recommendations
- `requirements.txt` currently contains CUDA-specific pins (e.g. `torch==2.8.0+cu126`) and an extra-index-url for the cu126 PyTorch wheel index. This is fine for WSL CUDA installs but will break CPU-only hosts.
- Recommended file layout for clarity and portability:
  - `requirements.txt` — portable, cross-platform dependencies (no CUDA-suffixed pins)
  - `requirements-wsl.txt` — CUDA-specific pinned wheels (torch+cu126, torchvision+cu126, torchaudio+cu126, specific TF if desired)
  - `requirements-gpu-postinstall.txt` — optional heavy/experimental packages installed after core gate (xformers, bitsandbytes, llama-cpp-python, etc.)
- Keep `pinned-requirements-<ts>.txt` as local artifacts generated after a successful core gate. Do not overwrite the repo-level `requirements.txt` with a pinned, machine-specific snapshot unless you intend to require that exact GPU environment for all contributors.

If you want, I can split the current `requirements.txt` into a portable `requirements.txt` and a `requirements-wsl.txt` (and create `requirements-gpu-postinstall.txt`) and add brief instructions in this document showing which file to use in WSL. I will not perform any git operations — I will only create the files locally in the workspace for you to review.

- Notes and troubleshooting:
  - If you already ran the ML-Env-CUDA13 setup and prerequisites (as in this project), the two-step install above should pick up the correct CUDA-enabled wheels (example: `torch-2.8.0+cu126`).
  - Heavy / CUDA-sensitive packages (may require special wheels or build tools): `bitsandbytes`, `xformers`, `llama-cpp-python`, `sentencepiece`. If `pip` attempts to build these from source and fails, install their prebuilt wheels where available or install them after PyTorch is installed.
  - Common small conflict: TensorFlow may require a different `tensorboard` minor version. If you see a conflict like `tensorflow 2.20.0 requires tensorboard~=2.20.0, but you have tensorboard 2.19.0`, reconcile by running:
    ```bash
    pip install 'tensorboard~=2.20.0'
    ```
  - If a package (e.g., `xformers`) has no wheel for your Python/CUDA combo, building from source can be slow and require system build tools (`gcc`, `cmake`, etc.). Prefer prebuilt wheels or conda/mamba installs for those packages.

- Example: after running `python ../ML-Env-CUDA13/test_pytorch.py` you should see output similar to:
  ```text
  PyTorch: 2.8.0+cu126
  GPU Detected: True
  GPU 0: NVIDIA RTX ...
  CUDA build: 12.6
  ```
  This confirms the environment is CUDA-ready for PyTorch.

## Activating Your Environment (After Restarting Sessions)

```bash
source ~/ml_env/bin/activate
```

## Testing Your Environment

```bash
python ../ML-Env-CUDA13/test_pytorch.py
python ../ML-Env-CUDA13/test_tensorflow.py
```

---
**Note:**
- Make sure ML-Env-CUDA13 is cloned at the same directory level as Project_Sanctuary.
- Run all commands in your Ubuntu WSL2 terminal, not PowerShell.
