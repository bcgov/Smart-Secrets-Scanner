# Task: Run ML-Env-CUDA13 Setup Script

**Status: Done**

## Prerequisites (Completed)

✅ **Task 01**: WSL2 Ubuntu configured  
✅ **Task 02**: NVIDIA drivers and CUDA Toolkit installed  
✅ **Task 03**: ML-Env-CUDA13 repository cloned  

## Description
Run the ML-Env-CUDA13 setup script in WSL2 Ubuntu to create the Python virtual environment and install base dependencies.

## Simple Steps
1. Open the Windows Start menu and type "Ubuntu" to launch your WSL Ubuntu terminal.
2. In the Ubuntu prompt, change directory to the ML-Env-CUDA13 folder:
   ```bash
   cd ~/repos/ML-Env-CUDA13
   ```
3. Run the setup script:
   ```bash
   bash setup_ml_env_wsl.sh
   ```
4. When finished, activate the environment:
   ```bash
   source ~/ml_env/bin/activate
   ```
5. Test your setup:
   ```bash
   python test_pytorch.py
   python test_tensorflow.py
   ```

## Steps
- Open WSL2 Ubuntu terminal
- Navigate to ML-Env-CUDA13 directory
- Run `bash setup_ml_env_wsl.sh`
- Activate environment: `source ~/ml_env/bin/activate`
- Verify with `python test_pytorch.py` and `python test_tensorflow.py`

## Resources
- ML-Env-CUDA13 README: Setup steps for WSL2/Ubuntu
