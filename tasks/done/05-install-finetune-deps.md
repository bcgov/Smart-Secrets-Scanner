# Task: Install Fine-Tuning Python Libraries

**Status: Backlog**

## Description
Install additional Python libraries required for LLM fine-tuning (transformers, peft, bitsandbytes, etc.) in the ML-Env-CUDA13 environment.

## Steps
- Activate ML-Env-CUDA13 Python environment
- Run `pip install --upgrade pip`
- Run `pip install transformers peft bitsandbytes accelerate sentencepiece torch --index-url https://download.pytorch.org/whl/cu130 --no-cache-dir`

## Resources
- Project README: install_deps.sh script
