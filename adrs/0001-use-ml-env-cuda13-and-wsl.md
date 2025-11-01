# ADR 0001: Use ML-Env-CUDA13 and WSL2 Ubuntu for GPU Fine-Tuning

## Status
Accepted

## Context
Fine-tuning large language models (LLMs) with GPU acceleration requires a reliable CUDA and Python environment. Native Windows support for CUDA and many ML libraries (PyTorch, TensorFlow, bitsandbytes, etc.) is limited and error-prone. The [ML-Env-CUDA13](https://github.com/bcgov/ML-Env-CUDA13) project provides a robust, reproducible setup for CUDA 13 and Python in WSL2 Ubuntu.

## Decision
- This project will **depend on ML-Env-CUDA13** for all GPU environment setup and management.
- All fine-tuning and inference scripts will be written for **WSL2 Ubuntu** (not Windows PowerShell or CMD).
- All Python libraries and CUDA dependencies will be installed and managed within the ML-Env-CUDA13 environment.
- The ML-Env-CUDA13 repo will be kept as a sibling directory, not copied as a subfolder.

## Consequences
- Users must clone and set up ML-Env-CUDA13 before running any scripts in this project.
- All documentation and scripts will assume a Linux (WSL2 Ubuntu) environment.
- This approach avoids common Windows-specific issues with ML libraries and GPU drivers.
- Future upgrades to CUDA or Python can be managed centrally in ML-Env-CUDA13.

## Alternatives Considered
- Native Windows setup (rejected due to poor compatibility and reliability)
- Docker-based environment (not chosen for simplicity and direct GPU access in WSL2)

---
