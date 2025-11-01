#!/bin/bash
# Runs the fine-tuning Python script
set -e

source ../ML-Env-CUDA13/cuda_clean_env/bin/activate
python fine_tune.py
