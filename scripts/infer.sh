#!/bin/bash
# Runs inference on the fine-tuned model
set -e

source ../ML-Env-CUDA13/cuda_clean_env/bin/activate
python inference.py
