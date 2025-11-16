#!/bin/bash
# Runs inference on the fine-tuned model
set -e

source ~/ml_env/bin/activate
python inference.py
