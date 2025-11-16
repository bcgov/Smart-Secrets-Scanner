#!/bin/bash
# Runs the fine-tuning Python script
set -e

source ~/ml_env/bin/activate
python fine_tune.py
