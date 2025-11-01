# Task: Train LoRA Adapter

**Status: Backlog**

## Description
Run the training script to fine-tune a LoRA adapter on your custom dataset.

## Steps
- Ensure dataset is prepared in `data/` folder
- Activate ML-Env-CUDA13 Python environment
- Run `python fine_tune.py` or `bash scripts/fine_tune.sh`
- Monitor GPU usage with `nvidia-smi`
- Verify LoRA adapter saved in `outputs/` directory
- Review training logs for loss and performance metrics

## Dependencies
- Task 07 (Create Dataset) must be completed
- Task 10 (Create Training Script) must be completed

## Resources
- ADR 0002: Library requirements
