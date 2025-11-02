# Task: Train LoRA Adapter

**Status: Done**

## Prerequisites (Completed)

✅ **Task 01-05**: Environment setup  
✅ **Task 22**: Base model downloaded  
✅ **Task 20**: Dataset created  
✅ **Task 30**: Training config created  
✅ **Task 36**: Training script created  

**Note:** This was completed through multiple iterations (1-3). See Task 08 for Iteration 4.

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
