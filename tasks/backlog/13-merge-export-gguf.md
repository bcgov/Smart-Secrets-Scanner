# Task: Merge LoRA and Export to GGUF

**Status: Backlog**

## Prerequisites (Completed)

✅ **Task 01-05**: Environment setup (WSL2, NVIDIA, ML-Env-CUDA13, dependencies)  
✅ **Task 22**: Base model downloaded  

**Pending:**  
⏳ **Task 08**: Fine-tuning Iteration 4 (in-progress - need trained adapter)  
⏳ **Task 12/38**: Merge script created  
⏳ **Task 39**: GGUF conversion script created  

## Description
Merge the trained LoRA adapter with the base model and export to GGUF format for deployment.

## Steps
- Activate ML-Env-CUDA13 Python environment
- Run `python merge_and_export.py`
- Wait for merge process (may take several minutes)
- Wait for GGUF conversion and quantization
- Verify GGUF file created in output directory
- Check file size and format

## Dependencies
- Task 11 (Train LoRA Adapter) must be completed
- Task 12 (Create Merge Script) must be completed

## Resources
- ADR 0002: GGUF export tooling
