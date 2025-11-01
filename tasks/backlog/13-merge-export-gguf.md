# Task: Merge LoRA and Export to GGUF

**Status: Backlog**

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
