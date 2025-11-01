# Task 22: Download Base Model (Llama 3 8B)

**Status:** Backlog  
**Created:** 2025-11-01  
**Related to:** Phase 2: Model Fine-Tuning (Step 4)

## Description
Download the base Llama 3 8B model from Hugging Face to `models/base/Meta-Llama-3-8B/`.

## Requirements
- Hugging Face account and auth token
- ~15-30 GB of disk space
- Internet connection for download

## Acceptance Criteria
- [ ] Model downloaded to `models/base/Meta-Llama-3-8B/`
- [ ] All model files present (pytorch_model.bin, tokenizer, config.json)
- [ ] Model can be loaded in Python with transformers library

## Steps
1. Create Hugging Face account and get auth token
2. Run download script or use `huggingface-cli download`
3. Verify model integrity
4. Document model version and download date

## Dependencies
- Task 04: ML-Env-CUDA13 setup must be complete
- Task 05: Python dependencies installed

## Notes
- Base model is ~15-30 GB depending on precision
- Consider using Llama 3.2 1B for faster experimentation
- Alternative: Download from Meta directly with license agreement
