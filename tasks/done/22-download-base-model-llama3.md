# Task 22: Download Base Model (Llama 3.1 8B)

**Status:** Done ✅  
**Created:** 2025-11-01  
**Completed:** 2025-11-01  
**Related to:** Phase 2: Model Fine-Tuning (Step 4)

## Description
Download the base Llama 3.1 8B model from Hugging Face to `models/base/Meta-Llama-3.1-8B/`.

## Requirements
- Hugging Face account and auth token
- ~15-30 GB of disk space
- Internet connection for download

## Acceptance Criteria
- [x] Model downloaded to `models/base/Meta-Llama-3.1-8B/`
- [x] All model files present (safetensors, tokenizer, config.json)
- [x] Model can be loaded in Python with transformers library

## Steps
1. ✅ Created Hugging Face account and got auth token
2. ✅ Stored token in `.env` file
3. ✅ Updated download script to use Llama 3.1-8B
4. ✅ Ran `bash scripts/download_model.sh`
5. ✅ Verified model integrity (~30 GB downloaded)

## Dependencies
- Task 04: ML-Env-CUDA13 setup must be complete ✅
- Task 05: Python dependencies installed ✅

## Completion Notes
- **Model Version:** Meta-Llama-3.1-8B (upgraded from 3.0)
- **Download Date:** 2025-11-01
- **Size:** ~30 GB (4 safetensors files + original checkpoint)
- **Location:** `models/base/Meta-Llama-3.1-8B/`
- **Files Downloaded:**
  - `model-00001-of-00004.safetensors` through `model-00004-of-00004.safetensors`
  - `original/consolidated.00.pth`
  - Config files, tokenizer, and metadata
- **Token Storage:** Hugging Face token stored securely in `.env` (gitignored)
- **Script Fixed:** Updated `scripts/download_model.sh` to strip whitespace from token

## Next Steps
1. Validate dataset: `python scripts/validate_dataset.py data/processed/smart-secrets-scanner-train.jsonl` ✅
2. Start fine-tuning: `python scripts/fine_tune.py`
