# Task: Create LoRA Merge and GGUF Export Script

**Status: Backlog**

## Prerequisites (Completed)

✅ **Task 01-05**: Environment setup (WSL2, NVIDIA, ML-Env-CUDA13, dependencies)  
✅ **Task 22**: Base model downloaded  

**Pending:**  
⏳ **Task 08**: Fine-tuning Iteration 4 (in-progress)  
⏳ **Task 38**: Merge adapter script (to be created)  
⏳ **Task 39**: GGUF conversion script (to be created)  

## Description
Create a Python script to merge the LoRA adapter with the base model and export to GGUF format.

**🔗 Implemented by:**
- **Task 38**: Merge Adapter Script (`scripts/merge_adapter.py`)
- **Task 39**: GGUF Conversion Script (`scripts/convert_to_gguf.py`)

See those tasks for complete implementations.

## Steps
- Create `merge_and_export.py` script
- Load base model in bfloat16 precision
- Load and merge LoRA adapter from `outputs/`
- Save merged model to temporary directory
- Build llama.cpp tools (if not already built)
- Convert merged model to fp16 GGUF
- Quantize to q4_k_m GGUF format
- Save final GGUF file to `outputs/` or `models/` directory

## Resources
- ADR 0002: GGUF and llama.cpp requirements
- Example: Google Colab notebook merge and conversion steps
