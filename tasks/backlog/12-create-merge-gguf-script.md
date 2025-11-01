# Task: Create LoRA Merge and GGUF Export Script

**Status: Backlog**

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
