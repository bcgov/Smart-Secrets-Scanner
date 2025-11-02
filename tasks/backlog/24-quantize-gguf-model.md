# Task 24: Quantize GGUF Model

**Status:** Backlog  
**Created:** 2025-11-01  
**Related to:** Phase 3: Model Export (Step 9)

## Prerequisites (Completed)

✅ **Task 01-05**: Environment setup (llama.cpp tools available)  

**Pending:**  
⏳ **Task 08**: Fine-tuning Iteration 4 (in-progress)  
⏳ **Task 13/39**: GGUF model created (base GGUF needed for quantization)  

**Note:** Task 39 may already handle quantization as part of GGUF conversion.

## Description
Quantize the GGUF model to reduce size and improve inference speed while maintaining accuracy.

## Requirements
- GGUF model created (Task 13)
- llama.cpp quantization tools installed
- Understanding of quantization levels

## Acceptance Criteria
- [ ] Q4_K_M quantized model created (smaller, faster)
- [ ] Q8_0 quantized model created (larger, more accurate)
- [ ] Models saved to `models/gguf/` with clear naming
- [ ] Size reduction verified (e.g., 15 GB → 4-8 GB)

## Steps
1. Navigate to llama.cpp directory
2. Run quantization: `./quantize ../models/gguf/smart-secrets-scanner.gguf ../models/gguf/smart-secrets-scanner-q4_k_m.gguf Q4_K_M`
3. Run quantization: `./quantize ../models/gguf/smart-secrets-scanner.gguf ../models/gguf/smart-secrets-scanner-q8_0.gguf Q8_0`
4. Verify output files created
5. Document model sizes

## Dependencies
- Task 13: GGUF conversion must be complete
- llama.cpp installed and compiled

## Notes
- Q4_K_M: ~4 GB, faster inference, slight accuracy loss
- Q8_0: ~8 GB, slower, minimal accuracy loss
- Q4_K_M recommended for pre-commit scanning (speed priority)
- Q8_0 for production deployment (accuracy priority)
