# Task 58: Optimize Inference Script for GPU Utilization

**Status:** Backlog  
**Priority:** HIGH  
**Created:** 2025-11-18  
**Related to:** Phase 2: Model Testing (Step 6)  
**Depends on:** Task 37 (inference script created)

## Prerequisites (Completed)

✅ **Task 37**: Inference script created and working  
✅ **Task 36**: Fine-tuning completed  
✅ **WSL2 + NVIDIA GPU setup verified**

## Description
Optimize `scripts/inference.py` for better GPU utilization on NVIDIA A2000. Currently the model offloads to CPU due to VRAM limitations. Implement 4-bit quantization, YAML config integration, and generation parameter fixes to achieve 80-100% GPU usage.

## Requirements
- Enable 4-bit quantization by default to fit model in A2000 VRAM (~6GB)
- Integrate YAML config (`config/inference_config.yaml`) for quantization and generation settings
- Fix generation parameters (temperature/top_p being ignored due to do_sample=False)
- Force GPU device mapping instead of auto-detection fallback
- Add performance monitoring and benchmarking

## Acceptance Criteria
- [ ] Model loads in 4-bit quantization without CPU offloading
- [ ] GPU utilization reaches 70-95% during inference (monitor with nvidia-smi)
- [ ] Inference time reduced from 1-2 minutes to 10-20 seconds for 150 tokens
- [ ] YAML config properly integrated for quantization and generation settings
- [ ] Generation parameters work correctly (temperature, top_p when do_sample=True)
- [ ] Script defaults to GPU usage on compatible hardware

## Implementation Plan
1. Update `load_model()` to enable 4-bit quantization by default
2. Integrate YAML config loading in `main()`
3. Fix generation parameters in `run_inference()`
4. Add device forcing options
5. Test and benchmark performance improvements

## Expected Performance Improvements
| Metric | Current (CPU Offload) | Target (4-bit GPU) |
|--------|----------------------|-------------------|
| Load Time | 40s + offload | 20-30s |
| Inference Time | 1-2min | 10-20s |
| GPU Util | 0% | 70-95% |
| VRAM Used | N/A | ~4.5GB / 6GB |</content>
<parameter name="filePath">c:\Users\RICHFREM\source\repos\Smart-Secrets-Scanner\tasks\backlog\58-optimize-inference-gpu-utilization.md