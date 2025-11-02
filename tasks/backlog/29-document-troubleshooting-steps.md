# Task 29: Document Troubleshooting Steps for CUDA/Python Issues

**Status:** Backlog  
**Created:** 2025-11-01  
**Related to:** Documentation and Support

## Prerequisites (Completed)

✅ **Task 01-05**: Environment setup (experience with setup issues)  
✅ **Task 08**: Fine-tuning runs (experience with training issues)  

**Note:** `EXECUTION_GUIDE.md` already contains some troubleshooting guidance.

## Description
Create comprehensive troubleshooting documentation for common CUDA, Python, and fine-tuning issues.

## Requirements
- Experience with ML-Env-CUDA13 setup
- Knowledge of common CUDA/GPU errors
- Understanding of PyTorch/Transformers issues

## Acceptance Criteria
- [ ] `TROUBLESHOOTING.md` created in project root
- [ ] CUDA issues documented (driver mismatch, out of memory)
- [ ] Python issues documented (version conflicts, package errors)
- [ ] Fine-tuning issues documented (NaN loss, slow training)
- [ ] Solutions tested and verified

## Steps
1. Create `TROUBLESHOOTING.md` with sections:
   - CUDA Issues
   - Python Environment Issues
   - Fine-Tuning Issues
   - Deployment Issues
2. Document common problems:
   - "CUDA out of memory" → Reduce batch size, use gradient checkpointing
   - "No CUDA devices found" → Check drivers, WSL2 GPU passthrough
   - "RuntimeError: Expected all tensors on same device" → Move model/data to GPU
   - "ImportError: cannot import name 'X'" → Version mismatch, reinstall
   - "NaN loss during training" → Lower learning rate, check data
3. Add diagnostic commands:
   - `nvidia-smi` (check GPU)
   - `python -c "import torch; print(torch.cuda.is_available())"` (check CUDA)
   - `pip list | grep torch` (check versions)
4. Link to ML-Env-CUDA13 troubleshooting

## Dependencies
- Task 04: ML-Env-CUDA13 setup (learn from issues)
- Task 08, 11: Fine-tuning experience (document errors)

## Notes
- Update as new issues discovered during project
- Include WSL2-specific issues (e.g., /mnt/c/ vs /home/)
- Add links to Stack Overflow, GitHub issues
- Consider FAQ section in README
