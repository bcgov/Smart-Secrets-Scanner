# Task 28: Create requirements.txt for Python Dependencies

**Status:** Backlog  
**Created:** 2025-11-01  
**Related to:** Setup and Documentation

## Prerequisites (Completed)

✅ **Task 04-05**: ML-Env-CUDA13 environment setup with dependencies  

**Note:** A `requirements.txt` file likely already exists in the project root with the necessary dependencies.

## Description
Create a `requirements.txt` file documenting all Python dependencies needed for fine-tuning and deployment.

## Requirements
- ML-Env-CUDA13 environment set up (Task 04)
- Understanding of required libraries

## Acceptance Criteria
- [ ] `requirements.txt` created in project root
- [ ] All fine-tuning dependencies listed
- [ ] All deployment dependencies listed
- [ ] Version numbers specified for stability
- [ ] Installation tested in clean environment

## Steps
1. Document core dependencies:
   - transformers
   - datasets
   - accelerate
   - bitsandbytes
   - peft (for LoRA)
   - trl (optional, for training)
   - unsloth (optional, for faster training)
2. Add deployment dependencies:
   - ollama (Python client)
   - pre-commit
3. Specify versions: `transformers==4.36.0`
4. Test installation: `pip install -r requirements.txt`
5. Document Python version requirement (3.10+)

## Dependencies
- Task 05: Install fine-tune dependencies (use as reference)

## Notes
- Pin major versions to avoid breaking changes
- Consider separate files:
  - `requirements-train.txt` (fine-tuning)
  - `requirements-deploy.txt` (inference only)
- Document CUDA version dependency in README
- Alternative: Use `pyproject.toml` or `environment.yml`
