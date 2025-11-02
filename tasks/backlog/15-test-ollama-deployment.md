# Task: Deploy and Test Model in Ollama

**Status: Backlog**

## Prerequisites (Completed)

✅ **Task 01-05**: Environment setup (WSL2, NVIDIA, ML-Env-CUDA13, dependencies)  
✅ **Task 31**: Test dataset created  

**Pending:**  
⏳ **Task 08**: Fine-tuning Iteration 4 (in-progress)  
⏳ **Task 38**: Merge LoRA adapter  
⏳ **Task 39**: Convert to GGUF (Task 13)  
⏳ **Task 40**: Create Modelfile (Task 14)  

## Description
Deploy the fine-tuned GGUF model to Ollama and test inference locally.

## Steps
- Ensure Ollama is installed on WSL2 Ubuntu
- Navigate to directory with Modelfile and GGUF model
- Run `ollama create <model-name> -f ./Modelfile`
- Verify model created successfully
- Run `ollama run <model-name>`
- Test with sample prompts and verify responses
- Check GPU usage during inference with `nvidia-smi`
- Document performance and quality observations

## Dependencies
- Task 13 (Merge and Export GGUF) must be completed
- Task 14 (Create Modelfile) must be completed

## Resources
- Ollama documentation
- ADR 0002: llama.cpp and Ollama deployment
