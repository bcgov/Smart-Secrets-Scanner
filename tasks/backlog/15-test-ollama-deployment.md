# Task: Deploy and Test Model in Ollama

**Status: Backlog**

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
