# Task 78: Upload Smart-Secrets-Scanner to Hugging Face

## Status: 🔄 IN PROGRESS

## Description
Upload the fine-tuned Smart-Secrets-Scanner model, Modelfile, and documentation to Hugging Face for public access and sharing.

## Acceptance Criteria
- [ ] Create Hugging Face repository (if not exists)
- [ ] Upload GGUF model file: `smart-secrets-scanner-Q4_K_M.gguf`
- [ ] Upload Modelfile for Ollama compatibility
- [ ] Upload model card and README documentation
- [ ] Verify model can be downloaded and run from HF: `ollama run hf.co/username/repo:Q4_K_M`
- [ ] Test downloaded model functionality matches local version

## Files to Upload
- `models/fine-tuned/gguf/smart-secrets-scanner-Q4_K_M.gguf` (4GB)
- `Modelfile` (Ollama configuration)
- `huggingface/model_card.yaml` (model metadata)
- `huggingface/README.md` (usage instructions)
- `huggingface/README_LORA.md` (LoRA adapter info)

## Commands to Run
```bash
# Activate environment
source ~/ml_env/bin/activate

# Upload to Hugging Face (replace with your username/repo)
python scripts/upload_to_huggingface.py --repo yourusername/smart-secrets-scanner --gguf --modelfile --readme

# Test download from Hugging Face
ollama run hf.co/yourusername/smart-secrets-scanner:Q4_K_M
```

## Repository Setup
- **Repository Name**: `smart-secrets-scanner` or `Smart-Secrets-Scanner-Model`
- **Tags**: `llama`, `llama-3.1`, `security`, `code-analysis`, `secrets-detection`, `gguf`, `ollama`
- **License**: MIT or Apache 2.0
- **Description**: Fine-tuned Llama-3.1-8B model specialized in detecting hardcoded secrets in source code

## Model Card Content
- Model architecture: Llama-3.1-8B fine-tuned with LoRA
- Training data: 1000+ examples of code with/without secrets
- Use cases: Code security analysis, secret detection, secure coding education
- Limitations: May have false positives, focused on common secret patterns

## Testing After Upload
1. Download model from HF using Ollama
2. Test same prompts used in local testing
3. Verify responses match local model behavior
4. Confirm no corruption during upload/download

## Dependencies
- Task 77: Test model in Ollama (✅ COMPLETED)
- Hugging Face account and token in `.env`
- `upload_to_huggingface.py` script
- Ollama for testing downloaded model

## Expected Outcome
- Public model available at: https://huggingface.co/yourusername/smart-secrets-scanner
- Users can run: `ollama run hf.co/yourusername/smart-secrets-scanner:Q4_K_M`
- Model card provides clear usage instructions and context