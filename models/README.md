# Models Directory Structure

This directory contains all model files, including base models, fine-tuned adapters, merged models, and quantized formats.

## Subdirectories

### `base/`
- **Purpose**: Original pre-trained models downloaded from Hugging Face or other sources
- **Contents**: Base LLM weights (e.g., Llama 3 8B, Mistral 7B)
- **Example**: `meta-llama/Meta-Llama-3-8B/`, `mistralai/Mistral-7B-v0.1/`
- **Note**: These files are typically large (15-30 GB) and should not be committed to Git

### `fine-tuned/`
- **Purpose**: Fine-tuned adapters (LoRA/QLoRA) or full fine-tuned models
- **Contents**: Adapter weights, training checkpoints, model configs
- **Example**: `smart-secrets-scanner-lora/`, `custom-model-v1/`
- **Note**: Adapters are small (few hundred MB), full models are large

### `gguf/`
- **Purpose**: Quantized models in GGUF format for efficient inference
- **Contents**: GGUF files ready for llama.cpp or Ollama deployment
- **Example**: `smart-secrets-scanner-Q4_K_M.gguf`, `custom-model-Q8_0.gguf`
- **Note**: GGUF files are optimized for CPU/GPU inference with reduced memory usage

## Best Practices

1. **Don't commit large files**: Add `*.bin`, `*.safetensors`, `*.gguf` to `.gitignore`
2. **Use model cards**: Document each model's purpose, training data, and performance in `model_card.yaml`
3. **Version models**: Use descriptive names with dates or version numbers
4. **Track provenance**: Note which base model and dataset were used for each fine-tuned model
5. **Test before deployment**: Always validate models in `evaluation/` before deploying to production

## Workflow

1. Download base model → `base/`
2. Fine-tune with LoRA → Save adapter to `fine-tuned/`
3. Merge adapter + base → Save to `outputs/merged/`
4. Convert to GGUF → Save to `gguf/`
5. Deploy via Ollama or llama.cpp
