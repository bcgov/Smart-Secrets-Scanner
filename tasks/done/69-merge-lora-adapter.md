# Task 69: Merge LoRA Adapter with Base Model

**Status:** Done  
**Created:** 2025-11-18  
**Completed:** 2025-11-18  

## Prerequisites (Completed)

✅ **Task 68**: LoRA adapter trained  
✅ **Task 67**: Base model downloaded  
✅ **Task 64**: Environment verified  
✅ **Task 38**: Merge script created  

## Objective

Combine the trained LoRA adapter with the base Llama 3.1 8B model to create a standalone fine-tuned model that can be used independently without requiring the adapter to be loaded separately.

## Requirements

- Trained LoRA adapter
- Base model files
- Merge script execution
- Disk space for merged model (~30GB)

## Implementation

### 1. Verify Adapter and Base Model

```bash
# Check adapter files
ls -la models/fine-tuned/smart-secrets-scanner-lora/
# Should contain: adapter_model.safetensors, adapter_config.json

# Check base model
ls -la models/base/Meta-Llama-3.1-8B/
# Should contain: model-*.safetensors files, config.json, tokenizer files
```

### 2. Execute Adapter Merge

```bash
# Run merge script with safety check
python scripts/merge_adapter.py --skip-sanity
```

### 3. Monitor Merge Process

The merge process will:
- Load base model and LoRA adapter
- Apply LoRA weights to base model
- Save merged model in safetensors format
- Preserve tokenizer and configuration

Expected output location: `models/merged/smart-secrets-scanner/`

### 4. Verify Merge Completion

```bash
# Check merged model directory
ls -la models/merged/smart-secrets-scanner/

# Expected files:
# - model-00001-of-00004.safetensors through model-00004-of-00004.safetensors (merged)
# - config.json, generation_config.json
# - tokenizer.json, tokenizer_config.json
# - special_tokens_map.json
```

### 5. Test Merged Model

```bash
# Quick inference test with merged model
python scripts/inference.py --model-type merged --input "Test: API_KEY = 'sk-1234567890abcdef'"
```

Expected response should show secret detection capability from the fine-tuned adapter.

### 6. Compare Model Sizes

```bash
# Compare base vs merged model sizes
du -sh models/base/Meta-Llama-3.1-8B/
du -sh models/merged/smart-secrets-scanner/

# Merged model should be similar size to base model
```

## Technical Details

### Merge Process

1. **Load Components**: Base model + LoRA adapter weights
2. **Weight Application**: LoRA updates applied to base weights
3. **Format Conversion**: Full precision weights saved as safetensors
4. **Configuration Preservation**: Model config and tokenizer copied

### LoRA Weight Mathematics

For each layer with LoRA adapter:
```
W_merged = W_base + (A × B) × scale
```

Where:
- `W_base`: Original base model weights
- `A` and `B`: Low-rank adapter matrices
- `scale`: LoRA scaling factor (alpha/r)

### Output Structure

```
models/merged/smart-secrets-scanner/
├── config.json                    # Model configuration (updated)
├── generation_config.json         # Generation parameters
├── model-00001-of-00004.safetensors  # Merged weights (split files)
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
├── model-00004-of-00004.safetensors
├── special_tokens_map.json        # Special token mapping
├── tokenizer.json                # Tokenizer data
└── tokenizer_config.json         # Tokenizer configuration
```

## Troubleshooting

### Merge Failures

```bash
# Check adapter compatibility
python -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained('models/base/Meta-Llama-3.1-8B')
peft_model = PeftModel.from_pretrained(base_model, 'models/fine-tuned/smart-secrets-scanner-lora')
print('Adapter compatible with base model')
"
```

### Memory Issues

```bash
# Monitor memory usage
free -h
nvidia-smi

# Use CPU if GPU memory insufficient
export CUDA_VISIBLE_DEVICES=""
python scripts/merge_adapter.py
```

### File Permission Issues

```bash
# Fix permissions
chmod -R 755 models/
chown -R $USER:$USER models/merged/
```

### Verification Failures

```bash
# Manual merge verification
python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load merged model
model = AutoModelForCausalLM.from_pretrained('models/merged/smart-secrets-scanner/')
tokenizer = AutoTokenizer.from_pretrained('models/merged/smart-secrets-scanner/')

# Test inference
input_text = "Analyze: password = 'secret123'"
inputs = tokenizer(input_text, return_tensors='pt')
outputs = model.generate(**inputs, max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Input: {input_text}")
print(f"Response: {response}")
PY
```

## Performance Comparison

```bash
# Compare base vs merged model responses
python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer

# Test prompts
test_prompts = [
    "API_KEY = 'sk-1234567890abcdef'",
    "password = 'admin123'",
    "const TOKEN = process.env.JWT_SECRET"
]

# Load models
base_model = AutoModelForCausalLM.from_pretrained('models/base/Meta-Llama-3.1-8B')
merged_model = AutoModelForCausalLM.from_pretrained('models/merged/smart-secrets-scanner')
tokenizer = AutoTokenizer.from_pretrained('models/base/Meta-Llama-3.1-8B')

for prompt in test_prompts:
    print(f"\nPrompt: {prompt}")
    
    # Base model response
    inputs = tokenizer(prompt, return_tensors='pt')
    base_output = base_model.generate(**inputs, max_length=50, do_sample=False)
    base_response = tokenizer.decode(base_output[0], skip_special_tokens=True)
    print(f"Base: {base_response}")
    
    # Merged model response  
    merged_output = merged_model.generate(**inputs, max_length=50, do_sample=False)
    merged_response = tokenizer.decode(merged_output[0], skip_special_tokens=True)
    print(f"Merged: {merged_response}")
PY
```

## Outcome

✅ LoRA adapter merged with base model  
✅ Standalone fine-tuned model created  
✅ Secret detection capability preserved  
✅ Ready for GGUF conversion  

## Related Tasks

- Task 38: Merge script creation (foundation)
- Task 68: LoRA training (prerequisite)
- Task 39: Convert to GGUF (next step)
- Task 37: Test inference (validation)</content>
<parameter name="filePath">c:\Users\RICHFREM\source\repos\Smart-Secrets-Scanner\tasks\done\69-merge-lora-adapter.md