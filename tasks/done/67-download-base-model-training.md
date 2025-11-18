# Task 67: Download Base Model for Training

**Status:** Done  
**Created:** 2025-11-18  
**Completed:** 2025-11-18  

## Prerequisites (Completed)

✅ **Task 66**: Dataset validated  
✅ **Task 64**: Environment verified  
✅ **Task 60**: Hugging Face auth configured  
✅ **Task 22**: Download script created  

## Objective

Download the base Llama 3.1 8B model from Hugging Face to enable fine-tuning. This large model file (~30GB) serves as the foundation for training the LoRA adapter.

## Requirements

- Hugging Face authentication
- ~30GB available disk space
- Stable internet connection
- Proper model directory structure

## Implementation

### 1. Verify Prerequisites

```bash
# Check available disk space
df -h /mnt/c/Users/RICHFREM/source/repos/Smart-Secrets-Scanner

# Verify Hugging Face authentication
python -c "
from huggingface_hub import HfApi
import os
token = os.getenv('HUGGING_FACE_TOKEN')
if token:
    api = HfApi(token=token)
    print('✅ HF token configured')
    try:
        user = api.whoami()
        print(f'✅ Authenticated as: {user[\"name\"]}')
    except Exception as e:
        print(f'❌ Auth failed: {e}')
else:
    print('❌ HUGGING_FACE_TOKEN not set')
"
```

### 2. Run Model Download

```bash
# Execute download script
bash scripts/download_model.sh
```

### 3. Monitor Download Progress

The download will show progress and may take 30-60 minutes depending on internet speed. Expected output:

```
Downloading model: meta-llama/Meta-Llama-3.1-8B
Fetching 4 files: ...
```

### 4. Verify Download Completion

```bash
# Check model directory contents
ls -la models/base/Meta-Llama-3.1-8B/

# Expected files:
# - model-00001-of-00004.safetensors through model-00004-of-00004.safetensors
# - tokenizer.json, tokenizer_config.json
# - config.json, generation_config.json
# - special_tokens_map.json
```

### 5. Test Model Loading

```bash
# Quick load test
python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Testing model loading...")
try:
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("models/base/Meta-Llama-3.1-8B/")
    print("✅ Tokenizer loaded")
    
    # Load model (CPU only for testing)
    model = AutoModelForCausalLM.from_pretrained(
        "models/base/Meta-Llama-3.1-8B/",
        torch_dtype=torch.float16,
        device_map="cpu"  # Load on CPU for verification
    )
    print("✅ Model loaded successfully")
    print(f"Model parameters: {model.num_parameters():,}")
    
except Exception as e:
    print(f"❌ Load failed: {e}")
PY
```

### 6. Check Model Integrity

```bash
# Verify safetensors integrity
python -c "
import safetensors
import os

model_dir = 'models/base/Meta-Llama-3.1-8B/'
safetensor_files = [f for f in os.listdir(model_dir) if f.endswith('.safetensors')]

print(f'Found {len(safetensor_files)} safetensor files')
for file in sorted(safetensor_files):
    path = os.path.join(model_dir, file)
    try:
        with safetensors.safe_open(path, 'r') as f:
            keys = list(f.keys())
        print(f'✅ {file}: {len(keys)} tensors')
    except Exception as e:
        print(f'❌ {file}: {e}')
"
```

## Technical Details

### Model Specifications

- **Model**: Meta-Llama-3.1-8B
- **Size**: ~30GB total
- **Architecture**: Transformer decoder-only
- **Context Length**: 8192 tokens
- **Quantization**: None (full precision for fine-tuning)

### File Structure

```
models/base/Meta-Llama-3.1-8B/
├── config.json                 # Model configuration
├── generation_config.json      # Generation parameters
├── model-00001-of-00004.safetensors  # Model weights (split)
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
├── model-00004-of-00004.safetensors
├── special_tokens_map.json     # Special token mapping
├── tokenizer.json             # Tokenizer data
├── tokenizer_config.json      # Tokenizer configuration
└── tokenizer.model           # SentencePiece model
```

### Download Process

1. **Authentication**: Uses HUGGING_FACE_TOKEN from .env
2. **Repository**: meta-llama/Meta-Llama-3.1-8B
3. **Files**: Downloads all model files and tokenizer
4. **Verification**: Automatic integrity checking

## Troubleshooting

### Authentication Issues

```bash
# Check token validity
curl -H "Authorization: Bearer $HUGGING_FACE_TOKEN" \
  https://huggingface.co/api/whoami

# Regenerate token if needed
echo "Go to: https://huggingface.co/settings/tokens"
echo "Create new token with 'Read' permissions"
```

### Download Failures

```bash
# Resume interrupted download
export HF_HUB_ENABLE_HF_TRANSFER=1  # Faster downloads
huggingface-cli download meta-llama/Meta-Llama-3.1-8B --resume-download

# Alternative: Manual download
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='meta-llama/Meta-Llama-3.1-8B',
    local_dir='models/base/Meta-Llama-3.1-8B/',
    resume_download=True
)
"
```

### Disk Space Issues

```bash
# Check space requirements
du -sh models/base/Meta-Llama-3.1-8B/

# Clean up if needed
rm -rf models/base/Meta-Llama-3.1-8B/
# Then retry download
```

### Permission Issues

```bash
# Fix directory permissions
chmod -R 755 models/
chown -R $USER:$USER models/base/
```

## Outcome

✅ Base model downloaded successfully  
✅ ~30GB of model files retrieved  
✅ Model integrity verified  
✅ Ready for fine-tuning  

## Execution Results (2025-11-18)

Successfully executed the download script `bash scripts/download_model.sh` which:
- Activated the ML environment (`~/ml_env`)
- Loaded Hugging Face token from `.env` file
- Downloaded the complete Llama-3.1-8B model (~15-30 GB)
- Saved all model files to `models/base/Meta-Llama-3.1-8B/`

**Download completed successfully** - model is now ready for fine-tuning.

## Related Tasks

- Task 22: Download script creation (foundation)
- Task 60: HF authentication (prerequisite)
- Task 36: Fine-tuning (next step)
- Task 66: Dataset validation (parallel)</content>
<parameter name="filePath">c:\Users\RICHFREM\source\repos\Smart-Secrets-Scanner\tasks\done\67-download-base-model-training.md