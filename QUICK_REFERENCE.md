# Quick Reference Card

**Smart Secrets Scanner Fine-Tuning - Command Cheatsheet**

> ⚠️ Important: this quick reference is for reproducible experiments and demos. The Smart Secrets Scanner examples are intended to demonstrate CUDA-accelerated fine-tuning and model export only. They are not a production-grade secret-scanning solution and should not replace commercial or enterprise scanners (for example, Snyk or Wiz).

## Setup (One-time)

```bash
# Clone repos
cd ~/repos
git clone https://github.com/bcgov/ML-Env-CUDA13.git
git clone <your-repo> Smart-Secrets-Scanner
git clone https://github.com/ggerganov/llama.cpp.git

# Setup environment
cd Smart-Secrets-Scanner
bash scripts/setup_env.sh
bash scripts/install_deps.sh
bash scripts/download_model.sh

# Build llama.cpp
cd ../llama.cpp && make LLAMA_CUBLAS=1
```

## Training Workflow

```bash
cd ~/repos/Smart-Secrets-Scanner
source ~/ml_env/bin/activate

# 1. Validate data
python scripts/validate_dataset.py data/processed/smart-secrets-scanner-train.jsonl

# 2. Train model
python scripts/fine_tune.py

# 3. Monitor training (in another terminal)
tensorboard --logdir outputs/logs
```

## Export Workflow

```bash
# 4. Merge adapter
python scripts/merge_adapter.py

# 5. Convert to GGUF
python scripts/convert_to_gguf.py

# 6. Create Modelfile
python scripts/create_modelfile.py
```

## Deployment

```bash
# 7. Import to Ollama
ollama serve &
ollama create smart-secrets-scanner -f Modelfile

# 8. Test
ollama run smart-secrets-scanner "Analyze: password='secret123'"

# 9. Setup pre-commit
pip install pre-commit
pre-commit install
```

## Testing Commands

```bash
# Test inference
python scripts/inference.py --input "api_key = 'sk_test_123'"

# Evaluate model
python scripts/evaluate.py

# Scan files
python scripts/scan_secrets.py --file test.py

# Run all pre-commit checks
pre-commit run --all-files
```

## Key File Locations

```
config/training_config.yaml          # Training hyperparameters
data/processed/*.jsonl               # Training/validation data
models/base/Meta-Llama-3.1-8B/         # Base model (15-30 GB)
models/fine-tuned/*-lora/            # LoRA adapter (~200 MB)
models/merged/smart-secrets-scanner/ # Merged model (15-30 GB)
models/gguf/*.gguf                   # Quantized models (4-15 GB)
Modelfile                            # Ollama deployment config
```

## Common Issues

```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU
nvidia-smi

# Check Ollama models
ollama list

# View logs
tail -f outputs/logs/training_log.txt

# Check environment
pip list | grep -E "transformers|peft|torch"
```

## Task Reference

| Task | Script | Purpose |
|------|--------|---------|
| 30 | `config/training_config.yaml` | Hyperparameters |
| 34 | `scripts/validate_dataset.py` | Data validation |
| 36 | `scripts/fine_tune.py` | Training |
| 38 | `scripts/merge_adapter.py` | Merge adapter |
| 39 | `scripts/convert_to_gguf.py` | GGUF export |
| 32 | `scripts/evaluate.py` | Evaluation |
| 37 | `scripts/inference.py` | Inference |
| 40 | `scripts/create_modelfile.py` | Modelfile |
| 41 | `scripts/scan_secrets.py` | Pre-commit |

## Documentation

- **EXECUTION_GUIDE.md** - Detailed step-by-step guide
- **SCRIPTS_TASKS_MAPPING.md** - Complete script reference
- **README.md** - Project overview
- **tasks/backlog/** - Individual task specs
