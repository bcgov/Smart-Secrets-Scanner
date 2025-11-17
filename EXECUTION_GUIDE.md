# Execution Guide: Smart Secrets Scanner Fine-Tuning

**Last Updated**: 2025-11-16  
**Estimated Total Time**: 4-6 hours (depending on GPU)

This guide provides step-by-step instructions to fine-tune Llama 3.1 for secret detection and deploy it to Ollama.

> ⚠️ Important: this guide and the accompanying scripts are provided for demonstration and research purposes only. The "Smart Secrets Scanner" examples are intended to show how CUDA-accelerated fine-tuning and model export can be performed. They are not a production-grade secret-scanning solution and should not be used as a replacement for established commercial or enterprise secret-scanning tools (for example, Snyk or Wiz). Use these instructions to reproduce experiments and validate workflows; rely on proven scanning products for operational security.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Phase 1: Environment Setup](#phase-1-environment-setup)
3. [Phase 2: Data Preparation](#phase-2-data-preparation)
4. [Phase 3: Model Fine-Tuning](#phase-3-model-fine-tuning)
5. [Phase 4: Model Export](#phase-4-model-export)
6. [Phase 5: Testing & Evaluation](#phase-5-testing--evaluation)
7. [Phase 6: Deployment](#phase-6-deployment)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- **OS**: WSL2 Ubuntu 20.04+ on Windows
- **GPU**: NVIDIA GPU with 12GB+ VRAM (RTX 3060 or better)
- **RAM**: 32GB+ recommended
- **Disk Space**: 50GB free (models are large)
- **CUDA**: 11.8+ with cuDNN

### Software Requirements
- **Git**: For cloning repositories
- **Python**: 3.10+ (managed by ML-Env-CUDA13)
- **WSL2**: Windows Subsystem for Linux 2
- **NVIDIA Drivers**: Latest GPU drivers with WSL support

### Repository Setup
```bash
# Clone repositories side-by-side
cd ~/repos  # or your preferred location

# 1. Clone ML-Env-CUDA13 (environment manager)
git clone https://github.com/bcgov/ML-Env-CUDA13.git

# 2. Clone this project
git clone <your-repo-url> Smart-Secrets-Scanner

# 3. Clone llama.cpp (for GGUF conversion)
git clone https://github.com/ggerganov/llama.cpp.git

# Verify structure
ls -la
# Should see: ML-Env-CUDA13/ Smart-Secrets-Scanner/ llama.cpp/
```

---

## Phase 1: Environment Setup

**Estimated Time**: 30-60 minutes

### Step 1.1: Setup CUDA ML Environment

```bash
cd ~/repos/Smart-Secrets-Scanner

# Run the unified setup script (requires sudo for system packages)
sudo python3 scripts/setup_cuda_env.py --staged --recreate
```

**What this does**:
- Installs system prerequisites (Python 3.11, etc.) if missing
- Creates a clean Python virtual environment at `~/ml_env`
- Installs CUDA-enabled PyTorch and all ML dependencies from `requirements-wsl.txt`
- Handles staged installation to prevent pip conflicts

**Expected Output**:
```
=== CUDA ML Environment Setup ===
Installing system prerequisites...
✓ Python 3.11 installed
✓ Virtual environment created at ~/ml_env
✓ CUDA-enabled PyTorch installed
✓ All dependencies installed
✓ Setup complete!
```
```
=== Setting up ML-Env-CUDA13 Environment ===
Running ML-Env-CUDA13 WSL setup script...
✓ CUDA installed
✓ Python virtual environment created
✓ Setup complete!
```

### WSL / CUDA-specific install (recommended for WSL users)

This repository uses a unified setup script that handles the entire CUDA ML environment setup in one command. Follow `CUDA-ML-ENV-SETUP.md` for complete WSL + CUDA instructions.

### STEP 1: WSL CUDA INSTALL SETUP

**Quick Setup (WSL/Ubuntu)**:

```bash
# 1) Start with a clean slate
deactivate 2>/dev/null || true
rm -rf ~/ml_env

# 2) Run the all-in-one setup script (requires sudo)
sudo python3 scripts/setup_cuda_env.py --staged --recreate

# 3) Activate and verify
source ~/ml_env/bin/activate
python scripts/test_torch_cuda.py
python scripts/test_xformers.py
python scripts/test_pytorch.py
```

**What the script does**:
- Installs system prerequisites (Python 3.11, etc.)
- Creates virtual environment at `~/ml_env`
- Installs CUDA-enabled PyTorch from `requirements-wsl.txt`
- Performs staged installation to avoid pip conflicts

**Notes**:
- The script uses `requirements-wsl.txt` with CUDA-pinned wheels (e.g., `torch==2.8.0+cu126`)
- Do not use these pins on CPU-only or non-WSL systems
- For large file support, install git-lfs: `sudo apt update && sudo apt install git-lfs`

### STEP 2: Step Verify Environment Setup

```bash

# Verify CUDA availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
# Expected: CUDA Available: True

# Run verification scripts
python scripts/test_torch_cuda.py
python scripts/test_xformers.py
python scripts/test_pytorch.py

# Check installed packages
pip list | grep -E "transformers|peft|trl|torch"
```

**What this does**:
- Activates the virtual environment created by the setup script
- Verifies PyTorch can access CUDA
- Runs additional verification tests for key libraries
- Confirms all ML dependencies are installed

### STEP 3:  Download Base Model

```bash
# Download Llama 3.1 8B (15-30 GB)
bash scripts/download_model.sh
```

**Alternative**: Manual download with Hugging Face CLI:
```bash
# Install huggingface_hub
pip install huggingface-cli

# Login to Hugging Face (requires account)
huggingface-cli login

# Download model
huggingface-cli download meta-llama/Meta-Llama-3.1-8B --local-dir models/base/Meta-Llama-3.1-8B
```

**Expected Output**:
```
models/base/Meta-Llama-3.1-8B/
├── config.json
├── generation_config.json
├── model-00001-of-00004.safetensors
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
├── model-00004-of-00004.safetensors
├── model.safetensors.index.json
├── special_tokens_map.json
├── tokenizer.json
└── tokenizer_config.json
```

### STEP 4: Build llama.cpp (for GGUF conversion)

```bash
cd ~/repos/llama.cpp

# Build with CUDA support
make LLAMA_CUBLAS=1

# Verify build
./main --version
```

---

### STEP 5: Data Preparation

**Estimated Time**: 15-30 minutes

### Step 2.1: Verify Training Data

```bash
cd ~/repos/Smart-Secrets-Scanner

# Check existing datasets
ls -lh data/processed/
# Should see:
# - smart-secrets-scanner-train.jsonl (56 examples)
# - smart-secrets-scanner-val.jsonl (16 examples)
```

### STEP 6: Validate Data Quality

```bash
# Run data validation script
python scripts/validate_dataset.py data/processed/smart-secrets-scanner-train.jsonl
```

**Expected Output**:
```
Validating data/processed/smart-secrets-scanner-train.jsonl...

1. Checking JSONL syntax...
  ✅ All lines are valid JSON

2. Checking schema...
  ✅ All required fields present

3. Checking class balance...
  Total: 56 examples
  Secrets (ALERT): 28 (50.0%)
  Safe: 28 (50.0%)
  ✅ Dataset is balanced

4. Checking token lengths...
  Min: 120 tokens
  Max: 450 tokens
  Avg: 250.5 tokens
  ✅ All examples fit in context window

5. Checking for duplicates...
  ✅ No duplicate inputs found

=== Validation Complete ===
✅ Dataset is ready for training!
```

### STEP 7: Create Test Dataset (Optional but Recommended)

**Task**: Generate 20 new examples for final evaluation

```bash
# Create test set following Task 31 guidelines
# Examples should be DIFFERENT from train/val data
# Include challenging edge cases
```

See `tasks/backlog/31-create-evaluation-test-dataset.md` for detailed instructions.

### STEP 8: Create Training Configuration

Create `config/training_config.yaml`:

```yaml
model:
  name: "models/base/Meta-Llama-3.1-8B"
  max_seq_length: 2048
  load_in_4bit: true

lora:
  r: 16
  alpha: 32
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
  dropout: 0.05
  bias: "none"

training:
  output_dir: "./outputs/checkpoints"
  num_train_epochs: 5
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2e-4
  warmup_steps: 10
  logging_steps: 5
  save_steps: 50
  eval_steps: 25
  save_total_limit: 3
  fp16: true
  optim: "paged_adamw_8bit"
  report_to: "tensorboard"

data:
  train: "data/processed/smart-secrets-scanner-train.jsonl"
  validation: "data/processed/smart-secrets-scanner-val.jsonl"
  test: "data/evaluation/smart-secrets-scanner-test.jsonl"
```

---

### STEP 9: Model Fine-Tuning

**Estimated Time**: 1-3 hours (depends on GPU and epochs)

#### Step 9.1: Start Training (Enhanced Script)

```bash
# Activate environment
source ~/ml_env/bin/activate

# Start fine-tuning with enhanced monitoring
python scripts/fine_tune.py
```

**What the enhanced script does**:
1. **System Diagnostics**: Checks CUDA availability, GPU memory, CPU cores
2. **Smart Authentication**: Loads Hugging Face token from `.env` file
3. **Resume Capability**: Automatically detects and resumes from existing checkpoints
4. **Optimized Loading**: Uses 4-bit quantization with improved dtype handling
5. **Professional Logging**: Structured logging with timestamps and progress tracking
6. **Data Validation**: Auto-creates validation split if missing (90/10)
7. **Performance Monitoring**: Tracks training metrics and GPU utilization
8. **Error Recovery**: Comprehensive error handling with actionable messages

**Expected Enhanced Output**:
```
🚀 Smart Secrets Scanner - Fine-Tuning Script (Optimized v2.0)
⏰ Start time: 2025-11-16 14:30:00

🔍 System Diagnostics:
   CUDA available: True
   GPU count: 1
   GPU 0: NVIDIA RTX A2000 (8192 MB)
   CPU cores: 12 logical, 15.2% used

🔐 Authenticating with Hugging Face...
✅ Authenticated successfully

📊 Loading and formatting dataset...
✅ Loaded 56 training examples
✅ Loaded 16 validation examples

📝 Formatting prompts with Alpaca template...
✅ Datasets formatted

🔽 Loading base model: models/base/Meta-Llama-3.1-8B
   Using 4-bit quantization for efficient training
✅ Model loaded and quantized

🔧 Configuring LoRA adapters...
📊 Trainable parameters: 0.49% (8,388,608 / 1,709,526,528)
✅ LoRA configuration applied

⚙️  Configuring training arguments...
✅ Training configuration:
   Epochs: 15
   Batch size: 1 (effective: 8)
   Learning rate: 0.0002
   Max sequence length: 256
   Optimizer: paged_adamw_8bit
   Scheduler: cosine

🏋️  Starting Training...
📁 Found checkpoint to resume from: outputs/checkpoints/checkpoint-200

✅ Training Complete!
⏰ Start time:  2025-11-16 14:30:00
⏰ End time:    2025-11-16 16:45:00
⏱️  Total time:  2h 15m 0s

📁 LoRA adapter saved to: models/fine-tuned/smart-secrets-scanner-lora
📁 Training checkpoints: outputs/checkpoints
📁 Training logs: outputs/logs

📊 Next steps:
  1. Review training logs: tensorboard --logdir outputs/logs
  2. Merge adapter with base model: python scripts/merge_adapter.py
  3. Convert to GGUF: python scripts/convert_to_gguf.py
  4. Evaluate model: python scripts/evaluate.py
```

#### Step 9.2: Monitor Training (in another terminal)

```bash
# View TensorBoard
tensorboard --logdir outputs/logs

# Open browser to http://localhost:6006
```

**What to watch**:
- **Training Loss**: Should decrease from ~2.0 to < 0.5
- **Validation Loss**: Should track training loss (not increase)
- **Learning Rate**: Should follow warmup schedule
- **GPU Memory**: Monitor with `nvidia-smi`

#### Step 9.3: Review Training Logs

```bash
# Check final metrics
cat outputs/logs/training_log.txt | tail -20

# Check checkpoints
ls -lh outputs/checkpoints/
# checkpoint-50/
# checkpoint-100/
# checkpoint-best/  (lowest validation loss)
```

---

### STEP 10: Model Export

**Estimated Time**: 30-60 minutes

#### Step 10.1: Merge LoRA Adapter with Base Model

```bash
# Merge adapter into full model
python scripts/merge_adapter.py \
  --base-model models/base/Meta-Llama-3.1-8B \
  --adapter models/fine-tuned/smart-secrets-scanner-lora \
  --output models/merged/smart-secrets-scanner \
  --verify
```

**Expected Output**:
```
🔗 Merging LoRA Adapter with Base Model
🔽 Loading base model...
🔽 Loading LoRA adapter...
🔗 Merging weights...
💾 Saving merged model...

✅ Merge Complete!
📁 Merged model: models/merged/smart-secrets-scanner/ (~15 GB)

🧪 Verifying model...
✅ Model generates output successfully
```

#### Step 10.2: Convert to GGUF Format

```bash
# Convert to GGUF with quantization
python scripts/convert_to_gguf.py \
  --model models/merged/smart-secrets-scanner \
  --output models/gguf/smart-secrets-scanner.gguf \
  --quantize Q4_K_M Q8_0
```

**Expected Output**:
```
📦 Converting to GGUF Format
🔄 Converting to F16 GGUF... ✅ (15 GB)
🔧 Quantizing to Q4_K_M... ✅ (4.2 GB)
🔧 Quantizing to Q8_0... ✅ (8.1 GB)

✅ GGUF Conversion Complete!
Created files:
  📦 smart-secrets-scanner-f16.gguf (15.2 GB)
  📦 smart-secrets-scanner-q4_k_m.gguf (4.2 GB)
  📦 smart-secrets-scanner-q8_0.gguf (8.1 GB)
```

---

### STEP 11: Testing & Evaluation

**Estimated Time**: 30-60 minutes

#### Step 11.1: Test Inference with Merged Model

```bash
# Quick test with merged model
python scripts/inference.py \
  --model models/merged/smart-secrets-scanner \
  --input 'api_key = "sk_live_1234567890abcdef"'
```

**Expected Output**:
```
🔍 Smart Secrets Scanner - Inference Results
📄 Source: <direct input>
🚨 ALERT: Stripe API key detected. This appears to be a live Stripe
    secret key (sk_live_...) that should not be hardcoded.
```

#### Step 11.2: Run Full Evaluation on Test Set

```bash
# Evaluate on test dataset
python scripts/evaluate.py \
  --model models/merged/smart-secrets-scanner \
  --test-data data/evaluation/smart-secrets-scanner-test.jsonl
```

**Expected Output**:
```
=== Evaluation Results ===
Accuracy:  0.950
Precision: 0.923
Recall:    0.962
F1 Score:  0.942

Classification Report:
              precision    recall  f1-score
Safe              0.95      0.93      0.94
Secret            0.93      0.96      0.94
```

**Target Metrics**:
- Precision: > 90% (few false positives)
- Recall: > 95% (catch almost all secrets)
- F1 Score: > 92%

#### Step 11.3: Test with Real Code Files

```bash
# Test on actual source files
python scripts/inference.py \
  --batch data/raw/python/ \
  --model models/merged/smart-secrets-scanner
```

---

### STEP 12: Deployment

**Estimated Time**: 15-30 minutes

#### Step 12.1: Create Ollama Modelfile

```bash
# Generate Modelfile
python scripts/create_modelfile.py \
  --gguf models/gguf/smart-secrets-scanner-q4_k_m.gguf \
  --output Modelfile
```

**Expected Output**:
```
📝 Creating Ollama Modelfile
✅ Modelfile created: Modelfile
   GGUF: models/gguf/smart-secrets-scanner-q4_k_m.gguf
   Quantization: Q4_K_M
```

#### Step 12.2: Import to Ollama

```bash
# Start Ollama (if not running)
ollama serve &

# Import model
ollama create smart-secrets-scanner -f Modelfile
```

**Expected Output**:
```
transferring model data
creating model layer
using already created layer sha256:abc123...
writing manifest
success
```

#### Step 12.3: Test Ollama Deployment

```bash
# Interactive test
ollama run smart-secrets-scanner

# Enter test at prompt:
>>> Analyze: aws_access_key = "AKIAIOSFODNN7EXAMPLE"

# Expected response:
ALERT: AWS access key detected. This is a hardcoded AWS IAM access
key (AKIA...) that should be stored in environment variables or AWS
Secrets Manager.

>>> /bye
```

### Step 13: Setup Pre-Commit Hooks (NOT RELATED TO MODEL TRAINING)

```bash
# Install pre-commit framework
pip install pre-commit

# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: local
    hooks:
      - id: smart-secrets-scanner
        name: Smart Secrets Scanner (LLM)
        entry: python scripts/scan_secrets.py
        language: python
        pass_filenames: true
        types: [python, javascript, yaml]
EOF

# Install hook
pre-commit install

# Test on all files
pre-commit run --all-files
```

### Step 14: Test Pre-Commit Hook (NOT RELATED TO MODEL TRAINING)

```bash
# Create test file with secret
echo 'password = "admin123"' > test_secret.py
git add test_secret.py

# Try to commit (should be blocked)
git commit -m "test commit"
```

**Expected Output**:
```
🔍 Smart Secrets Scanner - Pre-Commit Check
📄 Scanning 1 file(s)...
  Checking: test_secret.py... 🚨 ALERT

❌ SECRETS DETECTED - COMMIT BLOCKED
📁 test_secret.py
   ALERT: Hardcoded password detected. The string "admin123"
   appears to be a password that should not be committed.
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
```yaml
# Edit config/training_config.yaml
training:
  per_device_train_batch_size: 2  # Reduce from 4
  gradient_accumulation_steps: 8  # Increase to maintain effective batch size
```

Or use gradient checkpointing:
```python
# In scripts/fine_tune.py
model.gradient_checkpointing_enable()
```

#### 2. No CUDA Devices Found
**Error**: `CUDA Available: False`

**Solution**:
```bash
# Check WSL GPU access
nvidia-smi

# Update WSL2 kernel
wsl --update

# Reinstall NVIDIA drivers for WSL
# Download from: https://developer.nvidia.com/cuda/wsl
```

#### 3. Ollama Model Not Found
**Error**: `Error: model 'smart-secrets-scanner' not found`

**Solution**:
```bash
# Verify Ollama is running
ollama list

# Recreate model
ollama create smart-secrets-scanner -f Modelfile

# Check GGUF path in Modelfile is absolute
cat Modelfile | grep FROM
```

#### 4. Training Loss is NaN
**Error**: Loss becomes NaN during training

**Solution**:
```yaml
# Reduce learning rate
training:
  learning_rate: 1e-4  # Reduce from 2e-4

# Or add gradient clipping
max_grad_norm: 1.0
```

#### 5. Import Errors
**Error**: `ImportError: No module named 'peft'`

**Solution**:
```bash
# Ensure correct environment is activated
source ~/ml_env/bin/activate

# Reinstall dependencies
bash scripts/install_deps.sh

# Verify installation
pip list | grep peft
```

---

## Next Steps

After successful deployment:

1. **Monitor Performance**: Track false positives/negatives in production
2. **Expand Dataset**: Add more examples for edge cases discovered
3. **Fine-Tune Again**: Retrain with expanded dataset
4. **Integrate with CI/CD**: Add to GitHub Actions, GitLab CI, etc.
5. **Compare with Baselines**: Test against GitGuardian, TruffleHog
6. **Optimize for Speed**: Use smaller quantization (Q4_K_S) if needed
7. **Deploy to Team**: Share Modelfile with team members

---

## Reference

- **Detailed Task List**: See `tasks/backlog/` for individual task specifications
- **Script Documentation**: See `SCRIPTS_TASKS_MAPPING.md` for complete script reference
- **Architecture Decisions**: See `adrs/` for design rationale
- **Data Documentation**: See `data/README.md` for dataset structure

---

## Support

For issues or questions:
1. Check `TROUBLESHOOTING.md` (when created - Task 29)
2. Review task files in `tasks/backlog/`
3. Open an issue on GitHub

---

**🎉 Congratulations! You've successfully fine-tuned and deployed a custom LLM for secret detection!**
