# Scripts to Tasks Mapping

**Last Updated**: 2025-11-16  
**Task Counter**: 42

This document maps all scripts in `scripts/` to their corresponding task implementations.

## ✅ Existing Scripts (Shell Wrappers)

| Script | Status | Task | Purpose |
|--------|--------|------|---------|
| `setup_cuda_env.py` | ✅ Exists | Task 04, 05 | Unified CUDA environment setup |
| `download_model.sh` | ✅ Exists | Task 06, 22 | Download base Llama 3 model |
| `fine_tune.sh` | ⚠️ Incomplete | Task 08, 11 | Wrapper for `fine_tune.py` (missing) |
| `infer.sh` | ⚠️ Incomplete | Task 09 | Wrapper for `inference.py` (missing) |

## 🆕 New Python Scripts (To Be Created)

### **CRITICAL Priority** (Required for Training)

| Script | Task | Priority | Purpose |
|--------|------|----------|---------|
| `scripts/fine_tune.py` | **Task 36** | 🔴 CRITICAL | Main training script with LoRA/QLoRA |
| `scripts/validate_dataset.py` | **Task 34** | 🔴 CRITICAL | Pre-training data validation |
| `config/training_config.yaml` | **Task 30** | 🔴 CRITICAL | Hyperparameters configuration |

### **HIGH Priority** (Required for Deployment)

| Script | Task | Priority | Purpose |
|--------|------|----------|---------|
| `scripts/merge_adapter.py` | **Task 38** | 🟠 HIGH | Merge base model + LoRA adapter |
| `scripts/convert_to_gguf.py` | **Task 39** | 🟠 HIGH | Convert to GGUF for Ollama |
| `scripts/inference.py` | **Task 37** | 🟠 HIGH | Run inference on code snippets |
| `scripts/evaluate.py` | **Task 32** | 🟠 HIGH | Calculate precision/recall/F1 metrics |

### **MEDIUM Priority** (Required for Production)

| Script | Task | Priority | Purpose |
|--------|------|----------|---------|
| `scripts/create_modelfile.py` | **Task 40** | 🟡 MEDIUM | Generate Ollama Modelfile |
| `scripts/scan_secrets.py` | **Task 41** | 🟡 MEDIUM | Pre-commit hook scanner |

## 📋 Complete Workflow Execution Order

### **Phase 1: Setup & Preparation**
```bash
# 1. Setup environment & install dependencies (Task 04, 05)
sudo python3 scripts/setup_cuda_env.py --staged --recreate

# 2. Download base model (Task 06, 22)
bash scripts/download_model.sh

# 3. Validate training data (Task 34) - NEW
python scripts/validate_dataset.py data/processed/smart-secrets-scanner-train.jsonl
```

### **Phase 2: Fine-Tuning**
```bash
# 5. Fine-tune model (Task 36) - NEW
python scripts/fine_tune.py

# Alternative: Use wrapper
bash scripts/fine_tune.sh
```

### **Phase 3: Export & Conversion**
```bash
# 6. Merge adapter with base model (Task 38) - NEW
python scripts/merge_adapter.py

# 7. Convert to GGUF (Task 39) - NEW
python scripts/convert_to_gguf.py --quantize Q4_K_M Q8_0

# 8. Create Ollama Modelfile (Task 40) - NEW
python scripts/create_modelfile.py
```

### **Phase 4: Testing & Deployment**
```bash
# 9. Evaluate model (Task 32) - NEW
python scripts/evaluate.py

# 10. Test inference (Task 37) - NEW
python scripts/inference.py --input "api_key = 'test123'"

# 11. Deploy to Ollama (Task 14, 15)
ollama create smart-secrets-scanner -f Modelfile

# 12. Setup pre-commit hooks (Task 41) - NEW
pip install pre-commit
pre-commit install
```

## 📊 Task Mapping Summary

### Scripts Already Created (Shell Wrappers)
- ✅ 4 shell scripts exist (`setup_cuda_env.py`, `download_model.sh`, `fine_tune.sh`, `infer.sh`)
- ⚠️ 2 are incomplete (call missing Python files)

### Scripts to Create (Python Implementations)
- 🔴 3 CRITICAL scripts needed before training
- 🟠 4 HIGH priority scripts needed for deployment
- 🟡 2 MEDIUM priority scripts for production use
- **Total: 9 new Python scripts + 1 config file**

### Task Relationships

**Original Tasks** → **New Implementation Tasks**:
- Task 10 (Create training script) → **Task 36** (fine_tune.py)
- Task 09 (Run inference) → **Task 37** (inference.py)
- Task 12 (Merge/GGUF script) → **Task 38** (merge) + **Task 39** (GGUF)
- Task 14 (Create Modelfile) → **Task 40** (create_modelfile.py)
- Task 27 (Pre-commit integration) → **Task 41** (scan_secrets.py)

**New Foundational Tasks**:
- **Task 30**: Training config (required by Task 36)
- **Task 31**: Test dataset (required by Task 32)
- **Task 32**: Evaluation script (validates model performance)
- **Task 33**: Experiment tracking (W&B/TensorBoard)
- **Task 34**: Data validation (pre-training quality checks)
- **Task 35**: Model card (documentation)

## 🎯 Next Steps

### **Immediate Actions** (Before Training)
1. ✅ Create **Task 30**: `config/training_config.yaml`
2. ✅ Create **Task 36**: `scripts/fine_tune.py`
3. ✅ Create **Task 34**: `scripts/validate_dataset.py`
4. ✅ Create **Task 31**: Test dataset (20 examples)

### **After Training**
5. Create **Task 38**: `scripts/merge_adapter.py`
6. Create **Task 39**: `scripts/convert_to_gguf.py`
7. Create **Task 32**: `scripts/evaluate.py`
8. Create **Task 37**: `scripts/inference.py`

### **For Deployment**
9. Create **Task 40**: `scripts/create_modelfile.py`
10. Create **Task 41**: `scripts/scan_secrets.py`

## 🔗 Dependencies Graph

```
Task 04 (setup_cuda_env.py - unified setup & deps)
  └─> Task 06/22 (download_model.sh)
       └─> Task 30 (training_config.yaml)
            └─> Task 36 (fine_tune.py)
                 └─> Task 38 (merge_adapter.py)
                      └─> Task 39 (convert_to_gguf.py)
                           └─> Task 40 (create_modelfile.py)
                                └─> Task 41 (scan_secrets.py)

Task 31 (test dataset) --> Task 32 (evaluate.py)
Task 34 (validate_dataset.py) --> Task 36 (fine_tune.py)
Task 37 (inference.py) --> Task 26 (test with raw files)
```

## 📝 Notes

- All new tasks (30-41) have been created in `tasks/backlog/`
- Original tasks (04-27) have been updated to reference new implementations
- Shell scripts (`*.sh`) act as wrappers and are already functional
- Python scripts (`*.py`) are the actual implementations and need to be created
- Task counter updated to **42**
