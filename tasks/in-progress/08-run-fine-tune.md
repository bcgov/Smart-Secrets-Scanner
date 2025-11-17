# Task 08: Run Fine-Tuning - Iteration 4

**Status:** In-Progress  
**Priority:** HIGH  
**Created:** 2025-11-02  
**Updated:** 2025-11-02  
**Related to:** Task 46 (Enhance Training Data with Edge Cases), Task 47 (1000 Examples Dataset)

## Infrastructure Setup (Completed)

All environment and dependencies are ready from completed tasks:

✅ **Task 01**: WSL2 Ubuntu environment configured  
✅ **Task 02**: NVIDIA drivers (581.42) and CUDA Toolkit 13.0.2 installed  
✅ **Task 03**: ML-Env-CUDA13 repository cloned (sibling directory)  
✅ **Task 04**: ML-Env-CUDA13 Python environment activated  
✅ **Task 05**: Fine-tuning dependencies installed (transformers, peft, bitsandbytes, trl)  
✅ **Task 22**: Llama 3.1-8B base model downloaded to `models/base/`  
✅ **Task 30**: Training configuration file created (`config/training_config.yaml`)  
✅ **Task 36**: Fine-tuning Python script created (`scripts/fine_tune.py`)  
✅ **Task 37**: Inference script created (`scripts/inference.py`)  
✅ **Task 47**: 1000-example dataset generated (v3: 800 train/200 val)

## Current Training Run

### Iteration 4 - 1000 Examples (Running Now)
```bash
python scripts/fine_tune.py
```

**Training Configuration:**
- **Model**: Llama 3.1-8B (local at `models/base/Meta-Llama-3.1-8B`)
- **Dataset**: `smart-secrets-scanner-train-v3.jsonl` (800 examples)
- **Validation**: `smart-secrets-scanner-val-v3.jsonl` (200 examples)
- **Total**: 1000 examples (500 ALERT + 500 SAFE)
- **Method**: LoRA with 4-bit quantization (QLoRA)
- **Epochs**: 15
- **Batch size**: 1 (effective 8 with gradient accumulation)
- **Learning rate**: 2e-4
- **Precision**: FP16 (optimized for A2000)
- **Sequence Length**: 256 (optimized for 3-4x speedup)
- **Optimizations**: Gradient checkpointing, optimized dataloader, narrowed LoRA targets

### Previous Iterations (from Task 46)

| Iteration | Examples | Balance | Epochs | Accuracy | Precision | Recall | F1 | Issue |
|-----------|----------|---------|--------|----------|-----------|--------|----|----|
| 1 | 100 | N/A | 3 | Low | - | Low | - | Too verbose, missed secrets |
| 2 | 300 | 200/100 | 10 | 50% | Low | High | - | Flagged everything |
| 3 | 300 | 150/150 | 10 | 65% | 58.8% | 100% ✅ | 74.1% | Too many false positives |
| **4** | **1000** | **500/500** | **15** | **Running...** | - | - | - | **Current** |

### Key Improvements in Iteration 4
- **10x more data** (100 → 1000 examples)
- **Better balance** (500/500 instead of 150/150)
- **Edge cases included**:
  - ✅ Public API keys (Firebase, Google Maps, Stripe publishable)
  - ✅ Documentation placeholders (`YOUR_API_KEY_HERE`)
  - ✅ API endpoint URLs (without credentials)
  - ✅ Safe configuration values

### Target Metrics (from Task 46)
- **Accuracy**: ≥ 85% (up from 65%)
- **Precision**: ≥ 80% (up from 58.8%)
- **Recall**: ≥ 95% (maintain 100%)
- **F1 Score**: ≥ 85% (up from 74.1%)

## Monitoring Training

### Check Progress
```bash
# Watch training logs
tail -f outputs/logs/events.out.tfevents.*

# Monitor GPU usage
nvidia-smi -l 1
```

### View TensorBoard
```bash
tensorboard --logdir outputs/logs
```

## After Training Completes

### 1. Evaluate Model
```bash
# Full evaluation on test set
python scripts/evaluate.py --load-in-4bit

# Quick test (10 examples)
python scripts/evaluate.py --load-in-4bit --max-examples 10
```

### 2. Test Specific Cases
```bash
# Test public API key (should be SAFE)
python scripts/inference.py --input "apiKey = 'AIzaSyB-PUBLIC123'" --load-in-4bit

# Test real secret (should be ALERT)
python scripts/inference.py --input "apiKey = 'sk_live_REAL123'" --load-in-4bit
```

### 3. Review Results
- Compare metrics to Iteration 3
- Check false positive rate
- Verify edge cases are handled correctly

## Success Criteria
- [ ] Training completes without errors (15 epochs)
- [ ] Final loss < 0.05
- [ ] Accuracy ≥ 85%
- [ ] Precision ≥ 80%
- [ ] Recall ≥ 95%
- [ ] Model saved to `models/fine-tuned/smart-secrets-scanner-lora`

## Next Steps After Completion
- Update Task 46 with results
- If metrics met: Move to Task 38 (Merge adapter)
- If metrics not met: Iterate again with more data/epochs

## Files & Outputs
- **Config**: `config/training_config.yaml`
- **Training data**: `data/processed/smart-secrets-scanner-train-v3.jsonl`
- **Validation data**: `data/processed/smart-secrets-scanner-val-v3.jsonl`
- **Checkpoints**: `outputs/checkpoints/smart-secrets-scanner/`
- **Logs**: `outputs/logs/`
- **Final adapter**: `models/fine-tuned/smart-secrets-scanner-lora/`

## Related Tasks
- **Task 46**: Parent task with full context and edge case details
- **Task 30**: Training config file
- **Task 36**: Fine-tuning script implementation
- **Task 37**: Inference script for testing
