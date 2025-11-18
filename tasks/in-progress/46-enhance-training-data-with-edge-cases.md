# Task 46: Enhance Training Data with Edge Cases

**Status:** In-Progress  
**Priority:** HIGH  
**Created:** 2025-11-01  
**Updated:** 2025-11-02  
**Related to:** Phase 3: Model Training - Iteration 4  
**Depends on:** Task 11 (LoRA training completed 3 times)

## Prerequisites (Completed)

✅ **Task 01-05**: Environment setup (WSL2, NVIDIA, ML-Env-CUDA13, dependencies)  
✅ **Task 22**: Llama 3.1-8B base model downloaded  
✅ **Task 30**: Training configuration file created  
✅ **Task 31**: Evaluation test dataset created  
✅ **Task 36**: Fine-tuning Python script created  
✅ **Task 37**: Inference script created  
✅ **Task 47**: 1000-example dataset (v3) generated with edge cases  
✅ **Task 08**: Iteration 4 training in-progress (1000 examples, 15 epochs)  

## Session Summary - What We've Done

### ✅ Completed Infrastructure
- Created `requirements.txt` with all dependencies
- Fixed API compatibility issues (eval_strategy, processing_class)
- Added HuggingFace authentication to training script
- Configured local model path (no re-downloads)
- Created inference script with 4-bit quantization support
- Created comprehensive evaluation script with metrics

### ✅ Fine-Tuning Iterations
1. **Iteration 1** (100 examples, 3 epochs): Too verbose, didn't detect secrets
2. **Iteration 2** (300 examples: 200 ALERT + 100 SAFE, 10 epochs): 50% accuracy, flagged everything
3. **Iteration 3** (300 examples: 150 ALERT + 150 SAFE, 10 epochs): **65% accuracy, 100% recall**

### 📊 Current Model Performance (Iteration 3)
- **Accuracy**: 65.0%
- **Precision**: 58.8%
- **Recall**: 100.0% ✅ (catches ALL real secrets!)
- **F1 Score**: 74.1%
- **False Positives**: 7/20 test examples

### ⚠️ Current False Positive Examples
1. **Firebase public API keys** - Meant to be public but flagged as secrets
2. **Documentation placeholders** - `"YOUR_API_KEY_HERE"` flagged incorrectly
3. **API endpoint URLs** - `"https://api.example.com"` flagged incorrectly

## Problem Statement

The model has **100% recall** (catches all real secrets) but **58.8% precision** (too many false positives). It needs to learn nuanced cases where credentials appear in code but are actually safe:
- Public API keys (Firebase, Google Maps)
- Placeholder/example values in documentation
- API endpoints without embedded credentials
- Configuration examples in README files

## Acceptance Criteria

- [ ] Generate 50+ additional edge case examples using LLM direct creation
- [ ] Create balanced dataset: 200 ALERT + 200 SAFE (400 total) via LLM
- [ ] Include nuanced safe examples:
  - [ ] Firebase public keys
  - [ ] Documentation placeholders
  - [ ] API endpoint URLs
  - [ ] Example configuration files
- [ ] Retrain model for 10-15 epochs
- [ ] Evaluate and achieve:
  - [ ] Accuracy ≥ 85%
  - [ ] Precision ≥ 80%
  - [ ] Recall ≥ 95%
  - [ ] F1 Score ≥ 85%

## Implementation Steps

### 1. LLM-Generated Edge Cases

Use LLM to directly generate additional training examples focusing on edge cases that cause false positives:

**Safe Pattern Categories to Generate:**
```json
// Public API keys (safe to commit)
{"instruction": "...", "input": "firebase_key = 'AIzaSyB-PUBLIC123-ReadOnlyKey'", "output": "No secrets detected."}

// Documentation placeholders (safe examples)
{"instruction": "...", "input": "API_KEY = 'YOUR_API_KEY_HERE'", "output": "No secrets detected."}

// API endpoints without credentials (safe)
{"instruction": "...", "input": "api_url = 'https://api.example.com/v1/users'", "output": "No secrets detected."}

// Configuration examples (safe)
{"instruction": "...", "input": "DEBUG = True; PORT = 5432", "output": "No secrets detected."}
```

**Prompt for LLM Generation:**
```
Generate 50 JSONL examples for secret detection training. Focus on edge cases that commonly cause false positives:

1. Public API keys (Firebase, Google Maps, Stripe publishable keys)
2. Documentation placeholders (YOUR_API_KEY_HERE, <token>, example values)
3. API endpoints and URLs without embedded secrets
4. Configuration values (ports, debug flags, hostnames)
5. Environment variable usage patterns
6. Test fixtures and mock data

For each example, create realistic code snippets that look like secrets but are actually safe. Label them as "No secrets detected." with appropriate explanations.
```

### 3. Retrain Model

```bash
# Generate new training data
python scripts/generate_simple_training_data.py

# Update config to use 15 epochs (more learning time)
# Edit config/training_config.yaml: num_train_epochs: 15

# Retrain
python scripts/fine_tune.py
```

### 4. Evaluate

```bash
# Quick test (10 examples)
python scripts/evaluate.py --load-in-4bit --max-examples 10

# Full evaluation (all 20 test examples)
python scripts/evaluate.py --load-in-4bit
```

## Expected Results

After retraining with enhanced data:
- **Accuracy**: 85-90% (up from 65%)
- **Precision**: 80-85% (up from 58.8%)
- **Recall**: 95-100% (maintain high recall)
- **F1 Score**: 85-90% (up from 74.1%)
- **False Positives**: <3 out of 20 test examples

## Key Learnings from Previous Iterations

1. ✅ **Simple outputs work best** - "ALERT" or "SAFE" instead of long explanations
2. ✅ **Balance is critical** - 50/50 split prevents bias (200/200 not 200/100)
3. ✅ **High recall is non-negotiable** - Must catch all real secrets (100% achieved)
4. ⚠️ **Edge cases need explicit training** - Model can't infer context without examples
5. ✅ **Iterative improvement works** - 0% → 50% → 65% → (target 85%+)

## Files to Modify

- `data/processed/smart-secrets-scanner-train-v2.jsonl` - Create new dataset with edge cases
- `config/training_config.yaml` - Increase epochs to 15
- `scripts/fine_tune.py` - Update dataset path to use v2 dataset
- (Optional) `scripts/evaluate.py` - Add breakdown by error type

## Success Metrics

### Training Metrics
- Loss < 0.05 (achieved in iteration 3)
- Token accuracy > 98% (achieved in iteration 3)

### Evaluation Metrics
- **Must Have**: Recall ≥ 95% (catch all real secrets)
- **Should Have**: Precision ≥ 80% (minimize false positives)
- **Nice to Have**: Accuracy ≥ 90% (overall correctness)

## Timeline Estimate

- **Data enhancement**: 30 minutes
- **Training**: 50 minutes (15 epochs)
- **Evaluation**: 10 minutes
- **Total**: ~1.5 hours to production-ready model

## Next Steps After This Task

Once model achieves 85%+ accuracy:
1. **Task 32**: Mark evaluation script as done
2. **Task 38**: Merge LoRA adapter with base model
3. **Task 39**: Convert to GGUF format
4. **Task 40**: Create Modelfile for Ollama
5. **Task 14/15**: Deploy to Ollama and test

## Notes

- For security tools, **high recall > high precision** (better to over-alert than miss secrets)
- Current 65% accuracy is usable with manual review workflow
- Target 85%+ makes it production-ready for automated scanning
- Consider collecting real-world examples for continuous improvement

## Commands Reference

```bash
# Generate balanced training data with edge cases
python scripts/generate_simple_training_data.py

# Retrain model (15 epochs, ~50 minutes)
python scripts/fine_tune.py

# Quick evaluation (10 examples)
python scripts/evaluate.py --load-in-4bit --max-examples 10

# Full evaluation (all test examples)
python scripts/evaluate.py --load-in-4bit

# Test specific example
python scripts/inference.py --input "apiKey = 'AIzaSyB-PUBLIC123'" --load-in-4bit
```

## Related Tasks

- **Task 11**: Train LoRA adapter (completed 3 iterations)
- **Task 32**: Create evaluation script (in-progress)
- **Task 37**: Create inference script (done)
- **Task 38**: Merge adapter with base model (backlog)
- **Task 39**: Convert to GGUF (backlog)

## Architecture Reference

This task relates to the LLM-driven dataset creation approach documented in:  
**ADR 0007: LLM-Driven Dataset Creation for Secret Detection Training** (`adrs/0007-llm-driven-dataset-creation.md`)
