# Task 46: Enhance Training Data with Edge Cases

**Status:** In-Progress  
**Priority:** HIGH  
**Created:** 2025-11-01  
**Related to:** Phase 3: Model Training - Iteration 4  
**Depends on:** Task 11 (LoRA training completed 3 times)

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

- [ ] Add 50+ edge case examples to training data generation script
- [ ] Generate balanced dataset: 200 ALERT + 200 SAFE (400 total)
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

### 1. Update `scripts/generate_simple_training_data.py`

Add new safe pattern categories:

```python
# Public API keys (safe to commit)
PUBLIC_API_KEYS = [
    ('apiKey', 'AIzaSyB-PUBLIC123-ReadOnlyKey'),  # Firebase public
    ('GOOGLE_MAPS_KEY', 'AIzaSyPublicKeyForWebClient'),
    ('STRIPE_PUBLISHABLE_KEY', 'pk_test_51H...'),  # Stripe public
]

# Documentation placeholders (safe examples)
PLACEHOLDER_VALUES = [
    ('API_KEY', 'YOUR_API_KEY_HERE'),
    ('SECRET_TOKEN', '<your-token-here>'),
    ('PASSWORD', 'your_password'),
    ('AWS_KEY', 'AKIAIOSFODNN7EXAMPLE'),  # AWS example key
]

# API endpoints without credentials (safe)
API_ENDPOINTS = [
    ('API_URL', 'https://api.example.com/v1/users'),
    ('ENDPOINT', 'https://jsonplaceholder.typicode.com/posts'),
    ('BASE_URL', 'https://api.github.com'),
]

# Configuration examples (safe)
CONFIG_EXAMPLES = [
    ('db_host', 'localhost'),
    ('PORT', '5432'),
    ('DEBUG', 'True'),
    ('MAX_CONNECTIONS', '100'),
]
```

### 2. Update Generation Logic

```python
def generate_training_data(num_alert=200, num_safe=200):
    """Generate training data with edge cases"""
    examples = []
    
    # Generate alert examples (actual secrets)
    for _ in range(num_alert):
        secret_type = random.choice(list(SECRETS.keys()))
        var_name, value = random.choice(SECRETS[secret_type])
        examples.append(generate_alert_example(secret_type, var_name, value))
    
    # Generate safe examples (mix of all categories)
    safe_categories = [
        (SAFE_PATTERNS, 0.4),          # 40%: env vars, config.get()
        (PUBLIC_API_KEYS, 0.2),        # 20%: public keys
        (PLACEHOLDER_VALUES, 0.2),     # 20%: placeholders
        (API_ENDPOINTS, 0.1),          # 10%: endpoints
        (CONFIG_EXAMPLES, 0.1),        # 10%: config
    ]
    
    for _ in range(num_safe):
        category, _ = random.choices(
            safe_categories, 
            weights=[w for _, w in safe_categories]
        )[0]
        pattern = random.choice(category)
        examples.append(generate_safe_example(pattern))
    
    random.shuffle(examples)
    return examples
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

- `scripts/generate_simple_training_data.py` - Add edge case examples
- `config/training_config.yaml` - Increase epochs to 15
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
