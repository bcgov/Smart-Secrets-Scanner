# Task 47: Extend Training Data to 1000 Examples (Balanced, Diverse, Context-Aware)

**Status:** ✅ Done  
**Priority:** HIGH  
**Created:** 2025-11-02  
**Completed:** 2025-11-02  
**Related to:** Task 46 (edge case enhancement), Task 11 (LoRA training), Task 32 (evaluation)

## Prerequisites (Completed)

✅ **Task 00**: Use case defined  
✅ **Task 17**: Data directory structure  
✅ **Task 18**: JSONL format defined  
✅ **Task 20**: Initial 100-example dataset created  
✅ **Task 46**: Edge case research completed (informed v3 dataset design)  
✅ **Task 48**: Edge case improvements researched  

## Objective

Expand the training dataset to **1000 examples** that are:
- **Balanced:** 500 ALERT + 500 SAFE
- **Diverse:** Covers real-world secret patterns and edge cases
- **Context-aware:** Explicitly includes nuanced SAFE examples to reduce false positives
- **Production-ready:** Targets ≥85% accuracy and precision

## Acceptance Criteria

- [ ] Update `scripts/generate_simple_training_data.py` to generate 500 ALERT and 500 SAFE examples
- [ ] SAFE examples must include:
  - [ ] Public API keys (Firebase, Google Maps, Stripe publishable)
  - [ ] Documentation placeholders (`YOUR_API_KEY_HERE`, `<your-token-here>`)
  - [ ] API endpoints without credentials
  - [ ] Example config values
  - [ ] Public keys/certs, test credentials, sample data
  - [ ] Secrets loaded from env vars or secret managers
- [ ] ALERT examples must include:
  - [ ] Hardcoded secrets in code, config, comments, JSON/YAML, Dockerfiles, CI/CD configs
  - [ ] Obfuscated, encoded, or commented secrets
- [ ] Multi-language support (Python, JS, YAML, Dockerfile, etc.)
- [ ] Retrain model for 15 epochs
- [ ] Achieve evaluation metrics:
  - [ ] Accuracy ≥ 85%
  - [ ] Precision ≥ 85%
  - [ ] Recall ≥ 95%
  - [ ] F1 Score ≥ 85%

## Implementation Steps

1. **Update Data Generation Script**
   - Expand pattern lists for both ALERT and SAFE categories
   - Ensure edge cases and context-aware examples are included
   - Support multi-language formats

2. **Generate Dataset**
   ```bash
   python scripts/generate_simple_training_data.py
   ```

3. **Retrain Model**
   - Update config for 15 epochs
   ```bash
   python scripts/fine_tune.py
   ```

4. **Evaluate**
   - Quick test (10 examples)
   ```bash
   python scripts/evaluate.py --load-in-4bit --max-examples 10
   ```
   - Full evaluation (all test examples)
   ```bash
   python scripts/evaluate.py --load-in-4bit
   ```

## Expected Results

- **Accuracy:** ≥85%
- **Precision:** ≥85%
- **Recall:** ≥95%
- **F1 Score:** ≥85%
- **Significantly reduced false positives on edge cases**

## Notes

- Use patterns and insights from GitHub Secret Scanning, TruffleHog, Gitleaks, OWASP, and recent research papers.
- Continue to iterate and add new edge cases as needed.
- This task builds directly on Task 46 and is required for production readiness.

## Related Tasks

- **Task 46:** Enhance training data with edge cases (foundation for this task)
- **Task 11:** LoRA training (completed)
- **Task 32:** Evaluation script (in-progress)
- **Task 38:** Merge adapter with base model (backlog)
