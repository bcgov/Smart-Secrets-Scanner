# Task 25: Test with Evaluation JSONL

**Status:** Backlog  
**Created:** 2025-11-01  
**Related to:** Phase 4: Testing & Deployment (Step 10)

## Description
Test the fine-tuned model using evaluation JSONL data to calculate precision, recall, and F1 score.

## Requirements
- Fine-tuned model deployed (Task 14, 15)
- Evaluation dataset created in `data/evaluation/test.jsonl`
- Python script to calculate metrics

## Acceptance Criteria
- [ ] Model tested on all examples in `data/evaluation/test.jsonl`
- [ ] Precision, recall, F1 score calculated
- [ ] Results documented in `outputs/evaluation/metrics.json`
- [ ] Confusion matrix generated (TP, TN, FP, FN)

## Steps
1. Create test set in `data/evaluation/test.jsonl` (if not exists)
2. Run inference on each test example
3. Compare model output to expected output
4. Calculate metrics:
   - Precision = TP / (TP + FP)
   - Recall = TP / (TP + FN)
   - F1 = 2 * (Precision * Recall) / (Precision + Recall)
5. Document results

## Dependencies
- Task 15: Ollama deployment complete
- Evaluation dataset created

## Notes
- Test set should be different from training/validation data
- Target: >90% precision, >85% recall for secrets detection
- False positives acceptable if they're borderline cases
- False negatives are critical (missed secrets)
