# Task 44: Create Model Evaluation Notebook

**Status:** Backlog  
**Priority:** MEDIUM  
**Created:** 2025-11-01  
**Related to:** Phase 3-4: Model Export & Testing (Steps 7-11), Approach 2: Jupyter Notebooks

## Prerequisites (Completed)

✅ **Task 01-05**: Environment setup (WSL2, NVIDIA, ML-Env-CUDA13, dependencies)  
✅ **Task 22**: Base model downloaded  
✅ **Task 31**: Evaluation test dataset created  
✅ **Task 37**: Inference script created (reference implementation)  
✅ **Task 08**: Fine-tuning run (Iteration 4 in-progress)  

## Description
Create `notebooks/03_model_evaluation.ipynb` - an interactive notebook for merging the LoRA adapter, converting to GGUF, and comprehensively evaluating the fine-tuned model.

## Requirements
- Merge LoRA adapter with base model
- Convert to GGUF format
- Run evaluation on test dataset
- Calculate precision, recall, F1 score
- Test inference on code samples
- Generate confusion matrix and visualizations

## Acceptance Criteria
- [ ] `notebooks/03_model_evaluation.ipynb` created
- [ ] Merges adapter with base model
- [ ] Converts to GGUF (calls llama.cpp)
- [ ] Loads test dataset
- [ ] Runs inference on all test examples
- [ ] Calculates classification metrics
- [ ] Generates confusion matrix visualization
- [ ] Tests on real code files
- [ ] Displays false positives/negatives for analysis
- [ ] Well-documented with insights

## Notebook Sections

### 1. Introduction
- Overview of evaluation process
- Load necessary libraries
- Check for required files (adapter, base model)

### 2. Merge LoRA Adapter
- Load base model and adapter
- Merge weights
- Save merged model to `outputs/merged/`
- Verify merge with test inference

### 3. Convert to GGUF
- Call llama.cpp conversion script
- Create quantized versions (Q4_K_M, Q8_0)
- Display file sizes and compression ratios
- Test GGUF model loading

### 4. Load Test Dataset
- Load evaluation JSONL
- Display test set statistics
- Show example test cases

### 5. Run Inference
- Test merged model on all test examples
- Collect predictions
- Display progress with results preview

### 6. Calculate Metrics
- Compute accuracy, precision, recall, F1
- Generate classification report
- Create confusion matrix heatmap
- Display metrics comparison table

### 7. Error Analysis
- Show false positives (safe code flagged as secret)
- Show false negatives (secrets missed)
- Analyze patterns in errors
- Suggestions for improvement

### 8. Real Code Testing
- Test on complete Python/JavaScript files
- Show detection results with line numbers
- Interactive code viewer with highlights
- Performance metrics (inference time per file)

### 9. Comparison (Optional)
- Compare with baseline (regex patterns)
- Compare different quantization levels
- A/B testing visualization

### 10. Summary & Recommendations
- Overall model performance
- Deployment recommendations (which quantization to use)
- Next steps for production

## Dependencies
- Task 38: Merge adapter script (reference)
- Task 39: GGUF conversion script (reference)
- Task 32: Evaluation script (reference)
- Task 31: Test dataset created
- Python packages: transformers, sklearn, matplotlib, seaborn, plotly

## Visualizations
```python
# Confusion Matrix Heatmap
import seaborn as sns
import matplotlib.pyplot as plt

# ROC Curve (optional)
from sklearn.metrics import roc_curve, auc

# Interactive plots
import plotly.express as px
```

## Usage
```bash
# Launch notebook (after training complete)
jupyter notebook notebooks/03_model_evaluation.ipynb

# Requires:
# - Trained LoRA adapter
# - Test dataset
# - llama.cpp built
```

## Expected Outputs
- Merged model: `outputs/merged/smart-secrets-scanner/`
- GGUF files: `models/gguf/smart-secrets-scanner-*.gguf`
- Metrics JSON: `outputs/evaluation/metrics.json`
- Confusion matrix visualization
- Error analysis report
- Performance benchmarks

## Success Criteria
- Achieves target metrics (>90% precision, >95% recall)
- Clear visualizations of model performance
- Errors are analyzed and documented
- Users understand model strengths/weaknesses
- Actionable recommendations provided

## Advanced Features (Optional)
- Interactive confusion matrix (click to see examples)
- Confidence score calibration plots
- Per-secret-type performance breakdown
- Comparison with GitGuardian/TruffleHog
- Inference speed benchmarks

## Related Tasks
- Task 32: Evaluation script (CLI equivalent)
- Task 38: Merge adapter script
- Task 39: GGUF conversion script
- Task 25: Test with evaluation JSONL
- Task 26: Test with raw files
- Task 45: Deployment testing notebook (next step)
