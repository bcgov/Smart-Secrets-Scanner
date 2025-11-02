# Task 32: Create Model Evaluation Script

**Status:** Backlog  
**Priority:** HIGH  
**Created:** 2025-11-01  
**Related to:** Phase 4: Testing & Deployment (Step 10)

## Prerequisites (Completed)

✅ **Task 01-05**: Environment setup (WSL2, NVIDIA, ML-Env-CUDA13, dependencies)  
✅ **Task 31**: Evaluation test dataset created  
✅ **Task 37**: Inference script created (reference implementation)  

**Pending:**  
⏳ **Task 08**: Fine-tuning Iteration 4 (in-progress)  
⏳ **Task 38**: Merge LoRA adapter with base model  

## Description
Create an automated evaluation script (`scripts/evaluate.py`) to test the fine-tuned model on the test dataset and generate comprehensive metrics.

## Requirements
- Python script that loads model and test data
- Calculates precision, recall, F1 score, accuracy
- Generates confusion matrix and classification report
- Saves results to JSON for tracking over time

## Acceptance Criteria
- [ ] `scripts/evaluate.py` created and executable
- [ ] Loads model from `outputs/merged/` or Ollama
- [ ] Loads test data from `data/evaluation/test.jsonl`
- [ ] Calculates all required metrics
- [ ] Generates confusion matrix visualization
- [ ] Saves results to `outputs/evaluation/metrics.json`
- [ ] Displays sample predictions for manual review
- [ ] Handles errors gracefully

## Script Structure

```python
#!/usr/bin/env python3
"""
Evaluate fine-tuned Smart Secrets Scanner model
"""
import json
import argparse
from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    accuracy_score, confusion_matrix, classification_report
)
from transformers import pipeline
from datasets import load_dataset

def load_test_data(test_path):
    """Load test JSONL dataset"""
    dataset = load_dataset('json', data_files={'test': test_path})
    return dataset['test']

def run_inference(model_path, test_examples):
    """Run model inference on test examples"""
    generator = pipeline('text-generation', model=model_path)
    predictions = []
    
    for example in test_examples:
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n"
        output = generator(prompt, max_new_tokens=100)[0]['generated_text']
        predictions.append(output)
    
    return predictions

def parse_predictions(predictions):
    """Convert model outputs to binary labels"""
    labels = []
    for pred in predictions:
        # Check if "ALERT" appears in response
        labels.append(1 if "ALERT" in pred else 0)
    return labels

def calculate_metrics(y_true, y_pred, output_dir):
    """Calculate and save evaluation metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }
    
    # Save to JSON
    output_path = Path(output_dir) / 'metrics.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print report
    print("\n=== Evaluation Results ===")
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1 Score:  {metrics['f1_score']:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Safe', 'Secret']))
    
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='./outputs/merged/smart-secrets-scanner')
    parser.add_argument('--test-data', default='data/evaluation/smart-secrets-scanner-test.jsonl')
    parser.add_argument('--output-dir', default='outputs/evaluation')
    args = parser.parse_args()
    
    # Load test data
    print(f"Loading test data from {args.test_data}...")
    test_data = load_test_data(args.test_data)
    
    # Run inference
    print(f"Running inference with model {args.model}...")
    predictions = run_inference(args.model, test_data)
    
    # Parse predictions and ground truth
    y_pred = parse_predictions(predictions)
    y_true = [1 if "ALERT" in ex['output'] else 0 for ex in test_data]
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, args.output_dir)
    
    print(f"\nResults saved to {args.output_dir}/metrics.json")

if __name__ == '__main__':
    main()
```

## Dependencies
- Task 31: Test dataset must exist
- Task 13: Model must be merged/exported
- Task 28: requirements.txt (sklearn, transformers, datasets)

## Notes
- For Ollama deployment, modify to use Ollama API instead of transformers
- Consider adding ROC curve and precision-recall curve visualizations
- Track metrics over time to monitor model drift
- Include timestamp in metrics.json for versioning
- Option: Compare against baseline (regex-based scanner, GitGuardian)

## Success Criteria
- Precision > 90% (few false positives)
- Recall > 95% (catch almost all secrets)
- F1 > 92% (balanced performance)
