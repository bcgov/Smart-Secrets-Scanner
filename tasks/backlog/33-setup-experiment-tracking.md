# Task 33: Setup Experiment Tracking (Weights & Biases or TensorBoard)

**Status:** Backlog  
**Priority:** HIGH  
**Created:** 2025-11-01  
**Related to:** Phase 2: Model Fine-Tuning (Step 6)

## Prerequisites (Completed)

✅ **Task 30**: Training configuration created (includes TensorBoard setup)  
✅ **Task 36**: Fine-tuning script created (reports to TensorBoard)  

**Note:** TensorBoard is already configured in `config/training_config.yaml` (`report_to: "tensorboard"`).

## Description
Integrate experiment tracking to systematically log hyperparameters, metrics, and artifacts during training runs.

## Requirements
- Choose between Weights & Biases (W&B), MLflow, or TensorBoard
- Integration with training script
- Track loss curves, learning rate, system metrics
- Enable comparison of multiple training runs

## Acceptance Criteria
- [ ] Experiment tracking tool selected (recommend W&B for ease of use)
- [ ] Tool installed and configured
- [ ] Training script updated to log metrics
- [ ] Dashboard accessible for viewing results
- [ ] Documented in README and ADRs

## Recommended: Weights & Biases (W&B)

### Installation
```bash
pip install wandb
wandb login
```

### Integration in Training Script
```python
import wandb

# Initialize tracking
wandb.init(
    project="smart-secrets-scanner",
    name="llama3-lora-run1",
    config={
        "learning_rate": 2e-4,
        "epochs": 5,
        "batch_size": 4,
        "lora_r": 16,
        "lora_alpha": 32,
        "model": "meta-llama/Meta-Llama-3-8B"
    }
)

# During training (automatic with Transformers)
training_args = TrainingArguments(
    ...,
    report_to="wandb"  # Enables automatic logging
)

# Manual logging
wandb.log({"custom_metric": value})

# Save model artifact
wandb.save("outputs/checkpoints/checkpoint-best/*")
```

### What to Track
1. **Hyperparameters**: Learning rate, batch size, epochs, LoRA settings
2. **Training Metrics**: Loss (train/val), learning rate schedule, gradient norms
3. **System Metrics**: GPU utilization, memory usage, training time
4. **Model Artifacts**: Checkpoints, merged model, GGUF files
5. **Dataset Info**: Number of examples, split ratios, secret types
6. **Evaluation Results**: Precision, recall, F1, confusion matrix

## Alternative: TensorBoard (Offline)

### Installation
```bash
pip install tensorboard
```

### Integration
```python
training_args = TrainingArguments(
    ...,
    logging_dir="./outputs/logs",
    report_to="tensorboard"
)
```

### View Dashboard
```bash
tensorboard --logdir outputs/logs
```

## Dependencies
- Task 10: Training script creation
- Task 30: Training config file
- Task 28: requirements.txt update

## Notes
- **W&B Pros**: Cloud-based, beautiful UI, collaboration features, model versioning
- **W&B Cons**: Requires internet, account needed (free tier available)
- **TensorBoard Pros**: Offline, no account needed, integrated with PyTorch
- **TensorBoard Cons**: Less polished UI, no cloud features
- **MLflow**: Good for production, overkill for POC

## Decision Criteria
- **POC/Research**: Use W&B (fastest setup, best UX)
- **Production**: Consider MLflow or custom solution
- **No internet**: Use TensorBoard

## Architecture Decision Record
Create ADR: `adrs/0007-experiment-tracking-wandb.md` documenting choice

## Success Criteria
- Can view loss curves in real-time during training
- Can compare multiple runs side-by-side
- Hyperparameters automatically logged
- Training can resume from checkpoints
