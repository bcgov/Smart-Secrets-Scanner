# Task 30: Create Training Configuration File

**Status:** Done ✅  
**Priority:** CRITICAL  
**Created:** 2025-11-01  
**Completed:** 2025-11-01  
**Related to:** Phase 2: Model Fine-Tuning (Step 5)

## Prerequisites (Completed)

✅ **Task 00**: Use case defined (understood hyperparameters needed)  
✅ **Task 20**: Initial dataset created (knew dataset size)  
✅ **Task 22**: Base model downloaded (knew model architecture)  

## Description
Create a centralized configuration file (`config/training_config.yaml`) to manage all hyperparameters and training settings.

## Requirements
- Understanding of LoRA/QLoRA hyperparameters
- Knowledge of optimal settings for small datasets (56 examples)
- YAML or JSON format for easy modification

## Acceptance Criteria
- [ ] `config/training_config.yaml` created
- [ ] All key hyperparameters documented with comments
- [ ] Default values set based on best practices for 56 examples
- [ ] Training script updated to load from config file
- [ ] Config file version-controlled

## Configuration Parameters to Include

### Model Settings
```yaml
model:
  name: "meta-llama/Meta-Llama-3-8B"
  max_seq_length: 2048
  load_in_4bit: true
  
lora:
  r: 16                    # LoRA rank
  alpha: 32                # LoRA alpha (typically 2*r)
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
  dropout: 0.05
  bias: "none"
```

### Training Settings
```yaml
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
```

### Dataset Paths
```yaml
data:
  train: "data/processed/smart-secrets-scanner-train.jsonl"
  validation: "data/processed/smart-secrets-scanner-val.jsonl"
  test: "data/evaluation/smart-secrets-scanner-test.jsonl"
```

## Dependencies
- Task 28: requirements.txt (for PyYAML dependency)
- Task 10: Training script creation (will reference this config)

## Notes
- Learning rate: 2e-4 is standard for LoRA fine-tuning
- Batch size 4 with gradient accumulation 4 = effective batch size 16
- Small dataset (56 examples) → more epochs needed (5)
- Save config alongside checkpoints for reproducibility
- Consider separate configs for experimentation vs production

## References
- Hugging Face TrainingArguments docs
- Unsloth recommended hyperparameters
- LoRA paper (Hu et al., 2021)
