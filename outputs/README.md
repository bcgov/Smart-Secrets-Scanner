# Outputs Directory Structure

This directory contains all outputs from fine-tuning runs, including checkpoints, logs, and merged models.

## Subdirectories

### `checkpoints/`
- **Purpose**: Training checkpoints saved during fine-tuning
- **Contents**: Model states at various training steps for recovery and comparison
- **Example**: `checkpoint-500/`, `checkpoint-1000/`, `checkpoint-final/`
- **Note**: Useful for resuming training or selecting the best checkpoint

### `logs/`
- **Purpose**: Training logs, metrics, and debugging information
- **Contents**: TensorBoard logs, text logs, loss curves, evaluation metrics
- **Example**: `training-2025-11-01.log`, `tensorboard/`, `metrics.json`
- **Note**: Essential for monitoring training progress and debugging issues

### `merged/`
- **Purpose**: Fully merged models (base + adapter) ready for inference or conversion
- **Contents**: Complete model weights after merging LoRA adapters with base models
- **Example**: `smart-secrets-scanner-merged/`, `custom-model-full/`
- **Note**: These are used as input for GGUF conversion or direct deployment

## Best Practices

1. **Clean up checkpoints**: Delete intermediate checkpoints after selecting the best one to save disk space
2. **Archive logs**: Compress and archive old logs for future reference
3. **Version outputs**: Use timestamps or run IDs to track different training runs
4. **Monitor disk usage**: Checkpoints and merged models can consume significant space
5. **Backup important runs**: Keep backups of successful training runs and their configs

## Workflow

1. Fine-tuning → Saves checkpoints to `checkpoints/` and logs to `logs/`
2. Select best checkpoint → Merge with base model → Save to `merged/`
3. Convert merged model to GGUF → Save to `models/gguf/`
4. Deploy model via Ollama or llama.cpp
