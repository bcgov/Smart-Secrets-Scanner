# ADR 0006: Data Directory Structure for Fine-Tuning Projects

**Date**: 2025-11-01  
**Status**: Accepted  
**Deciders**: Project Team  

## Context

When fine-tuning LLMs, organizing training data, models, and outputs is critical for:
- **Reproducibility**: Clear separation of raw and processed data
- **Collaboration**: Multiple users can contribute datasets and models
- **Version control**: Large model files must be excluded from Git
- **Workflow efficiency**: Standard locations for scripts to read/write data
- **Template reusability**: Structure works for any fine-tuning project

## Decision

We will use the following directory structure for all fine-tuning projects:

```
project/
├── data/
│   ├── raw/              # Original unprocessed data (CSV, TXT, JSON, etc.)
│   ├── processed/        # Training-ready JSONL files
│   └── evaluation/       # Test sets and benchmarks
├── models/
│   ├── base/             # Downloaded pre-trained models
│   ├── fine-tuned/       # LoRA adapters or fine-tuned weights
│   └── gguf/             # Quantized models for deployment
├── outputs/
│   ├── checkpoints/      # Training checkpoints
│   ├── logs/             # Training logs and metrics
│   └── merged/           # Merged models (base + adapter)
├── scripts/              # Bash scripts for automation
├── notebooks/            # Jupyter notebooks for exploration
├── tasks/                # Task tracking
└── adrs/                 # Architecture decisions
```

### Key Principles

1. **Separation of concerns**: Raw data is never modified; processed data is derived from raw
2. **Git-friendly**: Large files (models, GGUF, checkpoints) are excluded via `.gitignore`
3. **Self-documenting**: Each directory has a README explaining its purpose
4. **Standard locations**: Scripts know where to find data and models
5. **Portable**: Structure works for any fine-tuning project (coding, secrets scanning, etc.)

### JSONL Training Data Location

- **All training data** goes in `data/processed/` as JSONL files
- **Format**: Alpaca instruction-input-output format or ChatML
- **Naming convention**: `<use-case>-train.jsonl`, `<use-case>-val.jsonl`
- **Example**: `smart-secrets-scanner-train.jsonl`

### Model Workflow

1. Download base model → `models/base/`
2. Fine-tune with LoRA → `models/fine-tuned/` (adapter only)
3. Checkpoints saved → `outputs/checkpoints/`
4. Merge adapter + base → `outputs/merged/`
5. Convert to GGUF → `models/gguf/`
6. Deploy via Ollama or llama.cpp

## Consequences

### Positive

- **Clear workflow**: Everyone knows where to put data and models
- **Git efficiency**: Only small files (scripts, configs, docs) are tracked
- **Reusable**: Same structure for all future fine-tuning projects
- **Collaborative**: Multiple team members can work without conflicts
- **Documented**: README files guide new users

### Negative

- **More directories**: Slightly more complex than flat structure
- **Manual management**: Users must remember to put files in correct locations
- **Initial setup**: Requires creating all directories upfront

### Mitigation

- Use `.gitkeep` files to preserve directory structure in Git
- Provide example JSONL files as templates
- Document structure clearly in main README
- Scripts can auto-create directories if missing

## References

- [Hugging Face Datasets Documentation](https://huggingface.co/docs/datasets)
- [Fine-tuning Best Practices](https://huggingface.co/blog/fine-tune-llms)
- ADR 0001: Use ML-Env-CUDA13 and WSL2
- ADR 0004: Jupyter Notebook Template
