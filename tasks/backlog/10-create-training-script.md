# Task: Create Python Training Script

**Status: Backlog**

## Description
Create a Python script (`fine_tune.py`) to train a LoRA adapter on your dataset using the base model.

**🔗 Implemented by Task 36: Create Fine-Tuning Python Script**

See `tasks/backlog/36-create-fine-tuning-python-script.md` for the complete implementation of `scripts/fine_tune.py`.

## Steps
- Create `fine_tune.py` in the project root
- Load base model with 4-bit quantization
- Configure LoRA parameters (r=16, alpha=32, target modules)
- Load and format dataset
- Set up SFTTrainer with appropriate hyperparameters
- Train and save LoRA adapter to `outputs/` directory

## Resources
- Project README: fine_tune.sh script
- ADR 0002: Library requirements (transformers, peft, unsloth, trl)
- Example: Google Colab notebook from previous work
