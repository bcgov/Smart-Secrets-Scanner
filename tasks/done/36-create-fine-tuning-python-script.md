# Task 36: Create Fine-Tuning Python Script

**Status:** Backlog  
**Priority:** CRITICAL  
**Created:** 2025-11-01  
**Related to:** Phase 2: Model Fine-Tuning (Step 5)  
**Depends on:** Task 30 (training config), Task 05 (dependencies installed)

## Description
Create `scripts/fine_tune.py` - the main Python script for fine-tuning Llama 3 with LoRA adapters using the Smart Secrets Scanner dataset.

## Requirements
- Load training/validation JSONL datasets
- Load base Llama 3 model
- Apply LoRA configuration
- Train with proper hyperparameters
- Save checkpoints and final adapter
- Integration with experiment tracking (W&B/TensorBoard)

## Acceptance Criteria
- [ ] `scripts/fine_tune.py` created and executable
- [ ] Loads config from `config/training_config.yaml`
- [ ] Loads datasets from `data/processed/*.jsonl`
- [ ] Applies LoRA to base model
- [ ] Trains with gradient accumulation, mixed precision
- [ ] Saves checkpoints to `outputs/checkpoints/`
- [ ] Saves final adapter to `models/fine-tuned/`
- [ ] Logs metrics to W&B or TensorBoard
- [ ] Handles errors gracefully with informative messages
- [ ] Can be called by `scripts/fine_tune.sh`

## Script Implementation
Create `scripts/fine_tune.py`:

```python
#!/usr/bin/env python3
"""
Fine-tune Llama 3 with LoRA for Smart Secrets Scanner
"""
import os
import sys
import yaml
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

def load_config(config_path="config/training_config.yaml"):
    """Load training configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_datasets(config):
    """Load training and validation datasets"""
    print("📊 Loading datasets...")
    dataset = load_dataset('json', data_files={
        'train': config['data']['train'],
        'validation': config['data']['validation']
    })
    
    print(f"  Training examples: {len(dataset['train'])}")
    print(f"  Validation examples: {len(dataset['validation'])}")
    return dataset

def format_prompt(example):
    """Format examples as Alpaca prompts"""
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    }

def setup_model_and_tokenizer(config):
    """Load base model with 4-bit quantization and prepare for LoRA"""
    print(f"🔽 Loading base model: {config['model']['name']}")
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config['model']['load_in_4bit'],
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def setup_lora(model, config):
    """Apply LoRA configuration to model"""
    print("🔧 Configuring LoRA adapters...")
    
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['dropout'],
        bias=config['lora']['bias'],
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

def main():
    print("=" * 60)
    print("🚀 Smart Secrets Scanner - Fine-Tuning Script")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    print(f"✅ Loaded config from config/training_config.yaml")
    
    # Create output directories
    Path(config['training']['output_dir']).mkdir(parents=True, exist_ok=True)
    Path("models/fine-tuned").mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    dataset = load_datasets(config)
    
    # Format datasets
    print("📝 Formatting prompts...")
    dataset = dataset.map(format_prompt)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Apply LoRA
    model = setup_lora(model, config)
    
    # Training arguments
    print("⚙️  Configuring training arguments...")
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        warmup_steps=config['training']['warmup_steps'],
        logging_steps=config['training']['logging_steps'],
        save_steps=config['training']['save_steps'],
        eval_steps=config['training']['eval_steps'],
        evaluation_strategy="steps",
        save_total_limit=config['training']['save_total_limit'],
        fp16=config['training']['fp16'],
        optim=config['training']['optim'],
        logging_dir="./outputs/logs",
        report_to=config['training'].get('report_to', 'tensorboard'),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_safetensors=True,
    )
    
    # Setup trainer
    print("🎓 Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        args=training_args,
        dataset_text_field="text",
        tokenizer=tokenizer,
        max_seq_length=config['model']['max_seq_length'],
        packing=False,
    )
    
    # Train!
    print("\n" + "=" * 60)
    print("🏋️  Starting Training...")
    print("=" * 60)
    
    trainer.train()
    
    # Save final model
    print("\n💾 Saving final LoRA adapter...")
    output_path = "models/fine-tuned/smart-secrets-scanner-lora"
    trainer.model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"📁 LoRA adapter saved to: {output_path}")
    print(f"📁 Checkpoints saved to: {config['training']['output_dir']}")
    print(f"📁 Logs saved to: outputs/logs/")
    print("\nNext steps:")
    print("  1. Review training logs: tensorboard --logdir outputs/logs")
    print("  2. Merge adapter: python scripts/merge_adapter.py")
    print("  3. Convert to GGUF: python scripts/convert_to_gguf.py")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
```

## Dependencies
- Task 30: Training config file must exist
- Task 22: Base model downloaded
- Task 05: Python dependencies installed (transformers, peft, trl, datasets, bitsandbytes)

## Testing
```bash
# Validate script runs
python scripts/fine_tune.py

# Or via shell wrapper
bash scripts/fine_tune.sh
```

## Success Criteria
- Script runs without errors
- Training begins and checkpoints are saved
- Validation loss decreases over epochs
- Final adapter saved to `models/fine-tuned/`
- Compatible with existing `scripts/fine_tune.sh`

## Related Files
- Called by: `scripts/fine_tune.sh`
- Reads: `config/training_config.yaml`
- Reads: `data/processed/smart-secrets-scanner-{train,val}.jsonl`
- Writes: `models/fine-tuned/smart-secrets-scanner-lora/`
- Writes: `outputs/checkpoints/checkpoint-*/`
- Writes: `outputs/logs/`
