#!/usr/bin/env python3
"""
Fine-tune Llama 3 8B for Smart Secrets Scanner using LoRA/QLoRA

Usage:
    python scripts/fine_tune.py
    python scripts/fine_tune.py --config config/custom_config.yaml
"""
import os
import sys
import argparse
import yaml
import torch
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from huggingface_hub import login

# Load Hugging Face token from .env file
def load_hf_token():
    """Load Hugging Face token from .env file"""
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if line.startswith('HUGGING_FACE_TOKEN='):
                    token = line.split('=', 1)[1].strip().strip('"').strip("'")
                    return token
    return None

def load_config(config_path="config/training_config.yaml"):
    """Load training configuration from YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def format_alpaca_prompt(example, prompt_template):
    """Format example as Alpaca prompt"""
    return {
        "text": prompt_template.format(
            instruction=example['instruction'],
            input=example['input'],
            output=example['output']
        )
    }

def load_datasets(config):
    """Load training and validation datasets"""
    print(f"📊 Loading datasets...")
    print(f"   Train: {config['data']['train_file']}")
    print(f"   Val:   {config['data']['val_file']}")
    
    # Load datasets
    dataset = load_dataset('json', data_files={
        'train': config['data']['train_file'],
        'validation': config['data']['val_file']
    })
    
    print(f"✅ Loaded {len(dataset['train'])} training examples")
    print(f"✅ Loaded {len(dataset['validation'])} validation examples")
    
    return dataset

def setup_model_and_tokenizer(config):
    """Load base model with 4-bit quantization and prepare for LoRA"""
    print(f"\n🔽 Loading base model: {config['model']['model_name']}")
    print(f"   Using 4-bit quantization for efficient training")
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config['quantization']['load_in_4bit'],
        bnb_4bit_quant_type=config['quantization']['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype=getattr(torch, config['quantization']['bnb_4bit_compute_dtype']),
        bnb_4bit_use_double_quant=config['quantization']['bnb_4bit_use_double_quant']
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['model_name'],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=config['model'].get('cache_dir')
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['model_name'],
        cache_dir=config['model'].get('cache_dir')
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    print(f"✅ Model loaded and quantized")
    
    return model, tokenizer

def setup_lora(model, config):
    """Apply LoRA configuration to model"""
    print("\n🔧 Configuring LoRA adapters...")
    
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        task_type=config['lora']['task_type']
    )
    
    model = get_peft_model(model, lora_config)
    
    print("✅ LoRA configuration applied")
    print("\n📊 Trainable parameters:")
    model.print_trainable_parameters()
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Fine-tune Llama 3 for Smart Secrets Scanner')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Path to training configuration file')
    args = parser.parse_args()
    
    # Record start time
    start_time = datetime.now()
    
    print("=" * 70)
    print("🚀 Smart Secrets Scanner - Fine-Tuning Script")
    print("=" * 70)
    print(f"⏰ Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Authenticate with Hugging Face
    print("\n🔐 Authenticating with Hugging Face...")
    token = load_hf_token()
    if token:
        login(token=token)
        print("✅ Authenticated successfully")
    else:
        print("⚠️  No token found in .env, attempting without authentication...")
    
    # Load configuration
    if not Path(args.config).exists():
        print(f"❌ Error: Config file not found: {args.config}")
        sys.exit(1)
    
    config = load_config(args.config)
    print(f"✅ Loaded configuration from {args.config}")
    
    # Validate data files exist
    if not Path(config['data']['train_file']).exists():
        print(f"❌ Error: Training file not found: {config['data']['train_file']}")
        sys.exit(1)
    if not Path(config['data']['val_file']).exists():
        print(f"❌ Error: Validation file not found: {config['data']['val_file']}")
        sys.exit(1)
    
    # Create output directories
    Path(config['training']['output_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['training']['logging_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['output']['adapter_path']).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory: {config['training']['output_dir']}")
    print(f"📁 Logging directory: {config['training']['logging_dir']}")
    
    # Load datasets
    dataset = load_datasets(config)
    
    # Format datasets with Alpaca prompt template
    print(f"\n📝 Formatting prompts with Alpaca template...")
    prompt_template = config['prompt_template']
    dataset = dataset.map(
        lambda x: format_alpaca_prompt(x, prompt_template),
        remove_columns=dataset['train'].column_names
    )
    
    print(f"✅ Datasets formatted")
    print(f"\nExample formatted prompt:")
    print("-" * 70)
    print(dataset['train'][0]['text'][:500] + "...")
    print("-" * 70)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Apply LoRA
    model = setup_lora(model, config)
    
    # Training arguments
    print("\n⚙️  Configuring training arguments...")
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        max_grad_norm=config['training']['max_grad_norm'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        warmup_ratio=config['training']['warmup_ratio'],
        logging_steps=config['training']['logging_steps'],
        logging_dir=config['training']['logging_dir'],
        logging_strategy=config['training']['logging_strategy'],
        save_steps=config['training']['save_steps'],
        save_strategy=config['training']['save_strategy'],
        save_total_limit=config['training']['save_total_limit'],
        eval_strategy=config['training']['evaluation_strategy'],  # Fixed: eval_strategy instead of evaluation_strategy
        eval_steps=config['training']['eval_steps'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        metric_for_best_model=config['training']['metric_for_best_model'],
        greater_is_better=config['training']['greater_is_better'],
        fp16=config['training']['fp16'],
        bf16=config['training']['bf16'],
        gradient_checkpointing=config['training']['gradient_checkpointing'],
        optim=config['training']['optim'],
        seed=config['training']['seed'],
        report_to=config['training']['report_to'],
        save_safetensors=True,
    )
    
    effective_batch_size = (
        config['training']['per_device_train_batch_size'] * 
        config['training']['gradient_accumulation_steps']
    )
    
    print(f"✅ Training configuration:")
    print(f"   Epochs: {config['training']['num_train_epochs']}")
    print(f"   Batch size: {config['training']['per_device_train_batch_size']} (effective: {effective_batch_size})")
    print(f"   Learning rate: {config['training']['learning_rate']}")
    print(f"   Max sequence length: {config['sft']['max_seq_length']}")
    print(f"   Optimizer: {config['training']['optim']}")
    print(f"   Scheduler: {config['training']['lr_scheduler_type']}")
    
    # Setup SFT trainer
    print("\n🎓 Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        args=training_args,
        processing_class=tokenizer,  # Updated API: tokenizer -> processing_class
    )
    
    print(f"✅ Trainer initialized")
    
    # Train!
    print("\n" + "=" * 70)
    print("🏋️  Starting Training...")
    print("=" * 70)
    print("\nMonitor training with: tensorboard --logdir " + config['training']['logging_dir'])
    print()
    
    trainer.train()
    
    # Save final model
    print("\n" + "=" * 70)
    print("💾 Saving final LoRA adapter...")
    output_path = config['output']['adapter_path']
    trainer.model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Record end time and calculate duration
    end_time = datetime.now()
    duration = end_time - start_time
    hours, remainder = divmod(duration.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    print(f"\n⏰ Start time:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏰ End time:    {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Total time:  {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"\n📁 LoRA adapter saved to: {output_path}")
    print(f"📁 Training checkpoints: {config['training']['output_dir']}")
    print(f"📁 Training logs: {config['training']['logging_dir']}")
    
    print("\n📊 Next steps:")
    print("  1. Review training logs:")
    print(f"     tensorboard --logdir {config['training']['logging_dir']}")
    print("  2. Merge adapter with base model:")
    print("     python scripts/merge_adapter.py")
    print("  3. Convert to GGUF:")
    print("     python scripts/convert_to_gguf.py")
    print("  4. Evaluate model:")
    print("     python scripts/evaluate.py")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
