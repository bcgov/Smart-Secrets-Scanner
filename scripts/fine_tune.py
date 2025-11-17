#!/usr/bin/env python3
"""
Fine-tune Llama 3 8B for Smart Secrets Scanner using LoRA/QLoRA

Optimized version with improved logging, error handling, and performance features.

Usage:
    python scripts/fine_tune.py
    python scripts/fine_tune.py --config config/custom_config.yaml
"""
import os
import sys
import json
import argparse
import yaml
import torch
import logging
import psutil
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from huggingface_hub import login

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("smart_secrets_scanner.fine_tune")

# --- Determine Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "training_config.yaml"

# Load Hugging Face token from .env file
def load_hf_token():
    """Load Hugging Face token from .env file"""
    env_path = PROJECT_ROOT / '.env'
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if line.startswith('HUGGING_FACE_TOKEN='):
                    token = line.split('=', 1)[1].strip().strip('"').strip("'")
                    return token
    return None

def get_torch_dtype(kind: str):
    """Safely map string to torch dtype."""
    kind = kind.lower()
    if kind in ("float16", "fp16"):
        return torch.float16
    if kind in ("float32", "fp32"):
        return torch.float32
    if kind in ("bfloat16", "bf16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype '{kind}' for bitsandbytes compute dtype")

def ensure_train_val_files(train_file_path, val_file_path=None):
    """Ensure train and validation files exist, create validation split if needed."""
    if val_file_path is None or not val_file_path:
        logger.info("No val_file provided; skipping split.")
        return train_file_path, None

    if val_file_path.exists():
        logger.info("Found existing val_file: %s", val_file_path)
        return train_file_path, val_file_path

    # Only split if val_file_path is explicitly requested but missing
    logger.info("Validation file not found. Creating split (train/val = 90/10)")
    with open(train_file_path, 'r', encoding='utf-8') as f:
        lines = [line for line in f if line.strip()]
    import random
    random.seed(42)
    random.shuffle(lines)
    split_idx = int(0.9 * len(lines))
    new_train = train_file_path.with_suffix('.train.jsonl')
    new_val = val_file_path
    # write out new files (don't overwrite original train file)
    with open(new_train, 'w', encoding='utf-8') as f:
        f.writelines(lines[:split_idx])
    with open(new_val, 'w', encoding='utf-8') as f:
        f.writelines(lines[split_idx:])
    logger.info("Split complete. Train: %d examples, Val: %d examples.", split_idx, len(lines) - split_idx)
    return new_train, new_val

def tokenize_and_cache(dataset, tokenizer, max_length, cache_path=None):
    """Tokenize dataset and optionally cache to disk."""
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    if cache_path:
        tokenized.save_to_disk(str(cache_path))
        logger.info("Tokenized dataset cached to: %s", cache_path)
    return tokenized

def load_config(config_path=DEFAULT_CONFIG_PATH):
    """Loads the training configuration from a YAML file with validation."""
    logger.info("🔩 Loading configuration from: %s", config_path)
    if not config_path.exists():
        logger.error("Configuration file not found: %s", config_path)
        sys.exit(1)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set defaults
    config.setdefault('max_seq_length', 256)
    config.setdefault('use_bf16', False)
    if 'training' not in config:
        logger.error("Missing 'training' section in config")
        sys.exit(1)

    # Convert and validate training parameters
    training = config['training']
    try:
        training['learning_rate'] = float(training.get('learning_rate', 2e-4))
        training['warmup_ratio'] = float(training.get('warmup_ratio', 0.03))
        training['max_grad_norm'] = float(training.get('max_grad_norm', 0.3))
        training['num_train_epochs'] = int(training.get('num_train_epochs', 3))
        training['per_device_train_batch_size'] = int(training.get('per_device_train_batch_size', 1))
        training['gradient_accumulation_steps'] = int(training.get('gradient_accumulation_steps', 8))
        training['logging_steps'] = int(training.get('logging_steps', 20))
    except Exception as e:
        logger.exception("Invalid training config: %s", e)
        sys.exit(1)

    logger.info("✅ Configuration loaded successfully.")
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

def get_torch_dtype(dtype_str):
    """Convert string dtype to torch dtype."""
    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
        'auto': 'auto'
    }
    return dtype_map.get(dtype_str, torch.float16)


def main():
    """Main function to execute the fine-tuning process."""
    parser = argparse.ArgumentParser(description='Fine-tune Llama 3 for Smart Secrets Scanner')
    parser.add_argument('--config', type=str, default=str(DEFAULT_CONFIG_PATH),
                       help='Path to training configuration file')
    args = parser.parse_args()

    # Record start time
    start_time = datetime.now()

    logger.info("=" * 70)
    logger.info("🚀 Smart Secrets Scanner - Fine-Tuning Script (Optimized v2.0)")
    logger.info("=" * 70)
    logger.info(f"⏰ Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")

    # System diagnostics
    logger.info("🔍 System Diagnostics:")
    logger.info("   CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("   GPU count: %d", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info("   GPU %d: %s (%d MB)", i, torch.cuda.get_device_name(i), props.total_memory // 1024**2)
    logger.info("   CPU cores: %d logical, %.1f%% used", psutil.cpu_count(logical=True), psutil.cpu_percent(interval=0.5))
    logger.info("")

    # Authenticate with Hugging Face
    logger.info("🔐 Authenticating with Hugging Face...")
    token = load_hf_token()
    if token:
        login(token=token)
        logger.info("✅ Authenticated successfully")
    else:
        logger.info("⚠️  No token found in .env, attempting without authentication...")

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("❌ Error: Config file not found: %s", config_path)
        sys.exit(1)

    config = load_config(config_path)
    logger.info("✅ Loaded configuration from %s", config_path)

    # Validate data files exist
    train_file_path = PROJECT_ROOT / config['data']['train_file']
    val_file_path = PROJECT_ROOT / config['data']['val_file'] if config['data'].get('val_file') else None

    if not train_file_path.exists():
        logger.error("❌ Error: Training file not found: %s", train_file_path)
        sys.exit(1)

    # Ensure train/val files exist
    train_file_path, val_file_path = ensure_train_val_files(train_file_path, val_file_path)

    # Create output directories
    output_dir = PROJECT_ROOT / config['training']['output_dir']
    logging_dir = PROJECT_ROOT / config['training']['logging_dir']
    adapter_path = PROJECT_ROOT / config['output']['adapter_path']

    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)
    adapter_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("📁 Output directory: %s", output_dir)
    logger.info("📁 Logging directory: %s", logging_dir)
    logger.info("")

    # Load and format dataset
    logger.info("📊 Loading and formatting dataset...")
    dataset_files = {'train': str(train_file_path)}
    if val_file_path:
        dataset_files['validation'] = str(val_file_path)

    dataset = load_dataset('json', data_files=dataset_files)

    logger.info("✅ Loaded %d training examples", len(dataset['train']))
    if 'validation' in dataset:
        logger.info("✅ Loaded %d validation examples", len(dataset['validation']))

    # Format datasets with Alpaca prompt template
    logger.info("📝 Formatting prompts with Alpaca template...")
    prompt_template = config['prompt_template']
    dataset = dataset.map(
        lambda x: format_alpaca_prompt(x, prompt_template),
        remove_columns=dataset['train'].column_names
    )

    logger.info("✅ Datasets formatted")
    logger.info("")
    logger.info("Example formatted prompt:")
    logger.info("-" * 70)
    example_text = dataset['train'][0]['text'][:500] + "..."
    logger.info(example_text)
    logger.info("-" * 70)
    logger.info("")

    # Setup model and tokenizer with improved quantization
    logger.info("🔽 Loading base model: %s", config['model']['model_name'])
    logger.info("   Using 4-bit quantization for efficient training")

    # 4-bit quantization config with improved dtype handling
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config['quantization']['load_in_4bit'],
        bnb_4bit_quant_type=config['quantization']['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype=get_torch_dtype(config['quantization']['bnb_4bit_compute_dtype']),
        bnb_4bit_use_double_quant=config['quantization']['bnb_4bit_use_double_quant']
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['model_name'],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=config['model'].get('cache_dir'),
        torch_dtype=get_torch_dtype(config.get('torch_dtype', 'auto')),
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

    logger.info("✅ Model loaded and quantized")

    # Apply LoRA with improved configuration
    logger.info("🔧 Configuring LoRA adapters...")

    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        task_type=config['lora']['task_type']
    )

    model = get_peft_model(model, lora_config)

    logger.info("✅ LoRA configuration applied")
    logger.info("")
    logger.info("📊 Trainable parameters:")
    model.print_trainable_parameters()
    logger.info("")

    # Training arguments with all optimized parameters
    logger.info("⚙️  Configuring training arguments...")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
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
        logging_dir=str(logging_dir),
        logging_strategy=config['training']['logging_strategy'],
        save_steps=config['training']['save_steps'],
        save_strategy=config['training']['save_strategy'],
        save_total_limit=config['training']['save_total_limit'],
        eval_strategy=config['training']['evaluation_strategy'],
        eval_steps=config['training']['eval_steps'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        metric_for_best_model=config['training']['metric_for_best_model'],
        greater_is_better=config['training']['greater_is_better'],
        fp16=config['training']['fp16'],
        bf16=config['training']['bf16'] or config.get('use_bf16', False),
        gradient_checkpointing=config['training']['gradient_checkpointing'],
        optim=config['training']['optim'],
        seed=config['training']['seed'],
        report_to=config['training']['report_to'],
        # Add optimized dataloader parameters
        dataloader_num_workers=config['training'].get('dataloader_num_workers', 0),
        dataloader_pin_memory=config['training'].get('dataloader_pin_memory', False),
        dataloader_persistent_workers=config['training'].get('dataloader_persistent_workers', False),
        group_by_length=config['training'].get('group_by_length', True),
        save_safetensors=True,
    )

    effective_batch_size = (
        config['training']['per_device_train_batch_size'] *
        config['training']['gradient_accumulation_steps']
    )

    logger.info("✅ Training configuration:")
    logger.info("   Epochs: %d", config['training']['num_train_epochs'])
    logger.info("   Batch size: %d (effective: %d)", config['training']['per_device_train_batch_size'], effective_batch_size)
    logger.info("   Learning rate: %s", config['training']['learning_rate'])
    logger.info("   Max sequence length: %d", config['data']['max_seq_length'])
    logger.info("   Optimizer: %s", config['training']['optim'])
    logger.info("   Scheduler: %s", config['training']['lr_scheduler_type'])
    logger.info("")

    # Check for resume capability
    last_checkpoint = None
    if output_dir.exists():
        checkpoints = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint")])
        if checkpoints:
            last_checkpoint = str(checkpoints[-1])
            logger.info("📁 Found checkpoint to resume from: %s", last_checkpoint)

    # Setup SFT trainer with improved data handling
    logger.info("🎓 Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'] if 'validation' in dataset else None,
        args=training_args,
        processing_class=tokenizer,
        max_seq_length=config['data']['max_seq_length'],
        packing=config.get('sft', {}).get('packing', False),
        dataset_text_field=config.get('sft', {}).get('dataset_text_field', 'text'),
    )

    logger.info("✅ Trainer initialized")

    # Train!
    logger.info("")
    logger.info("=" * 70)
    logger.info("🏋️  Starting Training...")
    logger.info("=" * 70)
    logger.info("")
    if last_checkpoint:
        logger.info("Resuming from checkpoint: %s", last_checkpoint)
    logger.info("Monitor training with: tensorboard --logdir %s", logging_dir)
    logger.info("")

    try:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    except Exception as e:
        logger.exception("Training failed with exception: %s", e)
        # Try to save whatever we have
        try:
            logger.info("Attempting best-effort save of current adapter to: %s", adapter_path)
            trainer.model.save_pretrained(str(adapter_path))
        except Exception as e2:
            logger.exception("Failed to save adapter: %s", e2)
        raise  # re-raise so caller knows training failed

    # Save final model
    logger.info("")
    logger.info("=" * 70)
    logger.info("💾 Saving final LoRA adapter...")
    trainer.model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    # Record end time and calculate duration
    end_time = datetime.now()
    duration = end_time - start_time
    hours, remainder = divmod(duration.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)

    logger.info("")
    logger.info("=" * 70)
    logger.info("✅ Training Complete!")
    logger.info("=" * 70)
    logger.info("⏰ Start time:  %s", start_time.strftime('%Y-%m-%d %H:%M:%S'))
    logger.info("⏰ End time:    %s", end_time.strftime('%Y-%m-%d %H:%M:%S'))
    logger.info("⏱️  Total time:  %dh %dm %ds", int(hours), int(minutes), int(seconds))
    logger.info("")
    logger.info("📁 LoRA adapter saved to: %s", adapter_path)
    logger.info("📁 Training checkpoints: %s", output_dir)
    logger.info("📁 Training logs: %s", logging_dir)
    logger.info("")
    logger.info("📊 Next steps:")
    logger.info("  1. Review training logs:")
    logger.info("     tensorboard --logdir %s", logging_dir)
    logger.info("  2. Merge adapter with base model:")
    logger.info("     python scripts/merge_adapter.py")
    logger.info("  3. Convert to GGUF:")
    logger.info("     python scripts/convert_to_gguf.py")
    logger.info("  4. Evaluate model:")
    logger.info("     python scripts/evaluate.py")

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.warning("⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("❌ Error during training: %s", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)
