#!/usr/bin/env python3
"""
Run inference on code snippets using fine-tuned Smart Secrets Scanner
"""
import argparse
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model(model_path, use_lora=False, base_model_path=None, device="auto", offload_folder="offload", load_in_4bit=False):
    """Load fine-tuned model and tokenizer"""
    print(f"🔽 Loading model from {model_path}...")
    
    if use_lora:
        # Load base model + LoRA adapter
        if not base_model_path:
            raise ValueError("base_model_path required when use_lora=True")
        
        print(f"  Loading base model: {base_model_path}")
        
        if load_in_4bit:
            # Use 4-bit quantization to fit in GPU
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                device_map=device,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                offload_folder=offload_folder,
                offload_state_dict=True
            )
        
        print(f"  Loading LoRA adapter: {model_path}")
        model = PeftModel.from_pretrained(
            base_model, 
            model_path,
            offload_folder=offload_folder if not load_in_4bit else None
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    else:
        # Load merged model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    
    print("✅ Model loaded successfully")
    return model, tokenizer

def format_prompt(code_snippet):
    """Format code as Alpaca prompt"""
    instruction = "Analyze the following code snippet and identify any secrets or sensitive credentials that should not be committed to version control."
    
    prompt = f"""### Instruction:
{instruction}

### Input:
{code_snippet}

### Response:
"""
    return prompt

def run_inference(model, tokenizer, code_snippet, max_new_tokens=150):
    """Run inference on a single code snippet"""
    prompt = format_prompt(code_snippet)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,  # Low temperature for consistent detection
            do_sample=False,  # Greedy decoding for deterministic output
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the response part
    if "### Response:" in response:
        response = response.split("### Response:")[1].strip()
    
    return response

def main():
    parser = argparse.ArgumentParser(description="Run inference with Smart Secrets Scanner")
    parser.add_argument(
        '--model',
        default='models/fine-tuned/smart-secrets-scanner-lora',
        help='Path to fine-tuned model or LoRA adapter (default: models/fine-tuned/smart-secrets-scanner-lora)'
    )
    parser.add_argument(
        '--base-model',
        default='models/base/Meta-Llama-3.1-8B',
        help='Path to base model (required if using LoRA adapter)'
    )
    parser.add_argument(
        '--use-lora',
        action='store_true',
        default=True,
        help='Load LoRA adapter instead of merged model (default: True)'
    )
    parser.add_argument(
        '--input',
        help='Code snippet to analyze (direct input)'
    )
    parser.add_argument(
        '--file',
        help='File containing code to analyze'
    )
    parser.add_argument(
        '--batch',
        help='Directory of files to analyze'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=150,
        help='Maximum tokens to generate (default: 150)'
    )
    parser.add_argument(
        '--offload-folder',
        default='offload',
        help='Folder for offloading model weights (default: offload)'
    )
    parser.add_argument(
        '--load-in-4bit',
        action='store_true',
        help='Load model in 4-bit quantization (faster inference, less memory)'
    )
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(
        args.model,
        use_lora=args.use_lora,
        base_model_path=args.base_model if args.use_lora else None,
        offload_folder=args.offload_folder,
        load_in_4bit=args.load_in_4bit
    )
    
    # Determine input source
    if args.input:
        # Direct input
        code_snippets = [args.input]
        sources = ["<direct input>"]
    elif args.file:
        # Single file
        with open(args.file, 'r', encoding='utf-8') as f:
            code_snippets = [f.read()]
        sources = [args.file]
    elif args.batch:
        # Batch mode - multiple files
        batch_path = Path(args.batch)
        files = list(batch_path.rglob('*.py')) + list(batch_path.rglob('*.js')) + list(batch_path.rglob('*.yaml'))
        code_snippets = []
        sources = []
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                code_snippets.append(f.read())
            sources.append(str(file_path))
    else:
        # Read from stdin
        print("Enter code snippet (Ctrl+D or Ctrl+Z when done):")
        code_snippets = [sys.stdin.read()]
        sources = ["<stdin>"]
    
    # Run inference
    print("\n" + "=" * 60)
    print("🔍 Smart Secrets Scanner - Inference Results")
    print("=" * 60)
    
    for code, source in zip(code_snippets, sources):
        print(f"\n📄 Source: {source}")
        print("-" * 60)
        
        result = run_inference(model, tokenizer, code, args.max_tokens)
        
        # Highlight alerts
        if "ALERT" in result.upper() or "SECRET" in result.upper() or "CREDENTIAL" in result.upper():
            print("🚨 " + result)
        else:
            print("✅ " + result)
        
        print("-" * 60)
    
    print("\n✅ Inference complete!")

if __name__ == "__main__":
    main()
