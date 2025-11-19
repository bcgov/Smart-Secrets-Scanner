#!/usr/bin/env python3
"""
Run inference on code snippets using fine-tuned Smart Secrets Scanner
"""
import argparse
import json
import sys
import torch
import yaml
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("⚠️  llama-cpp-python not available, GGUF model support disabled")

def load_config(config_path="config/inference_config.yaml"):
    """Load inference configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Loaded config from {config_path}")
        return config
    except FileNotFoundError:
        print(f"⚠️  Config file {config_path} not found, using defaults")
        return {}
    except Exception as e:
        print(f"⚠️  Error loading config: {e}, using defaults")
        return {}

def load_model(model_path, device="cuda", load_in_4bit=True):
    """Load the fine-tuned model with optional 4-bit quantization, or GGUF model."""
    
    # Check if this is a GGUF file
    if str(model_path).endswith('.gguf'):
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python is required to load GGUF models. Install with: pip install llama-cpp-python")
        
        print(f"Loading GGUF model from {model_path}...")
        model = Llama(
            model_path=str(model_path),
            n_ctx=2048,  # Context window
            n_threads=4,  # CPU threads
            n_gpu_layers=-1 if device == "cuda" else 0,  # Use GPU if available
            verbose=False
        )
        tokenizer = None  # GGUF models handle tokenization internally
        print("✅ GGUF model loaded successfully")
        return model, tokenizer
    
    # Original Hugging Face model loading logic
    print(f"Loading Hugging Face model from {model_path} on device {device}...")
    
    # Check if this is a LoRA adapter directory
    adapter_config_path = Path(model_path) / "adapter_config.json"
    if adapter_config_path.exists():
        print("Detected LoRA adapter, loading base model + adapter...")
        # Load base model first
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        base_model_path = adapter_config.get('base_model_name_or_path', 'models/base/Meta-Llama-3.1-8B')
        
        if load_in_4bit:
            print("Using 4-bit quantization for faster inference...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        else:
            print("Loading full precision base model...")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                device_map=device,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        
        # Load LoRA adapter
        print(f"Loading LoRA adapter from {model_path}...")
        model = PeftModel.from_pretrained(base_model, model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    else:
        # Load as merged model
        print("Loading merged model...")
        if load_in_4bit:
            print("Using 4-bit quantization for faster inference...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        else:
            print("Loading full precision model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Hugging Face model loaded successfully")
    return model, tokenizer

def format_prompt(code_snippet, system_prompt=None):
    """Format code as Alpaca prompt"""
    if system_prompt:
        instruction = system_prompt
    else:
        instruction = "Analyze the following code snippet and identify any secrets or sensitive credentials that should not be committed to version control."
    
    prompt = f"""### Instruction:
{instruction}

### Input:
{code_snippet}

### Response:
"""
    return prompt

def run_inference(model, tokenizer, code_snippet, max_new_tokens=150, system_prompt=None, config=None, do_sample=False, temperature=0.1, top_p=0.9):
    """Run inference on a single code snippet"""
    prompt = format_prompt(code_snippet, system_prompt)
    
    # Check if this is a GGUF model (no tokenizer)
    if tokenizer is None:
        # GGUF model inference
        print("Running inference with GGUF model...")
        
        # Get generation parameters from config or use defaults
        if config:
            gen_config = config.get('generation', {})
            temperature = temperature if temperature != 0.1 else gen_config.get('temperature', 0.1)
            do_sample = do_sample if do_sample else gen_config.get('do_sample', False)
            top_p = top_p if top_p != 0.9 else gen_config.get('top_p', 0.9)
        else:
            temperature = temperature
            do_sample = do_sample
            top_p = top_p
        
        # GGUF inference
        output = model(
            prompt,
            max_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0.1,
            top_p=top_p if do_sample else 1.0,
            echo=False
        )
        
        response = output['choices'][0]['text'].strip()
        
        # Extract only the response part
        if "### Response:" in response:
            response = response.split("### Response:")[1].strip()
        
        return response
    
    # Original Hugging Face model inference
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Use provided parameters or fall back to config
    if config:
        gen_config = config.get('generation', {})
        temperature = temperature if temperature != 0.1 else gen_config.get('temperature', 0.1)
        do_sample = do_sample if do_sample else gen_config.get('do_sample', False)
        top_p = top_p if top_p != 0.9 else gen_config.get('top_p', 0.9)
        use_cache = gen_config.get('use_cache', True)
        repetition_penalty = gen_config.get('repetition_penalty', 1.1)
        early_stopping = gen_config.get('early_stopping', True)
    else:
        use_cache = True
        repetition_penalty = 1.1
        early_stopping = True
    
    # Generate with optimized parameters for speed
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p if do_sample else 1.0,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=use_cache,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=early_stopping
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
        help='Path to model: LoRA adapter directory, merged model directory, or .gguf file (default: models/fine-tuned/smart-secrets-scanner-lora)'
    )
    parser.add_argument(
        '--device',
        default='cuda',
        help='Device to load model on (default: cuda, use "cpu" for CPU-only)'
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
        '--load-in-4bit',
        action='store_true',
        default=True,
        help='Load model in 4-bit quantization (default: True, faster inference, less memory)'
    )
    parser.add_argument(
        '--do-sample',
        action='store_true',
        default=False,
        help='Enable sampling during generation (default: False, uses greedy decoding)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='Temperature for sampling (default: 0.1, ignored if do_sample=False)'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.9,
        help='Top-p for sampling (default: 0.9, ignored if do_sample=False)'
    )
    
    args = parser.parse_args()
    
    # Load inference config
    config = load_config()
    
    # Get system prompt from config (fallback to training config if needed)
    system_prompt = config.get('system_prompt')
    if not system_prompt:
        try:
            import yaml
            config_path = Path(__file__).parent.parent / "config" / "training_config.yaml"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    training_config = yaml.safe_load(f)
                system_prompt = training_config.get('system_prompt', {}).get('system_prompt')
        except:
            pass
    
    # Load model with config defaults
    model, tokenizer = load_model(
        args.model,
        device=args.device,
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
        
        result = run_inference(model, tokenizer, code, args.max_tokens, system_prompt, config, args.do_sample, args.temperature, args.top_p)
        
        # Highlight alerts
        if "ALERT" in result.upper() or "SECRET" in result.upper() or "CREDENTIAL" in result.upper():
            print("🚨 " + result)
        else:
            print("✅ " + result)
        
        print("-" * 60)
    
    print("\n✅ Inference complete!")

if __name__ == "__main__":
    main()
