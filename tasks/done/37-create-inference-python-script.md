# Task 37: Create Inference Python Script

**Status:** Done ✅  
**Priority:** HIGH  
**Created:** 2025-11-01  
**Completed:** 2025-11-01  
**Related to:** Phase 4: Testing & Deployment (Step 11)  
**Depends on:** Task 36 (fine-tuning complete), Task 13 (model merged)

## Prerequisites (Completed)

✅ **Task 01-05**: Environment setup  
✅ **Task 22**: Base model downloaded  
✅ **Task 30**: Training config created (for prompt formatting)  
✅ **Task 36**: Fine-tuning script created (reference implementation)  

## Description
Create `scripts/inference.py` - script for running inference on code snippets using the fine-tuned model.

## Acceptance Criteria
- [ ] `scripts/inference.py` created and executable
- [ ] Can load merged model or LoRA adapter
- [ ] Accepts code input via stdin, file, or command-line arg
- [ ] Formats prompts correctly (Alpaca format)
- [ ] Returns detection results
- [ ] Can be called by `scripts/infer.sh`
- [ ] Supports batch inference on multiple files

## Script Implementation
Create `scripts/inference.py`:

```python
#!/usr/bin/env python3
"""
Run inference on code snippets using fine-tuned Smart Secrets Scanner
"""
import argparse
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def load_model(model_path, device="auto"):
    """Load fine-tuned model and tokenizer"""
    print(f"🔽 Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
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
        default='outputs/merged/smart-secrets-scanner',
        help='Path to fine-tuned model (default: outputs/merged/smart-secrets-scanner)'
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
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model)
    
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
        if "ALERT" in result:
            print("🚨 " + result)
        else:
            print("✅ " + result)
        
        print("-" * 60)
    
    print("\n✅ Inference complete!")

if __name__ == "__main__":
    main()
```

## Usage Examples
```bash
# Direct input
python scripts/inference.py --input 'api_key = "sk-1234567890abcdef"'

# From file
python scripts/inference.py --file examples/test_code.py

# From stdin
echo 'aws_key = "AKIAIOSFODNN7EXAMPLE"' | python scripts/inference.py

# Batch mode (scan directory)
python scripts/inference.py --batch data/raw/python/

# Custom model path
python scripts/inference.py --model models/gguf/smart-secrets-scanner.gguf --file test.py
```

## Dependencies
- Task 13: Merged model exists at `outputs/merged/`
- Task 28: requirements.txt (transformers, torch)

## Testing
```bash
# Test with known secret
python scripts/inference.py --input 'SECRET_KEY = "hardcoded_password_123"'

# Should output: ALERT: Hardcoded secret detected...

# Test with safe code
python scripts/inference.py --input 'api_key = os.getenv("API_KEY")'

# Should output: No secrets detected...
```

## Success Criteria
- Can load and run inference on fine-tuned model
- Correctly identifies secrets in test cases
- Handles multiple input formats (direct, file, batch)
- Output is clear and actionable
- Compatible with existing `scripts/infer.sh`

## Related Files
- Called by: `scripts/infer.sh`
- Reads: `outputs/merged/smart-secrets-scanner/` or `models/fine-tuned/`
- Alternative: Reads GGUF via llama-cpp-python
