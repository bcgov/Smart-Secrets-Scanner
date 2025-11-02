# Task 40: Create Modelfile Generation Script

**Status:** Backlog  
**Priority:** MEDIUM  
**Created:** 2025-11-01  
**Related to:** Phase 4: Testing & Deployment (Step 12)  
**Depends on:** Task 39 (GGUF created)

## Prerequisites (Completed)

✅ **Task 01-05**: Environment setup (WSL2, NVIDIA, ML-Env-CUDA13, dependencies)  

**Pending:**  
⏳ **Task 08**: Fine-tuning Iteration 4 (in-progress)  
⏳ **Task 38**: Merge LoRA adapter  
⏳ **Task 39**: Convert to GGUF format (creates GGUF file needed for Modelfile)  

## Description
Create `scripts/create_modelfile.py` - script to generate an Ollama Modelfile for deploying the fine-tuned model.

## Acceptance Criteria
- [ ] `scripts/create_modelfile.py` created and executable
- [ ] Generates Ollama Modelfile with correct GGUF path
- [ ] Includes appropriate system prompt for secret detection
- [ ] Sets optimal parameters for consistent detection
- [ ] Saves Modelfile to project root
- [ ] Provides instructions for importing to Ollama

## Script Implementation
Create `scripts/create_modelfile.py`:

```python
#!/usr/bin/env python3
"""
Generate Ollama Modelfile for Smart Secrets Scanner deployment
"""
import argparse
from pathlib import Path

MODELFILE_TEMPLATE = """# Modelfile for Smart Secrets Scanner
# Fine-tuned Llama 3 for detecting secrets in code

FROM {gguf_path}

# System prompt for secret detection
SYSTEM \"\"\"
You are a security-focused code analyzer specialized in detecting hardcoded secrets, API keys, passwords, and sensitive credentials in source code.

Your task is to analyze code snippets and identify:
- API keys and tokens (AWS, GitHub, Stripe, etc.)
- Database credentials and connection strings
- Private keys and certificates
- OAuth tokens and JWT secrets
- Hardcoded passwords
- Any sensitive data that should not be committed to version control

For each code snippet, respond with either:
1. "ALERT: [Description of the secret found]" if you detect a secret
2. "No secrets detected. This code uses secure practices." if the code is safe

Be thorough but avoid false positives. Environment variable usage (os.getenv, process.env) is SAFE. Placeholder values like "YOUR_API_KEY_HERE" are SAFE documentation examples.
\"\"\"

# Parameters optimized for consistent secret detection
PARAMETER temperature 0.1        # Low temperature for deterministic output
PARAMETER top_p 0.9              # Focused sampling
PARAMETER top_k 40               # Limit token choices
PARAMETER repeat_penalty 1.1     # Reduce repetition
PARAMETER num_ctx 2048           # Context window size
PARAMETER stop "### Instruction:"  # Stop at next instruction
PARAMETER stop "### Input:"        # Stop at next input

# Template for instruction format
TEMPLATE \"\"\"
### Instruction:
{{ .System }}

### Input:
{{ .Prompt }}

### Response:
\"\"\"
"""

def create_modelfile(gguf_path, output_path="Modelfile", quantization="q4_k_m"):
    """Generate Ollama Modelfile"""
    print("=" * 60)
    print("📝 Creating Ollama Modelfile")
    print("=" * 60)
    
    # Resolve GGUF path
    gguf_file = Path(gguf_path)
    if not gguf_file.exists():
        print(f"⚠️  Warning: GGUF file not found: {gguf_path}")
        print("   Modelfile will be created but won't work until GGUF exists")
    
    # Use absolute path in Modelfile
    gguf_absolute = gguf_file.resolve()
    
    # Generate Modelfile content
    modelfile_content = MODELFILE_TEMPLATE.format(gguf_path=str(gguf_absolute))
    
    # Write Modelfile
    with open(output_path, 'w') as f:
        f.write(modelfile_content)
    
    print(f"\n✅ Modelfile created: {output_path}")
    print(f"   GGUF path: {gguf_absolute}")
    print(f"   Quantization: {quantization.upper()}")
    
    # Instructions
    print("\n" + "=" * 60)
    print("📦 Deployment Instructions")
    print("=" * 60)
    print("\n1. Import model to Ollama:")
    print(f"   ollama create smart-secrets-scanner -f {output_path}")
    print("\n2. Test the model:")
    print('   ollama run smart-secrets-scanner "Analyze: api_key = \'sk-12345\'"')
    print("\n3. Use in pre-commit hooks:")
    print("   # See scripts/scan_secrets.py for integration")
    print("\n4. API usage:")
    print("   curl http://localhost:11434/api/generate -d '{")
    print('     "model": "smart-secrets-scanner",')
    print('     "prompt": "aws_key = \\"AKIAIOSFODNN7EXAMPLE\\""')
    print("   }'")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Generate Ollama Modelfile")
    parser.add_argument(
        '--gguf',
        default='models/gguf/smart-secrets-scanner-q4_k_m.gguf',
        help='Path to GGUF model file (default: Q4_K_M quantization)'
    )
    parser.add_argument(
        '--output',
        default='Modelfile',
        help='Output Modelfile path (default: Modelfile)'
    )
    parser.add_argument(
        '--quantization',
        default='q4_k_m',
        choices=['f16', 'q8_0', 'q5_k_m', 'q4_k_m', 'q4_k_s'],
        help='Quantization level indicator (default: q4_k_m)'
    )
    
    args = parser.parse_args()
    
    create_modelfile(args.gguf, args.output, args.quantization)

if __name__ == "__main__":
    main()
```

## Usage
```bash
# Default (Q4_K_M quantization)
python scripts/create_modelfile.py

# Use Q8_0 quantization (higher quality)
python scripts/create_modelfile.py --gguf models/gguf/smart-secrets-scanner-q8_0.gguf --quantization q8_0

# Custom output path
python scripts/create_modelfile.py --output deployment/Modelfile
```

## Generated Modelfile Example
```dockerfile
# Modelfile for Smart Secrets Scanner
FROM /absolute/path/to/models/gguf/smart-secrets-scanner-q4_k_m.gguf

SYSTEM """
You are a security-focused code analyzer...
"""

PARAMETER temperature 0.1
PARAMETER top_p 0.9
...

TEMPLATE """
### Instruction:
{{ .System }}
...
"""
```

## Testing Deployed Model
```bash
# Import to Ollama
ollama create smart-secrets-scanner -f Modelfile

# Test with secret
ollama run smart-secrets-scanner "Analyze this code: stripe_key = 'sk_live_abc123'"

# Expected output: ALERT: Stripe API key detected...

# Test with safe code
ollama run smart-secrets-scanner "Analyze: api_key = os.getenv('API_KEY')"

# Expected output: No secrets detected...
```

## Dependencies
- Task 39: GGUF file must exist
- Ollama installed (`curl -fsSL https://ollama.com/install.sh | sh`)

## Success Criteria
- Modelfile generated with correct GGUF path
- System prompt optimized for secret detection
- Parameters tuned for consistent output
- Can be imported to Ollama successfully
- Model responds correctly to test queries

## Related Tasks
- Task 14: Create Modelfile (this implements that task)
- Task 15: Test Ollama deployment (uses this Modelfile)
- Task 27: Pre-commit hook integration (uses deployed model)
- Task 41: Scan secrets script (uses Ollama API)
