# Task 38: Create Merge Adapter Script

**Status:** Backlog  
**Priority:** HIGH  
**Created:** 2025-11-01  
**Related to:** Phase 3: Model Export (Step 7)  
**Depends on:** Task 36 (fine-tuning complete)

## Description
Create `scripts/merge_adapter.py` - script to merge base model weights with LoRA adapter to create a standalone model.

## Acceptance Criteria
- [ ] `scripts/merge_adapter.py` created and executable
- [ ] Loads base model and LoRA adapter
- [ ] Merges weights correctly
- [ ] Saves merged model to `outputs/merged/`
- [ ] Verifies merged model is functional
- [ ] Cleans up temporary files

## Script Implementation
Create `scripts/merge_adapter.py`:

```python
#!/usr/bin/env python3
"""
Merge base Llama 3 model with LoRA adapter
"""
import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_lora_adapter(base_model_path, adapter_path, output_path):
    """Merge LoRA adapter with base model"""
    print("=" * 60)
    print("🔗 Merging LoRA Adapter with Base Model")
    print("=" * 60)
    
    # Load base model
    print(f"\n🔽 Loading base model from {base_model_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load tokenizer
    print(f"🔽 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Load LoRA adapter
    print(f"\n🔽 Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Merge adapter into base model
    print(f"\n🔗 Merging adapter weights into base model...")
    merged_model = model.merge_and_unload()
    
    # Save merged model
    print(f"\n💾 Saving merged model to {output_path}...")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    
    print("\n" + "=" * 60)
    print("✅ Merge Complete!")
    print("=" * 60)
    print(f"📁 Merged model saved to: {output_path}")
    print("\nModel files:")
    for file in Path(output_path).iterdir():
        print(f"  - {file.name}")
    
    print("\nNext steps:")
    print("  1. Test merged model: python scripts/inference.py --model " + output_path)
    print("  2. Convert to GGUF: python scripts/convert_to_gguf.py")
    
    return merged_model, tokenizer

def verify_merged_model(model, tokenizer):
    """Quick verification that merged model works"""
    print("\n🧪 Verifying merged model...")
    
    test_input = "api_key = 'test123'"
    prompt = f"### Instruction:\nAnalyze this code\n\n### Input:\n{test_input}\n\n### Response:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"✅ Model generates output successfully")
    print(f"   Sample response: {response[:100]}...")

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument(
        '--base-model',
        default='models/base/Meta-Llama-3-8B',
        help='Path to base model'
    )
    parser.add_argument(
        '--adapter',
        default='models/fine-tuned/smart-secrets-scanner-lora',
        help='Path to LoRA adapter'
    )
    parser.add_argument(
        '--output',
        default='outputs/merged/smart-secrets-scanner',
        help='Output path for merged model'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify merged model with test inference'
    )
    
    args = parser.parse_args()
    
    # Merge
    merged_model, tokenizer = merge_lora_adapter(
        args.base_model,
        args.adapter,
        args.output
    )
    
    # Verify if requested
    if args.verify:
        verify_merged_model(merged_model, tokenizer)
    
    print("\n✅ Done!")

if __name__ == "__main__":
    main()
```

## Usage
```bash
# Default paths
python scripts/merge_adapter.py

# Custom paths
python scripts/merge_adapter.py \
  --base-model models/base/Meta-Llama-3-8B \
  --adapter models/fine-tuned/smart-secrets-scanner-lora \
  --output outputs/merged/smart-secrets-scanner

# With verification
python scripts/merge_adapter.py --verify
```

## Dependencies
- Task 36: LoRA adapter must be trained
- Task 22: Base model downloaded
- Task 28: requirements.txt (transformers, peft, torch)

## Output
```
outputs/merged/smart-secrets-scanner/
├── config.json
├── generation_config.json
├── model-00001-of-00004.safetensors
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
├── model-00004-of-00004.safetensors
├── model.safetensors.index.json
├── special_tokens_map.json
├── tokenizer.json
└── tokenizer_config.json
```

## Success Criteria
- Merged model created successfully
- All model shards saved (safetensors format)
- Tokenizer included
- Model can be loaded for inference
- File size ~15-30 GB (full precision merged model)

## Related Tasks
- Task 12: Create merge/GGUF script (this implements merge part)
- Task 13: Merge and export GGUF (this is step 1 of that workflow)
- Task 39: Convert to GGUF (next step after merge)
