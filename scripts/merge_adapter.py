"""
Merge LoRA Adapter with Base Model

This script merges a trained LoRA adapter with the base Llama model,
creating a standalone merged model for inference or GGUF conversion.

Usage:
    python scripts/merge_adapter.py
    python scripts/merge_adapter.py --verify
    python scripts/merge_adapter.py --base-model path/to/base --adapter path/to/adapter
"""

import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_lora_adapter(base_model_path, adapter_path, output_path):
    """
    Merge LoRA adapter with base model
    
    Args:
        base_model_path: Path to base model
        adapter_path: Path to LoRA adapter
        output_path: Path to save merged model
    
    Returns:
        merged_model, tokenizer
    """
    print("=" * 60)
    print("🔗 Merging LoRA Adapter with Base Model")
    print("=" * 60)
    print(f"Base model: {base_model_path}")
    print(f"LoRA adapter: {adapter_path}")
    print(f"Output: {output_path}")
    print()
    
    # Load base model
    print(f"🔽 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
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
        default='models/base/Meta-Llama-3.1-8B',
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
