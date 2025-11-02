# Task 39: Create GGUF Conversion Script

**Status:** Backlog  
**Priority:** HIGH  
**Created:** 2025-11-01  
**Related to:** Phase 3: Model Export (Step 8)  
**Depends on:** Task 38 (merged model created)

## Prerequisites (Completed)

✅ **Task 01-05**: Environment setup (WSL2, NVIDIA, ML-Env-CUDA13, dependencies)  
✅ **Task 22**: Base model downloaded  

**Pending:**  
⏳ **Task 08**: Fine-tuning Iteration 4 (in-progress)  
⏳ **Task 38**: Merge LoRA adapter (creates merged model needed for GGUF)  

## Description
Create `scripts/convert_to_gguf.py` - script to convert merged Hugging Face model to GGUF format for deployment with Ollama/llama.cpp.

## Acceptance Criteria
- [ ] `scripts/convert_to_gguf.py` created and executable
- [ ] Converts merged model to GGUF format
- [ ] Supports multiple quantization levels (Q4_K_M, Q8_0, F16)
- [ ] Saves GGUF files to `models/gguf/`
- [ ] Validates conversion success
- [ ] Reports file sizes and compression ratios

## Script Implementation
Create `scripts/convert_to_gguf.py`:

```python
#!/usr/bin/env python3
"""
Convert merged model to GGUF format for Ollama/llama.cpp deployment
"""
import argparse
import subprocess
import shutil
from pathlib import Path

def check_llama_cpp():
    """Check if llama.cpp is available"""
    llama_cpp_path = Path("../llama.cpp")
    
    if not llama_cpp_path.exists():
        print("❌ llama.cpp not found!")
        print("\nPlease clone and build llama.cpp:")
        print("  cd ..")
        print("  git clone https://github.com/ggerganov/llama.cpp")
        print("  cd llama.cpp")
        print("  make")
        return None
    
    convert_script = llama_cpp_path / "convert-hf-to-gguf.py"
    if not convert_script.exists():
        print(f"❌ Conversion script not found: {convert_script}")
        return None
    
    return llama_cpp_path

def convert_to_gguf(model_path, output_path, llama_cpp_path):
    """Convert HuggingFace model to GGUF format"""
    print("=" * 60)
    print("📦 Converting Model to GGUF Format")
    print("=" * 60)
    
    convert_script = llama_cpp_path / "convert-hf-to-gguf.py"
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        "python",
        str(convert_script),
        str(model_path),
        "--outfile", str(output_path),
        "--outtype", "f16"  # Start with FP16, then quantize
    ]
    
    print(f"\n🔄 Running conversion...")
    print(f"   Command: {' '.join(cmd)}\n")
    
    # Run conversion
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ Conversion failed!")
        print("STDERR:", result.stderr)
        return False
    
    print(result.stdout)
    print(f"\n✅ Converted to GGUF (F16): {output_path}")
    
    return True

def quantize_gguf(input_gguf, output_gguf, quant_type, llama_cpp_path):
    """Quantize GGUF model to specified quantization level"""
    print(f"\n🔧 Quantizing to {quant_type}...")
    
    quantize_bin = llama_cpp_path / "quantize"
    if not quantize_bin.exists():
        quantize_bin = llama_cpp_path / "quantize.exe"  # Windows
    
    cmd = [
        str(quantize_bin),
        str(input_gguf),
        str(output_gguf),
        quant_type
    ]
    
    print(f"   Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Quantization to {quant_type} failed!")
        print("STDERR:", result.stderr)
        return False
    
    print(result.stdout)
    
    # Report file size
    if Path(output_gguf).exists():
        size_gb = Path(output_gguf).stat().st_size / (1024**3)
        print(f"✅ Quantized to {quant_type}: {output_gguf}")
        print(f"   File size: {size_gb:.2f} GB")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Convert model to GGUF format")
    parser.add_argument(
        '--model',
        default='outputs/merged/smart-secrets-scanner',
        help='Path to merged HuggingFace model'
    )
    parser.add_argument(
        '--output',
        default='models/gguf/smart-secrets-scanner.gguf',
        help='Output GGUF file path'
    )
    parser.add_argument(
        '--quantize',
        nargs='+',
        default=['Q4_K_M', 'Q8_0'],
        help='Quantization types to create (default: Q4_K_M Q8_0)'
    )
    parser.add_argument(
        '--llama-cpp-path',
        default='../llama.cpp',
        help='Path to llama.cpp directory'
    )
    
    args = parser.parse_args()
    
    # Check llama.cpp
    llama_cpp_path = Path(args.llama_cpp_path)
    if not llama_cpp_path.exists():
        llama_cpp_path = check_llama_cpp()
        if llama_cpp_path is None:
            return 1
    
    # Convert to GGUF (F16 first)
    base_output = Path(args.output)
    f16_output = base_output.with_stem(base_output.stem + "-f16")
    
    success = convert_to_gguf(args.model, f16_output, llama_cpp_path)
    if not success:
        return 1
    
    # Quantize to requested formats
    print("\n" + "=" * 60)
    print("🔧 Creating Quantized Versions")
    print("=" * 60)
    
    for quant_type in args.quantize:
        quant_output = base_output.with_stem(f"{base_output.stem}-{quant_type.lower()}")
        quantize_gguf(f16_output, quant_output, quant_type, llama_cpp_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ GGUF Conversion Complete!")
    print("=" * 60)
    print("\nCreated files:")
    
    gguf_dir = Path("models/gguf")
    if gguf_dir.exists():
        for gguf_file in sorted(gguf_dir.glob("smart-secrets-scanner*.gguf")):
            size_gb = gguf_file.stat().st_size / (1024**3)
            print(f"  📦 {gguf_file.name} ({size_gb:.2f} GB)")
    
    print("\nNext steps:")
    print("  1. Test GGUF: llama.cpp/main -m models/gguf/smart-secrets-scanner-q4_k_m.gguf -p 'Test prompt'")
    print("  2. Create Modelfile: python scripts/create_modelfile.py")
    print("  3. Import to Ollama: ollama create smart-secrets-scanner -f Modelfile")

if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
```

## Prerequisites
Clone and build llama.cpp:
```bash
cd ..
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make  # or: cmake -B build && cmake --build build
```

## Usage
```bash
# Default: Create Q4_K_M and Q8_0 quantizations
python scripts/convert_to_gguf.py

# Custom quantizations
python scripts/convert_to_gguf.py --quantize Q4_K_M Q5_K_M Q8_0

# Custom paths
python scripts/convert_to_gguf.py \
  --model outputs/merged/smart-secrets-scanner \
  --output models/gguf/smart-secrets-scanner.gguf \
  --llama-cpp-path ../llama.cpp
```

## Quantization Types
- **F16**: Full 16-bit precision (~15 GB) - base for quantization
- **Q8_0**: 8-bit quantization (~8 GB) - high quality
- **Q5_K_M**: 5-bit medium (~5 GB) - balanced
- **Q4_K_M**: 4-bit medium (~4 GB) - **recommended for pre-commit scanning**
- **Q4_K_S**: 4-bit small (~3.5 GB) - faster but lower quality

## Dependencies
- Task 38: Merged model must exist
- llama.cpp cloned and built
- Python 3.10+

## Output
```
models/gguf/
├── smart-secrets-scanner-f16.gguf      (~15 GB)
├── smart-secrets-scanner-q4_k_m.gguf  (~4 GB)
└── smart-secrets-scanner-q8_0.gguf    (~8 GB)
```

## Success Criteria
- GGUF files created successfully
- File sizes match expected quantization levels
- Models can be loaded by llama.cpp/Ollama
- Q4_K_M model suitable for fast inference

## Related Tasks
- Task 12: Create merge/GGUF script (this implements GGUF part)
- Task 13: Merge and export GGUF (this is step 2 of that workflow)
- Task 24: Quantize GGUF (this implements that task)
- Task 40: Create Modelfile script (next step)
