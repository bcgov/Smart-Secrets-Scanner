#!/usr/bin/env python3
"""
Convert merged model to GGUF format for Ollama/llama.cpp deployment

This script converts a merged Hugging Face model to GGUF format with multiple
quantization levels for efficient deployment.

Usage:
    python scripts/convert_to_gguf.py
    python scripts/convert_to_gguf.py --quantize Q4_K_M Q8_0
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
        print("  make LLAMA_CUBLAS=1")
        return None

    convert_script = llama_cpp_path / "convert-hf-to-gguf.py"
    if not convert_script.exists():
        print(f"❌ Conversion script not found: {convert_script}")
        print("Please build llama.cpp first: cd ../llama.cpp && make")
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

    if not quantize_bin.exists():
        print(f"❌ Quantize binary not found: {quantize_bin}")
        return False

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
        default='models/merged/smart-secrets-scanner',
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

    args = parser.parse_args()

    # Check prerequisites
    llama_cpp_path = check_llama_cpp()
    if not llama_cpp_path:
        return 1

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return 1

    # Convert to base GGUF
    base_gguf = Path(args.output)
    if not convert_to_gguf(model_path, base_gguf, llama_cpp_path):
        return 1

    # Create quantized versions
    for quant_type in args.quantize:
        quant_output = base_gguf.parent / f"{base_gguf.stem}-{quant_type.lower()}{base_gguf.suffix}"
        if not quantize_gguf(base_gguf, quant_output, quant_type, llama_cpp_path):
            print(f"⚠️  Failed to create {quant_type} quantization")

    print("\n" + "=" * 60)
    print("✅ GGUF Conversion Complete!")
    print("=" * 60)
    print(f"📁 GGUF files saved to: {base_gguf.parent}")
    print("\n🚀 Next steps:")
    print("  1. Create Ollama modelfile: python scripts/create_modelfile.py")
    print("  2. Test with Ollama: ollama create smart-secrets-scanner -f Modelfile")
    print("  3. Run inference: ollama run smart-secrets-scanner")

    return 0

if __name__ == "__main__":
    exit(main())