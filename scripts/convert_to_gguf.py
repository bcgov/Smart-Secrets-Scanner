#!/usr/bin/env python3
"""
Convert merged model to GGUF format for Ollama/llama.cpp deployment

Enhanced version with config-driven approach, robust error handling,
GGUF verification, and automatic cleanup.

Usage:
    python scripts/convert_to_gguf.py
    python scripts/convert_to_gguf.py --quantize Q4_K_M Q8_0
    python scripts/convert_to_gguf.py --config config/gguf_config.yaml
"""

import argparse
import logging
import subprocess
import shutil
import sys
from pathlib import Path

import yaml

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "gguf_config.yaml"


# --------------------------------------------------------------------------- #
# Config Loader
# --------------------------------------------------------------------------- #
def load_config(config_path: Path):
    """Load GGUF conversion configuration from YAML file"""
    log.info(f"Loading GGUF config from {config_path}")
    if not config_path.exists():
        log.warning(f"Config not found: {config_path}")
        log.info("Using default configuration...")
        # Return default config if file doesn't exist
        return {
            "model": {
                "merged_path": "models/merged/smart-secrets-scanner",
                "gguf_output_dir": "models/fine-tuned/gguf",
                "gguf_model_name": "smart-secrets-scanner"
            },
            "quantization": {
                "default_types": ["Q4_K_M"],
                "use_cuda": True
            }
        }

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


# --------------------------------------------------------------------------- #
# Run CLI with error capture
# --------------------------------------------------------------------------- #
def run_command(cmd: list, desc: str):
    """Run a command with proper error handling and logging"""
    log.info(f"{desc}: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"{desc} failed:\n{result.stderr}")
        return None
    else:
        log.info(f"{desc} completed.")
    return result.stdout


# --------------------------------------------------------------------------- #
# Verify GGUF file
# --------------------------------------------------------------------------- #
def verify_gguf(file_path: Path):
    """Verify GGUF file integrity"""
    try:
        import gguf  # pip install gguf
        reader = gguf.GGUFReader(str(file_path))
        log.info(f"✅ GGUF valid: {file_path.name} | tensors: {len(reader.tensors)} | metadata: {len(reader.metadata)}")
        return True
    except ImportError:
        log.warning("GGUF verification library not available (pip install gguf)")
        # Fallback: check file exists and has reasonable size
        if file_path.exists() and file_path.stat().st_size > 1000000:  # > 1MB
            log.info(f"✅ GGUF file exists and has reasonable size: {file_path}")
            return True
        else:
            log.warning("GGUF file verification failed - file missing or too small")
            return False
    except Exception as e:
        log.warning(f"GGUF verification failed: {e}")
        return False


# --------------------------------------------------------------------------- #
# Check llama.cpp availability
# --------------------------------------------------------------------------- #
def check_llama_cpp_tools():
    """Find llama.cpp conversion and quantization tools"""
    try:
        convert_script = shutil.which("convert-hf-to-gguf.py")
        quantize_script = shutil.which("llama-quantize")
        if not convert_script or not quantize_script:
            raise FileNotFoundError
        return convert_script, quantize_script
    except:
        log.error("llama.cpp CLI tools not found in PATH.")
        log.info("Install with: pip install 'llama-cpp-python[cli]'")
        log.info("Or build from: https://github.com/ggerganov/llama.cpp")
        return None, None

def convert_to_gguf(model_path: Path, output_path: Path, convert_script: str, model_name: str, use_cuda: bool = True):
    """Convert HuggingFace model to GGUF format (F16)"""
    log.info("📦 Converting Model to GGUF Format")
    log.info(f"   Model: {model_path}")
    log.info(f"   Output: {output_path}")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        "python", convert_script,
        str(model_path),
        "--outfile", str(output_path),
        "--model-name", model_name,
    ]

    if use_cuda:
        cmd.extend(["--use-cuda"])
        log.info("   Using CUDA acceleration")
    else:
        log.info("   Using CPU-only conversion")

    # Run conversion
    result = run_command(cmd, "[1/3] HF → GGUF (f16)")
    if result is None:
        return False

    if output_path.exists():
        size_gb = output_path.stat().st_size / (1024**3)
        log.info(f"✅ Converted to GGUF (F16): {output_path} ({size_gb:.2f} GB)")
    else:
        log.error(f"❌ Output file not created: {output_path}")
        return False

    return True


def quantize_gguf(input_gguf: Path, output_gguf: Path, quant_type: str, quantize_script: str):
    """Quantize GGUF model to specified quantization level"""
    log.info(f"🔧 Quantizing to {quant_type}")
    log.info(f"   Input: {input_gguf}")
    log.info(f"   Output: {output_gguf}")

    cmd = [
        quantize_script,
        str(input_gguf),
        str(output_gguf),
        quant_type
    ]

    result = run_command(f"[2/3] Quantize → {quant_type}", cmd)
    if result is None:
        return False

    # Report file size
    if output_gguf.exists():
        size_gb = output_gguf.stat().st_size / (1024**3)
        log.info(f"✅ Quantized to {quant_type}: {output_gguf} ({size_gb:.2f} GB)")
        return True
    else:
        log.error(f"❌ Quantized file not created: {output_gguf}")
        return False

def main():
    """Main function for GGUF conversion and quantization"""
    parser = argparse.ArgumentParser(description="Convert merged HF model to GGUF + quantize")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                       help="Path to GGUF config YAML")
    parser.add_argument("--merged", type=str,
                       help="Override merged model directory")
    parser.add_argument("--output-dir", type=str,
                       help="Override GGUF output directory")
    parser.add_argument("--quantize", nargs='+',
                       help="Quantization types to create")
    parser.add_argument("--force", action="store_true",
                       help="Overwrite existing files")
    parser.add_argument("--no-cuda", action="store_true",
                       help="Disable CUDA (CPU only)")
    parser.add_argument("--no-verify", action="store_true",
                       help="Skip GGUF verification")
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)

    # Resolve paths
    merged_dir = PROJECT_ROOT / (args.merged or cfg["model"]["merged_path"])
    output_dir = PROJECT_ROOT / (args.output_dir or cfg["model"]["gguf_output_dir"])
    model_name = cfg["model"]["gguf_model_name"]

    # Get quantization types
    quant_types = args.quantize or cfg["quantization"]["default_types"]
    use_cuda = not args.no_cuda and cfg["quantization"].get("use_cuda", True)

    # Define output paths
    f16_gguf = output_dir / f"{model_name}.gguf"
    final_ggufs = [output_dir / f"{model_name}-{quant_type}.gguf" for quant_type in quant_types]

    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== GGUF Conversion & Quantization ===")
    log.info(f"Merged model: {merged_dir}")
    log.info(f"Output dir: {output_dir}")
    log.info(f"Model name: {model_name}")
    log.info(f"Quantizations: {', '.join(quant_types)}")
    log.info(f"CUDA: {'enabled' if use_cuda else 'disabled'}")

    # --- Validation ---
    if not merged_dir.exists():
        log.error(f"Merged model not found: {merged_dir}")
        log.info("Run merge_adapter.py first.")
        return 1

    # --- Check for existing files ---
    existing_files = [f for f in [f16_gguf] + final_ggufs if f.exists()]
    if existing_files and not args.force:
        log.error("Existing files found:")
        for f in existing_files:
            log.error(f"  - {f}")
        log.info("Use --force to overwrite.")
        return 1

    # --- Find llama.cpp tools ---
    convert_script, quantize_script = check_llama_cpp_tools()
    if not convert_script or not quantize_script:
        return 1

    # --- Step 1: Convert HF → GGUF (f16) ---
    if not convert_to_gguf(merged_dir, f16_gguf, convert_script, model_name, use_cuda):
        return 1

    # --- Step 2: Quantize ---
    success_count = 0
    for quant_type in quant_types:
        final_gguf = output_dir / f"{model_name}-{quant_type}.gguf"
        if quantize_gguf(f16_gguf, final_gguf, quant_type, quantize_script):
            success_count += 1
        else:
            log.warning(f"Failed to create {quant_type} quantization")

    # --- Step 3: Verify ---
    if not args.no_verify:
        log.info("[3/3] Verifying final GGUF files...")
        for quant_type in quant_types:
            final_gguf = output_dir / f"{model_name}-{quant_type}.gguf"
            if final_gguf.exists():
                verify_gguf(final_gguf)

    # --- Cleanup intermediate ---
    if f16_gguf.exists():
        f16_gguf.unlink()
        log.info(f"🧹 Cleaned up intermediate: {f16_gguf}")

    log.info("=== GGUF Conversion Complete ===")
    log.info(f"✅ Successfully created {success_count}/{len(quant_types)} quantizations")
    log.info(f"📁 GGUF files saved to: {output_dir}")

    log.info("\n🚀 Next steps:")
    log.info("  1. Create Modelfile: python scripts/create_modelfile.py")
    log.info("  2. Test with Ollama: ollama create smart-secrets-scanner -f Modelfile")
    log.info("  3. Run inference: ollama run smart-secrets-scanner")

    return 0 if success_count > 0 else 1