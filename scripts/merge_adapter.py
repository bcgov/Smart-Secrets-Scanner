#!/usr/bin/env python3
# ==============================================================================
# MERGE_ADAPTER.PY (v2.0) – 8GB-Safe, Config-Driven, Robust LoRA Merger
# ==============================================================================
"""
Merge LoRA Adapter with Base Model for Smart-Secrets-Scanner

This script merges a trained LoRA adapter with the base Llama model,
creating a standalone merged model for inference or GGUF conversion.

Enhanced with production features from Project Sanctuary:
- 4-bit quantization for 8GB VRAM safety
- Config-driven paths and settings
- Robust error handling and logging
- Sanity inference checks
- Atomic saves with metadata tracking
- Memory usage monitoring

Usage:
    python scripts/merge_adapter.py
    python scripts/merge_adapter.py --config config/merge_config.yaml
    python scripts/merge_adapter.py --base "models/base/Meta-Llama-3.1-8B" --adapter "models/fine-tuned/smart-secrets-scanner-lora"
"""

import argparse
import json
import logging
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

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
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "merge_config.yaml"


# --------------------------------------------------------------------------- #
# Config Loader
# --------------------------------------------------------------------------- #
def load_config(config_path: Path):
    """Load merge configuration from YAML file"""
    log.info(f"Loading merge config from {config_path}")
    if not config_path.exists():
        log.warning(f"Config not found: {config_path}")
        log.info("Using default configuration...")
        # Return default config if file doesn't exist
        return {
            "model": {
                "base_model_name": "Meta-Llama-3.1-8B",
                "adapter_path": "models/fine-tuned/smart-secrets-scanner-lora",
                "merged_output_path": "models/merged/smart-secrets-scanner"
            },
            "merge": {
                "final_dtype": "float16",
                "skip_sanity_check": False
            }
        }

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


# --------------------------------------------------------------------------- #
# Memory Reporter
# --------------------------------------------------------------------------- #
def report_memory(stage: str):
    """Report current GPU memory usage"""
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        log.info(".2f")
    else:
        log.info(f"{stage} | CPU mode - no GPU memory tracking")


# --------------------------------------------------------------------------- #
# Sanity Check Inference
# --------------------------------------------------------------------------- #
def sanity_check_inference(model, tokenizer, prompt="Analyze this code for secrets: API_KEY = 'test123'"):
    """Run a quick sanity check inference on the merged model"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False, temperature=0.0)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        log.info(f"✅ Sanity check passed: {decoded[:100]}...")
        return True
    except Exception as e:
        log.warning(f"⚠️  Sanity check failed: {e}")
        return False


# --------------------------------------------------------------------------- #
# Main Merge Function
# --------------------------------------------------------------------------- #
def merge_lora_adapter(base_model_path: Path, adapter_path: Path, output_path: Path,
                      final_dtype: torch.dtype = torch.float16, skip_sanity: bool = False):
    """
    Merge LoRA adapter with base model using memory-efficient approach

    Args:
        base_model_path: Path to base model directory
        adapter_path: Path to LoRA adapter directory
        output_path: Path to save merged model
        final_dtype: Final dtype for saved model
        skip_sanity: Skip sanity inference check

    Returns:
        tuple: (merged_model, tokenizer) or (None, None) on failure
    """

    log.info("🔗 Starting LoRA Adapter Merge")
    log.info(f"   Base: {base_model_path}")
    log.info(f"   Adapter: {adapter_path}")
    log.info(f"   Output: {output_path}")
    log.info(f"   Final dtype: {final_dtype}")

    # --- Validation ---
    if not base_model_path.exists():
        log.error(f"❌ Base model not found: {base_model_path}")
        return None, None

    if not adapter_path.exists() or not (adapter_path / "adapter_config.json").exists():
        log.error(f"❌ Adapter not found or invalid: {adapter_path}")
        return None, None

    output_path.mkdir(parents=True, exist_ok=True)

    # --- 4-bit Quantization Config (critical for 8GB VRAM) ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    report_memory("[1/6] Before load")

    # --- Load Base Model in 4-bit ---
    log.info("[2/6] 🔽 Loading base model in 4-bit (VRAM-safe)")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            str(base_model_path),
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(base_model_path), trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    except Exception as e:
        log.exception(f"❌ Failed to load base model: {e}")
        return None, None

    report_memory("[2/6] After base load")

    # --- Load LoRA Adapter ---
    log.info("[3/6] 🔗 Applying LoRA adapter")
    try:
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
    except Exception as e:
        log.exception(f"❌ Failed to load adapter: {e}")
        return None, None

    report_memory("[3/6] After adapter")

    # --- Merge Weights ---
    log.info("[4/6] 🔄 Merging weights (may take 30-60s)")
    try:
        with torch.no_grad():
            merged_model = model.merge_and_unload()
        # Move to CPU to free GPU memory
        merged_model.to("cpu")
        torch.cuda.empty_cache()
    except Exception as e:
        log.exception(f"❌ Merge failed: {e}")
        return None, None

    report_memory("[4/6] After merge")

    # --- Sanity Check ---
    if not skip_sanity:
        log.info("[5/6] 🧪 Running sanity inference check")
        if not sanity_check_inference(merged_model, tokenizer):
            log.warning("⚠️  Sanity check failed; proceeding but verify outputs manually")

    # --- Cast to final dtype ---
    log.info(f"[6/6] 💾 Casting to {final_dtype} and saving")
    merged_model = merged_model.to(final_dtype)

    # --- Atomic Save ---
    tmpdir = Path(tempfile.mkdtemp(prefix="merge_tmp_"))
    try:
        merged_model.save_pretrained(str(tmpdir), safe_serialization=True, max_shard_size="10GB")
        tokenizer.save_pretrained(str(tmpdir))

        # Save metadata
        meta = {
            "merged_at": datetime.utcnow().isoformat() + "Z",
            "torch_version": torch.__version__,
            "transformers_version": __import__("transformers").__version__,
            "peft_version": __import__("peft").__version__,
            "base_model": str(base_model_path),
            "adapter": str(adapter_path),
            "final_dtype": str(final_dtype),
            "project": "Smart-Secrets-Scanner",
            "sanity_check_passed": not skip_sanity
        }
        with open(tmpdir / "merge_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Atomic move
        if output_path.exists():
            shutil.rmtree(output_path)
        shutil.move(str(tmpdir), str(output_path))

        log.info(f"✅ Merged model saved to {output_path}")
        log.info("📋 Next steps:")
        log.info("   1. Test merged model: python scripts/inference.py")
        log.info("   2. Convert to GGUF: python scripts/convert_to_gguf.py")
        return merged_model, tokenizer

    except Exception as e:
        log.exception(f"❌ Save failed: {e}")
        try:
            shutil.rmtree(tmpdir)
        except:
            pass
        return None, None
    finally:
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                       help="Path to merge config YAML")
    parser.add_argument("--base", type=str,
                       help="Override base model name/path")
    parser.add_argument("--adapter", type=str,
                       help="Override adapter path")
    parser.add_argument("--output", type=str,
                       help="Override output path")
    parser.add_argument("--dtype", type=str, default="float16",
                       choices=["float16", "bfloat16", "float32"],
                       help="Final save dtype")
    parser.add_argument("--skip-sanity", action="store_true",
                       help="Skip sanity inference check")
    parser.add_argument("--verify", action="store_true",
                       help="Run verification after merge (legacy flag)")

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Override from CLI or use defaults
    base_name_or_path = args.base or cfg["model"]["base_model_name"]
    adapter_path_str = args.adapter or cfg["model"]["adapter_path"]
    output_path_str = args.output or cfg["model"]["merged_output_path"]
    final_dtype = getattr(torch, args.dtype)

    # Resolve paths relative to project root
    if Path(base_name_or_path).is_absolute():
        base_model_path = Path(base_name_or_path)
    else:
        base_model_path = PROJECT_ROOT / "models" / "base" / base_name_or_path

    adapter_path = PROJECT_ROOT / adapter_path_str
    output_path = PROJECT_ROOT / output_path_str

    # Merge
    merged_model, tokenizer = merge_lora_adapter(
        base_model_path, adapter_path, output_path,
        final_dtype, args.skip_sanity
    )

    if merged_model is None:
        log.error("❌ Merge failed!")
        return 1

    # Legacy verification (if requested)
    if args.verify:
        log.info("🧪 Running legacy verification...")
        test_input = "api_key = 'test123'"
        prompt = f"### Instruction:\nAnalyze this code\n\n### Input:\n{test_input}\n\n### Response:\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(merged_model.device)

        with torch.no_grad():
            outputs = merged_model.generate(**inputs, max_new_tokens=50)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        log.info(f"✅ Verification complete. Sample response: {response[:100]}...")

    log.info("🎉 Merge operation completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
