#!/usr/bin/env python3
# ==============================================================================
# CREATE_MODELFILE.PY (v6.0) – STANDARD ALPACA FORMAT
# Matches the generic training format for broad compatibility
# ==============================================================================
import sys
import yaml
import os
import json
from pathlib import Path
from datetime import datetime, timezone

# --- Load environment variables ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
env_path = PROJECT_ROOT / '.env'
if env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        pass

# --- Paths ---
FORGE_ROOT = PROJECT_ROOT

# --- Load Configuration ---
CONFIG_PATH = FORGE_ROOT / "config" / "gguf_config.yaml"
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)
    GGUF_DIR = PROJECT_ROOT / os.environ.get('SANCTUARY_GGUF_OUTPUT_DIR', cfg["model"]["gguf_output_dir"])
    MODEL_NAME_PATTERN = os.environ.get('SANCTUARY_GGUF_MODEL_NAME', cfg["model"]["gguf_model_name"])
else:
    # Fallback defaults if config missing
    GGUF_DIR = PROJECT_ROOT / "models/fine-tuned/gguf"
    MODEL_NAME_PATTERN = "smart-secrets-scanner"

# Auto-pick newest Smart-Secrets-Scanner GGUF
gguf_files = list(GGUF_DIR.glob(f"{MODEL_NAME_PATTERN}*.gguf"))
if not gguf_files:
    print(f"ERROR: No {MODEL_NAME_PATTERN}*.gguf found in {GGUF_DIR}/")
    sys.exit(1)

GGUF_MODEL_PATH = max(gguf_files, key=lambda p: p.stat().st_mtime)
OUTPUT_MODELFILE_PATH = PROJECT_ROOT / "Modelfile"

# --- SYSTEM PROMPT (The Training Preamble) ---
# Matches formatting_prompts_func in fine_tune.py
SYSTEM_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."""

# --- TEMPLATE (Standard Alpaca) ---
# This is the standard template for Alpaca models. 
# It maps the User's input to the 'Instruction' slot.
# We leave 'Input' empty here because in a chat context, the user's prompt 
# usually contains the full context.
TEMPLATE_CONTENT = """{{ if .System }}{{ .System }}

{{ end }}### Instruction:
{{ .Prompt }}

### Input:

### Response:
"""

# --- FINAL MODELFILE ---
MODELFILE_CONTENT = f'''# ==============================================================================
# Ollama Modelfile – {MODEL_NAME_PATTERN}
# Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")} UTC
# ==============================================================================

FROM {GGUF_MODEL_PATH.name}

# Standard Alpaca Preamble
SYSTEM """{SYSTEM_PROMPT}"""

# Standard Alpaca Template
TEMPLATE """{TEMPLATE_CONTENT}"""

# Standard Alpaca Stop Tokens
PARAMETER stop "### Instruction:"
PARAMETER stop "### Input:"
PARAMETER stop "### Response:"
PARAMETER stop "<|end_of_text|>"

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER num_ctx 8192
PARAMETER num_predict 256
'''

def main():
    print("Ollama Modelfile Generator v6.0 — STANDARD ALPACA FORMAT")
    
    try:
        # 1. Write the local Modelfile
        OUTPUT_MODELFILE_PATH.write_text(MODELFILE_CONTENT.lstrip(), encoding="utf-8")
        print(f"SUCCESS → Modelfile created at {OUTPUT_MODELFILE_PATH}")
        
        # 2. Write separate files for Hugging Face
        system_file = PROJECT_ROOT / "system"
        template_file = PROJECT_ROOT / "template"
        params_file = PROJECT_ROOT / "params.json"
        
        system_file.write_text(SYSTEM_PROMPT, encoding="utf-8")
        template_file.write_text(TEMPLATE_CONTENT, encoding="utf-8")
        
        params = {
            "temperature": 0.1, 
            "top_p": 0.9,
            "num_ctx": 8192,
            "num_predict": 256,
            "stop": ["### Instruction:", "### Input:", "### Response:", "<|end_of_text|>"]
        }
        params_file.write_text(json.dumps(params, indent=2), encoding="utf-8")
        
        print(f"SUCCESS → Config files created (system, template, params.json)")

    except Exception as e:
        print(f"Failed to write files: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()