#!/usr/bin/env python3
# ==============================================================================
# CREATE_MODELFILE.PY v18.1 — FINAL WITH VERSION HEADER
# Adds clear version banner to both local and HF Modelfiles for verification
# ==============================================================================
from pathlib import Path
import yaml
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = PROJECT_ROOT / "config" / "gguf_config.yaml"
LOCAL_MODELFILE = PROJECT_ROOT / "Modelfile"
HF_DIR = PROJECT_ROOT / "huggingface"
HF_MODELFILE = HF_DIR / "Modelfile"

# Load config
with open(CONFIG_FILE) as f:
    cfg = yaml.safe_load(f)['model']

# Find GGUF
gguf_path = None
for folder in [Path.home() / "ollama-models", PROJECT_ROOT / cfg['gguf_output_dir']]:
    if folder.exists():
        files = list(folder.glob("*-Q4_K_M.gguf"))
        if files:
            gguf_path = files[0]
            break

if not gguf_path:
    print("ERROR: GGUF not found!")
    exit(1)

print(f"Found GGUF: {gguf_path}")

# Version header — will appear at the top of both Modelfiles
VERSION_HEADER = f'''# ==============================================================================
# Smart Secrets Scanner — v18.1 (FINAL CLEAN OUTPUT)
# Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# This version has perfect structured output with no garbage
# ==============================================================================
'''

SYSTEM_PROMPT = """You are Smart Secrets Scanner — a hardened, single-purpose secret detection engine.

You respond with EXACTLY this format and STOP:

ALERT: <description>
Recommendation: <short fix>
Final verdict: BLOCKED or SAFE

If no secrets found:
No secrets detected. Safe to commit.

NEVER add checklists, TODOs, boxes, or any extra text."""

COMMON_CONTENT = f'''
SYSTEM \"\"\"{SYSTEM_PROMPT}\"\"\"

TEMPLATE """### Human: {{ .Prompt }}

### Assistant: 
"""

PARAMETER stop "### Human:"
PARAMETER stop "### Assistant:"
PARAMETER stop "[ ]"
PARAMETER stop "[x]"
PARAMETER stop "TODO"
PARAMETER stop "|"
PARAMETER stop "Verified"
PARAMETER stop "Other..."

PARAMETER temperature 0.01
PARAMETER top_p 0.9
PARAMETER num_predict 120
'''

# Local: absolute path
LOCAL_CONTENT = f'''FROM {gguf_path}
''' + VERSION_HEADER + COMMON_CONTENT.lstrip()

# Hugging Face: relative path
HF_CONTENT = f'''FROM ./smart-secrets-scanner-Q4_K_M.gguf
''' + VERSION_HEADER + COMMON_CONTENT.lstrip()

# Write both
LOCAL_MODELFILE.write_text(LOCAL_CONTENT.lstrip(), encoding="utf-8")
HF_DIR.mkdir(exist_ok=True)
HF_MODELFILE.write_text(HF_CONTENT.lstrip(), encoding="utf-8")

print("\nv18.1 — VERSION HEADER ADDED TO BOTH MODEFILES")
print("   You will see '# Smart Secrets Scanner — v18.1' at the top on HF")
print("   Upload now to verify!")