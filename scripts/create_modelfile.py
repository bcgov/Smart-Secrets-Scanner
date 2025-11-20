#!/usr/bin/env python3
# ==============================================================================
# CREATE_MODELFILE.PY v17.0 — FINAL & PERFECT FOR LOCAL + HUGGING FACE
# Generates two correct Modelfiles with different FROM paths
# ==============================================================================
from pathlib import Path
import yaml

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

# WINNING CONTENT — identical behavior, different FROM
COMMON_CONTENT = '''
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
''' + COMMON_CONTENT

# Hugging Face: relative path (standard Ollama format)
HF_CONTENT = f'''FROM ./smart-secrets-scanner-Q4_K_M.gguf
''' + COMMON_CONTENT

# Write both
LOCAL_MODELFILE.write_text(LOCAL_CONTENT.lstrip(), encoding="utf-8")
HF_DIR.mkdir(exist_ok=True)
HF_MODELFILE.write_text(HF_CONTENT.lstrip(), encoding="utf-8")

print("\nv17.0 — PERFECT DUAL MODELFIL ES GENERATED")
print("   Local Modelfile → ready")
print("   huggingface/Modelfile → ready for upload")