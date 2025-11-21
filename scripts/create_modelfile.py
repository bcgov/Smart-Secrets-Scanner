#!/usr/bin/env python3
# ==============================================================================
# CREATE_MODELFILE.PY v18.4 — FINAL WITH VERSION HEADER
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
# Smart Secrets Scanner — v18.4 (FINAL CLEAN OUTPUT)
# Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# This version has perfect structured output with no garbage
# ==============================================================================
'''

# EXACT SYSTEM PROMPT FROM TRAINING (scripts/fine_tune.py)
SYSTEM_PROMPT = (
    "You are Smart-Secrets-Scanner, a hyper-specialized AI model for code security. "
    "Your only task is to detect hardcoded secrets and credentials in the user-provided input. "
    "Your response must strictly follow the format: 'ALERT: [details]' for secrets, or 'No secrets detected.' for safe code."
)

# Llama 3 Template Structure
# Note: Quadruple braces {{{{ }}}} are needed for f-string escaping to produce {{ }} in output
COMMON_CONTENT = f'''
SYSTEM \"\"\"{SYSTEM_PROMPT}\"\"\"

TEMPLATE """{{{{ if .System }}}}<|start_header_id|>system<|end_header_id|>

{{{{ .System }}}}<|eot_id|>{{{{ end }}}}{{{{ if .Prompt }}}}<|start_header_id|>user<|end_header_id|>

{{{{ .Prompt }}}}<|eot_id|>{{{{ end }}}}<|start_header_id|>assistant<|end_header_id|>

{{{{ .Response }}}}<|eot_id|>"""

PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|reserved_special_token_"
PARAMETER stop "user"
PARAMETER stop "assistant"
PARAMETER stop "Analyze"
PARAMETER stop "[ ]"
PARAMETER stop "[x]"
PARAMETER stop "TODO"
PARAMETER stop "Verified"

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
#!/usr/bin/env python3
# ==============================================================================
# CREATE_MODELFILE.PY v18.4 — FINAL WITH VERSION HEADER
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
# Smart Secrets Scanner — v18.4 (FINAL CLEAN OUTPUT)
# Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# This version has perfect structured output with no garbage
# ==============================================================================
'''

# EXACT SYSTEM PROMPT FROM TRAINING (scripts/fine_tune.py)
SYSTEM_PROMPT = (
    "You are Smart-Secrets-Scanner, a hyper-specialized AI model for code security. "
    "Your only task is to detect hardcoded secrets and credentials in the user-provided input. "
    "Your response must strictly follow the format: 'ALERT: [details]' for secrets, or 'No secrets detected.' for safe code."
)

# Llama 3 Template Structure
# Note: Quadruple braces {{{{ }}}} are needed for f-string escaping to produce {{ }} in output
COMMON_CONTENT = f'''
SYSTEM \"\"\"{SYSTEM_PROMPT}\"\"\"

TEMPLATE """{{{{ if .System }}}}<|start_header_id|>system<|end_header_id|>

{{{{ .System }}}}<|eot_id|>{{{{ end }}}}{{{{ if .Prompt }}}}<|start_header_id|>user<|end_header_id|>

{{{{ .Prompt }}}}<|eot_id|>{{{{ end }}}}<|start_header_id|>assistant<|end_header_id|>

{{{{ .Response }}}}<|eot_id|>"""

PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|reserved_special_token_"
PARAMETER stop "user"
PARAMETER stop "assistant"
PARAMETER stop "Analyze"
PARAMETER stop "[ ]"
PARAMETER stop "[x]"
PARAMETER stop "TODO"
PARAMETER stop "Verified"

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

print("\nv18.4 — VERSION HEADER ADDED TO BOTH MODEFILES")
print("   You will see '# Smart Secrets Scanner — v18.4' at the top on HF")
print("\n=== NEXT STEPS ===")
print("1. Upload the fixed Modelfile:")
print("   python scripts/upload_to_huggingface.py --repo richfrem/smart-secrets-scanner-gguf --files huggingface/Modelfile")
print("\n2. Run model directly from Hugging Face (Ollama will auto-download config):")
print("   ollama rm hf.co/richfrem/smart-secrets-scanner-gguf:Q4_K_M")
print("   ollama run hf.co/richfrem/smart-secrets-scanner-gguf:Q4_K_M")