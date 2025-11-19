#!/usr/bin/env python3
# ==============================================================================
# UPLOAD_TO_HUGGINGFACE.PY (v1.0) – Automated Hugging Face Upload Script
# ==============================================================================
import argparse
import logging
import os
import sys
from pathlib import Path
import atexit

try:
    from dotenv import load_dotenv
except ImportError:
    print("python-dotenv not installed. Install with: pip install python-dotenv")
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("PyYAML not installed. Install with: pip install PyYAML")
    sys.exit(1)

try:
    from huggingface_hub import HfApi, upload_file, upload_folder
except ImportError:
    print("huggingface_hub not installed. Install with: pip install huggingface_hub")
    sys.exit(1)

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

# Ensure log directory exists
log_dir = Path('../outputs/logs')
log_dir.mkdir(parents=True, exist_ok=True)

file_handler = logging.FileHandler('../outputs/logs/upload_to_huggingface.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
logging.getLogger().addHandler(file_handler)

log = logging.getLogger(__name__)
log.info("Upload to Hugging Face script started - logging to console and ../outputs/logs/upload_to_huggingface.log")

atexit.register(logging.shutdown)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
SCRIPT_DIR = Path(__file__).resolve().parent
FORGE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = FORGE_ROOT.parent.parent

# --------------------------------------------------------------------------- #
# Load Config
# --------------------------------------------------------------------------- #
def load_config():
    config_path = FORGE_ROOT / "config" / "upload_config.yaml"
    if not config_path.exists():
        log.warning(f"Config file not found at {config_path}, using defaults.")
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        log.info(f"Loaded config from {config_path}")
        return config
def load_environment():
    env_path = FORGE_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        log.info(f"Loaded environment from {env_path}")
    else:
        log.warning(f"No .env file found at {env_path}")

    token = os.getenv("HUGGING_FACE_TOKEN")
    username = os.getenv("HUGGING_FACE_USERNAME")
    repo_name = os.getenv("HUGGING_FACE_REPO")
    
    if not token:
        log.error("HUGGING_FACE_TOKEN not found in environment variables.")
        log.info("Please set HUGGING_FACE_TOKEN in your .env file.")
        sys.exit(1)
    
    if not username or not repo_name:
        log.warning("HUGGING_FACE_USERNAME or HUGGING_FACE_REPO not set. Will require --repo argument.")
    
    return token, username, repo_name

# --------------------------------------------------------------------------- #
# Upload Function
# --------------------------------------------------------------------------- #
def make_modelfile_portable(modelfile_path, gguf_filename):
    """Create a portable version of the Modelfile for upload."""
    with open(modelfile_path, 'r') as f:
        content = f.read()
    
    # Replace the FROM line with relative path
    import re
    content = re.sub(r'FROM .*', f'FROM {gguf_filename}', content)
    
    # Write to a temp file
    temp_path = modelfile_path.with_suffix('.temp')
    with open(temp_path, 'w') as f:
        f.write(content)
    
    return temp_path

def upload_to_hf(repo_id, file_paths, token, private=False, gguf_filename=None):
    api = HfApi(token=token)
    
    # Create repo if it doesn't exist
    try:
        api.repo_info(repo_id)
        log.info(f"Repository {repo_id} exists.")
    except Exception:
        log.info(f"Creating repository {repo_id}...")
        api.create_repo(repo_id, private=private)
    
    temp_files = []
    for file_path in file_paths:
        path_obj = Path(file_path)
        if not path_obj.exists():
            log.warning(f"File not found: {file_path}, skipping.")
            continue
        
        if path_obj.is_file():
            # Special handling for Modelfile to make it portable
            if path_obj.name == 'Modelfile' and gguf_filename:
                log.info(f"Making Modelfile portable for upload...")
                temp_file = make_modelfile_portable(path_obj, gguf_filename)
                temp_files.append(temp_file)
                upload_path = str(temp_file)
                repo_name = 'Modelfile'
            else:
                upload_path = str(path_obj)
                repo_name = path_obj.name
            
            log.info(f"Uploading file: {repo_name}")
            upload_file(
                path_or_fileobj=upload_path,
                path_in_repo=repo_name,
                repo_id=repo_id,
                token=token
            )
        elif path_obj.is_dir():
            log.info(f"Uploading folder: {path_obj.name}")
            upload_folder(
                folder_path=str(path_obj),
                repo_id=repo_id,
                token=token
            )
        else:
            log.warning(f"Unknown path type: {file_path}, skipping.")
    
    # Clean up temp files
    for temp_file in temp_files:
        temp_file.unlink()
    
    log.info(f"Upload complete. Repository: https://huggingface.co/{repo_id}")

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Upload files to Hugging Face repository.")
    parser.add_argument("--repo", help="Hugging Face repository ID (e.g., username/repo-name). If not provided, uses HUGGING_FACE_USERNAME/HUGGING_FACE_REPO from .env or config")
    parser.add_argument("--files", nargs="+", help="Paths to files or folders to upload")
    parser.add_argument("--private", action="store_true", help="Create private repository")
    parser.add_argument("--gguf", action="store_true", help="Upload GGUF file from config location")
    parser.add_argument("--modelfile", action="store_true", help="Upload Modelfile from config location")
    
    # --- ADD THESE NEW ARGUMENTS ---
    parser.add_argument("--system", action="store_true", help="Upload 'system' file for Ollama configuration")
    parser.add_argument("--template", action="store_true", help="Upload 'template' file for Ollama configuration")
    parser.add_argument("--params", action="store_true", help="Upload 'params.json' file for Ollama configuration")
    # -------------------------------
    
    parser.add_argument("--readme", action="store_true", help="Upload README.md from config location")
    parser.add_argument("--model-card", action="store_true", help="Upload model_card.yaml from config location")
    parser.add_argument("--lora", action="store_true", help="Upload LoRA adapter from config location")
    parser.add_argument("--merged", action="store_true", help="Upload merged model from config location")
    args = parser.parse_args()

    # Load configuration
    config = load_config()
    token, username, repo_name = load_environment()

    if args.repo:
        repo_id = args.repo
    elif username and repo_name:
        repo_id = f"{username}/{repo_name}"
        log.info(f"Using default repo from .env: {repo_id}")
    elif config.get('repository', {}).get('default_repo'):
        repo_id = config['repository']['default_repo']
        log.info(f"Using default repo from config: {repo_id}")
    else:
        log.error("No repository specified. Use --repo or set HUGGING_FACE_USERNAME and HUGGING_FACE_REPO in .env or config")
        sys.exit(1)

    file_paths = args.files or []

    # Use config paths for file locations
    files_config = config.get('files', {})
    gguf_filename = None

    if args.gguf:
        gguf_path = FORGE_ROOT / files_config.get('gguf_path', "models/fine-tuned/gguf/smart-secrets-scanner-Q4_K_M.gguf")
        file_paths.append(str(gguf_path))
        gguf_filename = Path(gguf_path).name

        # ----------------------------------------------------------------------
        # CRITICAL FIX: Ensure separate config files for Hugging Face are uploaded.
        # Ollama requires these files (system, template, params.json) 
        # to override baked-in GGUF model settings for consistent behavior on the Hub.
        # ----------------------------------------------------------------------
        log.info("Adding separate Ollama configuration files for Hub consistency.")
        
        # FORGE_ROOT is defined near the top and points to the project root directory
        project_root = FORGE_ROOT 
        
        # 1. Add 'system' file (for SYSTEM prompt) from huggingface folder
        file_paths.append(str(project_root / "huggingface" / "system"))
        # 2. Add 'template' file (for TEMPLATE) from huggingface folder
        file_paths.append(str(project_root / "huggingface" / "template"))
        # 3. Add 'params.json' file (for PARAMETERs) from huggingface folder
        file_paths.append(str(project_root / "huggingface" / "params.json"))

    if args.modelfile:
        modelfile_path = FORGE_ROOT / files_config.get('modelfile_path', "Modelfile")
        file_paths.append(str(modelfile_path))

    if args.lora:
        lora_path = FORGE_ROOT / files_config.get('lora_path', "models/fine-tuned/smart-secrets-scanner-lora")
        file_paths.append(str(lora_path))

    # --- ADDED: Separate Configuration File Uploads ---
    # Look for config files in huggingface folder
    if args.system:
        file_paths.append(str(FORGE_ROOT / "huggingface" / "system"))
        
    if args.template:
        file_paths.append(str(FORGE_ROOT / "huggingface" / "template"))
        
    if args.params:
        file_paths.append(str(FORGE_ROOT / "huggingface" / "params.json"))
    # --------------------------------------------------
        if args.lora:
            # Use LoRA-specific README for LoRA uploads
            readme_path = FORGE_ROOT / files_config.get('readme_lora_path', "huggingface/README_LORA.md")
        else:
            # Use standard README for other uploads
            readme_path = FORGE_ROOT / files_config.get('readme_path', "huggingface/README.md")
        file_paths.append(str(readme_path))

    if args.model_card:
        model_card_path = FORGE_ROOT / files_config.get('model_card_path', "huggingface/model_card.yaml")
        file_paths.append(str(model_card_path))

    if args.lora:
        lora_path = FORGE_ROOT / files_config.get('lora_path', "models/fine-tuned/smart-secrets-scanner-lora")
        file_paths.append(str(lora_path))

    if args.merged:
        merged_path = FORGE_ROOT / files_config.get('merged_path', "outputs/merged/smart-secrets-scanner")
        file_paths.append(str(merged_path))

    if not file_paths:
        log.error("No files specified. Use --files, --gguf, --modelfile, --readme, --model-card, or --lora")
        sys.exit(1)

    log.info("=== Hugging Face Upload ===")
    log.info(f"Repository: {repo_id}")
    log.info(f"Files: {file_paths}")

    upload_to_hf(repo_id, file_paths, token, args.private, gguf_filename)

    log.info("=== Upload Complete ===")

if __name__ == "__main__":
    main()