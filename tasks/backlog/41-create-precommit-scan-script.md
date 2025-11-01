# Task 41: Create Pre-Commit Scan Script

**Status:** Backlog  
**Priority:** MEDIUM  
**Created:** 2025-11-01  
**Related to:** Phase 4: Testing & Deployment (Step 13)  
**Depends on:** Task 40 (Ollama deployment)

## Description
Create `scripts/scan_secrets.py` - Python script that scans staged files for secrets using the deployed Ollama model, for use in pre-commit hooks.

## Acceptance Criteria
- [ ] `scripts/scan_secrets.py` created and executable
- [ ] Reads staged git files
- [ ] Calls Ollama API with code content
- [ ] Parses model response for ALERT signals
- [ ] Exits with error code if secrets detected
- [ ] User-friendly error messages with line numbers
- [ ] Fast enough for developer workflow (<5 seconds per file)
- [ ] Can be integrated with pre-commit framework

## Script Implementation
Create `scripts/scan_secrets.py`:

```python
#!/usr/bin/env python3
"""
Pre-commit hook script to scan code for secrets using Smart Secrets Scanner
"""
import sys
import subprocess
import json
import requests
import argparse
from pathlib import Path
from typing import List, Tuple

OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "smart-secrets-scanner"

def check_ollama_running():
    """Check if Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_staged_files() -> List[str]:
    """Get list of staged files from git"""
    result = subprocess.run(
        ['git', 'diff', '--cached', '--name-only', '--diff-filter=ACM'],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        return []
    
    # Filter for code files
    extensions = {'.py', '.js', '.ts', '.java', '.go', '.rb', '.php', '.yaml', '.yml', '.json', '.env'}
    files = []
    for line in result.stdout.strip().split('\n'):
        if line and Path(line).suffix in extensions:
            files.append(line)
    
    return files

def scan_code_with_ollama(code: str, file_path: str = "") -> Tuple[bool, str]:
    """Scan code using Ollama API"""
    prompt = f"Analyze the following code snippet and identify any secrets or sensitive credentials that should not be committed to version control.\n\n{code}"
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 150
        }
    }
    
    try:
        response = requests.post(OLLAMA_API, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        response_text = result.get('response', '')
        
        # Check for ALERT signal
        has_secret = "ALERT" in response_text.upper()
        
        return has_secret, response_text
        
    except requests.exceptions.RequestException as e:
        print(f"⚠️  Warning: Failed to scan {file_path}: {e}")
        return False, ""

def scan_file(file_path: str) -> Tuple[bool, str]:
    """Scan a single file for secrets"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Skip empty files
        if not content.strip():
            return False, ""
        
        # Skip very large files (>100KB)
        if len(content) > 100_000:
            print(f"⚠️  Skipping large file: {file_path} ({len(content)} bytes)")
            return False, ""
        
        return scan_code_with_ollama(content, file_path)
        
    except Exception as e:
        print(f"⚠️  Warning: Could not read {file_path}: {e}")
        return False, ""

def main():
    parser = argparse.ArgumentParser(description="Scan staged files for secrets")
    parser.add_argument(
        'files',
        nargs='*',
        help='Files to scan (default: all staged files)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Scan all files in repository, not just staged'
    )
    parser.add_argument(
        '--model',
        default=MODEL_NAME,
        help=f'Ollama model name (default: {MODEL_NAME})'
    )
    
    args = parser.parse_args()
    
    global MODEL_NAME
    MODEL_NAME = args.model
    
    print("🔍 Smart Secrets Scanner - Pre-Commit Check")
    print("=" * 60)
    
    # Check Ollama is running
    if not check_ollama_running():
        print("❌ Error: Ollama is not running!")
        print("\nPlease start Ollama:")
        print("  ollama serve")
        print(f"\nAnd ensure model is available:")
        print(f"  ollama list | grep {MODEL_NAME}")
        return 1
    
    # Get files to scan
    if args.files:
        files_to_scan = args.files
    elif args.all:
        # Scan all tracked files
        result = subprocess.run(
            ['git', 'ls-files'],
            capture_output=True,
            text=True
        )
        files_to_scan = result.stdout.strip().split('\n')
    else:
        files_to_scan = get_staged_files()
    
    if not files_to_scan:
        print("✅ No files to scan")
        return 0
    
    print(f"\n📄 Scanning {len(files_to_scan)} file(s)...\n")
    
    # Scan each file
    secrets_found = []
    
    for file_path in files_to_scan:
        if not file_path:
            continue
        
        print(f"  Checking: {file_path}...", end=' ')
        sys.stdout.flush()
        
        has_secret, message = scan_file(file_path)
        
        if has_secret:
            print("🚨 ALERT")
            secrets_found.append((file_path, message))
        else:
            print("✅")
    
    # Report results
    print("\n" + "=" * 60)
    
    if secrets_found:
        print("❌ SECRETS DETECTED - COMMIT BLOCKED")
        print("=" * 60)
        print(f"\nFound {len(secrets_found)} file(s) with potential secrets:\n")
        
        for file_path, message in secrets_found:
            print(f"📁 {file_path}")
            print(f"   {message.strip()}\n")
        
        print("⚠️  Please remove hardcoded secrets before committing.")
        print("\nRecommended actions:")
        print("  1. Move secrets to environment variables")
        print("  2. Use a secret management service (AWS Secrets Manager, etc.)")
        print("  3. Add to .gitignore if it's a config file")
        print("  4. Use git-filter-repo to remove from history if already committed")
        
        return 1
    else:
        print("✅ No secrets detected - safe to commit!")
        print("=" * 60)
        return 0

if __name__ == "__main__":
    sys.exit(main())
```

## Pre-Commit Integration
Create `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: local
    hooks:
      - id: smart-secrets-scanner
        name: Smart Secrets Scanner (LLM-powered)
        entry: python scripts/scan_secrets.py
        language: python
        pass_filenames: true
        types: [python, javascript, yaml, json]
        stages: [commit]
```

## Installation
```bash
# Install pre-commit framework
pip install pre-commit

# Install hook
pre-commit install

# Test on all files
pre-commit run --all-files

# Test on staged files
pre-commit run
```

## Usage
```bash
# Scan staged files (pre-commit use)
python scripts/scan_secrets.py

# Scan specific files
python scripts/scan_secrets.py file1.py file2.js

# Scan all files in repo
python scripts/scan_secrets.py --all

# Use custom model
python scripts/scan_secrets.py --model my-custom-scanner
```

## Example Output
```
🔍 Smart Secrets Scanner - Pre-Commit Check
============================================================

📄 Scanning 3 file(s)...

  Checking: src/config.py... 🚨 ALERT
  Checking: src/utils.py... ✅
  Checking: tests/test_auth.py... ✅

============================================================
❌ SECRETS DETECTED - COMMIT BLOCKED
============================================================

Found 1 file(s) with potential secrets:

📁 src/config.py
   ALERT: AWS access key detected in line 12. This appears to be a
   hardcoded AWS credential (AKIA...) that should not be committed.

⚠️  Please remove hardcoded secrets before committing.
```

## Dependencies
- Task 40: Ollama model deployed
- Task 15: Ollama running and model available
- Python packages: requests (add to requirements.txt)

## Performance Optimization
- Skip files >100KB
- Skip binary files
- Batch small files together
- Cache results for unchanged files (future enhancement)

## Success Criteria
- Scans staged files in <5 seconds each
- Correctly blocks commits with secrets
- Allows commits with safe code
- Clear error messages guide developers
- Works with standard pre-commit framework

## Related Tasks
- Task 27: Integrate pre-commit hooks (this implements the scanner)
- Task 15: Test Ollama deployment (prerequisite)
- Task 26: Test with raw files (validation of this scanner)
