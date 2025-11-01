# Task 27: Integrate with Pre-Commit Hooks

**Status:** Backlog  
**Created:** 2025-11-01  
**Related to:** Phase 4: Testing & Deployment (Step 13)

## Description
Integrate the fine-tuned Smart Secrets Scanner model with git pre-commit hooks to scan code before commits.

**Note**: The scanner script is now implemented by **Task 41: Create Pre-Commit Scan Script**.

## Implementation
See **Task 41** for the complete implementation of `scripts/scan_secrets.py`.

## Quick Setup
```bash
# Install pre-commit
pip install pre-commit

# Create .pre-commit-config.yaml (see Task 41 for template)

# Install hook
pre-commit install

# Test
pre-commit run --all-files
```

## Requirements
- Ollama deployment complete (Task 15)
- Pre-commit framework installed
- Python script to call model via Ollama API

## Acceptance Criteria
- [ ] `.pre-commit-config.yaml` created
- [ ] Python script scans staged files via Ollama API
- [ ] Commit blocked if secrets detected
- [ ] User-friendly error messages with line numbers
- [ ] Fast enough for developer workflow (<5 seconds)

## Steps
1. Install pre-commit: `pip install pre-commit`
2. Create `.pre-commit-config.yaml`:
   ```yaml
   repos:
     - repo: local
       hooks:
         - id: smart-secrets-scanner
           name: Smart Secrets Scanner (LLM)
           entry: python scripts/scan_secrets.py
           language: python
           pass_filenames: true
           types: [python, javascript, yaml]
   ```
3. Create `scripts/scan_secrets.py`:
   - Read staged files
   - Call Ollama API with model
   - Parse output for ALERT signals
   - Exit with error code if secrets found
4. Test: `pre-commit run --all-files`
5. Install: `pre-commit install`

## Dependencies
- Task 15: Ollama deployment
- Task 25, 26: Model tested and validated

## Notes
- Performance target: <5 seconds per file
- Consider batching small files
- Option: Run only on changed lines (git diff)
- Whitelist certain files (.env.example, test fixtures)
- Integration with GitGuardian/TruffleHog for comparison
