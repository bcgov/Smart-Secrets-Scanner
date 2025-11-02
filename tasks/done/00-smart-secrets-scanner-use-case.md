# Task: Define and Validate Smart Secrets Scanner Use Case

**Status: Done**

## Prerequisites

This was the foundational task - no prerequisites needed.

## Description
Define the use case and requirements for a Smart Secrets Scanner - an ML-powered pre-commit hook that intelligently detects real secrets while minimizing false positives from documentation, tests, and placeholder values.

## Problem Statement
Current regex-based secret scanners (e.g., git-secrets, detect-secrets, gitleaks) generate excessive false positives because they cannot understand context. This leads to:
- Alert fatigue from flagging documentation examples, test fixtures, and placeholders
- Developers bypassing or ignoring scanner warnings
- Real secrets potentially being missed in the noise

## Proposed Solution
Fine-tune a small LLM to analyze code lines and classify them as:
1. **REAL SECRET** - Actual credentials that should not be committed
2. **FALSE POSITIVE** - Safe patterns (placeholders, docs, env var usage, test data)
3. **SUSPICIOUS** - Requires human review

The model will understand:
- Context clues (file type, variable names, comments)
- Placeholder patterns ("your-key-here", "example", "fake", "test")
- Safe patterns (environment variable reads, config templates)
- Secret formats (AWS keys, GitHub tokens, API keys, passwords)

## Dataset Characteristics
- **Size:** 100-150 examples (sufficient for LoRA fine-tuning)
- **Format:** JSONL with instruction/input/output structure
- **Balance:** 50% real secrets, 50% false positives
- **Secret Types:** AWS, GitHub, Stripe, database passwords, API keys, JWT tokens
- **False Positive Types:** Documentation, test fixtures, placeholders, env var usage

## Training Data Sources
1. Real secret patterns from security research and breach databases (anonymized)
2. Common placeholder patterns from popular frameworks and documentation
3. Test fixture examples from open-source projects
4. Environment variable usage patterns
5. Configuration file templates

## Model Requirements
- **Base Model:** Small instruction-tuned model (e.g., Llama 3 8B, Qwen 7B)
- **Fine-Tuning Method:** LoRA adapters for efficiency
- **Deployment:** GGUF format for local pre-commit hook usage
- **Inference Speed:** < 1 second per file for practical use
- **Accuracy Target:** > 90% precision, > 85% recall on validation set

## Integration Plan
1. Fine-tune model using ML-Env-CUDA13 GPU environment
2. Export to GGUF format for efficient local inference
3. Create pre-commit hook script that:
   - Reads staged files
   - Runs inference using llama.cpp or Ollama
   - Reports findings with severity levels
   - Allows manual override for false positives
4. Package as installable pre-commit hook

## Success Metrics
- Reduce false positive rate by > 70% compared to regex scanners
- Maintain or improve detection of real secrets
- Inference time < 1 second per file on standard hardware
- Positive developer feedback on reduced alert fatigue

## Next Steps
1. ✅ Document use case and requirements (this document)
2. ⏳ Generate initial training dataset (100-150 examples)
3. ⏳ Create dataset in proper format (JSONL with Alpaca/ChatML template)
4. ⏳ Fine-tune model using project workflow
5. ⏳ Evaluate model performance on test set
6. ⏳ Create pre-commit hook integration
7. ⏳ Test with real repositories
8. ⏳ Document usage and deployment

## Dependencies
- ML-Env-CUDA13 environment setup (Task 04)
- Fine-tuning dependencies installed (Task 05)
- Base model downloaded (Task 06)

## Resources
- Common secret patterns: https://github.com/trufflesecurity/trufflehog
- Pre-commit hooks: https://pre-commit.com/
- Example regex scanners for comparison benchmarking

## Notes
This use case is ideal for demonstrating the value of fine-tuning because:
- Clear, measurable improvement over existing regex-based solutions
- Immediately practical for development teams
- Small dataset requirement (100-150 examples)
- Fast inference requirement validates GGUF export workflow
- On-premises deployment requirement aligns with ADR 0005
