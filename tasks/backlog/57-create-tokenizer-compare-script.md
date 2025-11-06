# Task 57: Create tokenizer/token-id comparison script

Status: backlog

Description

Create a small utility that compares tokenization outputs between the local tokenizer (used with the base/merged model) and the tokenizer configuration used by the hosted model (if accessible via HF or by downloading tokenizer files). The script should help detect differences in token ids, special tokens, and tokenization boundaries that may explain output divergence.

Acceptance criteria

- `scripts/compare_tokenizers.py` or similar exists and can:
  - Load a local tokenizer from `models/base/<model>` or from a Hugging Face hub path.
  - Tokenize a list of prompts and save token ids and decoded tokens.
  - (Optional) If the hosted tokenizer is accessible via HF, fetch tokenizer files and compare token ids for the same prompts.
  - Output a human-readable diff or CSV showing tokens, ids, and positions where they differ.
- Usage example documented in the script header and in `tasks/backlog/57-create-tokenizer-compare-script.md`.

Notes

- Avoid committing any HF API tokens. Use environment variables for credentials.
- This script complements Task 55 and helps isolate tokenizer-related causes of mismatched outputs.
