# Task 55: Compare local GGUF outputs vs hosted model

Status: backlog

Description

Create a reproducible comparison between local GGUF/ollama model outputs and the hosted Hugging Face model that the Modelfile references. The goal is to produce a deterministic, repeatable test harness that saves prompts, tokenization, raw outputs and token ids for side-by-side diff and investigation.

Acceptance criteria

- A script `scripts/compare_models.py` (or similar) exists that:
  - Accepts a small set of prompts (file or directory) and generation params.
  - Runs inference locally (via `python scripts/inference.py --model <path>` or `ollama run`).
  - Calls the Hugging Face Inference API (configurable model and token) with matching generation parameters where supported.
  - Saves tokenizer dumps, raw responses, and token ids to `outputs/compare/<run-timestamp>/`.
  - Produces a small markdown report summarizing differences (counts of changed outputs, sample diffs).
- A short README section or `tasks/in-progress/55-compare-gguf-vs-hosted.md` describing how to run the comparison locally (WSL and PowerShell examples).

Notes / Implementation hints

- Use fixed seeds and temperature=0 for deterministic outputs where possible.
- Include exact prompt wrapper used by `scripts/inference.py` so prompts are identical.
- Keep HF calls rate-limited and respect API quotas; support a `--dry-run` mode that only tokenizes and does not call HF.
- Ensure API keys are read from env `HF_TOKEN` to avoid committing secrets.
