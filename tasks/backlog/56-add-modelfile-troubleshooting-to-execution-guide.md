# Task 56: Add Modelfile troubleshooting section to EXECUTION_GUIDE.md

Status: backlog

Description

Add a dedicated troubleshooting subsection to `EXECUTION_GUIDE.md` documenting steps to diagnose differences between local GGUF model runs and hosted model deployments referenced by Modelfile. Include exact commands/examples for:

- `ollama run` vs `curl` to Hugging Face Inference API (with placeholders for tokens)
- How to verify Modelfile `FROM` entries and model revisions
- Generation parameter parity (temperature, top_p, seed, max tokens)
- Tokenizer and special token checks
- Merge/LoRA adapter validation steps

Acceptance criteria

- `EXECUTION_GUIDE.md` contains a new subsection "Modelfile troubleshooting" with reproducible commands and a checklist as above.
- Cross-reference to `README.md` note added earlier.
- Example commands for both WSL/Bash and Windows PowerShell are provided.
