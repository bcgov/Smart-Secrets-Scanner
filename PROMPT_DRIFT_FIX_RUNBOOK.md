# Final Verified Runbook: Fixing Prompt Drift in Smart-Secrets-Scanner

This runbook documents the root cause and resolution of model hallucinations caused by "Prompt Drift"—where the inference template deviates from the training data structure. The fix involved aligning the Ollama deployment template with the exact Alpaca training format used during fine-tuning.

## Root Cause
The model was "overfit" to the specific Alpaca instruction format used in `fine_tune.py`. During inference, the template mismatch caused confusion, leading to hallucinations like "### Instruction:" artifacts and run-on text instead of consistent "ALERT" responses.

## Resolution Steps

### 1. Updated create_modelfile.py to v5.0
**Change:** Hardcoded the specific Alpaca training instruction ("Analyze the following code...") into the TEMPLATE variable, ensuring the model sees the exact patterns it memorized during fine-tuning.

**Ref:** scripts/create_modelfile.py (updated to v5 logic).

### 2. Regenerated Configuration Files
This script detects the latest GGUF and writes the Modelfile, system, template, and params.json with the new v5 logic.

```bash
python scripts/create_modelfile.py
```

### 3. Copied GGUF File to Project Root
Since the new Modelfile uses a relative path (FROM smart-secrets-scanner...) for portability, the GGUF file must be in the same directory as the Modelfile during creation.

```bash
cp models/fine-tuned/gguf/smart-secrets-scanner-Q4_K_M.gguf .
```

### 4. Created Local Ollama Model
This compiles the model in your local Ollama registry using the corrected Modelfile.

```bash
ollama create smart-secrets-scanner -f Modelfile
```

### 5. Uploaded Config Files to Hugging Face
We uploaded only the lightweight configuration files. This updates the repo's behavior without requiring a re-upload of the heavy GGUF binary.

```bash
python scripts/upload_to_huggingface.py --repo richfrem/smart-secrets-scanner-gguf --system --template --params
```

### 6. Tested with Code Snippet (Pipe Input)
Verified that the model accepts piped input without generating conversational filler.

```bash
echo "API_KEY = 'sk-1234567890abcdef'" | ollama run smart-secrets-scanner
```

**Result:** ALERT: Hardcoded API key detected

### 7. Interactive Testing
Manual validation of the "No secrets detected" path.

```bash
ollama run smart-secrets-scanner
>>> import os; key = os.getenv('API_KEY')
```

**Result:** No secrets detected

## Key Takeaway
The critical fix was acknowledging that the model had "overfit" to the specific Alpaca instruction format used in `fine_tune.py`. By hardcoding that exact instruction into the inference template, you aligned the deployment environment with the model's internal weights, eliminating the confusion that caused the hallucinations.

## Validation
- Local Ollama model: Consistent ALERT responses for secrets, no hallucinations.
- Hugging Face model: Identical behavior when run via `ollama run hf.co/richfrem/smart-secrets-scanner-gguf:Q4_K_M`.
- Both positive (secrets detected) and negative (no secrets) cases validated.