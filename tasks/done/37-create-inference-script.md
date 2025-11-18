# Task 37: Create Inference Script

**Status:** Done ✅  
**Priority:** HIGH  
**Created:** 2025-11-01  
**Completed:** 2025-11-18  
**Related to:** Phase 2: Model Testing (Step 6)  
**Depends on:** Task 36 (fine-tuning complete)

## Prerequisites (Completed)

✅ **Task 01-05**: Environment setup (WSL2, NVIDIA, ML-Env-CUDA13, dependencies)  
✅ **Task 36**: Fine-tuning completed and adapter saved  

## Description
Create `scripts/inference.py` - the main Python script for running inference on the fine-tuned LoRA adapter or merged model.

## Requirements
- Load fine-tuned LoRA adapter or merged model
- Run inference on code snippets
- Support different model types (lora, merged)
- Handle input from command line or files
- Output security analysis results

## Acceptance Criteria
- [x] `scripts/inference.py` created and executable
- [x] Loads LoRA adapter from `models/fine-tuned/smart-secrets-scanner-lora/`
- [x] Supports `--model-type lora` and `--model-type merged`
- [x] Accepts `--input` for direct text input
- [x] Generates security analysis output
- [x] Handles errors gracefully with informative messages
- [x] Can be called by `scripts/infer.sh`

## Script Implementation
Create `scripts/inference.py`:

```python
#!/usr/bin/env python3
"""
Inference script for Smart Secrets Scanner
"""
import os
import sys
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_model(model_type, base_model_path, lora_path=None):
    # Implementation for loading model based on type
    pass

def run_inference(model, tokenizer, input_text):
    # Implementation for running inference
    pass

if __name__ == "__main__":
    # Argument parsing and main logic
    pass
```

## Verification Results
- ✅ Successfully loads LoRA adapter
- ✅ Generates output without errors
- ✅ Model responds appropriately to test prompts
- ✅ Ready for merge phase</content>
<parameter name="filePath">c:\Users\RICHFREM\source\repos\Smart-Secrets-Scanner\tasks\done\37-create-inference-script.md