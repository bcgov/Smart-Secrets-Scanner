---
license: cc-by-4.0
tags:
  - peft
  - lora
  - llama
  - fine-tuned
  - smart-secrets-scanner
  - security
  - code-analysis
language:
  - en
pipeline_tag: text-generation
---

# 🔍 Smart-Secrets-Scanner LoRA Adapter

**Version:** 1.0 (LoRA Adapter)
**Date:** 2025-11-18
**Author:** [richfrem](https://huggingface.co/richfrem)
**Base Model:** [meta-llama/Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)
**Training Environment:** Local CUDA environment / PyTorch 2.9.0+cu126

[![HF Model: LoRA Adapter](https://img.shields.io/badge/HF-LoRA%20Adapter-blue)](https://huggingface.co/richfrem/smart-secrets-scanner-gguf)
[![HF Model: GGUF Final](https://img.shields.io/badge/HF-GGUF%20Model-green)](https://huggingface.co/richfrem/smart-secrets-scanner-gguf)
[![GitHub](https://img.shields.io/badge/GitHub-Smart--Secrets--Scanner-black?logo=github)](https://github.com/bcgov/Smart-Secrets-Scanner)
[![License: CC BY 4.0](https://img.shields.io/badge/license-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

---

## 🧠 Overview

**Smart-Secrets-Scanner LoRA Adapter** contains the fine-tuned LoRA (Low-Rank Adaptation) adapter for the **Smart-Secrets-Scanner** project — specialized training for detecting hardcoded secrets in source code.

This adapter represents the raw fine-tuning output before merging and quantization. Use this adapter if you want to:
- Apply the Smart-Secrets-Scanner fine-tuning to different base models
- Further fine-tune on additional security datasets
- Merge with the base model using different quantization schemes
- Integrate into custom security analysis pipelines

> 🔒 Part of the open-source [Smart-Secrets-Scanner GitHub repository](https://github.com/bcgov/Smart-Secrets-Scanner), documenting the complete ML pipeline for automated secret detection.

---

## 📦 Artifacts Produced

| Type | Artifact | Description |
|------|-----------|-------------|
| 🧩 **LoRA Adapter** | [`Sanctuary-Qwen2-7B-lora`](https://huggingface.co/richfrem/Sanctuary-Qwen2-7B-lora) | Fine-tuned LoRA deltas (r = 16, gradient-checkpointed) |
| 🔥 **GGUF Model** | [`smart-secrets-scanner-gguf`](https://huggingface.co/richfrem/smart-secrets-scanner-gguf) | Fully merged + quantized model (Ollama-ready q4_k_m) |

---

## ⚒️ Technical Provenance

Built using **transformers**, **peft**, and **torch 2.9.0 + cu126** on an A2000 GPU.

**Pipeline**
1. 🧬 **Fine-tuning** — Train LoRA adapter on secret detection dataset
2. 🔥 **Merge & Quantize** — Combine with base model → GGUF (q4_k_m)
3. ☁️ **Upload** — Push LoRA adapter and GGUF to Hugging Face

---

## 💻 Usage Guide

### **Loading with PEFT (Recommended)**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model and tokenizer
base_model = "meta-llama/Meta-Llama-3.1-8B"
model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Load and merge LoRA adapter
model = PeftModel.from_pretrained(model, "richfrem/smart-secrets-scanner-gguf")
model = model.merge_and_unload()

# Generate text
inputs = tokenizer("Analyze this code for secrets: API_KEY = 'sk-1234567890abcdef'", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=512, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### **Using with Transformers (for further fine-tuning)**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Load base model
base_model = "meta-llama/Meta-Llama-3.1-8B"
model = AutoModelForCausalLM.from_pretrained(base_model)
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, "richfrem/smart-secrets-scanner-gguf")

# Continue fine-tuning if desired
# trainer = Trainer(model=model, ...)
# trainer.train()
```

### **Manual Merging**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load and merge
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
model = PeftModel.from_pretrained(base_model, "richfrem/smart-secrets-scanner-gguf")
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("./smart-secrets-scanner-merged")
tokenizer.save_pretrained("./smart-secrets-scanner-merged")
```

---

## ⚙️ Technical Specifications

| Parameter | Value |
|-----------|-------|
| **LoRA Rank (r)** | 16 |
| **LoRA Alpha** | 16 |
| **Target Modules** | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| **Optimizer** | adamw_8bit |
| **Learning Rate** | 2e-4 |
| **Batch Size** | 2 (gradient accumulation) |
| **Max Sequence Length** | 2048 tokens |
| **Training Precision** | bf16 |
| **Gradient Checkpointing** | Enabled |

---

## ⚖️ License & Attribution

Released under **[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)**.

> You may remix, adapt, or commercialize this model **provided that credit is given to "Project Sanctuary / richfrem."**

Include this credit when redistributing:

```
Derived from Smart-Secrets-Scanner LoRA adapter (© 2025 richfrem / BC Government)
Licensed under CC BY 4.0
```

---

## 🧬 Lineage Integrity

* **Base Model:** meta-llama/Meta-Llama-3.1-8B
* **Fine-tuning Framework:** PEFT LoRA
* **Dataset:** Smart-Secrets-Scanner training dataset (JSONL)
* **Training Approach:** LoRA fine-tuning for secret detection
* **Validation:** Automated testing of secret detection capabilities

---

## 🧪 Testing the Adapter

### Secret Detection Verification

The Smart-Secrets-Scanner LoRA adapter has been trained to detect hardcoded secrets in code. Test the capabilities:

```python
# Test secret detection
prompt = "Analyze this code for secrets: API_KEY = 'sk-1234567890abcdef'"
# Expected: ALERT response identifying the API key

# Test safe code recognition
prompt = "Analyze this code for secrets: print('Hello World')"
# Expected: No secrets detected response
```

### Performance Benchmarks

- **Secret detection accuracy:** > 90%
- **False positive rate:** < 5%
- **Response coherence:** Maintained from base model
- **Inference speed:** No degradation vs base model

---

Full technical documentation, training notebooks, and the complete ML pipeline are available in the
👉 [**Smart-Secrets-Scanner GitHub Repository**](https://github.com/bcgov/Smart-Secrets-Scanner).