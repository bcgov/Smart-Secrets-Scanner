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

[![HF Model: LoRA Adapter](https://img.shields.io/badge/HF-LoRA%20Adapter-blue)](https://huggingface.co/richfrem/smart-secrets-scanner-lora)
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
| 🧩 **LoRA Adapter** | [`smart-secrets-scanner-lora`](https://huggingface.co/richfrem/smart-secrets-scanner-lora) | Fine-tuned LoRA deltas (r = 16, gradient-checkpointed) |
| 🔥 **GGUF Model** | [`smart-secrets-scanner-gguf`](https://huggingface.co/richfrem/smart-secrets-scanner-gguf) | Fully merged + quantized model (Ollama-ready q4_k_m) |

---

## ⚒️ Technical Provenance

Built using **transformers**, **peft**, and **torch 2.9.0 + cu126** on an A2000 GPU.

**Pipeline**
1. 🧬 **Fine-tuning** — Train LoRA adapter on secret detection dataset
2. 🔥 **Merge & Quantize** — Combine with base model → GGUF (q4_k_m)
3. ☁️ **Upload** — Push LoRA adapter and GGUF to Hugging Face

---

## 💻 Usage Guide (Hugging Face)

If you are loading the LoRA adapter directly from the Hub for merging or continued training:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Load the base model
base_model = "meta-llama/Meta-Llama-3.1-8B" 
tokenizer = AutoTokenizer.from_pretrained(base_model)

# 2. Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "richfrem/smart-secrets-scanner-lora")

# 3. Manual Merging example
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./smart-secrets-scanner-merged")
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

## ⚖️ Governance and Source

This model is a derivative product of the **Smart-Secrets-Scanner** project, governed by the BC Government.

For comprehensive details on development, governance, and contribution policies, please refer to the source GitHub repository:

| Document | Link |
| :--- | :--- |
| **GitHub Source** | [bcgov/Smart-Secrets-Scanner](https://github.com/bcgov/Smart-Secrets-Scanner) |
| **License** | [LICENSE](https://github.com/bcgov/Smart-Secrets-Scanner/blob/main/LICENSE) |
| **Code of Conduct** | [CODE_OF_CONDUCT.md](https://github.com/bcgov/Smart-Secrets-Scanner/blob/main/CODE_OF_CONDUCT.md) |
| **Contributing** | [CONTRIBUTING.md](https://github.com/bcgov/Smart-Secrets-Scanner/blob/main/CONTRIBUTING.md) |

---

## ⚖️ License & Attribution

This model is licensed under the **Creative Commons Attribution 4.0 International Public License (CC BY 4.0)**.

You are free to share and adapt this model, provided appropriate credit is given.

**Required Attribution:**

Derived from Smart-Secrets-Scanner (© 2025 richfrem / BC Government)Source: https://github.com/bcgov/Smart-Secrets-ScannerLicensed under CC BY 4.0

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