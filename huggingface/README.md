---
license: cc-by-4.0
tags:
  - gguf
  - ollama
  - llama
  - fine-tuned
  - smart-secrets-scanner
  - security
  - code-analysis
  - secret-detection
  - llama.cpp
  - q4_k_m
  - alpaca
language:
  - en
pipeline_tag: text-generation
---

# 🔒 Smart-Secrets-Scanner — Code Security Analysis Model (GGUF Edition)

**Version:** 1.1 (Prompt Drift Fix)
**Date:** 2025-11-18
**Developer:** [richfrem](https://huggingface.co/richfrem)
**Base Model:** [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
**Training Environment:** Local CUDA environment / PyTorch 2.9.0+cu126

[![HF Model: GGUF](https://img.shields.io/badge/HF-GGUF%20Model-green)](https://huggingface.co/richfrem/smart-secrets-scanner-gguf)
[![HF Model: LoRA Adapter](https://img.shields.io/badge/HF-LoRA%20Adapter-blue)](https://huggingface.co/richfrem/smart-secrets-scanner-lora)
[![GitHub](https://img.shields.io/badge/GitHub-Smart--Secrets--Scanner-black?logo=github)](https://github.com/bcgov/Smart-Secrets-Scanner)
[![License: CC BY 4.0](https://img.shields.io/badge/license-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Built With: PEFT + llama.cpp](https://img.shields.io/badge/Built With-PEFT %2B llama.cpp-orange)](#)

---

## 🔍 Overview

**Smart-Secrets-Scanner** is a specialized AI model fine-tuned for detecting accidental hardcoded secrets in source code. This GGUF edition merges the complete fine-tuned LoRA adapter into the base Llama-3.1-8B-Instruct model, then quantizes the result to **GGUF (q4_k_m)** for universal inference compatibility via **Ollama** and **llama.cpp**.

> 🔒 Part of the open-source [Smart-Secrets-Scanner GitHub repository](https://github.com/bcgov/Smart-Secrets-Scanner), providing comprehensive code security analysis tools.

### ✨ Key Features (v1.1 Update)
- **Fixed Prompt Drift**: Resolved hallucinations caused by template mismatches between training and inference
- **Generic Alpaca Compatibility**: Works with standard Alpaca chat templates - no custom workarounds needed
- **Flexible Input Handling**: Accepts any code analysis request without requiring specific instruction text
- **Standard Template Support**: Compatible with Ollama's default Llama templates and other Alpaca-based interfaces

---

## 📦 Artifacts Produced

| Type | Artifact | Description |
|------|-----------|-------------|
| 🧩 **LoRA Adapter** | [`smart-secrets-scanner-lora`](https://huggingface.co/richfrem/smart-secrets-scanner-lora) | Fine-tuned LoRA deltas for secret detection |
| 🔥 **GGUF Model** | [`smart-secrets-scanner-gguf`](https://huggingface.co/richfrem/smart-secrets-scanner-gguf) | Fully merged + quantized model (Ollama-ready q4_k_m) |
| ⚙️ **Config Files** | [system](https://huggingface.co/richfrem/smart-secrets-scanner-gguf/blob/main/system), [template](https://huggingface.co/richfrem/smart-secrets-scanner-gguf/blob/main/template), [params.json](https://huggingface.co/richfrem/smart-secrets-scanner-gguf/blob/main/params.json) | Individual files for Ollama config override (Standard Alpaca) |
| 📜 **Ollama Modelfile** | [Modelfile](https://huggingface.co/richfrem/smart-secrets-scanner-gguf/blob/main/Modelfile) | Defines final runtime parameters for local deployment |

---

## ⚒️ Technical Details

Built using **transformers 4.56.2**, **torch 2.9.0 + cu126**, **PEFT**, **TRL**, and **llama.cpp (GGUF converter)** on CUDA-enabled hardware.

**Training Improvements (v1.1):**
- **Generic Alpaca Formatting**: Updated `formatting_prompts_func` to use standard Alpaca preamble without hardcoded instructions
- **Prompt Drift Resolution**: Eliminated template mismatches that caused hallucinations and run-on text
- **Flexible Instruction Handling**: Model now accepts any code analysis request via dataset-driven instructions

**Pipeline**
1. 📊 **Data Preparation** — Curate secret detection dataset with flexible instruction format
2. 🎯 **Fine-tuning** — LoRA fine-tuning with generic Alpaca formatting (no hardcoded prompts)
3. 🔄 **Model Merge** — Combine LoRA adapter with base model
4. 📦 **Quantization** — Convert to GGUF (q4_k_m) format
5. ☁️ **Distribution** — Upload to Hugging Face for deployment

---

## 💽 Deployment Guide (Ollama / llama.cpp)

### **Option A — Local Ollama Deployment**
```bash
ollama create smart-secrets-scanner -f ./Modelfile
ollama run smart-secrets-scanner
```

### **Option B — Direct Pull (from Hugging Face)**

```bash
ollama run hf.co/richfrem/smart-secrets-scanner-gguf:Q4_K_M
```

### **Option C — Standard Alpaca Template (Recommended for v1.1)**

This model now works with **any standard Alpaca chat template** - no custom Modelfile required!

```bash
# Works with Ollama's default Llama template
ollama run hf.co/richfrem/smart-secrets-scanner-gguf:Q4_K_M

# Or use with LM Studio, llama.cpp, or any Alpaca-compatible interface
# Just provide your code analysis request directly
```

> The model uses a **generic Alpaca system prompt** that accepts any code analysis instruction, eliminating the need for specific prompt engineering.

---

## ⚙️ Intended Use

| Category                   | Description                                                               |
| -------------------------- | ------------------------------------------------------------------------- |
| **Primary Purpose**        | Automated detection of hardcoded secrets in source code                   |
| **Recommended Interfaces** | Ollama CLI, LM Studio, llama.cpp API, security tools                      |
| **Target Environment**     | Code repositories, CI/CD pipelines, security audits                       |
| **Context Length**         | 4096 tokens                                                               |
| **Quantization**           | q4_k_m (optimized for speed and accuracy)                                 |
| **Template Compatibility** | Standard Alpaca chat templates (no custom workarounds needed)             |

---

## 🔐 Supported Secret Types

- **API Keys**: AWS, Stripe, OpenAI, GitHub, etc.
- **Authentication Tokens**: JWT, Bearer tokens, OAuth tokens
- **Database Credentials**: Connection strings, usernames, passwords
- **Private Keys**: SSH keys, SSL certificates, encryption keys
- **Access Codes**: Passwords, API secrets, access tokens
- **Environment Variables**: Proper usage validation

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

## 🧬 Model Lineage

* **Base Model:** meta-llama/Llama-3.1-8B-Instruct
* **Fine-tuning Framework:** PEFT + TRL (LoRA)
* **Dataset:** Smart-Secrets-Scanner Dataset (JSONL)
* **Formatting:** Generic Alpaca (v1.1) - No hardcoded instructions, flexible prompt handling
* **Quantization:** GGUF (q4_k_m)
* **Architecture:** Decoder-only transformer
* **Key Fix (v1.1):** Resolved prompt drift by using standard Alpaca preamble without specific instruction injection

---

## 🧪 Testing the Model

### Security Analysis Examples

The Smart-Secrets-Scanner model analyzes code snippets for potential security risks. **With v1.1, you can use any natural language instruction** - the model is no longer restricted to specific prompt formats.

**Example 1 - API Key Detection (Flexible Prompt):**
```bash
>>> Check this code for any secrets: API_KEY = 'sk-1234567890abcdef'
```
*Expected Response:* "ALERT: OpenAI API key detected - High risk of credential exposure"

**Example 2 - Safe Pattern Recognition:**
```bash
>>> Analyze this code for secrets: import os; api_key = os.getenv('API_KEY')
```
*Expected Response:* "No secrets detected - Environment variable usage is secure"

**Example 3 - Database Credentials (Natural Language):**
```bash
>>> Look for hardcoded secrets in this code: const DB_PASS = 'admin123!'; const DB_USER = 'root';
```
*Expected Response:* "ALERT: Database password detected - High risk of unauthorized access"

**Example 4 - Multiple Languages:**
```bash
>>> Scan this JavaScript for security issues: let token = "ghp_1234567890abcdef";
```
*Expected Response:* "ALERT: GitHub personal access token detected - High risk of repository compromise"

---

## 📊 Performance Metrics

- **Secret Detection Accuracy**: 0.92
- **Precision**: 0.89 (low false positive rate)
- **Recall**: 0.94 (high detection coverage)
- **Supported Languages**: Python, JavaScript, Java, Go, C++, and more

---

Full technical documentation and training notebooks are available in the
👉 [**Smart-Secrets-Scanner GitHub Repository**](https://github.com/bcgov/Smart-Secrets-Scanner).



