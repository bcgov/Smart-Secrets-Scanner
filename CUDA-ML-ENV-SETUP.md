# Smart-Secrets-Scanner: Meta Llama 3.1 (8B) Fine-Tuning Protocol
**Version:** 2.2 (Clarified Llama.cpp Build)

This guide provides the single, authoritative protocol for setting up the environment, preparing the training dataset, executing the full fine-tuning pipeline, and preparing the model for local deployment with Ollama.


---

## Phase 0: One-Time System & Repository Setup

These steps only need to be performed once per machine.

### 1. System Prerequisites (WSL2 & NVIDIA Drivers)

*   **Install WSL2 and Ubuntu:** Ensure you have a functional WSL2 environment with Ubuntu installed.
*   **Install NVIDIA Drivers:** You must have the latest NVIDIA drivers for Windows that support WSL2.
*   **Verify GPU Access:** Open an Ubuntu terminal and run `nvidia-smi`. You must see your GPU details before proceeding.


### 2. Verify Repository Structure

This project's workflow depends on the `llama.cpp` repository for model conversion. It must be located as a **sibling directory** to your `Smart-Secrets-Scanner` folder.

**If the `llama.cpp` directory is missing,** run the following command from your `Smart-Secrets-Scanner` root to clone it into the correct location:

```bash
# Clone llama.cpp into the parent directory
git clone https://github.com/ggerganov/llama.cpp.git ../llama.cpp
```

### 3. Build `llama.cpp` Tools (The "Engine")

This step compiles the core `llama.cpp` C++/CUDA application from source. This creates powerful, machine-optimized command-line executables (like `quantize`) that are used by our Python scripts for heavy-lifting tasks.

**Note:** This is a one-time, long-running compilation process (5-15 minutes). You do not need to repeat it unless you update the `llama.cpp` repository. This build is separate from and not affected by your Python virtual environment (`~/ml_env`)s.

The tools within `llama.cpp` must be compiled using `cmake`. This process builds the executables required for model conversion and quantization. The `GGML_CUDA=ON` flag is crucial as it enables GPU support.

> **Note:** This is a one-time, long-running compilation process (5-15 minutes). You do not need to repeat it unless you update the `llama.cpp` repository.

```bash
# Navigate to the llama.cpp directory from your project root
cd ../llama.cpp

# Step 1: Configure the build with CMake, enabling CUDA support
cmake -B build -DGGML_CUDA=ON

# Step 2: Build the executables using the configuration
cmake --build build --config Release

# (Optional) Verify the build by checking the main executable's version
./build/bin/llama-cli --version

# Return to your project directory
cd ../Smart-Secrets-Scanner
```

### 4. Hugging Face Authentication

Ensure you have a `.env` file in the root of this project (`Smart-Secrets-Scanner`) containing your Hugging Face token. The file should include:

```code
HUGGING_FACE_TOKEN=your_actual_token_here
```

If the `.env` file doesn't exist or is missing the token, create/update it with your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

---

## Phase 1: Project Environment Setup

This phase builds the project's specific Python environment. It can be re-run at any time to create a clean environment.

### 1. Run the All-in-One Setup Script

From your `Smart-Secrets-Scanner` root directory, execute the `setup_cuda_env.py` script.
Note: Run this with sudo as it automatically installs system packages like python3.11 and git-lfs if they are missing.

```bash
sudo python3 scripts/setup_cuda_env.py --staged --recreate
```

This script creates (`~/ml_env`)  and installs all Python dependencies from requirements.txt, including the llama-cpp-python library.

- **Core ML Libraries:** PyTorch 2.9.0+cu126, transformers, peft, accelerate, bitsandbytes, trl, datasets, xformers
- **Model Conversion:** llama-cpp-python with CUDA support
- **System Tools:** Git LFS, CUDA toolkit components
- **Development Tools:** Jupyter, various utility packages

### 2. Activate the Environment

```bash
source ~/ml_env/bin/activate
```

### 3. Build the `llama-cpp-python` "Bridge"
The `llama-cpp-python` package is the Python "bridge" that allows your Python code (like inference.py) to communicate with the GGUF model. We must ensure this bridge is also built with CUDA support.

The `setup_cuda_env.py` script installs a version of this package, but running the command below is a crucial verification step to force-rebuild it with CUDA flags enabled within your activated environment.

```bash
# While your (ml_env) is active:
CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall --no-cache-dir llama-cpp-python --no-deps
```

### 4. Verify the Complete Environment

Run the full suite of verification scripts to confirm everything is perfectly configured.

```bash
# From the Smart-Secrets-Scanner root, with (ml_env) active:
python scripts/test_torch_cuda.py
python scripts/test_xformers.py
python scripts/test_tensorflow.py
python scripts/test_llama_cpp.py
```

**All tests must pass before proceeding.**

---

## Phase 2: Data & Model Training Workflow

Ensure your `(ml_env)` is active for all subsequent commands.

### 1. Prepare Training Dataset

**Note:** The Smart-Secrets-Scanner project uses pre-prepared JSONL datasets for training. The training data should already be available in the `data/processed/` directory as `smart-secrets-scanner-train.jsonl` and `smart-secrets-scanner-val.jsonl`. If you need to create or validate datasets, use the available scripts:

```bash
# Validate existing dataset
python scripts/validate_dataset.py path/to/your/dataset.jsonl
```

### 3. Download the Base Model

Run the download script to download Meta Llama 3.1 (8B) from Hugging Face. This will only download the large model files once (15-30 GB).

```bash
bash scripts/download_model.sh
```

### 4. Fine-Tune the LoRA Adapter

With the data prepared and the base Meta Llama 3.1 (8B) model downloaded, execute the main fine-tuning script. This will apply LoRA/QLoRA adapters to fine-tune the model for secret detection. **This is a long-running process (1-3 hours).**

```bash
python scripts/fine_tune.py
```
The final LoRA adapter will be saved to `models/fine-tuned/smart-secrets-scanner-lora/`.

### 5. Merge the Adapter

Combine the trained LoRA adapter with the base Meta Llama 3.1 (8B) model to create a full, standalone fine-tuned model.

```bash
python scripts/merge_adapter.py
```
The merged model will be saved to `models/merged/smart-secrets-scanner/`.

---

## Phase 3: Deployment Preparation & Verification

### 1. Convert to GGUF Format

**Note:** GGUF conversion script needs to be implemented. The merged model can be converted to GGUF format for Ollama deployment using llama.cpp tools.

```bash
# TODO: Implement convert_to_gguf.py script
# python scripts/convert_to_gguf.py
```


### 2. Deploy to Ollama

**a. Create a `Modelfile` in your project root:**
```
# ===================================================================
# Canonical Modelfile for Smart-Secrets-Scanner (Llama 3.1 8B)
# ===================================================================

# 1. Specifies the local GGUF model file to use as the base.
FROM ./models/gguf/smart-secrets-scanner-Q4_K_M.gguf

# 2. Defines the prompt template for Llama 3 models.
TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

"""

# 3. Sets the system prompt for secret detection.
SYSTEM """You are a Smart Secrets Scanner, an AI assistant specialized in detecting hardcoded secrets and sensitive information in source code. Your task is to analyze code and identify potential security vulnerabilities related to exposed secrets like API keys, passwords, tokens, and other sensitive data."""

# 4. Defines stop tokens to prevent the model from hallucinating extra turns.
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
```

**b. Import and run the model with Ollama:**
```bash
ollama create smart-secrets-scanner -f Modelfile
ollama run smart-secrets-scanner
```

### 3. Verify Model Performance

**a. Quick Inference Test:**
Use the `inference.py` script for a quick spot-check.
```bash
python scripts/inference.py --input "Analyze this code for secrets: api_key = 'sk_live_1234567890abcdef'"
```

**b. (Recommended) Full Evaluation:**
Run a full evaluation against a held-out test set to get objective performance metrics.
```bash
python scripts/evaluate.py
```

**c. (Crucial) Test with Real Code Examples:**
Use the inference script to test the model against real source code files containing potential secrets.
```bash
python scripts/inference.py --file data/raw/example.py
```

