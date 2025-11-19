# Smart-Secrets-Scanner: Canonical CUDA ML Environment & Fine-Tuning Protocol
**Version:** 2.2 (Clarified Llama.cpp Build)

This guide provides the single, authoritative protocol for setting up the environment, forging the training dataset, executing the full fine-tuning pipeline, and preparing the model for local deployment with Ollama.


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

**Quick Check (Recommended First Step):** If you have an existing `~/ml_env` from another compatible project, try this first:

### 1. Activate and Verify Existing Environment

```bash
source ~/ml_env/bin/activate
```

Run the verification scripts to check if your existing environment is compatible:

```bash
python scripts/test_torch_cuda.py
python scripts/test_pytorch.py
python scripts/test_xformers.py
python scripts/test_llama_cpp.py
```

**If all tests pass** → Your existing environment is compatible! Skip to Phase 2.

**If any tests fail** → Run the full Phase 1 setup below.

### Full Setup (Run if verification fails)

### 0. Clear Environment (Optional)

To ensure a completely clean start, you can manually delete the existing `~/ml_env` virtual environment before running the setup script. The setup script with `--recreate` will do this automatically, but this step gives you explicit control.

```bash
# Manually delete the existing environment (optional, as --recreate does this)
deactivate 2>/dev/null || true
rm -rf ~/ml_env
```


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

### 2b. Install Critical CUDA Binaries (Surgical Strike)

Certain low-level libraries like `bitsandbytes`, `triton`, and `xformers` require a specific installation order to link correctly with a CUDA-enabled PyTorch. A standard pip install can often fail or install a CPU-only version.

This "surgical strike" process ensures these critical binaries are installed correctly after your main environment is set up. Execute these commands one by one from your activated `(ml_env)`.

**Pre-flight Check:** Before you begin, confirm that the correct PyTorch is installed. Run this command:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

It should return 2.9.0+cu126 (or the CUDA-enabled build you targeted). If it doesn't, re-run the main setup script (setup_cuda_env.py) and re-check.

The Surgical Installation Protocol (ordered & deterministic)

NOTE: run each line/section sequentially and paste the verification outputs if anything errors. This protocol was validated to work with PyTorch 2.9.0+cu126, resulting in triton 3.5.0 and bitsandbytes 0.48.2 with CUDA support.

# A: confirm env basics (do this first)
```bash
which python
python -V
pip --version
python -c "import torch; print('torch:', torch.__version__, 'cuda_available:', torch.cuda.is_available())"
```

# B: clean slate
```bash
pip uninstall -y bitsandbytes triton xformers || true
pip install --upgrade pip setuptools wheel
```

# C: install Triton 3.1.0 (this will be overridden by xformers to 3.5.0, which is compatible and works)
```bash
pip install --force-reinstall "triton==3.1.0"
```

# Quick verify Triton import
```bash
python - <<'PY'
try:
    import triton
    print("triton OK:", triton.__version__)
except Exception as e:
    print("triton import failed:", repr(e))
    raise
PY
```

# D: diagnostic — show which bitsandbytes wheels pip can see on the extra indexes
```bash
pip index versions bitsandbytes --extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu126
```

# E: install bitsandbytes with CUDA support (use version 0.48.2, which includes CUDA126 native lib)
```bash
pip install --force-reinstall --no-cache-dir bitsandbytes==0.48.2 --no-deps \
  --extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu126
```

# F: install xformers (this will pull triton 3.5.0, which is compatible and provides triton.ops)
```bash
pip install xformers
```

# G: known fsspec/datasets compatibility mitigation (optional)
```bash
pip install "fsspec<=2024.3.1"
```

# H: verification snippet — verifies triton and bitsandbytes plus native libs
```bash
python - <<'PY'
import importlib, pathlib
def try_import(name):
    try:
        m = importlib.import_module(name)
        print(f"{name} imported, ver:", getattr(m,'__version__', None), "file:", getattr(m,'__file__', None))
    except Exception as e:
        print(f"{name} import failed:", repr(e))

try_import('triton')
try_import('bitsandbytes')

# list any native libbitsandbytes files next to the package
try:
    import bitsandbytes as bnb
    p = pathlib.Path(bnb.__file__).parent
    found = False
    for f in p.glob("libbitsandbytes*"):
        print("native lib:", f)
        found = True
    if not found:
        print("no libbitsandbytes native libs found (likely CPU-only install)")
except Exception as e:
    print("bitsandbytes inspect failed:", repr(e))
PY
```

### Troubleshooting: Accelerator Version Conflicts

If you encounter `TypeError: Accelerator.unwrap_model() got an unexpected keyword argument 'keep_torch_compile'` during training initialization, update accelerate to ensure compatibility with the installed transformers version:

```bash
pip install --upgrade accelerate
```

This resolves version mismatches that can occur after the surgical strike installations.

### Troubleshooting: Training Configuration Errors

If you encounter `TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'`, update the config to use the newer argument name:

In `config/training_config.yaml`, change:
```yaml
evaluation_strategy: "steps"
```
To:
```yaml
eval_strategy: "steps"
```

This ensures compatibility with the current transformers version. Also, remove any deprecated arguments like `group_by_length` or `dataloader_persistent_workers` if present.

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
python scripts/test_pytorch.py
python scripts/test_xformers.py
python scripts/test_tensorflow.py
python scripts/test_llama_cpp.py
```

**All tests must pass before proceeding.**

---

## Phase 2: Data & Model Forging Workflow

Ensure your `(ml_env)` is active for all subsequent commands.

### 1. Prepare Your Training Dataset

The training dataset is created directly by the LLM using an innovative approach that eliminates traditional data engineering pipelines (see ADR 0007: LLM-Driven Dataset Creation). This dataset consists of approximately **1000 rows** of high-quality, diverse training examples.

**Dataset Requirements:**
- **Format**: JSONL (JSON Lines) format with one training example per line
- **Size**: Target of 1000+ examples for effective fine-tuning
- **Structure**: Each example contains three fields:
  - `instruction`: The task description for the model
  - `input`: Code snippet to analyze for secrets
  - `output`: Expected analysis result (ALERT or no secrets detected)

**Example Dataset Entry:**
```json
{
  "instruction": "Analyze this code snippet for hardcoded secrets such as API keys, passwords, tokens, and other sensitive information. Respond with ALERT if secrets are found, otherwise respond with no secrets detected.",
  "input": "API_KEY = 'sk-1234567890abcdef'",
  "output": "ALERT: Hardcoded API key detected. This appears to be an OpenAI API key that should be stored securely in environment variables."
}
```

**Dataset Coverage:**
- Multiple programming languages (Python, JavaScript, Java, etc.)
- Various secret types (API keys, passwords, tokens, certificates, database credentials)
- Edge cases (environment variables, test data, placeholders, obfuscated secrets)
- Both positive examples (containing secrets) and negative examples (safe code)

The dataset should be placed in the `data/processed` directory as `smart-secrets-scanner-dataset.jsonl`. This is the **essential first step** before training can begin.

### 2. Validate the Dataset

After creating the dataset, run the validation script to check it for errors.

```bash
python scripts/validate_dataset.py data/processed/smart-secrets-scanner-dataset.jsonl
```

### 3. Download the Base Model

Run the download script. This will only download the large model files once.

```bash
bash scripts/download_model.sh
```

### 4. Fine-Tune the LoRA Adapter

With the data forged and the base model downloaded, execute the optimized fine-tuning script. This script now includes advanced features like structured logging, automatic resume from checkpoints, pre-tokenization for faster starts, and robust error handling. **This is a process that takes 1-3 hours for full datasets (1000+ examples), but ~2 minutes for small test runs (8 examples).**

```bash
python scripts/fine_tune.py
```
The final LoRA adapter will be saved to `models/fine-tuned/smart-secrets-scanner-lora/`.

**Verification:** After completion, verify the adapter is saved correctly by checking the directory contents:
```bash
ls -la models/fine-tuned/smart-secrets-scanner-lora/
```
Ensure `adapter_model.safetensors` and `adapter_config.json` are present. For a quick integrity test, run: (5 mins)
```bash
python scripts/inference.py --input "Test prompt"
```
If it loads and generates output without errors, the adapter is valid.

### 5. Merge the Adapter

Combine the trained adapter with the base model to create a full, standalone fine-tuned model. (about 5 mins)

```bash
python scripts/merge_adapter.py --skip-sanity
```
The merged model will be saved to `models/merged/smart-secrets-scanner/`.

**Verification:** After completion, verify the merged model by testing it:
```bash
python scripts/inference.py --model outputs/merged/smart-secrets-scanner --input "Test prompt"
```
If it loads and generates output without errors, the merged model is valid and ready for GGUF conversion.

---

## Phase 3: Deployment Preparation & Verification

### setup for gguf
Llama-3.1-8B uses SentencePiece tokenizer → convert_hf_to_gguf.py requires the sentencepiece Python package or it dies exactly where you saw it.
Run this right now in your activated (ml_env):

```bash
pip install sentencepiece protobuf
```

### 1.  Convert to GGUF Format

Convert the merged model to the GGUF format required by Ollama.

```bash
python scripts/convert_to_gguf.py --quant Q4_K_M --force
```
The final quantized `.gguf` file will be saved to `models/fine-tuned/gguf/smart-secrets-scanner.gguf`.

next test the gguf model with the following
```bash
python scripts/inference.py --model models/fine-tuned/gguf/smart-secrets-scanner-Q4_K_M.gguf --input "Test prompt"
```

---

### 2. Test gguf file locally with ollama

**2a. Generate Modelfile:**

Run the bulletproof Modelfile generator script:

```bash
python scripts/create_modelfile.py
```

This creates a production-ready Modelfile with auto-detected GGUF path, official Llama-3.1-8B template, full Smart-Secrets-Scanner system prompt, and optimized parameters.

**2b. Import to Ollama:**
```bash
ollama create smart-secrets-scanner -f Modelfile
```

**2c. Run locally in Ollama:**
```bash
ollama run smart-secrets-scanner
```
---

**2d. Test Both Interaction Modes:**

After running `ollama run smart-secrets-scanner`, you can test the model's dual-mode capability:

**Mode 1 - Plain Language Conversational Mode (Default):**
The model responds naturally and helpfully to direct questions and requests.
```bash
>>> Analyze this code for secrets: API_KEY = 'sk-1234567890abcdef'
>>> What types of secrets should I look for in code?
>>> Explain how to securely handle API keys
>>> Who is the Smart-Secrets-Scanner?
```

**Mode 2 - Structured Analysis Mode:**
When provided with code input, the model switches to generating security analysis for secret detection.
```bash
>>> {"task_type": "secret_scan", "code_snippet": "const API_KEY = 'sk-1234567890abcdef'; const DB_PASS = 'admin123';", "analysis_type": "comprehensive"}
```
*Expected Response:* The model outputs a structured analysis identifying potential security risks.

This demonstrates Smart-Secrets-Scanner's ability to handle both human conversation and automated code analysis seamlessly.

---

### 3. Verify Model Performance

**Note:** This section tests the local merged model (created in Phase 2) using Python inference scripts for comprehensive evaluation. For Ollama-based chat testing, see Section 2 above. After uploading to Hugging Face, compare performance with Section 5 (HF download testing).

**3a. Quick Inference Test:**
Use the `inference.py` script for a quick spot-check.
```bash
python scripts/inference.py --input "Analyze this code for secrets: API_KEY = 'sk-1234567890abcdef'"
```

**3b. (Recommended) Full Evaluation:**
Run a full evaluation against a held-out test set to get objective performance metrics.

```bash
pip install evaluate rouge-score
```


```bash
python scripts/evaluate.py --load-in-4bit --test-data data/processed/smart-secrets-scanner-val.jsonl --max-examples 10
```

**3c. Test GGUF Model Locally:**
After creating the GGUF file, test it directly:
```bash
python scripts/inference.py --model models/fine-tuned/gguf/smart-secrets-scanner-Q4_K_M.gguf --input "Test prompt"
```


---

### 4. Upload to Hugging Face

Run the automated upload script to upload the GGUF model, system, template, params.json, and README to your Hugging Face repository:

```bash
#python scripts/upload_to_huggingface.py --repo yourusername/your-repo-name --gguf --system --template --params --readme

python scripts/upload_to_huggingface.py --repo richfrem/smart-secrets-scanner-gguf --gguf --system --template --params --readme

```

Replace `yourusername/your-repo-name` with your actual Hugging Face repository ID (e.g., `richfrem/Smart-Secrets-Scanner-Model`).

The script will:
- Authenticate using your `HUGGING_FACE_TOKEN` from `.env`
- Create the repository if it doesn't exist
- Upload the specified files

After upload, your model will be available at: https://huggingface.co/yourusername/your-repo-name

**Selective Upload Examples:**

```bash
# Upload README only
python scripts/upload_to_huggingface.py --repo richfrem/smart-secrets-scanner-gguf --readme

# Upload GGUF model only
python scripts/upload_to_huggingface.py --repo richfrem/smart-secrets-scanner-gguf --gguf

# Upload merged model only
python scripts/upload_to_huggingface.py --repo richfrem/smart-secrets-scanner-gguf --merged

# Upload config files only (system, template, params.json)
python scripts/upload_to_huggingface.py --repo richfrem/smart-secrets-scanner-gguf --system --template --params

# Upload all (GGUF + configs + README)
python scripts/upload_to_huggingface.py --repo richfrem/smart-secrets-scanner-gguf --gguf --system --template --params --readme
```

---

### 5. download and test hugging face model

**5a. Direct Run from Hugging Face (Recommended):**
Ollama can run the model directly from Hugging Face without downloading it first. This is the most convenient method:

```bash
ollama run hf.co/richfrem/smart-secrets-scanner-gguf:Q4_K_M
```

This command will automatically download and run the model from Hugging Face on-demand.

**5b. Download from Hugging Face:**
If you prefer to download the model files for local verification:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='richfrem/smart-secrets-scanner-gguf', local_dir='huggingface/downloaded_models')"
```

After downloading the model from Hugging Face, test it locally in Ollama to verify the upload/download process didn't corrupt the model and that inference works correctly. Compare performance with the local tests in Section 3 to ensure consistency.

**Note:** The repository includes separate configuration files (system, template, params.json) that Ollama automatically detects and applies when running `ollama run hf.co/richfrem/smart-secrets-scanner-gguf:Q4_K_M`. For local testing, you can create a Modelfile from these files or use them directly.



**5c. (Optional) Create Local Modelfile for Downloaded Model:**
If you prefer to create a local Ollama model from the downloaded files instead of using the direct HF run, create a Modelfile that references the downloaded system, template, and params.json:

```bash
# Create a local Modelfile
cat > Modelfile_HF << 'EOF'
FROM ./huggingface/downloaded_models/smart-secrets-scanner-Q4_K_M.gguf

SYSTEM "$(cat huggingface/downloaded_models/system)"

TEMPLATE "$(cat huggingface/downloaded_models/template)"

PARAMETER $(jq -r 'to_entries[] | "PARAMETER \(.key) \(.value)"' huggingface/downloaded_models/params.json | tr '\n' '\n')
EOF
```
```
FROM ./huggingface/downloaded_models/smart-secrets-scanner-Q4_K_M.gguf

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""

SYSTEM """You are a specialized code security analyzer trained to detect accidental hardcoded secrets (API keys, tokens, passwords, etc.) in source code.

Your task is to scan code snippets and identify potential security risks such as:
- API keys (AWS, Stripe, OpenAI, etc.)
- Authentication tokens (GitHub, JWT, Bearer tokens)
- Database credentials
- Private keys and certificates
- Passwords and secrets

For each finding, respond with "ALERT: [type of secret] detected" and explain the risk.
For safe code (environment variables, test data, placeholders), respond "No secrets detected" or "Safe pattern".

Be precise and minimize false positives while catching real security issues."""

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
```

**5d. (Optional) Import Local Model to Ollama:**
```bash
ollama create smart-secrets-scanner-HF -f Modelfile_HF
ollama run smart-secrets-scanner-HF
```

**5e. Test Inference:**
Then, provide test prompts to verify the model responds correctly, such as: "Analyze this code for secrets: API_KEY = 'sk-1234567890abcdef'".

