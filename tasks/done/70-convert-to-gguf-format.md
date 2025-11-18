# Task 70: Convert Merged Model to GGUF Format

**Status:** Done  
**Created:** 2025-11-18  
**Completed:** 2025-11-18  

## Prerequisites (Completed)

✅ **Task 69**: Adapter merged  
✅ **Task 68**: LoRA adapter trained  
✅ **Task 64**: Environment verified  
✅ **Task 63**: llama-cpp-python bridge built  
✅ **Task 39**: GGUF conversion script created  

## Objective

Convert the merged fine-tuned model to GGUF format for compatibility with Ollama and other GGUF-compatible inference engines. This enables efficient deployment and local execution.

## Requirements

- Merged model in safetensors format
- llama.cpp tools compiled
- GGUF conversion script
- Disk space for quantized model (~7-14GB)

## Implementation

### 1. Install Required Dependencies

```bash
# Activate environment
source ~/ml_env/bin/activate

# Install sentencepiece for Llama tokenizer
pip install sentencepiece protobuf
```

### 2. Verify Merged Model

```bash
# Check merged model files
ls -la models/merged/smart-secrets-scanner/

# Verify model can be loaded
python -c "
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('models/merged/smart-secrets-scanner/')
print(f'✅ Model loaded: {model.num_parameters():,} parameters')
"
```

### 3. Execute GGUF Conversion

```bash
# Convert to Q4_K_M quantization (recommended balance)
python scripts/convert_to_gguf.py --quant Q4_K_M --force
```

Alternative quantization options:
- `Q4_0`: Faster, less accurate (~7GB)
- `Q4_K_M`: Balanced quality/speed (~8GB) - **Recommended**
- `Q5_K_M`: Higher quality, slower (~10GB)
- `Q8_0`: High quality, largest (~14GB)

### 4. Monitor Conversion Process

The conversion will:
- Load merged model from safetensors
- Apply quantization algorithm
- Save in GGUF format
- Show progress and file size

Expected output: `models/fine-tuned/gguf/smart-secrets-scanner.gguf`

### 5. Verify GGUF File

```bash
# Check GGUF file creation
ls -la models/fine-tuned/gguf/smart-secrets-scanner.gguf

# Verify file integrity
file models/fine-tuned/gguf/smart-secrets-scanner.gguf

# Expected: GGUF model file, version 3
```

### 6. Test GGUF Loading

```bash
# Test with llama.cpp python bindings
python - <<'PY'
from llama_cpp import Llama

# Load GGUF model
model = Llama(
    model_path="models/fine-tuned/gguf/smart-secrets-scanner.gguf",
    n_ctx=2048,  # Context length
    n_threads=8  # CPU threads
)

print("✅ GGUF model loaded successfully")
print(f"Model metadata: {model.metadata}")

# Quick inference test
response = model("Analyze this code: API_KEY = 'sk-1234567890abcdef'", max_tokens=100)
print(f"Response: {response['choices'][0]['text']}")
PY
```

## Technical Details

### GGUF Format Benefits

- **Quantization**: Reduces model size by 75% (Q4_K_M)
- **Cross-Platform**: Works on CPU/GPU, multiple OS
- **Ollama Compatible**: Direct import to Ollama
- **Efficient Inference**: Optimized for speed vs accuracy tradeoff

### Quantization Types

| Quantization | Size | Quality | Speed | Use Case |
|-------------|------|---------|-------|----------|
| Q4_0 | ~7GB | Good | Fast | Quick testing |
| Q4_K_M | ~8GB | Better | Fast | **Recommended** |
| Q5_K_M | ~10GB | Best | Medium | High accuracy |
| Q8_0 | ~14GB | Excellent | Slow | Maximum quality |

### Conversion Process

1. **Model Loading**: Safetensors → PyTorch tensors
2. **Quantization**: FP16 → 4-bit integers with K-means clustering
3. **Metadata Addition**: Model config, tokenizer, special tokens
4. **GGUF Encoding**: Custom binary format with compression

## Troubleshooting

### Conversion Failures

```bash
# Check llama.cpp installation
which llama.cpp/convert_hf_to_gguf.py

# Verify model path
ls -la models/merged/smart-secrets-scanner/

# Check disk space
df -h models/fine-tuned/gguf/
```

### Memory Issues

```bash
# Monitor memory during conversion
free -h
nvidia-smi

# Use CPU-only conversion if GPU memory limited
export CUDA_VISIBLE_DEVICES=""
python scripts/convert_to_gguf.py --quant Q4_K_M
```

### Import Errors

```bash
# Reinstall llama-cpp-python
pip uninstall llama-cpp-python -y
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# Test import
python -c "from llama_cpp import Llama; print('Import successful')"
```

### GGUF Corruption

```bash
# Validate GGUF file
python -c "
import gguf
reader = gguf.GGUFReader('models/fine-tuned/gguf/smart-secrets-scanner.gguf')
print('GGUF file valid')
print('Metadata:', reader.metadata)
"
```

## Quality Verification

```bash
# Compare quantized vs full-precision responses
python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_cpp import Llama

# Load full-precision model
full_model = AutoModelForCausalLM.from_pretrained('models/merged/smart-secrets-scanner/')
tokenizer = AutoTokenizer.from_pretrained('models/merged/smart-secrets-scanner/')

# Load GGUF model
gguf_model = Llama('models/fine-tuned/gguf/smart-secrets-scanner.gguf')

test_input = "Check for secrets: password = 'admin123'"

# Full precision response
inputs = tokenizer(test_input, return_tensors='pt')
full_output = full_model.generate(**inputs, max_length=50, do_sample=False)
full_response = tokenizer.decode(full_output[0], skip_special_tokens=True)

# GGUF response
gguf_response = gguf_model(test_input, max_tokens=50)['choices'][0]['text']

print(f"Input: {test_input}")
print(f"Full precision: {full_response}")
print(f"GGUF Q4_K_M: {gguf_response}")
PY
```

## Outcome

✅ Model converted to GGUF format  
✅ Quantization applied (Q4_K_M)  
✅ File integrity verified  
✅ Ready for Ollama deployment  

## Related Tasks

- Task 39: GGUF conversion script (foundation)
- Task 69: Model merging (prerequisite)
- Task 40: Create Modelfile (next step)
- Task 14: Test Ollama deployment (validation)</content>
<parameter name="filePath">c:\Users\RICHFREM\source\repos\Smart-Secrets-Scanner\tasks\done\70-convert-to-gguf-format.md