# Task 71: Create Modelfile and Test with Ollama

**Status:** Done  
**Created:** 2025-11-18  
**Completed:** 2025-11-18  

## Prerequisites (Completed)

✅ **Task 70**: GGUF conversion completed  
✅ **Task 64**: Environment verified  
✅ **Task 40**: Modelfile script created  
✅ Ollama installed and running  

## Objective

Create an Ollama-compatible Modelfile for the GGUF model and test local inference to ensure the model works correctly in the Ollama environment before deployment.

## Requirements

- GGUF model file
- Modelfile generation script
- Ollama service running
- Model import and testing

## Implementation

### 1. Generate Modelfile Automatically

```bash
# Run the Modelfile generator script
python scripts/create_modelfile.py
```

This creates `Modelfile` with:
- Auto-detected GGUF path
- Llama 3.1 template
- Smart-Secrets-Scanner system prompt
- Optimized parameters

### 2. Review Generated Modelfile

```bash
# Check Modelfile contents
cat Modelfile
```

Expected structure:
```
FROM ./models/fine-tuned/gguf/smart-secrets-scanner.gguf

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""

SYSTEM """You are a specialized code security analyzer..."""

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
```

### 3. Import Model to Ollama

```bash
# Create the model in Ollama
ollama create smart-secrets-scanner -f Modelfile
```

### 4. Test Basic Functionality

```bash
# Start interactive session
ollama run smart-secrets-scanner
```

Then test prompts:
```
>>> Analyze this code for secrets: API_KEY = 'sk-1234567890abcdef'
>>> What types of secrets should I look for in code?
>>> Explain how to securely handle API keys
```

### 5. Test Dual-Mode Capability

**Mode 1 - Conversational Mode:**
```bash
>>> Who is the Smart-Secrets-Scanner?
>>> Explain how API keys work
```

**Mode 2 - Structured Analysis Mode:**
```bash
>>> {"task_type": "secret_scan", "code_snippet": "const API_KEY = 'sk-1234567890abcdef'; const DB_PASS = 'admin123';", "analysis_type": "comprehensive"}
```

### 6. Performance Verification

```bash
# Test response time and quality
time ollama run smart-secrets-scanner "Check: password = 'secret123'" --format json
```

## Technical Details

### Modelfile Components

- **FROM**: Path to GGUF file
- **TEMPLATE**: Chat format template (Llama 3.1 style)
- **SYSTEM**: Specialized system prompt for secret detection
- **PARAMETER**: Stop tokens and generation parameters

### System Prompt Content

```
You are a specialized code security analyzer trained to detect accidental hardcoded secrets (API keys, tokens, passwords, etc.) in source code.

Your task is to scan code snippets and identify potential security risks such as:
- API keys (AWS, Stripe, OpenAI, etc.)
- Authentication tokens (GitHub, JWT, Bearer tokens)
- Database credentials
- Private keys and certificates
- Passwords and secrets

For each finding, respond with "ALERT: [type of secret] detected" and explain the risk.
For safe code (environment variables, test data, placeholders), respond "No secrets detected" or "Safe pattern".

Be precise and minimize false positives while catching real security issues.
```

### Ollama Integration

- **Model Name**: `smart-secrets-scanner`
- **Base Architecture**: Llama 3.1 8B fine-tuned
- **Quantization**: Q4_K_M (GGUF)
- **Context Window**: 2048 tokens (configurable)

## Troubleshooting

### Modelfile Generation Issues

```bash
# Check GGUF file exists
ls -la models/fine-tuned/gguf/smart-secrets-scanner.gguf

# Regenerate Modelfile
rm Modelfile
python scripts/create_modelfile.py --force
```

### Ollama Import Failures

```bash
# Check Ollama service
ollama serve

# Verify Modelfile syntax
ollama create smart-secrets-scanner -f Modelfile --dry-run

# Check model list
ollama list
```

### Inference Problems

```bash
# Test with verbose output
ollama run smart-secrets-scanner --verbose "Test prompt"

# Check model loading
ollama show smart-secrets-scanner
```

### Response Quality Issues

```bash
# Compare with Python inference
python scripts/inference.py --model-type gguf --input "Test: API_KEY = 'sk-123'"

# Adjust parameters in Modelfile
echo "PARAMETER temperature 0.1" >> Modelfile
echo "PARAMETER top_p 0.9" >> Modelfile
ollama create smart-secrets-scanner -f Modelfile
```

## Quality Assurance Tests

### Functional Testing

```bash
# Test various secret types
ollama run smart-secrets-scanner << 'EOF'
Analyze: aws_access_key = 'AKIAIOSFODNN7EXAMPLE'
EOF

ollama run smart-secrets-scanner << 'EOF'
Analyze: github_token = 'ghp_1234567890abcdef'
EOF

ollama run smart-secrets-scanner << 'EOF'
Analyze: db_password = os.getenv('DB_PASS')
EOF
```

### Performance Benchmarking

```bash
# Measure response times
for i in {1..5}; do
  time ollama run smart-secrets-scanner "API_KEY = 'test'" --format raw
done
```

### Accuracy Validation

```bash
# Test with known good/bad examples
ollama run smart-secrets-scanner << 'EOF'
{"task_type": "secret_scan", "code_snippet": "const key = 'sk-1234567890abcdef';", "analysis_type": "comprehensive"}
EOF
```

## Outcome

✅ Modelfile generated and configured  
✅ Model imported to Ollama successfully  
✅ Local inference tested and functional  
✅ Dual-mode capability verified  
✅ Ready for deployment and distribution  

## Related Tasks

- Task 40: Modelfile script creation (foundation)
- Task 70: GGUF conversion (prerequisite)
- Task 14: Ollama deployment testing (validation)
- Task 16: Upload to Hugging Face (next deployment step)</content>
<parameter name="filePath">c:\Users\RICHFREM\source\repos\Smart-Secrets-Scanner\tasks\done\71-create-modelfile-test-ollama.md