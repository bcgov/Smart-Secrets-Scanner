# Task 74: Download and Test Hugging Face Model

**Status:** Done  
**Created:** 2025-11-18  
**Completed:** 2025-11-18  

## Prerequisites (Completed)

✅ **Task 73**: Model uploaded to Hugging Face  
✅ **Task 72**: Performance verified locally  
✅ **Task 71**: Ollama testing completed  
✅ **Task 15**: Ollama deployment testing framework  

## Objective

Download the model from Hugging Face and perform final end-to-end testing to ensure the upload/download process doesn't corrupt the model and that inference works correctly in the distributed version.

## Requirements

- Hugging Face repository accessible
- Ollama installed and configured
- Model download and import
- Comparative testing vs local version

## Implementation

### 1. Test Direct Hugging Face Access

```bash
# Run model directly from Hugging Face (no download)
ollama run hf.co/richfrem/Smart-Secrets-Scanner-Model:Q4_K_M
```

Test prompts:
```
>>> Analyze this code for secrets: API_KEY = 'sk-1234567890abcdef'
>>> What types of secrets should I look for in code?
>>> {"task_type": "secret_scan", "code_snippet": "const DB_PASS = 'admin123';", "analysis_type": "comprehensive"}
```

### 2. Download Model Files (Optional)

```bash
# Download GGUF file for local testing
huggingface-cli download richfrem/Smart-Secrets-Scanner-Model \
  smart-secrets-scanner.gguf \
  --local-dir downloaded_models \
  --local-dir-use-symlinks False
```

### 3. Create Local Modelfile for Downloaded Model

```bash
# Create Modelfile pointing to downloaded GGUF
cat > Modelfile_HF << 'EOF'
FROM ./downloaded_models/smart-secrets-scanner.gguf

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
EOF
```

### 4. Import Downloaded Model to Ollama

```bash
# Create model from downloaded files
ollama create smart-secrets-scanner-HF -f Modelfile_HF
```

### 5. Comprehensive Testing

```bash
# Test downloaded model
ollama run smart-secrets-scanner-HF
```

Run the same test prompts as with the direct HF version.

### 6. Performance Comparison

```bash
# Compare local vs downloaded model responses
echo "=== Local Model ==="
ollama run smart-secrets-scanner "API_KEY = 'sk-1234567890abcdef'"

echo "=== Downloaded Model ==="
ollama run smart-secrets-scanner-HF "API_KEY = 'sk-1234567890abcdef'"

echo "=== Direct HF Model ==="
ollama run hf.co/richfrem/Smart-Secrets-Scanner-Model:Q4_K_M "API_KEY = 'sk-1234567890abcdef'"
```

## Technical Details

### Testing Scenarios

1. **Direct HF Access**: `hf.co/username/repo:quant`
2. **Downloaded Model**: Local GGUF with custom Modelfile
3. **Local Model**: Original Modelfile from training

### Quality Assurance Checks

- **Response Consistency**: All three methods should give similar results
- **Performance**: Response time and quality maintained
- **Functionality**: Both conversational and structured analysis modes work
- **Accuracy**: Secret detection capability preserved

### Distribution Methods

| Method | Pros | Cons | Use Case |
|--------|------|------|----------|
| Direct HF | No download, always latest | Requires internet | Quick testing |
| Downloaded | Offline capable, faster | Manual updates | Production deployment |
| Local | Full control, fastest | Storage intensive | Development |

## Troubleshooting

### Direct HF Access Issues

```bash
# Check repository accessibility
curl https://huggingface.co/richfrem/Smart-Secrets-Scanner-Model/resolve/main/smart-secrets-scanner.gguf

# Verify quantization tag
ollama show hf.co/richfrem/Smart-Secrets-Scanner-Model:Q4_K_M
```

### Download Problems

```bash
# Check available space
df -h

# Download with progress
huggingface-cli download richfrem/Smart-Secrets-Scanner-Model \
  --local-dir downloaded_models \
  --local-dir-use-symlinks False \
  --resume-download
```

### Import Failures

```bash
# Validate Modelfile syntax
ollama create smart-secrets-scanner-HF -f Modelfile_HF --dry-run

# Check GGUF file integrity
ls -la downloaded_models/smart-secrets-scanner.gguf
```

### Performance Differences

```bash
# Compare response times
time ollama run smart-secrets-scanner "test prompt" --format raw
time ollama run smart-secrets-scanner-HF "test prompt" --format raw
time ollama run hf.co/richfrem/Smart-Secrets-Scanner-Model:Q4_K_M "test prompt" --format raw
```

## Final Validation Tests

### Comprehensive Test Suite

```bash
# Test various secret types
TEST_PROMPTS=(
    "API_KEY = 'sk-1234567890abcdef'"
    "password = 'admin123'"
    "github_token = 'ghp_abcd1234efgh5678'"
    "aws_secret = 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'"
    "db_connection = 'postgresql://user:password@localhost/db'"
    "const key = process.env.API_KEY"  # Safe pattern
    "placeholder = 'YOUR_API_KEY_HERE'"  # Safe pattern
)

for prompt in "${TEST_PROMPTS[@]}"; do
    echo "=== Testing: $prompt ==="
    ollama run hf.co/richfrem/Smart-Secrets-Scanner-Model:Q4_K_M "$prompt"
    echo
done
```

### Structured Analysis Testing

```bash
# Test JSON-structured input
ollama run hf.co/richfrem/Smart-Secrets-Scanner-Model:Q4_K_M << 'EOF'
{"task_type": "secret_scan", "code_snippet": "const API_KEY = 'sk-1234567890abcdef'; const DB_PASS = 'admin123'; const SAFE_VAR = process.env.NODE_ENV;", "analysis_type": "comprehensive"}
EOF
```

## Outcome

✅ Hugging Face model download tested  
✅ All access methods functional  
✅ Performance consistency verified  
✅ Model distribution ready  
✅ End-to-end workflow complete  

## Related Tasks

- Task 15: Ollama deployment testing (foundation)
- Task 73: HF upload (prerequisite)
- Task 55: Compare GGUF vs hosted (validation)
- Task 14: Local Ollama testing (complementary)

## Completion Summary

🎉 **Smart-Secrets-Scanner Development Complete!**

- ✅ **Phase 0**: Environment setup (Tasks 58-64)
- ✅ **Phase 1**: Repository and tools setup (Tasks 58-64)  
- ✅ **Phase 2**: Data & model forging (Tasks 65-69)
- ✅ **Phase 3**: Deployment & verification (Tasks 70-74)

**Model Available At:**
- **Direct**: `ollama run hf.co/richfrem/Smart-Secrets-Scanner-Model:Q4_K_M`
- **Repository**: https://huggingface.co/richfrem/Smart-Secrets-Scanner-Model
- **Local**: `ollama run smart-secrets-scanner`</content>
<parameter name="filePath">c:\Users\RICHFREM\source\repos\Smart-Secrets-Scanner\tasks\done\74-download-test-hugging-face-model.md