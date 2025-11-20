# Tests to run with model

After running `ollama run smart-secrets-scanner`, or `ollama run hf.co/richfrem/smart-secrets-scanner-gguf:Q4_K_M` you can test the model's dual-mode capability:

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