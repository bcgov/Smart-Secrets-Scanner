# Task 21: Generate Synthetic Raw Source Code Files

**Status**: Backlog  
**Created**: 2025-11-01  
**Priority**: Medium  
**Assignee**: TBD  

## Prerequisites (Completed)

✅ **Task 20, 47**: JSONL datasets generated (reference for patterns)  
✅ **Task 00**: Use case defined (Smart Secrets Scanner patterns)  

**Optional Enhancement:**  
💡 Could be used to generate more diverse training data  

## Objective

Create realistic synthetic source code files (.py, .js, .yaml, .go, .java) containing secrets and safe patterns. These files serve as:
1. **Source material** for the JSONL training data (raw → processed workflow)
2. **Test inputs** for the fine-tuned model (simulate real pre-commit scanning)
3. **Examples** for documentation and demonstrations

## Requirements

- Generate complete, runnable source code files (not just snippets)
- Include both **secrets** (synthetic) and **safe patterns** (env vars, test data)
- Cover multiple programming languages and frameworks
- Files should look like realistic production code
- Match the patterns already in the JSONL dataset (Task 20)

## File Structure to Create

```
data/raw/
├── python-examples/
│   ├── 01-real-secrets.py          # Flask app with hardcoded API keys
│   ├── 02-safe-config.py           # Django settings using env vars
│   ├── 03-obfuscated-secrets.py    # Base64, split strings, URL encoding
│   ├── 04-database-config.py       # SQLAlchemy with connection strings
│   └── 05-mixed-patterns.py        # Both safe and unsafe patterns
├── javascript-examples/
│   ├── 01-express-app.js           # Express server with hardcoded tokens
│   ├── 02-safe-config.js           # React app using process.env
│   ├── 03-api-client.js            # Various API key patterns
│   └── 04-edge-cases.js            # Secrets in comments, TODOs
├── yaml-examples/
│   ├── 01-k8s-secrets.yaml         # Kubernetes secrets (unsafe)
│   ├── 02-k8s-configmap.yaml       # ConfigMaps (safe)
│   ├── 03-docker-compose.yml       # Mixed patterns
│   └── 04-github-actions.yml       # CI/CD with secrets
├── java-examples/
│   ├── DatabaseConfig.java         # JDBC connection strings
│   ├── ApiClient.java              # REST client with tokens
│   └── SecureConfig.java           # Safe configuration patterns
├── go-examples/
│   ├── main.go                     # Go app with hardcoded keys
│   └── config.go                   # Safe config with env vars
└── edge-cases/
    ├── comments-with-keys.py       # Debug comments with secrets
    ├── url-encoded-secrets.js      # URL-encoded credentials
    ├── concatenated-strings.go     # Split/joined credential strings
    └── fallback-defaults.py        # Env vars with hardcoded fallbacks
```

## Secret Types to Include

Based on Task 20 JSONL examples:

### True Positives (Secrets to Embed):
- **Cloud**: AWS keys (AKIA*), Azure Storage keys, GCP service account JSON
- **Payment APIs**: Stripe (sk_live_*, sk_test_*), Square
- **Developer Tools**: GitHub PAT (ghp_*), GitLab tokens, Bitbucket (ATBB*)
- **Communication**: SendGrid (SG.*), Twilio (AC*, auth tokens), Slack webhooks
- **Databases**: PostgreSQL, MongoDB, Redis connection strings with passwords
- **Authentication**: JWT secrets, OAuth tokens, Bearer tokens
- **Monitoring**: Datadog, New Relic, PagerDuty, LaunchDarkly
- **Infrastructure**: Vault tokens (hvs.*), Firebase keys (AIza*), NPM tokens
- **Cryptographic**: RSA private keys, SSH keys, encryption keys

### True Negatives (Safe Patterns):
- Environment variables (os.getenv, process.env, ${VAR})
- Placeholder values (YOUR_API_KEY_HERE)
- Test fixtures (faker, mock data)
- Public values (UUIDs, version numbers, public keys)
- Runtime generation (secrets module, crypto.randomBytes)
- Secret manager integrations (AWS Secrets Manager, Vault client)

### Edge Cases:
- Base64/URL/hex encoded secrets
- Split strings (key_part1 + key_part2)
- Secrets in comments/TODOs
- Hardcoded fallback values
- Weak crypto (predictable keys, timestamp-based)

## Example Files to Generate

### `python-examples/01-real-secrets.py`
```python
# Flask API with embedded secrets (UNSAFE)
from flask import Flask, jsonify
import stripe

app = Flask(__name__)

# SECURITY ISSUE: Hardcoded Stripe API key
stripe.api_key = "sk_live_XXXXXXXXXXXXXXXXXXXX"

# SECURITY ISSUE: AWS credentials
AWS_ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"
AWS_SECRET_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"

@app.route('/charge', methods=['POST'])
def create_charge():
    # Payment processing logic
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True)
```

### `python-examples/02-safe-config.py`
```python
# Django settings using environment variables (SAFE)
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# SECURE: API keys from environment
STRIPE_API_KEY = os.getenv('STRIPE_API_KEY')
if not STRIPE_API_KEY:
    raise ValueError("STRIPE_API_KEY environment variable not set")

# SECURE: Database config from env
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.getenv('DB_NAME', 'myapp'),
        'USER': os.getenv('DB_USER'),
        'PASSWORD': os.getenv('DB_PASSWORD'),
        'HOST': os.getenv('DB_HOST', 'localhost'),
        'PORT': os.getenv('DB_PORT', '5432'),
    }
}

# Safe configuration values
DEBUG = os.getenv('DEBUG', 'False') == 'True'
ALLOWED_HOSTS = ['localhost', '127.0.0.1']
```

### `javascript-examples/01-express-app.js`
```javascript
// Express server with hardcoded secrets (UNSAFE)
const express = require('express');
const app = express();

// SECURITY ISSUE: Hardcoded GitHub token
const GITHUB_TOKEN = "ghp_1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef";

// SECURITY ISSUE: MongoDB connection with password
const mongoUri = "mongodb://admin:MyS3cr3tP@ss@prod-db.example.com:27017/myapp";

// SECURITY ISSUE: SendGrid API key
const sgMail = require('@sendgrid/mail');
sgMail.setApiKey('SG.1234567890abcdefghijklmnopqrstuvwxyz.ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890abcdefghij');

app.get('/api/data', async (req, res) => {
  // API logic here
  res.json({ message: 'Hello World' });
});

app.listen(3000, () => console.log('Server running on port 3000'));
```

### `yaml-examples/01-k8s-secrets.yaml`
```yaml
# Kubernetes Secret with hardcoded values (UNSAFE)
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
type: Opaque
data:
  # SECURITY ISSUE: Base64 is not encryption!
  api-key: c2stbGl2ZV9YWFhYWFhYWFhYWFhYWFhYWFhYWFhYWA==  # sk_live_XXXXXXXXXXXXXXXXXXXX
  db-password: TXlTM2NyM3RQQHNz  # MyS3cr3tP@ss
stringData:
  # SECURITY ISSUE: Plain text in YAML
  github-token: "ghp_9876543210ZYXWVUTSRQPONMLKJIHGFEDCBAzyxwvu"
```

## Deliverables

1. ✅ 20-30 complete source code files across multiple languages
2. ✅ Files organized in `data/raw/` subdirectories
3. ✅ Each file is runnable/valid syntax for its language
4. ✅ Realistic code structure (imports, functions, classes, configs)
5. ✅ Covers all secret types from JSONL dataset
6. ✅ Includes safe patterns and edge cases
7. ✅ Optional: Script to extract JSONL from raw files (demonstrate workflow)

## Success Criteria

- ✅ Can feed raw files to fine-tuned model for testing
- ✅ Files look realistic (not just test snippets)
- ✅ All patterns from Task 20 JSONL are represented
- ✅ Demonstrates raw → processed workflow
- ✅ Useful for documentation and demos

## Use Cases

1. **Model Testing**: Feed complete files to fine-tuned LLM, check detections
2. **Pre-commit Simulation**: Test scanning of full files before commit
3. **Documentation**: Show realistic examples in README/guides
4. **Benchmark Comparison**: Test regex tools vs LLM on same files
5. **Data Provenance**: Demonstrate how JSONL was extracted from source files

## Related Tasks

- Task 20: Generate JSONL dataset (done - used these patterns)
- Task 08: Run fine-tune (will train on JSONL)
- Future: Create extraction script (raw → processed automation)

## Next Steps After Completion

1. Move task to `done/`
2. Use raw files to test fine-tuned model
3. Optionally create extraction script to regenerate JSONL
4. Update TASK_COUNTER to 22
