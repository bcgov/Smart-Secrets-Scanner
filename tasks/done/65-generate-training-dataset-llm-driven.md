# Task 65: Generate Training Dataset (LLM-Driven)

**Status:** Done  
**Created:** 2025-11-18  
**Completed:** 2025-11-18  

## Prerequisites (Completed)

✅ **Task 64**: Environment verified  
✅ **Task 63**: llama-cpp-python bridge built  
✅ **Task 60**: Hugging Face auth configured  
✅ **Task 20**: Dataset generation approach defined (LLM-driven)  

## Objective

Generate the complete training dataset using LLM-driven creation approach. This replaces traditional scripted dataset generation with direct LLM content creation for higher quality and more diverse training examples.

## Requirements

- Access to LLM API (OpenAI/Hugging Face)
- Dataset generation script
- Quality validation
- Output in JSONL format for training

## Implementation

### 1. Prepare Generation Environment

```bash
# Activate ML environment
source ~/ml_env/bin/activate

# Verify LLM access (choose one method)
# Option A: OpenAI API
export OPENAI_API_KEY="your-key-here"

# Option B: Hugging Face Inference API
export HF_TOKEN="your-hf-token"
```

### 2. Run Dataset Generation

```bash
# Generate comprehensive dataset using LLM
python scripts/forge_whole_genome_dataset.py \
  --output data/processed/smart-secrets-scanner-dataset.jsonl \
  --target-size 1000 \
  --llm-provider openai \
  --model gpt-4 \
  --validate-output
```

### 3. Alternative: Use Test Dataset Generator

For initial testing or smaller datasets:

```bash
# Generate smaller test dataset
python scripts/forge_test_set.py \
  --output data/processed/test-dataset.jsonl \
  --size 50 \
  --include-edge-cases
```

### 4. Validate Generated Dataset

```bash
# Run validation script
python scripts/validate_dataset.py data/processed/smart-secrets-scanner-dataset.jsonl

# Check dataset statistics
python -c "
import json
count = 0
with open('data/processed/smart-secrets-scanner-dataset.jsonl', 'r') as f:
    for line in f:
        count += 1
print(f'Total examples: {count}')

# Sample first few examples
with open('data/processed/smart-secrets-scanner-dataset.jsonl', 'r') as f:
    for i, line in enumerate(f):
        if i >= 3: break
        data = json.loads(line)
        print(f'Example {i+1}: {data.keys()}')
"
```

### 5. Dataset Quality Checks

```bash
# Check for data quality issues
python - <<'PY'
import json
from collections import Counter

issues = []
secret_types = Counter()

with open('data/processed/smart-secrets-scanner-dataset.jsonl', 'r') as f:
    for i, line in enumerate(f):
        try:
            data = json.loads(line)
            
            # Check required fields
            if 'input' not in data or 'output' not in data:
                issues.append(f'Line {i+1}: Missing input/output fields')
            
            # Track secret types
            if 'secret_type' in data:
                secret_types[data['secret_type']] += 1
                
        except json.JSONDecodeError:
            issues.append(f'Line {i+1}: Invalid JSON')

print(f"Quality check complete. Found {len(issues)} issues:")
for issue in issues[:5]:  # Show first 5 issues
    print(f"  {issue}")

print(f"\nSecret type distribution: {dict(secret_types)}")
PY
```

## Technical Details

### LLM-Driven Generation Process

1. **Prompt Engineering**: Carefully crafted prompts for diverse secret types
2. **Quality Control**: LLM self-validation and filtering
3. **Edge Cases**: Explicit generation of hard-to-detect patterns
4. **Diversity**: Multiple secret types, languages, and contexts

### Dataset Format

```json
{
  "input": "Code snippet containing potential secret",
  "output": "Analysis result (ALERT/NEGATIVE)",
  "secret_type": "api_key|password|token|etc",
  "confidence": 0.95,
  "metadata": {
    "language": "python|javascript|etc",
    "context": "environment variable|hardcoded|etc"
  }
}
```

### Generation Strategies

- **Comprehensive Coverage**: All major secret types (API keys, passwords, tokens, certificates)
- **Contextual Variety**: Different programming languages and frameworks
- **False Positive Reduction**: Include safe patterns and legitimate uses
- **Edge Cases**: Obfuscated secrets, partial exposures, environment variables

## Troubleshooting

### LLM API Issues

```bash
# Check API connectivity
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models

# For Hugging Face
curl -H "Authorization: Bearer $HF_TOKEN" \
  https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct
```

### Generation Failures

```bash
# Check available models
python -c "
import openai
client = openai.OpenAI()
models = client.models.list()
print([m.id for m in models.data if 'gpt' in m.id])
"
```

### Quality Issues

```bash
# Regenerate specific secret types
python scripts/forge_whole_genome_dataset.py \
  --output data/processed/supplemental-dataset.jsonl \
  --secret-types api_keys passwords \
  --size 200
```

## Outcome

✅ Training dataset generated via LLM  
✅ 1000+ diverse examples created  
✅ JSONL format validated  
✅ Quality checks passed  
✅ Ready for Phase 2: Model training  

## Related Tasks

- Task 20: Dataset generation approach (foundation)
- Task 46: Edge case enhancement (complementary)
- Task 47: 1000-example expansion (complementary)
- Task 34: Dataset validation (next step)
- Task 36: Fine-tuning (uses this dataset)</content>
<parameter name="filePath">c:\Users\RICHFREM\source\repos\Smart-Secrets-Scanner\tasks\done\65-generate-training-dataset-llm-driven.md