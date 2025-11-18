# Task 66: Validate Training Dataset

**Status:** Done  
**Created:** 2025-11-18  
**Completed:** 2025-11-18  

## Prerequisites (Completed)

✅ **Task 65**: Training dataset generated  
✅ **Task 64**: Environment verified  
✅ **Task 34**: Validation script created  

## Objective

Run comprehensive validation on the generated training dataset to ensure quality, format correctness, and training readiness before proceeding to model fine-tuning.

## Requirements

- JSONL format validation
- Content quality checks
- Statistical analysis
- Error reporting and fixes

## Implementation

### 1. Run Dataset Validation

```bash
# Activate environment
source ~/ml_env/bin/activate

# Run validation script
python scripts/validate_dataset.py data/processed/smart-secrets-scanner-dataset.jsonl
```

### 2. Review Validation Output

The script will check for:
- JSONL syntax validity
- Required field presence (input, output)
- Data quality metrics
- Duplicate detection
- Format consistency

### 3. Manual Quality Inspection

```bash
# Sample dataset examples for manual review
python - <<'PY'
import json
import random

# Load and sample dataset
examples = []
with open('data/processed/smart-secrets-scanner-dataset.jsonl', 'r') as f:
    for line in f:
        examples.append(json.loads(line))

print(f"Total examples: {len(examples)}")

# Sample 10 random examples
sampled = random.sample(examples, min(10, len(examples)))
for i, ex in enumerate(sampled, 1):
    print(f"\n--- Example {i} ---")
    print(f"Input: {ex.get('input', 'MISSING')[:100]}...")
    print(f"Output: {ex.get('output', 'MISSING')}")
    print(f"Secret Type: {ex.get('secret_type', 'N/A')}")
PY
```

### 4. Statistical Analysis

```bash
# Analyze dataset composition
python - <<'PY'
import json
from collections import Counter

examples = []
with open('data/processed/smart-secrets-scanner-dataset.jsonl', 'r') as f:
    for line in f:
        examples.append(json.loads(line))

print(f"Dataset Statistics:")
print(f"Total examples: {len(examples)}")

# Analyze outputs
outputs = [ex.get('output', '') for ex in examples]
output_counts = Counter(outputs)
print(f"Output distribution: {dict(output_counts)}")

# Analyze secret types
secret_types = [ex.get('secret_type', 'unknown') for ex in examples if ex.get('secret_type')]
type_counts = Counter(secret_types)
print(f"Secret type distribution: {dict(type_counts)}")

# Check input lengths
input_lengths = [len(ex.get('input', '')) for ex in examples]
print(f"Input length stats: min={min(input_lengths)}, max={max(input_lengths)}, avg={sum(input_lengths)/len(input_lengths):.1f}")
PY
```

### 5. Fix Any Issues Found

If validation reveals issues:

```bash
# Example: Remove duplicates
python - <<'PY'
import json

seen_inputs = set()
cleaned = []

with open('data/processed/smart-secrets-scanner-dataset.jsonl', 'r') as f:
    for line in f:
        ex = json.loads(line)
        input_text = ex.get('input', '')
        if input_text not in seen_inputs:
            cleaned.append(ex)
            seen_inputs.add(input_text)

print(f"Removed {len(examples) - len(cleaned)} duplicates")

# Write cleaned dataset
with open('data/processed/smart-secrets-scanner-dataset-cleaned.jsonl', 'w') as f:
    for ex in cleaned:
        f.write(json.dumps(ex) + '\n')

print("Cleaned dataset saved")
PY
```

## Validation Criteria

### Format Checks
- [ ] All lines are valid JSON
- [ ] Each example has 'input' and 'output' fields
- [ ] No empty or null values in required fields
- [ ] UTF-8 encoding throughout

### Content Checks
- [ ] No exact duplicate inputs
- [ ] Reasonable input length distribution (10-2048 chars)
- [ ] Consistent output format (ALERT/NEGATIVE patterns)
- [ ] Diverse secret types represented

### Quality Metrics
- [ ] Balanced positive/negative examples
- [ ] No malformed code snippets
- [ ] Realistic secret patterns
- [ ] Appropriate context for each example

## Troubleshooting

### JSON Parsing Errors

```bash
# Find problematic lines
python -c "
import json
with open('data/processed/smart-secrets-scanner-dataset.jsonl', 'r') as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line: continue
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            print(f'Line {i}: {e}')
            print(f'Content: {line[:100]}...')
"
```

### Missing Fields

```bash
# Check for missing required fields
python -c "
import json
missing_input = 0
missing_output = 0

with open('data/processed/smart-secrets-scanner-dataset.jsonl', 'r') as f:
    for line in f:
        ex = json.loads(line)
        if 'input' not in ex: missing_input += 1
        if 'output' not in ex: missing_output += 1

print(f'Missing input: {missing_input}')
print(f'Missing output: {missing_output}')
"
```

### Quality Issues

```bash
# Regenerate problematic examples
# Use Task 65 generation script with specific filters
python scripts/forge_whole_genome_dataset.py \
  --output data/processed/supplemental-dataset.jsonl \
  --size 100 \
  --fix-quality-issues
```

## Outcome

✅ Dataset validation completed  
✅ Format and quality verified  
✅ Statistical analysis performed  
✅ Issues identified and resolved  
✅ Dataset ready for training  

## Related Tasks

- Task 65: Dataset generation (prerequisite)
- Task 34: Validation script creation (foundation)
- Task 36: Fine-tuning (next step)
- Task 22: Download base model (parallel preparation)</content>
<parameter name="filePath">c:\Users\RICHFREM\source\repos\Smart-Secrets-Scanner\tasks\done\66-validate-training-dataset.md