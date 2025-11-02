# Task 34: Create Data Validation Script

**Status:** Done ✅  
**Priority:** MEDIUM  
**Created:** 2025-11-01  
**Completed:** 2025-11-01  
**Related to:** Phase 1: Data Preparation

## Prerequisites (Completed)

✅ **Task 17**: Data directory structure created  
✅ **Task 18**: JSONL format defined  
✅ **Task 20**: Training dataset created (to validate)  

## Description
Create an automated validation script (`scripts/validate_dataset.py`) to check JSONL data quality before training.

## Requirements
- Python script to validate JSONL format and content
- Check for common data quality issues
- Generate validation report
- Run as pre-training sanity check

## Acceptance Criteria
- [ ] `scripts/validate_dataset.py` created
- [ ] Validates JSONL syntax
- [ ] Checks required fields (instruction, input, output)
- [ ] Detects duplicates
- [ ] Verifies class balance
- [ ] Checks token lengths
- [ ] Generates validation report

## Validation Checks

### 1. JSONL Syntax
- Each line is valid JSON
- No empty lines or malformed entries
- UTF-8 encoding

### 2. Schema Validation
- Required fields present: `instruction`, `input`, `output`
- No null/empty values
- Correct data types (all strings)

### 3. Content Quality
- Instruction consistency (should be same across examples)
- Input length within reasonable bounds (10-2048 tokens)
- Output format consistency ("ALERT" vs "No secrets")
- No exact duplicates in inputs
- Character encoding issues (invalid UTF-8)

### 4. Class Balance
- Count ALERT vs safe examples
- Report percentage split (should be ~50/50)
- Warn if imbalanced (>60% either way)

### 5. Token Length Analysis
- Max/min/average token counts
- Identify outliers (very long or very short)
- Check if any exceed model max_seq_length

### 6. Secret Type Coverage
- Parse outputs to identify secret types mentioned
- Report coverage (AWS, GitHub, Stripe, etc.)
- Flag if important categories missing

## Script Template

```python
#!/usr/bin/env python3
"""
Validate JSONL training data quality
"""
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from transformers import AutoTokenizer

def validate_jsonl_syntax(file_path):
    """Check each line is valid JSON"""
    errors = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                errors.append(f"Line {i}: Empty line")
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: Invalid JSON - {e}")
    return errors

def validate_schema(file_path):
    """Check required fields present"""
    errors = []
    required_fields = {'instruction', 'input', 'output'}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            try:
                obj = json.loads(line.strip())
                missing = required_fields - set(obj.keys())
                if missing:
                    errors.append(f"Line {i}: Missing fields {missing}")
                
                # Check for empty values
                for field in required_fields:
                    if field in obj and not obj[field]:
                        errors.append(f"Line {i}: Empty {field}")
            except:
                continue  # Already caught in syntax check
    
    return errors

def check_class_balance(file_path):
    """Count ALERT vs safe examples"""
    alert_count = 0
    safe_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line.strip())
            if 'ALERT' in obj.get('output', ''):
                alert_count += 1
            else:
                safe_count += 1
    
    total = alert_count + safe_count
    alert_pct = (alert_count / total * 100) if total > 0 else 0
    safe_pct = (safe_count / total * 100) if total > 0 else 0
    
    return {
        'total': total,
        'alert': alert_count,
        'safe': safe_count,
        'alert_pct': alert_pct,
        'safe_pct': safe_pct,
        'balanced': 40 <= alert_pct <= 60
    }

def check_token_lengths(file_path, model_name="meta-llama/Meta-Llama-3-8B"):
    """Analyze token length distribution"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except:
        print("Warning: Could not load tokenizer, skipping token analysis")
        return None
    
    lengths = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line.strip())
            full_text = f"{obj['instruction']}\n{obj['input']}\n{obj['output']}"
            tokens = tokenizer.encode(full_text)
            lengths.append(len(tokens))
    
    return {
        'min': min(lengths),
        'max': max(lengths),
        'avg': sum(lengths) / len(lengths),
        'over_2048': sum(1 for l in lengths if l > 2048)
    }

def check_duplicates(file_path):
    """Find duplicate inputs"""
    inputs_seen = defaultdict(list)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            obj = json.loads(line.strip())
            inputs_seen[obj['input']].append(i)
    
    duplicates = {k: v for k, v in inputs_seen.items() if len(v) > 1}
    return duplicates

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='JSONL file to validate')
    args = parser.parse_args()
    
    print(f"Validating {args.file}...\n")
    
    # Run all checks
    print("1. Checking JSONL syntax...")
    syntax_errors = validate_jsonl_syntax(args.file)
    if syntax_errors:
        print(f"  ❌ Found {len(syntax_errors)} syntax errors")
        for err in syntax_errors[:5]:  # Show first 5
            print(f"     {err}")
    else:
        print("  ✅ All lines are valid JSON")
    
    print("\n2. Checking schema...")
    schema_errors = validate_schema(args.file)
    if schema_errors:
        print(f"  ❌ Found {len(schema_errors)} schema errors")
        for err in schema_errors[:5]:
            print(f"     {err}")
    else:
        print("  ✅ All required fields present")
    
    print("\n3. Checking class balance...")
    balance = check_class_balance(args.file)
    print(f"  Total: {balance['total']} examples")
    print(f"  Secrets (ALERT): {balance['alert']} ({balance['alert_pct']:.1f}%)")
    print(f"  Safe: {balance['safe']} ({balance['safe_pct']:.1f}%)")
    if balance['balanced']:
        print("  ✅ Dataset is balanced")
    else:
        print("  ⚠️  Dataset is imbalanced")
    
    print("\n4. Checking token lengths...")
    token_stats = check_token_lengths(args.file)
    if token_stats:
        print(f"  Min: {token_stats['min']} tokens")
        print(f"  Max: {token_stats['max']} tokens")
        print(f"  Avg: {token_stats['avg']:.1f} tokens")
        if token_stats['over_2048'] > 0:
            print(f"  ⚠️  {token_stats['over_2048']} examples exceed 2048 tokens")
        else:
            print("  ✅ All examples fit in context window")
    
    print("\n5. Checking for duplicates...")
    duplicates = check_duplicates(args.file)
    if duplicates:
        print(f"  ⚠️  Found {len(duplicates)} duplicate inputs")
        for inp, lines in list(duplicates.items())[:3]:
            print(f"     Lines {lines}: {inp[:50]}...")
    else:
        print("  ✅ No duplicate inputs found")
    
    print("\n=== Validation Complete ===")
    if not (syntax_errors or schema_errors):
        print("✅ Dataset is ready for training!")
    else:
        print("❌ Please fix errors before training")

if __name__ == '__main__':
    main()
```

## Usage
```bash
python scripts/validate_dataset.py data/processed/smart-secrets-scanner-train.jsonl
python scripts/validate_dataset.py data/processed/smart-secrets-scanner-val.jsonl
python scripts/validate_dataset.py data/evaluation/smart-secrets-scanner-test.jsonl
```

## Dependencies
- Task 28: requirements.txt (transformers, collections)

## Notes
- Run before every training session
- Add to pre-commit hooks (optional)
- Consider adding to CI/CD pipeline
- Extend with custom checks for your use case

## Success Criteria
- Detects all common data quality issues
- Provides actionable error messages
- Runs in <5 seconds for 100 examples
- Zero false positives on validated datasets
