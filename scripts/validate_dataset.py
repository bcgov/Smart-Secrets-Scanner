#!/usr/bin/env python3
"""
Validate JSONL training data quality for fine-tuning

Usage:
    python scripts/validate_dataset.py data/processed/smart-secrets-scanner-train.jsonl
    python scripts/validate_dataset.py data/processed/smart-secrets-scanner-train.jsonl --strict
"""
import json
import argparse
import sys
from pathlib import Path
from collections import Counter, defaultdict

def validate_jsonl_syntax(file_path):
    """Check each line is valid JSON"""
    errors = []
    line_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line_count = i
            line = line.strip()
            if not line:
                errors.append(f"Line {i}: Empty line")
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: Invalid JSON - {e}")
    
    return errors, line_count

def validate_schema(file_path):
    """Check required fields present and valid"""
    errors = []
    required_fields = {'instruction', 'input', 'output'}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                obj = json.loads(line)
                
                # Check missing fields
                missing = required_fields - set(obj.keys())
                if missing:
                    errors.append(f"Line {i}: Missing fields {missing}")
                    continue
                
                # Check for empty values
                for field in required_fields:
                    if field in obj:
                        value = obj[field]
                        if not isinstance(value, str):
                            errors.append(f"Line {i}: Field '{field}' must be string, got {type(value).__name__}")
                        elif not value.strip():
                            errors.append(f"Line {i}: Empty {field}")
                            
            except json.JSONDecodeError:
                continue  # Already caught in syntax check
            except Exception as e:
                errors.append(f"Line {i}: Validation error - {e}")
    
    return errors

def check_duplicates(file_path):
    """Find duplicate inputs"""
    inputs_seen = {}
    duplicates = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                obj = json.loads(line)
                input_text = obj.get('input', '')
                
                if input_text in inputs_seen:
                    duplicates.append(f"Line {i} duplicates line {inputs_seen[input_text]}")
                else:
                    inputs_seen[input_text] = i
            except:
                continue
    
    return duplicates

def check_class_balance(file_path):
    """Count ALERT vs safe examples"""
    alert_count = 0
    safe_count = 0
    secret_types = Counter()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            try:
                obj = json.loads(line)
                output = obj.get('output', '')
                
                if 'ALERT' in output.upper():
                    alert_count += 1
                    # Extract secret type
                    if 'AWS' in output:
                        secret_types['AWS'] += 1
                    if 'Stripe' in output or 'stripe' in output:
                        secret_types['Stripe'] += 1
                    if 'GitHub' in output or 'github' in output:
                        secret_types['GitHub'] += 1
                    if 'OpenAI' in output or 'openai' in output:
                        secret_types['OpenAI'] += 1
                    if 'JWT' in output:
                        secret_types['JWT'] += 1
                    if 'API key' in output or 'API_KEY' in output:
                        secret_types['API Key'] += 1
                else:
                    safe_count += 1
            except:
                continue
    
    total = alert_count + safe_count
    alert_pct = (alert_count / total * 100) if total > 0 else 0
    safe_pct = (safe_count / total * 100) if total > 0 else 0
    
    return {
        'total': total,
        'alert': alert_count,
        'safe': safe_count,
        'alert_pct': alert_pct,
        'safe_pct': safe_pct,
        'balanced': 40 <= alert_pct <= 60,
        'secret_types': secret_types
    }

def estimate_token_lengths(file_path):
    """Estimate token lengths (rough approximation: ~4 chars per token)"""
    lengths = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            try:
                obj = json.loads(line)
                full_text = f"{obj['instruction']}\n{obj['input']}\n{obj['output']}"
                # Rough estimate: 4 chars per token on average
                estimated_tokens = len(full_text) // 4
                lengths.append(estimated_tokens)
            except:
                continue
    
    if not lengths:
        return None
    
    return {
        'min': min(lengths),
        'max': max(lengths),
        'avg': sum(lengths) / len(lengths),
        'over_2048': sum(1 for l in lengths if l > 2048)
    }

def check_instruction_consistency(file_path):
    """Check if instruction field is consistent"""
    instructions = Counter()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            try:
                obj = json.loads(line)
                instructions[obj.get('instruction', '')] += 1
            except:
                continue
    
    return instructions

def print_report(file_path, strict=False):
    """Generate and print validation report"""
    print(f"\n{'='*60}")
    print(f"Validation Report: {file_path}")
    print(f"{'='*60}\n")
    
    # 1. JSONL Syntax
    print("1. JSONL Syntax Validation")
    print("-" * 60)
    syntax_errors, line_count = validate_jsonl_syntax(file_path)
    if syntax_errors:
        print(f"❌ Found {len(syntax_errors)} syntax error(s):")
        for error in syntax_errors[:10]:  # Show first 10
            print(f"   - {error}")
        if len(syntax_errors) > 10:
            print(f"   ... and {len(syntax_errors) - 10} more")
    else:
        print(f"✅ All {line_count} lines are valid JSON")
    print()
    
    # 2. Schema Validation
    print("2. Schema Validation")
    print("-" * 60)
    schema_errors = validate_schema(file_path)
    if schema_errors:
        print(f"❌ Found {len(schema_errors)} schema error(s):")
        for error in schema_errors[:10]:
            print(f"   - {error}")
        if len(schema_errors) > 10:
            print(f"   ... and {len(schema_errors) - 10} more")
    else:
        print(f"✅ All required fields present and valid")
    print()
    
    # 3. Duplicates
    print("3. Duplicate Detection")
    print("-" * 60)
    duplicates = check_duplicates(file_path)
    if duplicates:
        print(f"⚠️  Found {len(duplicates)} duplicate input(s):")
        for dup in duplicates[:5]:
            print(f"   - {dup}")
        if len(duplicates) > 5:
            print(f"   ... and {len(duplicates) - 5} more")
    else:
        print(f"✅ No duplicate inputs found")
    print()
    
    # 4. Class Balance
    print("4. Class Balance Analysis")
    print("-" * 60)
    balance = check_class_balance(file_path)
    print(f"Total examples: {balance['total']}")
    print(f"ALERT examples: {balance['alert']} ({balance['alert_pct']:.1f}%)")
    print(f"Safe examples:  {balance['safe']} ({balance['safe_pct']:.1f}%)")
    
    if balance['balanced']:
        print(f"✅ Dataset is balanced (40-60% split)")
    else:
        print(f"⚠️  Dataset may be imbalanced")
    
    if balance['secret_types']:
        print(f"\nSecret types detected:")
        for secret_type, count in balance['secret_types'].most_common():
            print(f"   - {secret_type}: {count}")
    print()
    
    # 5. Token Length Estimation
    print("5. Token Length Analysis (Estimated)")
    print("-" * 60)
    token_stats = estimate_token_lengths(file_path)
    if token_stats:
        print(f"Min tokens:  {token_stats['min']}")
        print(f"Max tokens:  {token_stats['max']}")
        print(f"Avg tokens:  {token_stats['avg']:.1f}")
        
        if token_stats['over_2048'] > 0:
            print(f"⚠️  {token_stats['over_2048']} example(s) may exceed 2048 tokens")
        else:
            print(f"✅ All examples within 2048 token limit")
    print()
    
    # 6. Instruction Consistency
    print("6. Instruction Consistency")
    print("-" * 60)
    instructions = check_instruction_consistency(file_path)
    if len(instructions) == 1:
        print(f"✅ Single consistent instruction across all examples")
        print(f"   Instruction: \"{list(instructions.keys())[0][:100]}...\"")
    else:
        print(f"⚠️  Found {len(instructions)} different instruction(s):")
        for instr, count in instructions.most_common(3):
            print(f"   - ({count}x) \"{instr[:80]}...\"")
    print()
    
    # Overall Summary
    print("="*60)
    total_errors = len(syntax_errors) + len(schema_errors)
    total_warnings = len(duplicates) + (0 if balance['balanced'] else 1)
    
    if total_errors == 0 and total_warnings == 0:
        print("✅ VALIDATION PASSED - Dataset is ready for training!")
        return 0
    elif total_errors == 0:
        print(f"⚠️  VALIDATION PASSED WITH WARNINGS ({total_warnings} warning(s))")
        print("   Dataset can be used but consider addressing warnings")
        return 0 if not strict else 1
    else:
        print(f"❌ VALIDATION FAILED ({total_errors} error(s), {total_warnings} warning(s))")
        print("   Please fix errors before training")
        return 1

def main():
    parser = argparse.ArgumentParser(
        description='Validate JSONL training data for fine-tuning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/validate_dataset.py data/processed/train.jsonl
  python scripts/validate_dataset.py data/processed/train.jsonl --strict
        """
    )
    parser.add_argument('file', type=str, help='Path to JSONL file to validate')
    parser.add_argument('--strict', action='store_true', 
                       help='Fail on warnings (not just errors)')
    
    args = parser.parse_args()
    
    # Check file exists
    if not Path(args.file).exists():
        print(f"❌ Error: File not found: {args.file}")
        return 1
    
    # Run validation
    exit_code = print_report(args.file, strict=args.strict)
    return exit_code

if __name__ == "__main__":
    sys.exit(main())
