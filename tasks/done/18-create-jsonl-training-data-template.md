# Task 18: Create JSONL Training Data Template

**Status**: Done  
**Created**: 2025-11-01  
**Completed**: 2025-11-01  

## Objective

Provide an example JSONL file demonstrating the correct format for Smart Secrets Scanner training data.

## Requirements

- Use Alpaca instruction-input-output format
- Include examples of true positives (real secrets detected)
- Include examples of true negatives (safe code)
- Include diverse code patterns and languages
- Show clear, actionable output messages

## Implementation

Created `data/processed/example-smart-secrets-scanner.jsonl` with 5 example records:

1. **API Key Detection**: Stripe API key hardcoded in JavaScript
2. **Safe Configuration**: Database host/port (non-sensitive)
3. **Password Detection**: Hardcoded password with TODO comment
4. **Private Key Detection**: RSA private key embedded in code
5. **Secure Practice**: Correct use of environment variables

## JSONL Format

```json
{
  "instruction": "Analyze the following code snippet and identify any secrets or sensitive credentials that should not be committed to version control.",
  "input": "<code snippet>",
  "output": "<detection result with explanation>"
}
```

## Files Created

- `data/processed/example-smart-secrets-scanner.jsonl` (5 examples)

## Outcome

✅ Template file created with diverse examples  
✅ Format ready for expansion to 100-150 training examples  
✅ Clear pattern for creating more training data  

## Next Steps

- Expand to 100-150 examples covering all edge cases
- Balance positive/negative examples (50/50 split)
- Add examples from multiple programming languages

## Related Tasks

- Task 17: Create data directory structure
- Task 07: Create dataset (backlog)
