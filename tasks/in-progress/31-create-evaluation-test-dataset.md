# Task 31: Create Evaluation/Test Dataset

**Status:** Backlog  
**Priority:** CRITICAL  
**Created:** 2025-11-01  
**Related to:** Phase 1: Data Preparation (Step 3)

## Description
Create a held-out test dataset (`data/evaluation/smart-secrets-scanner-test.jsonl`) for final model evaluation with truly unseen data.

## Requirements
- Same JSONL format as training/validation data
- Different examples not seen during training
- Balanced positive/negative examples
- Edge cases and challenging scenarios

## Acceptance Criteria
- [ ] `data/evaluation/smart-secrets-scanner-test.jsonl` created
- [ ] 20-30 examples (10% of total dataset)
- [ ] 50/50 balance of secrets vs safe code
- [ ] No overlap with training or validation data
- [ ] Includes challenging edge cases

## Test Set Composition (20 examples recommended)

### Secrets to Detect (10 examples):
1. **Cloud providers**: AWS, Azure, GCP (not in train/val)
2. **CI/CD tokens**: CircleCI, Travis, Jenkins
3. **Database URLs**: MySQL, PostgreSQL with embedded passwords
4. **API keys**: Anthropic, OpenAI, Cohere (AI platforms)
5. **Certificates**: PEM-encoded private keys
6. **Obfuscation**: ROT13, XOR encoding, environment variable interpolation
7. **Multi-line secrets**: SSH keys spanning multiple lines
8. **Secrets in comments**: Accidentally committed in TODO/DEBUG comments
9. **Legacy patterns**: FTP credentials, SMTP passwords
10. **New platforms**: Vercel, Railway, Render tokens

### Safe Patterns (10 examples):
1. **Correct env usage**: os.environ.get with no defaults
2. **Key derivation**: PBKDF2, bcrypt password hashing
3. **Public tokens**: Read-only API keys, public Firebase config
4. **Placeholder examples**: Documentation with "YOUR_KEY_HERE"
5. **Test fixtures**: Clearly marked test data with "test" prefix
6. **Generated values**: uuid.uuid4(), secrets.token_hex()
7. **Non-sensitive URLs**: HTTP endpoints without credentials
8. **Config templates**: .env.example files
9. **Version strings**: Semantic versioning, commit hashes
10. **Feature flags**: Boolean/string configuration values

## Steps
1. Review existing train/val data to avoid duplicates
2. Research additional secret types not yet covered
3. Generate 20 examples following Alpaca format
4. Validate JSONL syntax
5. Document test set coverage in `data/SOURCES.md`
6. Update `data/README.md` with test set info

## Dependencies
- Task 20: Existing JSONL format reference
- Task 25: Evaluation script will use this dataset

## Notes
- Test set should be HARDER than training examples
- Include adversarial examples (secrets designed to evade detection)
- Consider real-world scenarios from public breach reports
- This is the "final exam" for the model - make it comprehensive
- Do NOT use this data for training or hyperparameter tuning

## Quality Checks
- No duplicates from train/val sets
- Diverse programming languages (Python, JS, Go, YAML, JSON)
- Mix of obvious and subtle secrets
- Edge cases: multi-line, obfuscated, in-code comments
