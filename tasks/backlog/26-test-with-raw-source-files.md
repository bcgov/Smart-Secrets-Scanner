# Task 26: Test with Raw Source Files

**Status:** Backlog  
**Created:** 2025-11-01  
**Related to:** Phase 4: Testing & Deployment (Step 11)

## Prerequisites (Completed)

✅ **Task 37**: Inference script created  

**Pending:**  
⏳ **Task 08**: Fine-tuning Iteration 4 (in-progress)  
⏳ **Task 15**: Ollama deployment (for API testing)  
⏳ **Task 21**: Synthetic source files generated (optional - or use real files)  

## Description
Test the fine-tuned model with complete source code files (.py, .js, .yaml) to validate real-world performance.

## Requirements
- Fine-tuned model deployed (Task 15)
- Raw source files in `data/raw/` or `data/evaluation/`
- Test cases with known secrets and safe code

## Acceptance Criteria
- [ ] Model scans complete .py, .js, .yaml files
- [ ] Detects all known secrets in test files
- [ ] No false positives on safe code
- [ ] Results documented with file paths and findings

## Steps
1. Collect or generate test files:
   - Files WITH secrets (AWS keys, API tokens, etc.)
   - Files WITHOUT secrets (env vars, test data, UUIDs)
2. Feed each file to model via Ollama API
3. Record model detections
4. Verify against known ground truth
5. Document accuracy and edge cases

## Dependencies
- Task 15: Ollama deployment complete
- Task 21: Synthetic raw source files (optional)

## Notes
- Use Task 21 output if available
- Real-world test: Clone public repos, scan for secrets
- Check for: line numbers, secret types, confidence scores
- Edge cases: multi-line secrets, obfuscated keys, comments
