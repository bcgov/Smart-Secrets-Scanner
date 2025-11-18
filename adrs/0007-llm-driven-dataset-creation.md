# ADR 0007: LLM-Driven Dataset Creation for Secret Detection Training

**Date**: 2025-11-18  
**Status**: Accepted  
**Deciders**: Project Team  

## Context

The Smart Secrets Scanner project requires a high-quality training dataset to fine-tune an LLM for detecting secrets and sensitive credentials in code. Traditional approaches to dataset creation include:

- **Rule-based scripts**: Deterministic extraction using regex patterns or static rules
- **Manual curation**: Human experts manually labeling examples
- **Crowdsourcing**: Distributed labeling through platforms
- **LLM-assisted generation**: Using LLMs to augment or accelerate traditional methods

For secret detection, the challenge is creating nuanced examples that require:
- Understanding context (what constitutes a "secret" vs. false positive)
- Recognizing various secret formats (API keys, passwords, tokens, certificates)
- Handling edge cases (test data, documentation examples, revoked secrets)
- Maintaining consistency across programming languages

**Key Learning**: Unlike traditional ML dataset creation that involves collecting raw data and then processing it through scripts or tools, this project used a direct LLM-driven approach where the LLM itself generated the complete labeled dataset without any intermediate processing steps.

## Decision

We will create the training dataset directly using LLM generation rather than any form of script-based processing or data extraction. The dataset will be created by having the LLM:

1. **Generate complete examples**: Create both code snippets and corresponding labels from scratch
2. **Apply contextual judgment**: Use understanding of programming and security to create realistic scenarios
3. **Format as JSONL**: Structure examples as instruction-input-output triples for fine-tuning
4. **Ensure diversity**: Cover multiple programming languages, secret types, and edge cases

### Key Architectural Difference

**Direct LLM Creation**: Unlike traditional workflows that collect raw data first and then process it, this approach eliminates the data collection and processing pipeline entirely. The LLM serves as both the data generator and labeler in a single step.

### Dataset Structure

Each training example follows this JSONL format:

```json
{
  "instruction": "Analyze the following code snippet and identify any secrets or sensitive credentials that should not be committed to version control.",
  "input": "<code snippet with potential secrets>",
  "output": "<ALERT: Found [secret type] in [location] | No secrets detected.>"
}
```

### Data Sources

- **LLM-generated examples**: Complete synthetic code snippets with embedded secrets
- **Realistic patterns**: Based on actual secret types and programming practices
- **Edge cases**: Context-dependent scenarios requiring judgment

## Consequences

### Positive

- **No pipeline complexity**: Eliminates data collection, extraction, and processing steps
- **Higher quality labels**: LLM can understand context and nuance better than regex
- **Faster development**: No need to develop and maintain complex extraction scripts
- **Consistency**: Single "expert" (LLM) applies consistent judgment across all examples
- **Flexibility**: Easy to generate new examples for specific scenarios
- **Cost-effective**: No manual labeling effort or infrastructure required

### Negative

- **Non-deterministic**: Results may vary between LLM runs and prompt variations
- **Black box**: Harder to audit the generation and labeling decisions
- **Dependency on LLM**: Requires access to capable language models
- **Potential bias**: LLM training data may influence example generation patterns
- **No provenance**: Generated examples don't trace back to real-world sources

### Risks

- **Label quality**: LLM might generate unrealistic examples or incorrect labels
- **Reproducibility**: Different LLM versions or prompts might produce different datasets
- **Scale limitations**: Manual LLM interaction may be slower than automated scripts
- **Overfitting potential**: Generated data might not reflect real-world distribution

## Alternatives Considered

### Option 1: Traditional Data Pipeline (Rejected)
- **Description**: Collect raw code samples → Extract snippets → Script-based labeling → JSONL format
- **Pros**: Reproducible, auditable, follows data engineering best practices
- **Cons**: Complex pipeline development, potential quality issues with automated labeling
- **Reason**: For this proof-of-concept, direct LLM generation was faster and produced higher quality results

### Option 2: Regex-Based Script (Rejected)
- **Pros**: Deterministic, fast, reproducible
- **Cons**: Poor at context understanding, high false positive/negative rates
- **Reason**: Secret detection requires nuanced judgment beyond pattern matching

### Option 3: Human Curation (Rejected)
- **Pros**: High accuracy, understandable decisions
- **Cons**: Time-consuming, expensive, inconsistent between annotators
- **Reason**: Project timeline and resource constraints; LLM quality proved sufficient

### Option 4: Hybrid LLM-Assisted (Deferred)
- **Description**: LLM generates candidates, humans review and correct
- **Pros**: Combines LLM speed with human accuracy
- **Cons**: Still requires human effort, slower than pure LLM generation
- **Reason**: Current pure LLM approach working well for initial dataset; consider for production expansion

## Implementation Notes

- **Direct LLM Creation**: Dataset generated entirely by LLM without intermediate processing steps
- **Dataset files**: `data/processed/smart-secrets-scanner-train.jsonl` (56 examples), `data/processed/smart-secrets-scanner-val.jsonl` (16 examples)
- **Validation**: Use `scripts/validate_dataset.py` to check JSONL format and basic consistency
- **Version control**: Dataset files committed to Git for reproducibility
- **Documentation**: See `data/README.md` for data directory structure and workflow, `data/HOW_TO_USE_DATASET.md` for usage examples, and `data/SOURCES.md` for detailed dataset statistics and creation methodology
- **Key Learning**: This approach eliminates traditional data engineering pipelines while maintaining quality
- **Future enhancement**: Consider hybrid human-LLM validation for production datasets or larger scales</content>
<parameter name="filePath">c:\Users\RICHFREM\source\repos\Smart-Secrets-Scanner\adrs\0007-llm-driven-dataset-creation.md