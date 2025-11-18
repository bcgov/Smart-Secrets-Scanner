# How to Use the Smart Secrets Scanner Dataset

## Dataset Creation Approach

**LLM-Driven Creation**: Unlike traditional datasets created through automated scripts or manual curation, this dataset was created directly by LLM analysis. The LLM applied human-like judgment to analyze code snippets and generate labeled examples, allowing for nuanced understanding of security context and edge cases.

**Key Advantages**:
- Context-aware labeling (beyond regex patterns)
- Natural language explanations
- Handles edge cases and obfuscation techniques
- No deterministic script required  

---

## JSONL Format

Each line is a valid JSON object with three fields:

```json
{
  "instruction": "Analyze the following code snippet and identify any secrets or sensitive credentials that should not be committed to version control.",
  "input": "<code snippet>",
  "output": "<ALERT: ... | No secrets detected. ...>"
}
```

### Field Descriptions

- **instruction**: Task description (identical across all examples for consistency)
- **input**: Code snippet to analyze (various programming languages)
- **output**: Expected model response (binary: ALERT or safe)

---

## Using with Fine-Tuning Scripts

### With Hugging Face Transformers

```python
from datasets import load_dataset

# Load JSONL files
dataset = load_dataset('json', data_files={
    'train': 'data/processed/smart-secrets-scanner-train.jsonl',
    'validation': 'data/processed/smart-secrets-scanner-val.jsonl'
})

print(f"Training examples: {len(dataset['train'])}")
print(f"Validation examples: {len(dataset['validation'])}")
```

### With Unsloth (for efficient fine-tuning)

```python
from unsloth import FastLanguageModel
from datasets import load_dataset

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Meta-Llama-3-8B",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Load dataset
dataset = load_dataset('json', data_files={
    'train': 'data/processed/smart-secrets-scanner-train.jsonl',
    'validation': 'data/processed/smart-secrets-scanner-val.jsonl'
})

# Format as Alpaca prompts
def format_prompts(examples):
    texts = []
    for instruction, input_text, output in zip(
        examples['instruction'], 
        examples['input'], 
        examples['output']
    ):
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(format_prompts, batched=True)
```

### With TRL (Transformer Reinforcement Learning)

```python
from trl import SFTTrainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./outputs/checkpoints",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=50,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    args=training_args,
    dataset_text_field="text",
    max_seq_length=2048,
)

trainer.train()
```

---

## Validation During Training

Monitor these metrics during fine-tuning:

### Training Loss
- Should decrease steadily over epochs
- Target: < 0.5 by epoch 3

### Validation Loss
- Should track training loss without diverging (no overfitting)
- If validation loss increases while training loss decreases → overfitting

### Accuracy Metrics
- **Precision**: Of detected secrets, how many are real? (Target: > 90%)
- **Recall**: Of real secrets, how many are detected? (Target: > 95%)
- **F1 Score**: Balance of precision and recall (Target: > 92%)

---

## Testing the Fine-Tuned Model

### Example Inference

```python
from transformers import pipeline

# Load fine-tuned model
generator = pipeline('text-generation', model='./models/merged/smart-secrets-scanner')

# Test input
test_code = 'aws_key = "AKIAIOSFODNN7EXAMPLE"'

# Generate prediction
prompt = f"""### Instruction:
Analyze the following code snippet and identify any secrets or sensitive credentials that should not be committed to version control.

### Input:
{test_code}

### Response:
"""

result = generator(prompt, max_new_tokens=150, temperature=0.1)
print(result[0]['generated_text'])
```

### Expected Output

```
ALERT: AWS credentials detected. The variable 'aws_key' contains an AWS access key (AKIA* pattern). This must be removed immediately. Use AWS IAM roles, environment variables, or AWS Secrets Manager.
```

---

## Benchmark Testing

Compare fine-tuned model against regex-based tools:

### Test Dataset Categories
1. **True Positives**: Real secrets (should detect)
2. **True Negatives**: Safe code (should NOT flag)
3. **Obfuscated Secrets**: Base64, split strings (should detect)
4. **Edge Cases**: Fallbacks, comments (context-dependent)

### Comparison Tools
- **detect-secrets** (Yelp)
- **GitGuardian ggshield**
- **TruffleHog**
- **Your fine-tuned LLM**

### Metrics to Compare
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)

---

## Expanding the Dataset

To improve model performance, consider adding:

### More Programming Languages
- Ruby (Rails apps)
- Rust (systems programming)
- C# (.NET applications)
- PHP (WordPress, Laravel)

### More Secret Types
- Shopify API keys
- Square payment tokens
- Zoom API credentials
- GraphQL tokens

### More Edge Cases
- Multi-line strings
- Encrypted values (still detectable patterns)
- Environment-specific patterns (.env files)
- CI/CD-specific secrets

### Recommended Size
- Start: 100-150 examples (current: 72)
- Good: 300-500 examples
- Excellent: 1000+ examples

---

## Integration with Pre-Commit Hooks

Once fine-tuned, integrate into a pre-commit hook:

```bash
# .git/hooks/pre-commit
#!/bin/bash

# Run GGUF quantized model via llama.cpp
git diff --cached --name-only | while read file; do
    result=$(llama-cli \
        --model models/gguf/smart-secrets-scanner-Q4_K_M.gguf \
        --prompt "### Instruction:\nAnalyze for secrets...\n### Input:\n$(git diff --cached $file)\n### Response:\n")
    
    if echo "$result" | grep -q "ALERT:"; then
        echo "⛔ Secret detected in $file"
        exit 1
    fi
done
```

---

## Next Steps

1. ✅ Dataset created (Task 20 - DONE)
2. ⏭️ Run fine-tuning (Task 08 - Next)
3. ⏭️ Merge and export to GGUF (Task 13)
4. ⏭️ Test Ollama deployment (Task 15)
5. ⏭️ Benchmark vs regex tools (New task)

**Ready to fine-tune! 🚀**

---

## Architecture Decision

For details on the LLM-driven dataset creation approach (vs. traditional script-based methods), see:  
**ADR 0007: LLM-Driven Dataset Creation for Secret Detection Training** (`adrs/0007-llm-driven-dataset-creation.md`)
