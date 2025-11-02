#!/usr/bin/env python3
"""
Evaluate fine-tuned Smart Secrets Scanner model
"""
import json
import argparse
from pathlib import Path
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

def load_model(model_path, use_lora=False, base_model_path=None, load_in_4bit=False):
    """Load fine-tuned model and tokenizer"""
    print(f"🔽 Loading model from {model_path}...")
    
    if use_lora:
        if not base_model_path:
            raise ValueError("base_model_path required when use_lora=True")
        
        print(f"  Loading base model: {base_model_path}")
        
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        
        print(f"  Loading LoRA adapter: {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    
    print("✅ Model loaded successfully")
    return model, tokenizer

def load_test_data(test_path):
    """Load test JSONL dataset"""
    print(f"📂 Loading test data from {test_path}...")
    examples = []
    with open(test_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"✅ Loaded {len(examples)} test examples")
    return examples

def format_prompt(instruction, code_input):
    """Format code as Alpaca prompt"""
    prompt = f"""### Instruction:
{instruction}

### Input:
{code_input}

### Response:
"""
    return prompt

def run_inference(model, tokenizer, example, max_new_tokens=150):
    """Run inference on a single example"""
    prompt = format_prompt(example['instruction'], example['input'])
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the response part
    if "### Response:" in response:
        response = response.split("### Response:")[1].strip()
    
    return response

def parse_label(text):
    """Parse whether text contains an alert (1) or is safe (0)"""
    text_upper = text.upper()
    if "ALERT" in text_upper:
        return 1
    elif "SAFE" in text_upper:
        return 0
    else:
        # If neither keyword, check for other secret indicators
        if any(word in text_upper for word in ["SECRET", "CREDENTIAL", "DETECTED", "HARDCODED"]):
            return 1
        return 0

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    # True positives, false positives, true negatives, false negatives
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
    tn = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 0)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'confusion_matrix': {
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        },
        'total_examples': len(y_true)
    }
    
    return metrics

def print_metrics(metrics):
    """Print metrics in a nice format"""
    print("\n" + "=" * 60)
    print("📊 Evaluation Results")
    print("=" * 60)
    print(f"Total Examples: {metrics['total_examples']}")
    print(f"\n🎯 Accuracy:  {metrics['accuracy']:.1%}")
    print(f"🎯 Precision: {metrics['precision']:.1%}")
    print(f"🎯 Recall:    {metrics['recall']:.1%}")
    print(f"🎯 F1 Score:  {metrics['f1_score']:.1%}")
    
    cm = metrics['confusion_matrix']
    print(f"\n📋 Confusion Matrix:")
    print(f"  True Positives:  {cm['true_positives']}")
    print(f"  False Positives: {cm['false_positives']}")
    print(f"  True Negatives:  {cm['true_negatives']}")
    print(f"  False Negatives: {cm['false_negatives']}")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Smart Secrets Scanner")
    parser.add_argument(
        '--model',
        default='models/fine-tuned/smart-secrets-scanner-lora',
        help='Path to fine-tuned model or LoRA adapter'
    )
    parser.add_argument(
        '--base-model',
        default='models/base/Meta-Llama-3.1-8B',
        help='Path to base model (required if using LoRA)'
    )
    parser.add_argument(
        '--use-lora',
        action='store_true',
        default=True,
        help='Load LoRA adapter instead of merged model'
    )
    parser.add_argument(
        '--load-in-4bit',
        action='store_true',
        help='Load model in 4-bit quantization'
    )
    parser.add_argument(
        '--test-data',
        default='data/evaluation/smart-secrets-scanner-test.jsonl',
        help='Path to test data'
    )
    parser.add_argument(
        '--output-dir',
        default='outputs/evaluation',
        help='Output directory for results'
    )
    parser.add_argument(
        '--max-examples',
        type=int,
        help='Maximum number of examples to evaluate (for quick tests)'
    )
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(
        args.model,
        use_lora=args.use_lora,
        base_model_path=args.base_model if args.use_lora else None,
        load_in_4bit=args.load_in_4bit
    )
    
    # Load test data
    test_examples = load_test_data(args.test_data)
    
    if args.max_examples:
        test_examples = test_examples[:args.max_examples]
        print(f"⚠️  Limiting to {args.max_examples} examples for quick test")
    
    # Run evaluation
    print(f"\n🔍 Running evaluation on {len(test_examples)} examples...")
    predictions = []
    ground_truth = []
    results = []
    
    for i, example in enumerate(tqdm(test_examples, desc="Evaluating")):
        # Get prediction
        prediction = run_inference(model, tokenizer, example)
        
        # Parse labels
        pred_label = parse_label(prediction)
        true_label = parse_label(example['output'])
        
        predictions.append(pred_label)
        ground_truth.append(true_label)
        
        # Store result
        results.append({
            'input': example['input'],
            'expected': example['output'],
            'predicted': prediction,
            'correct': pred_label == true_label
        })
    
    # Calculate metrics
    metrics = calculate_metrics(ground_truth, predictions)
    metrics['timestamp'] = datetime.now().isoformat()
    metrics['model_path'] = args.model
    metrics['test_data_path'] = args.test_data
    
    # Print metrics
    print_metrics(metrics)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    metrics_file = output_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n💾 Metrics saved to: {metrics_file}")
    
    # Save detailed results
    results_file = output_dir / 'detailed_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Detailed results saved to: {results_file}")
    
    # Print sample errors
    errors = [r for r in results if not r['correct']]
    if errors:
        print(f"\n⚠️  Found {len(errors)} errors. Sample errors:")
        for i, error in enumerate(errors[:3], 1):
            print(f"\n{i}. Input: {error['input'][:100]}...")
            print(f"   Expected: {error['expected'][:100]}...")
            print(f"   Predicted: {error['predicted'][:100]}...")
    else:
        print("\n✅ Perfect! No errors found!")
    
    print("\n✅ Evaluation complete!")

if __name__ == '__main__':
    main()
