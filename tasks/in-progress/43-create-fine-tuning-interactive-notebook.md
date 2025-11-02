# Task 43: Create Fine-Tuning Interactive Notebook

**Status:** Backlog  
**Priority:** MEDIUM  
**Created:** 2025-11-01  
**Related to:** Phase 2: Model Fine-Tuning (Steps 4-6), Approach 2: Jupyter Notebooks

## Prerequisites (Completed)

✅ **Task 01-05**: Environment setup (WSL2, NVIDIA, ML-Env-CUDA13, dependencies)  
✅ **Task 22**: Llama 3.1-8B base model downloaded  
✅ **Task 30**: Training configuration file created  
✅ **Task 36**: Fine-tuning Python script created (reference implementation)  
✅ **Task 47**: 1000-example dataset generated  

## Description
Create `notebooks/02_fine_tuning_interactive.ipynb` - an interactive Jupyter notebook for fine-tuning Llama 3 with LoRA, featuring real-time progress tracking and visualization.

## Requirements
- Setup ML-Env-CUDA13 environment
- Download base model (if needed)
- Configure LoRA parameters
- Train model with live progress bars
- Visualize training metrics in real-time
- Save checkpoints and final adapter

## Acceptance Criteria
- [ ] `notebooks/02_fine_tuning_interactive.ipynb` created
- [ ] Checks CUDA availability and GPU memory
- [ ] Loads configuration from `config/training_config.yaml`
- [ ] Implements LoRA fine-tuning with progress tracking
- [ ] Displays loss curves during training
- [ ] Shows validation metrics at each eval step
- [ ] Saves LoRA adapter to `models/fine-tuned/`
- [ ] Well-documented with markdown explanations
- [ ] Can resume from checkpoints

## Notebook Sections

### 1. Environment Setup
- Check CUDA and GPU availability
- Import libraries (transformers, peft, trl, datasets)
- Display system information (GPU memory, CUDA version)

### 2. Configuration
- Load training config from YAML
- Display hyperparameters in readable format
- Allow interactive parameter adjustment (optional)

### 3. Load Base Model
- Download Llama 3 8B if not present
- Load with 4-bit quantization
- Display model architecture and size

### 4. Prepare Dataset
- Load JSONL training and validation data
- Format as Alpaca prompts
- Show example formatted prompts
- Tokenize and display token statistics

### 5. Configure LoRA
- Apply LoRA configuration
- Display trainable parameters count
- Show which modules are being adapted

### 6. Training
- Setup TrainingArguments with progress tracking
- Initialize SFTTrainer
- Train with live updates:
  - Progress bar for epochs and steps
  - Real-time loss plotting
  - Validation metrics display
  - GPU memory monitoring

### 7. Results Visualization
- Plot training loss curve
- Plot validation loss curve
- Display learning rate schedule
- Show final metrics

### 8. Save & Next Steps
- Save LoRA adapter
- Display save location
- Instructions for merging (link to next notebook)

## Dependencies
- Task 30: Training config file
- Task 22: Base model downloaded
- Task 36: Fine-tuning script (reference implementation)
- Python packages: transformers, peft, trl, datasets, bitsandbytes, matplotlib, ipywidgets

## Interactive Features
```python
# Real-time loss plotting
from IPython.display import clear_output
import matplotlib.pyplot as plt

# Progress bars
from tqdm.notebook import tqdm

# Interactive widgets (optional)
import ipywidgets as widgets
```

## Usage
```bash
# Launch notebook
jupyter notebook notebooks/02_fine_tuning_interactive.ipynb

# Ensure you have enough GPU memory (12GB+)
# Training may take 1-3 hours depending on GPU
```

## Expected Outputs
- Real-time training progress with loss plots
- Validation metrics table
- GPU memory usage graphs
- Saved LoRA adapter in `models/fine-tuned/smart-secrets-scanner-lora/`

## Success Criteria
- Users can follow along and understand each step
- Training completes successfully
- Metrics are visualized clearly
- Adapter is saved and ready for merging
- Notebook is educational and interactive

## Advanced Features (Optional)
- Hyperparameter tuning widgets (sliders for learning rate, batch size)
- Experiment tracking integration (W&B dashboard)
- Checkpoint browser (load and compare checkpoints)
- Early stopping visualization

## Related Tasks
- Task 36: Fine-tuning Python script (CLI equivalent)
- Task 23: Review training logs (similar monitoring)
- Task 33: Experiment tracking (W&B integration)
- Task 44: Model evaluation notebook (next step)
