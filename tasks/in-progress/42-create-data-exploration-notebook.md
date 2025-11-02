# Task 42: Create Data Exploration Notebook

**Status:** Backlog  
**Priority:** MEDIUM  
**Created:** 2025-11-01  
**Related to:** Phase 1: Data Preparation (Steps 1-3), Approach 2: Jupyter Notebooks

## Prerequisites (Completed)

✅ **Task 01**: WSL2 Ubuntu environment configured  
✅ **Task 02**: NVIDIA drivers and CUDA Toolkit installed  
✅ **Task 17**: Data directory structure created  
✅ **Task 20**: JSONL dataset generated  
✅ **Task 47**: 1000-example dataset (v3) generated  
✅ **Task 34**: Data validation script created  

## Description
Create `notebooks/01_data_exploration.ipynb` - an interactive Jupyter notebook for exploring, validating, and visualizing the Smart Secrets Scanner training dataset.

## Requirements
- Load and inspect JSONL training/validation datasets
- Validate data quality (schema, balance, duplicates)
- Visualize dataset statistics and examples
- Provide interactive data exploration
- Document findings with narrative cells

## Acceptance Criteria
- [ ] `notebooks/01_data_exploration.ipynb` created
- [ ] Loads JSONL files with pandas or datasets library
- [ ] Validates data schema and quality
- [ ] Generates visualizations (class balance, token lengths, secret types)
- [ ] Displays sample examples (secrets and safe code)
- [ ] Interactive widgets for exploration (optional)
- [ ] Well-documented with markdown cells
- [ ] Can run from start to finish without errors

## Notebook Sections

### 1. Introduction & Setup
- Project overview
- Import required libraries
- Set paths and configurations

### 2. Load Training Data
- Load train.jsonl and val.jsonl
- Display dataset statistics
- Show data structure

### 3. Data Quality Checks
- Schema validation (instruction, input, output fields)
- Check for null/empty values
- Detect duplicates
- Token length analysis

### 4. Class Balance Analysis
- Count ALERT vs safe examples
- Visualize distribution (pie chart, bar chart)
- Verify 50/50 balance

### 5. Content Analysis
- Secret types covered (AWS, Stripe, GitHub, etc.)
- Programming languages represented
- Safe pattern categories

### 6. Example Showcase
- Display random secret examples
- Display random safe examples
- Interactive exploration of specific patterns

### 7. Recommendations
- Data quality summary
- Suggestions for additional examples
- Next steps for fine-tuning

## Dependencies
- Task 20: JSONL datasets created
- Python packages: pandas, matplotlib, seaborn, datasets (add to requirements.txt)
- Jupyter environment

## Usage
```bash
# Install Jupyter
pip install jupyter ipywidgets

# Launch notebook
jupyter notebook notebooks/01_data_exploration.ipynb

# Or use VS Code Jupyter extension
code notebooks/01_data_exploration.ipynb
```

## Expected Outputs
- Dataset statistics table
- Class balance visualizations
- Token length distribution histogram
- Secret types breakdown
- Example displays with syntax highlighting

## Success Criteria
- Users can understand dataset composition
- Data quality issues are identified
- Visualizations are clear and informative
- Notebook runs without errors
- Serves as documentation for the dataset

## Related Tasks
- Task 18: JSONL training data template (data format reference)
- Task 20: Generate JSONL dataset (data source)
- Task 34: Data validation script (similar checks, CLI version)
- Task 43: Fine-tuning notebook (next step in workflow)
