# Task 45: Create Deployment Testing Notebook

**Status:** Backlog  
**Priority:** MEDIUM  
**Created:** 2025-11-01  
**Related to:** Phase 4: Testing & Deployment (Steps 12-13), Approach 2: Jupyter Notebooks

## Prerequisites (Completed)

✅ **Task 01-05**: Environment setup (WSL2, NVIDIA, ML-Env-CUDA13, dependencies)  
✅ **Task 08**: Fine-tuning completed (Iteration 4)  
✅ **Task 31**: Test dataset created  
✅ **Task 37**: Inference script created  

**Pending:**  
⏳ **Task 38**: Merge LoRA adapter (needed for GGUF)  
⏳ **Task 39**: Convert to GGUF format  
⏳ **Task 40**: Create Modelfile for Ollama  

## Description
Create `notebooks/04_deployment_testing.ipynb` - an interactive notebook for deploying the model to Ollama and testing the pre-commit scanning workflow.

## Requirements
- Generate Ollama Modelfile
- Deploy model to Ollama
- Test Ollama API
- Simulate pre-commit scanning
- Validate end-to-end workflow
- Document deployment process

## Acceptance Criteria
- [ ] `notebooks/04_deployment_testing.ipynb` created
- [ ] Generates Modelfile with optimal settings
- [ ] Imports model to Ollama
- [ ] Tests Ollama API responses
- [ ] Simulates pre-commit hook workflow
- [ ] Tests scanning on sample files
- [ ] Measures inference performance
- [ ] Documents deployment checklist
- [ ] Provides troubleshooting guidance

## Notebook Sections

### 1. Introduction
- Deployment overview
- Prerequisites checklist
- Architecture diagram

### 2. Generate Modelfile
- Load GGUF path
- Create Modelfile with system prompt
- Configure parameters (temperature, context)
- Display generated Modelfile

### 3. Deploy to Ollama
- Check Ollama is running
- Import model with `ollama create`
- Verify model in Ollama list
- Test basic inference

### 4. Test Ollama API
- Send requests via Python API
- Test with secret examples
- Test with safe code examples
- Display response times
- Validate output format

### 5. Pre-Commit Workflow Simulation
- Create test repository structure
- Add sample files (with/without secrets)
- Simulate git add/commit flow
- Run scanner on staged files
- Show blocking behavior

### 6. Performance Testing
- Measure inference time per file
- Test batch scanning
- Monitor resource usage
- Compare quantization performance (Q4 vs Q8)

### 7. Integration Examples
- Python API usage
- Bash script integration
- Git hook setup
- CI/CD pipeline example

### 8. Production Checklist
- Deployment readiness assessment
- Security considerations
- Monitoring setup
- Rollback procedure
- Team onboarding guide

### 9. Troubleshooting
- Common issues and solutions
- Ollama not running
- Model not found
- Slow inference
- False positives handling

### 10. Next Steps
- Team deployment instructions
- Expand to other repos
- Continuous improvement plan
- Metrics collection strategy

## Dependencies
- Task 40: Create Modelfile script (reference)
- Task 41: Pre-commit scan script (reference)
- Task 39: GGUF model created
- Ollama installed
- Python packages: requests, subprocess

## Interactive Testing
```python
# Test Ollama API
import requests

def test_secret_detection(code_snippet):
    response = requests.post('http://localhost:11434/api/generate', json={
        'model': 'smart-secrets-scanner',
        'prompt': code_snippet
    })
    return response.json()

# Interactive widget for testing
import ipywidgets as widgets
from IPython.display import display

code_input = widgets.Textarea(
    placeholder='Paste code to scan...',
    layout=widgets.Layout(width='100%', height='200px')
)
scan_button = widgets.Button(description='Scan for Secrets')
output_display = widgets.Output()
```

## Usage
```bash
# Ensure Ollama is installed and running
ollama serve

# Launch notebook
jupyter notebook notebooks/04_deployment_testing.ipynb

# This notebook assumes model has been trained and converted to GGUF
```

## Expected Outputs
- Modelfile: `Modelfile`
- Ollama model: `smart-secrets-scanner` (imported)
- Performance benchmarks table
- Example API responses
- Pre-commit hook demo
- Deployment documentation

## Success Criteria
- Model successfully deployed to Ollama
- API responds correctly to test cases
- Pre-commit workflow validated
- Performance meets requirements (<5 sec per file)
- Clear documentation for team rollout
- Troubleshooting guide is comprehensive

## Advanced Features (Optional)
- Interactive code scanner widget
- Real-time API testing playground
- Performance comparison dashboard
- Multi-model testing (compare versions)
- Team feedback collection form

## Production Considerations
```python
# Model versioning
MODEL_VERSION = "v1.0.0"

# Performance targets
MAX_INFERENCE_TIME = 5.0  # seconds per file
MIN_PRECISION = 0.90
MIN_RECALL = 0.95

# Monitoring alerts
def check_performance_degradation():
    # Compare current metrics with baseline
    pass
```

## Related Tasks
- Task 40: Create Modelfile script
- Task 41: Pre-commit scan script
- Task 15: Test Ollama deployment
- Task 27: Integrate pre-commit hooks
- Task 14: Create Modelfile (reference)

## Deployment Checklist (in notebook)
- [ ] GGUF model created and tested
- [ ] Modelfile configured with optimal parameters
- [ ] Ollama running and accessible
- [ ] Model imported successfully
- [ ] API tests pass (100% success rate)
- [ ] Pre-commit hook tested locally
- [ ] Performance benchmarks meet targets
- [ ] Team documentation complete
- [ ] Rollback plan documented
- [ ] Monitoring setup (optional)
