# Task: Create Ollama Modelfile

**Status: Backlog**

## Prerequisites (Completed)

✅ **Task 01-05**: Environment setup (WSL2, NVIDIA, ML-Env-CUDA13, dependencies)  

**Pending:**  
⏳ **Task 08**: Fine-tuning Iteration 4 (in-progress)  
⏳ **Task 13/39**: GGUF model created  
⏳ **Task 40**: Modelfile generation script (to be created)  

## Description
Create a Modelfile to define the chat template, system prompt, and parameters for Ollama deployment.

**🔗 Implemented by Task 40: Create Modelfile Generation Script**

See `tasks/backlog/40-create-modelfile-script.md` for the complete implementation of `scripts/create_modelfile.py`.

## Steps
- Create `Modelfile` in the project root or outputs directory
- Specify the GGUF model path (use `FROM` directive)
- Define chat template (e.g., ChatML, Alpaca, etc.)
- Add system prompt with model persona/instructions
- Configure parameters (temperature, context length, stop tokens)
- Save the Modelfile

## Resources
- Example Modelfile from previous work
- Ollama documentation: https://github.com/ollama/ollama/blob/main/docs/modelfile.md
