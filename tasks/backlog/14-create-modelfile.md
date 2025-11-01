# Task: Create Ollama Modelfile

**Status: Backlog**

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
