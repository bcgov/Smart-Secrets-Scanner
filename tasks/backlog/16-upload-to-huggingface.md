# Task: Deploy Models for On-Premises Hosting (Optional)

**Status: Backlog**

## Description
Prepare and deploy the LoRA adapter and/or GGUF model for on-premises hosting and internal use. This is for models trained on sensitive data that cannot be published to public repositories like Hugging Face.

## Steps
- Package model artifacts (LoRA adapter, GGUF model, Modelfile)
- Create internal model registry entry with metadata
- Copy models to secure on-premises storage or model server
- Set up access controls and authentication
- Test deployment with internal Ollama/llama.cpp instance
- Document deployment path and access instructions
- Create internal model card with training details and usage notes
- Ensure compliance with data security and privacy policies

## Dependencies
- Task 11 (Train LoRA Adapter) must be completed
- Task 13 (Merge and Export GGUF) must be completed

## Resources
- Internal model hosting documentation
- On-premises Ollama deployment guide
- Data security and privacy policies
- Example model card in `notebooks/model_card.yaml`

## Note
For models trained on sensitive data, DO NOT upload to public repositories like Hugging Face. Use on-premises hosting and internal model registries only.
