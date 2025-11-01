# ADR 0005: On-Premises Model Deployment for Sensitive Data

## Status
Accepted

## Context
Many fine-tuned models will be trained on sensitive, proprietary, or confidential data that cannot be shared publicly. Public model repositories like Hugging Face Hub are not suitable for such models. The organization requires secure, on-premises hosting and deployment solutions for these models while maintaining the same workflow and deployment capabilities.

## Decision
- Models trained on sensitive data will NOT be uploaded to public repositories like Hugging Face Hub.
- All sensitive models will be deployed to on-premises infrastructure with appropriate access controls and security measures.
- The same GGUF export and Ollama deployment workflow will be used, but hosted internally.
- Model artifacts (LoRA adapters, GGUF files, Modelfiles) will be stored in secure internal storage or model registries.
- Internal documentation and model cards will be maintained for tracking and versioning.
- Access to models will be controlled through internal authentication and authorization mechanisms.

## Deployment Options
1. **On-Premises Ollama Server**: Deploy GGUF models to an internal Ollama instance accessible only within the organization network
2. **Internal Model Registry**: Store and version model artifacts in a private, secure model repository
3. **Self-Hosted Inference API**: Set up llama.cpp or other inference engines behind internal APIs
4. **Local Deployment**: Distribute GGUF models directly to authorized users for local Ollama deployment

## Consequences
- Enhanced data security and compliance with privacy policies
- Models remain under organizational control
- Access can be audited and restricted as needed
- No risk of sensitive training data exposure through public repositories
- Requires internal infrastructure and model hosting capabilities
- Model sharing limited to authorized internal users
- Version control and model registry must be managed internally

## Alternatives Considered
- Private Hugging Face Hub (rejected due to external dependency and cost)
- Public model hosting (rejected due to data sensitivity requirements)

---
