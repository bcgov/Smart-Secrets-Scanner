# ADR 0004: Jupyter Notebook Template for Fine-Tuning Workflow

## Status
Accepted

## Context
Interactive development and documentation are essential for reproducible machine learning workflows. Jupyter notebooks allow users to run, modify, and share code and documentation step-by-step. A template notebook provides a clear starting point for environment setup, data loading, model training, adapter merging, GGUF export, and inference.

## Decision
- The project will include a `notebooks/` subfolder containing a template Jupyter notebook for the full fine-tuning and deployment workflow.
- The notebook will be organized into markdown and code cells for each major step:
  - Environment setup and dependency installation
  - Data loading and preprocessing
  - Model loading and fine-tuning
  - Adapter merging and GGUF export
  - Inference and evaluation
  - Deployment instructions
- The notebook will serve as both documentation and an executable workflow for new users and contributors.

## Consequences
- Users can quickly reproduce, modify, and share the fine-tuning workflow.
- The notebook format supports both code and rich documentation.
- Future updates to the workflow can be reflected in the template notebook.

## Alternatives Considered
- Only providing scripts and markdown documentation (less interactive and reproducible)
- Using other interactive formats (Jupyter is the most widely supported)

---
