# ADR 0003: ML-Env-CUDA13 as Sibling Directory

## Status
Accepted

## Context
Managing Python and CUDA environments for deep learning projects can be complex and error-prone. The ML-Env-CUDA13 repository provides a robust, reusable setup for GPU-accelerated workflows. There are two main options for integrating this environment: copying it as a subfolder or referencing it as a sibling directory.

## Decision
- ML-Env-CUDA13 will be kept as a **separate sibling directory** at the same level as this project (not imported or copied as a subfolder).
- All scripts in this project will reference ML-Env-CUDA13 via relative paths (e.g., `../ML-Env-CUDA13/`).
- Environment setup, updates, and management will be performed in ML-Env-CUDA13 independently of this project.

## Consequences
- Easier updates and maintenance of the environment without affecting project files.
- Avoids duplication and potential version drift between projects.
- Promotes modularity and reuse of the environment across multiple ML projects.
- Documentation and scripts must clearly state the required directory structure and relative paths.

## Alternatives Considered
- Importing ML-Env-CUDA13 as a subfolder (rejected due to maintainability and update issues).
- Using a global system environment (not chosen for reproducibility and isolation).

---
