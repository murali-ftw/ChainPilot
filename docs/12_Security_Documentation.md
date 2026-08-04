# Document 12 — Security Documentation

## Out of Scope for the HADES Model-Development Prototype

Status: Out of scope

---

## 1. Why this document is empty

This document previously specified JWT auth, RBAC, encryption, secrets management, prompt-injection protection, RAG security, and MCP execution security for a served, multi-user application. This project is currently scoped to **model development only** — an offline training/evaluation pipeline with no users, no auth, no network-facing service, and no agentic execution. There is no attack surface here beyond ordinary local-development hygiene (not committing credentials, not committing the trained model artifacts or dataset if they contain anything sensitive — they currently don't, since the dataset is synthetic).

## 2. If this work resumes

`architecture.md` (Layer 5 — MCP Servers, human-approval gate) and the git history of the earlier product-scoped version of this document are the reference points if a served, multi-user application with agentic execution is built later.

## 3. What to read instead

- **`10_AI_ML_Documentation.md`** — what is actually being built.
- **`db/README.md`** — the synthetic dataset's provenance (it is synthetic; there is no real customer or supplier data in this repository).

---

## Document Control

- Superseded by the model-development rescope. Not maintained until/unless a served application is back in scope.
