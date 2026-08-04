# Document 8 — Backend Design

## Out of Scope for the HADES Model-Development Prototype

Status: Out of scope

---

## 1. Why this document is empty

This document previously specified a full application backend — controllers/services/repositories layering, REST API modules, caching, queues, auth. This project is currently scoped to **model development only**: a data pipeline and a training/evaluation codebase, not a served application. There is no backend to design in the product sense — the actual code organization for the ML pipeline (feature engineering, graph construction, model, training loop, evaluation) is specified in `10_AI_ML_Documentation.md` and built out step by step in `14_Model_Development_Roadmap.md`, not laid out as a controller/service/repository stack here.

## 2. If this work resumes

`architecture.md` (Layers 1–6) is the reference point for the full system's backend shape if a served application is built later.

## 3. What to read instead

- **`10_AI_ML_Documentation.md`** — the ML pipeline's actual structure.
- **`14_Model_Development_Roadmap.md`** — build order, including a lightweight code-layout recommendation for the training codebase.

---

## Document Control

- Superseded by the model-development rescope. Not maintained until/unless a served backend is back in scope.
