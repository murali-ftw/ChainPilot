# Document 9 — REST API Documentation

## Out of Scope for the HADES Model-Development Prototype

Status: Out of scope

---

## 1. Why this document is empty

This document previously specified REST endpoints for authentication, entity CRUD, predictions, chat, what-if simulation, recommendations, optimization, approvals, and alerts. This project is currently scoped to **model development only** — there is no served API. Training and evaluation run as offline scripts/notebooks against the database in `05_Database_Design.md`, not behind an API.

## 2. If this work resumes

`architecture.md` and the git history of the earlier product-scoped version of this document are the reference points if an API is built later to serve trained-model inference.

## 3. What to read instead

- **`10_AI_ML_Documentation.md`** — how inference actually runs in this prototype (a direct forward pass over a `HeteroData` snapshot, not a request/response cycle).
- **`14_Model_Development_Roadmap.md`** — build order.

---

## Document Control

- Superseded by the model-development rescope. Not maintained until/unless a served API is back in scope.
