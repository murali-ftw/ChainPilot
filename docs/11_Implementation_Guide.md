# Document 11 — Implementation Guide

## HADES Model-Development Prototype

Version: 2.0 — rescoped to a single-researcher Python/ML workflow (Docker, CI/CD, service deployment, and multi-contributor process removed)
Status: Baseline
Consistent with: `01_Product_Requirement_Document.md`–`10_AI_ML_Documentation.md`

---

## 1. Purpose

This document specifies how the ML pipeline is actually developed and run: environment setup, dependency management, code organization, Git practice, testing gates, and how to reproduce or roll back to a previous training run. There is no deployment step in the product sense — "shipping" here means a model version and its evaluation results are recorded in `model_registry`.

## 2. Scope

Covers the full research-prototype workflow: local Python environment, running the data/graph/training/evaluation pipeline, and the lightweight process discipline needed to keep every run reproducible.

## 3. Assumptions

- A single researcher, working intermittently, is the entire "team." There is no multi-contributor review process, no sprint cadence, and no cross-team ownership split to coordinate.
- Local development happens directly on the researcher's machine — no container orchestration is required at this scale.
- GitHub is the source host; CI, if used at all, is a lightweight lint/test check, not a deployment pipeline.

## 4. Dependencies

`02_Technical_Requirement_Specification.md` (technology stack), `05_Database_Design.md` (schema and migrations), `13_Testing_Documentation.md` (test suite).

## 5. Development Environment

| Requirement | Version/Tool |
|---|---|
| Python | 3.11+ |
| PostgreSQL client tools | `psql` 15+ (or run PostgreSQL itself in a single local container/service, since no multi-service stack is being orchestrated) |
| Package management | `uv` or `pip` + `venv` |
| Lint/format | `ruff` |
| Notebook/script environment | Jupyter or plain scripts — either is fine; this is a research pipeline, not a codebase with a UI |

## 6. Dependencies

| Category | Key Packages |
|---|---|
| ML core | `torch`, `torch-geometric` |
| Data processing | `pandas`, `numpy` |
| Evaluation | `scikit-learn`, `scipy` (for bootstrap/DeLong confidence intervals) |
| Database | `psycopg2` or `sqlalchemy` (read/write against the schema in `05_Database_Design.md`) |
| Testing | `pytest` |

Pinned in `pyproject.toml`/`requirements.txt`. There is no separate frontend, backend, RAG/LLM, optimization, or notification dependency group — none of that exists in this project (`02_Technical_Requirement_Specification.md` §4).

## 7. Database Setup

No Docker Compose stack. A single local PostgreSQL instance is sufficient:

```bash
createdb chainpilot
python db/load_data.py --dsn "postgresql://user:pass@localhost:5432/chainpilot"
```

`load_data.py` applies `db/schema.sql`, loads the synthetic CSVs, and runs the verification queries described in `db/README.md` — FK integrity, chronology, label-window containment. **Do not proceed to graph construction if this step reports a failure.**

## 8. Suggested Code Layout

Not prescriptive — this is a research pipeline, and the layout should stay easy to change. A reasonable starting point:

```
ml/
├── data/            # feature engineering, leakage contract (10_AI_ML_Documentation.md §6)
├── graph/           # graph snapshot assembly (06_Graph_Database_Design.md)
├── models/          # HGT encoder, depth selection, Transformer 2, prediction heads
├── train.py         # training loop, run inventory (10_AI_ML_Documentation.md §9.2)
├── evaluate.py       # metrics, confidence intervals, per-claim validation routes
└── tests/           # leakage test, unit tests (13_Testing_Documentation.md)
```

## 9. Testing Gate

Full detail: `13_Testing_Documentation.md`. Minimum bar before trusting any result: the leakage test passes (`10_AI_ML_Documentation.md` §9.5), and the pipeline's unit tests (feature derivation, as-of edge filtering, label-window invariants) are green. There is no CI/CD pipeline enforcing this automatically at this scale — it is a manual discipline, stated here so it doesn't get skipped under "I'll just try this one thing quickly" pressure.

## 10. Git Strategy

- **Trunk-based, no branch model overhead.** `main` (or the working branch) holds the current state of the pipeline; feature branches are optional, used when a change is large enough to want isolation before merging (e.g., building the depth gate).
- Commit messages reference what changed in plain terms; there is no multi-team requirement-ID tagging convention to maintain.
- **`git_commit` on every `model_registry` row is the actual discipline that matters here** — not the branch model. A training run's code state must be committed before the run, so the `git_commit` field in `model_registry` (`05_Database_Design.md` §6.22) is truthful.

## 11. Running a Training Run ("Deployment," for This Project)

| Step | Detail |
|---|---|
| Commit | Commit the current code state before starting a run, so `git_commit` is accurate |
| Build snapshots | Run graph snapshot construction (`04_Application_Flow.md` §5) for the relevant `t₀` range, if not already built |
| Train | Run the training script; it writes one `model_registry` row on completion |
| Evaluate | Run the evaluation script against the held-out split; it writes `model_evaluation_runs` rows with confidence intervals |
| Compare | Read `model_evaluation_runs` for the relevant claim (`10_AI_ML_Documentation.md` §9.4) and report the result, including a null result if that's what the evidence shows |

## 12. Reproducing or Rolling Back to a Previous Run

There is no live service to roll back — "rollback" here means re-running or re-inspecting a previous result:

| Step | Detail |
|---|---|
| Find the run | Query `model_registry` for the `model_version` of interest |
| Reproduce | `git checkout <that run's git_commit>`, rebuild the same dataset snapshot (`training_dataset` field), re-run with the same `hyperparameters` |
| Compare, don't overwrite | `model_evaluation_runs` is append-only (`05_Database_Design.md` §6.21) — a re-run produces a new row, never overwrites the old one, so results stay comparable across attempts |

## 13. Risks

| ID | Risk | Mitigation |
|---|---|---|
| IG-01 | A training run happens against uncommitted code, making `git_commit` inaccurate | Commit-before-run discipline (Section 11); spot-check `model_registry.git_commit` against `git log` periodically |
| IG-02 | Manual steps (Section 11) are error-prone without automation | Script the sequence (a `Makefile` or shell script) rather than running each step ad hoc by memory |
| IG-03 | `model_evaluation_runs`/`model_registry` schema (`05_Database_Design.md` §6.21–6.22) isn't actually built yet, and results get logged informally instead (e.g., only in a notebook) | Build these two tables early — before the first ablation run, per `14_Model_Development_Roadmap.md` — specifically to prevent this |

## 14. Future Extension

If this project resumes as a multi-contributor or served-application build, the earlier product-scoped version of this document (Docker Compose, CI/CD, trunk-based multi-branch strategy, staged deployment/rollback) is recoverable from git history and from `architecture.md`.

---

## Document Control

- **Purpose:** Section 1. **Scope:** Section 2. **Assumptions:** Section 3. **Dependencies:** Section 4. **Risks:** Section 13. **Future Extension:** Section 14.
- Baseline for `14_Model_Development_Roadmap.md`.
