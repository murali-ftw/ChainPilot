# Document 13 — Testing Documentation

## Graph Neural Network and Generative AI-Based Supply Chain Risk Prediction System

Version: 1.0
Status: Baseline
Consistent with: Documents 1–12

---

## 1. Purpose

This document specifies the testing strategy across unit, integration, system, API, UI, performance, security, UAT, and edge-case testing, and maps representative test cases to the requirements in Document 1 and the controls in Document 12. It defines the quality gate referenced in Document 11, Section 9 (CI/CD) and Section 12.

## 2. Scope

Covers test coverage for both phases; Phase 2 test cases are written when Phase 2 features are implemented, following the same test-type structure established for Phase 1 rather than a different testing approach.

## 3. Assumptions

- `pytest` (backend/ML) and `vitest`/`jest` + `react-testing-library` (frontend) are the test runners, per Document 11, Section 6.
- A dedicated test database (migrated fresh per test run or per suite) is used for integration/API tests — no test ever runs against the staging/demo database.
- UAT is conducted by the developer role-playing each persona from Document 1, Section 7, given the absence of real end users on an academic project, supplemented by the Academic Supervisor's review where available.

## 4. Dependencies

Document 1 (requirements/user stories test cases trace to), Document 5 (schema constraints validated), Document 8 (layers under test), Document 9 (API contracts under test), Document 12 (security controls under test).

## 5. Testing Pyramid

```mermaid
flowchart TD
    UAT["UAT\n(persona-driven acceptance)"]
    SYS["System Tests\n(end-to-end flows, Document 4)"]
    API["API Tests\n(contract-level, Document 9)"]
    UI["UI Tests\n(component + screen, Document 3)"]
    INT["Integration Tests\n(service + DB)"]
    UNIT["Unit Tests\n(service/repository logic, ML functions)"]

    UNIT --> INT --> API
    UNIT --> UI
    INT --> SYS
    API --> SYS
    UI --> SYS
    SYS --> UAT
```

## 6. Unit Testing

| Target | Example Test Case | Requirement Reference |
|---|---|---|
| `AuthService.verify_password` | Correct password returns True; incorrect returns False; hash never equals plaintext | FR-AUTH-01 |
| `AuthService.lock account` | 5th consecutive failed attempt sets `locked_until` | FR-AUTH-05 |
| `ApprovalService.reject` | Raises validation error if `reason` is empty | FR-MCP-04, Document 5 §6.17 |
| `RiskScoreRepository` mapping | Polymorphic `entity_type`/`entity_id` row maps to correct DTO | Document 5 §6.13 |
| Feature encoding functions (`ml/gnn/features.py`) | Given raw supplier record, output matches expected normalized feature vector | Document 10 §5 |
| GNNExplainer wrapper | Given a mock prediction, returns a subgraph with contribution weights summing within an expected tolerance | FR-GNN-05 |

**Delivery Phase:** Phase 1 (auth, entity, prediction, explainability units); Phase 2 (approval, chatbot intent classification, recommendation similarity units)

## 7. Integration Testing

| Test Case | Verifies | Requirement Reference |
|---|---|---|
| Graph Construction Service ingests a new shipment row → `HeteroData` snapshot updates incrementally | FR-GC-07, Document 6 §11 |
| Document upload → parsing → structured fields populate `documents.extracted_fields` | FR-GC-05 |
| GNN Inference Service scores a known graph fixture → scores within expected range | FR-GNN-01–03 |
| Approval Flow: approve action_request → MCP Execution Service invoked with correct payload | Document 4 §10–12 |
| RAG Retrieval Service: seeded evidence corpus → query returns expected top-k chunk | FR-RAG-02, Document 7 §9 |
| Alert Evaluation job: risk score crossing configured threshold creates exactly one `alerts` row | FR-MCP-05, Document 4 §11 |

**Delivery Phase:** Phase 1 (graph construction, inference); Phase 2 (RAG, approval, MCP, alerts)

## 8. System (End-to-End) Testing

Traces directly to the flows in Document 4:

| Flow Under Test | Scenario |
|---|---|
| Authentication → Dashboard | Login with valid credentials renders Dashboard within NFR-01/02 latency budget |
| Graph Construction → Prediction | New supplier delay record flows through to an updated risk score visible on the Risk Dashboard |
| Chatbot → Approval → MCP → ERP | User asks "what should we do," approves in chat, action executes against sandbox ERP, `action_log` reflects success |
| Alert → Notification | Threshold breach produces both an in-app alert and a (mocked) Slack/email delivery record |
| What-If Simulator | Perturbed lead time produces a different impact score than baseline, and the persisted graph is unchanged after the session |

**Delivery Phase:** Phase 1 (first two rows); Phase 2 (remaining rows)

## 9. API Testing

Contract tests against every endpoint in Document 9, using `httpx`/`pytest-asyncio`:

| Test Case | Verifies |
|---|---|
| `POST /auth/login` with valid/invalid/locked credentials | Correct status codes and response shapes (Document 9 §6.1) |
| `GET /suppliers` pagination and filter parameters | Envelope shape, `422` on invalid `risk_level` |
| `POST /approvals/{id}/reject` without `reason` | Returns `422 VALIDATION_ERROR` |
| `POST /approvals/{id}/approve` on an already-decided request | Returns `409 CONFLICT` (Document 9 §13, Document 12 §11) |
| `PUT /alert-thresholds/{entity}/{metric}` as non-admin | Returns `403 FORBIDDEN` |
| Every entity family (Suppliers, Warehouses, Orders, Shipments, Products, Inventory) | Shared pattern (Document 9 §8.1) tested once per entity to catch drift (Document 9, risk API-01) |

**Delivery Phase:** Phase 1 (auth, entity, prediction endpoints); Phase 2 (chat, simulate, recommendation, approval, alert endpoints)

## 10. UI Testing

Per screen (Document 3, Section 6), automated component tests plus manual verification of each state:

| Test Case | Screen | Verifies |
|---|---|---|
| Invalid login shows generic error, not field-specific | Login | Document 3 §6.1; NFR-08 |
| Risk Dashboard renders skeleton → data → error/retry states | Risk Dashboard | Document 3 §6.4 loading/error states |
| Graph view highlights explanation subgraph nodes in correct risk color | Supply Chain Graph | FR-EXP-02 |
| Chatbot shows streaming indicator, then renders citations as clickable links | Chatbot | Document 3 §6.11 |
| Reject action blocked until a reason is entered | Alerts | FR-MCP-04 |
| Every screen: role without permission sees access-denied, not a broken page | All | Document 3 §7 |

**Delivery Phase:** Phase 1 (core screens); Phase 2 (chatbot, simulator, recommendation, alerts screens)

## 11. Performance Testing

| Test Case | Target | Requirement Reference |
|---|---|---|
| Single-entity inference latency under load | p95 ≤ 2s | NFR-01 |
| Graph view render/interaction with 5,000-node fixture | p95 interaction ≤ 500ms | NFR-02 |
| Chatbot end-to-end response latency | p95 ≤ 5s | NFR-03 |
| Concurrent dashboard load (N simulated users) does not degrade inference latency beyond target | Cache effectiveness (Document 8 §10) verified under load | Phase 2 |

**Delivery Phase:** Phase 1 (inference, graph render); Phase 2 (chatbot, concurrency)

## 12. Security Testing

| Test Case | Verifies | Requirement Reference |
|---|---|---|
| Expired/invalid JWT rejected on every protected endpoint | Document 12 §5 | NFR-07 |
| Deactivated user's still-unexpired token is rejected | Document 12 §5 | FR-AUTH-04 |
| Role without `approver` cannot call approve/reject endpoints even with a valid token | Document 12 §6 | FR-MCP-03 |
| Password hash never appears in any API response or log | Document 12 §7–8 | NFR-08 |
| Injected instruction inside a mocked RAG evidence chunk does not alter the LLM's action-recommendation schema | Document 12 §9 | Phase 2 |
| Action cannot reach `executed` status without a prior `approved` status and `decided_by` | Document 12 §11 | NFR-09 |
| Rate limit returns `429` after threshold exceeded on `/auth/login` | Document 12 §12 | Phase 1 |

**Delivery Phase:** Phase 1 (auth, RBAC, secrets); Phase 2 (prompt injection, approval-gate integrity)

## 13. User Acceptance Testing (UAT)

Persona-driven acceptance walkthroughs (Document 1, Section 7), each mapped to its user stories (Document 1, Section 10):

| Persona | UAT Scenario | User Stories Verified |
|---|---|---|
| Priya (Analyst) | Log in, view Risk Dashboard, drill into a flagged supplier, view explanation | US-PRED-01–05, US-EXP-01 |
| Rahul (Procurement Manager) | View alternative-supplier recommendation, approve raising a PO | US-REC-01–03 |
| Meera (Operations Manager) | View shortage risk in Inventory, receive a proactive alert | US-PRED-02, US-MCP-05 |
| Arjun (Admin) | Create a user, configure an alert threshold | US-AUTH-03, US-MCP-06 |
| Devika (Executive) | View Dashboard summary and risk trend timeline | US-DASH-04, US-TREND-01 |
| Karan (Compliance Officer) | Review and reject a proposed action with a reason, review audit log | US-MCP-01–02, US-DASH-05 |

UAT sign-off for Phase 1 requires all Phase 1-tagged scenarios above to pass before Phase 2 work begins, enforcing Document 1's Delivery Phase discipline.

**Delivery Phase:** Phase 1 (Priya, Meera, Arjun core scenarios); Phase 2 (Rahul, Devika, Karan scenarios and remaining Phase 2 behavior)

## 14. Edge Cases

| Edge Case | Expected Behavior | Requirement Reference |
|---|---|---|
| Entity with no historical data (new supplier) | Prediction returns a low-confidence/neutral score rather than an error | NFR-17 |
| Malformed uploaded PDF | `documents.parse_status = 'failed'`, surfaced as a data-quality warning, not a silent drop | FR-GC-05, NFR-17 |
| Simultaneous approve attempts on the same `action_request` by two approvers | Second request receives `409 CONFLICT`; only one execution occurs | Document 12 §11 |
| RAG returns zero relevant chunks | Chatbot answer explicitly states low/no evidence rather than fabricating a citation | Document 7 §12 (VDB-02) |
| What-if simulation with an out-of-range feature value (e.g., negative lead time) | Blocked client-side and server-side with `422`, no scoring attempted | FR-SIM-01, Document 3 §6.12 |
| Graph query for a deactivated supplier | Entity still viewable (historical integrity) but flagged as inactive, excluded from new recommendations | Document 5 §6.2 |
| Alert threshold set to an extreme value causing alert flooding | Admin-configurable, but system does not crash or drop alerts under high volume — verified via a stress test | NFR-18 |
| Notification channel (Slack/email) temporarily unreachable | Delivery marked `failed`, alert remains visible in-app regardless | Document 4 §14 |

## 15. Risks

| ID | Risk | Mitigation | Delivery Phase |
|---|---|---|---|
| TD-01 | Solo developer authoring both code and tests risks confirmation bias (tests validate the implementation, not the requirement) | Test cases in this document are written from Document 1's requirement/user-story IDs first, traced explicitly in the tables above, not derived from the implementation after the fact | Phase 1 |
| TD-02 | Phase 2's generative/agentic surface (Sections 12–13) is inherently harder to test deterministically | Structured-output and approval-gate tests (Section 12) target the deterministic guardrails around the LLM, not the LLM's free-text output itself | Phase 2 |
| TD-03 | UAT without real end users (Section 3 assumption) may miss usability issues real personas would catch | Academic Supervisor review supplements self-conducted UAT where available | Phase 1 |

## 16. Future Extension

Automated load testing at scale beyond NFR-01/02's prototype targets, and fuzz testing of the LLM/chatbot input surface, are natural extensions once the system moves beyond an academic prototype (Document 1, Section 15), built on top of the same test-type structure (Sections 6–12) defined here.

---

## Document Control

- **Purpose:** Section 1. **Scope:** Section 2. **Assumptions:** Section 3. **Dependencies:** Section 4. **Risks:** Section 15. **Future Extension:** Section 16.
- Baseline for Document 14 (Project Roadmap — test milestones per sprint).
