# Document 9 — REST API Documentation

## Graph Neural Network and Generative AI-Based Supply Chain Risk Prediction System

Version: 1.0
Status: Baseline
Consistent with: Documents 1–8

---

## 1. Purpose

This document specifies every REST endpoint exposed by the API Gateway (Document 2, Section 4; Document 8, Section 7): method, URL, authentication requirement, headers, request shape, response shape, error responses, and example JSON. It is the authoritative contract between the React Dashboard (Document 3) and the backend (Document 8).

## 2. Scope

All endpoints are versioned under `/api/v1/`. Phase 1 endpoints cover auth, entity CRUD/read, predictions (incl. `scoring_method`, `confidence`, `risk_category`), customers, and evaluation/architecture-comparison/model-governance. Phase 2 endpoints add chat, simulation, recommendations, decision intelligence (routing/decision-trace), optimization (safety-stock, PO-split, customer allocation), approvals, alerts, and audit of approval/action events.

## 3. Assumptions

- All requests/responses use `application/json` unless otherwise noted (file upload uses `multipart/form-data`).
- All timestamps are ISO-8601 UTC.
- Pagination uses `?page=&page_size=` query parameters with a standard envelope (Section 5).
- Six structurally identical entity families (Suppliers, Warehouses, Orders, Shipments, Products, Inventory) share one documented pattern (Section 7) to avoid repeating an identical spec six times; only Suppliers is fully worked as the representative example.

## 4. Dependencies

Document 8 (controllers implementing these contracts), Document 5 (underlying schema), Document 12 (auth/rate-limiting detail referenced here).

## 5. Global Conventions

| Convention | Detail |
|---|---|
| Base URL | `/api/v1` |
| Authentication | `Authorization: Bearer <JWT>` header, required on every endpoint except `POST /auth/login` |
| Standard headers | `Content-Type: application/json`, `Accept: application/json` |
| Pagination envelope | `{ "items": [...], "page": 1, "page_size": 20, "total": 123 }` |
| Standard error envelope | `{ "error": { "code": "STRING_CODE", "message": "human-readable" } }` |
| Common error codes | `401 UNAUTHENTICATED`, `403 FORBIDDEN`, `404 NOT_FOUND`, `422 VALIDATION_ERROR`, `429 RATE_LIMITED`, `500 INTERNAL_ERROR` |

## 6. Authentication Endpoints (Phase 1)

### 6.1 `POST /api/v1/auth/login`

| Field | Value |
|---|---|
| Authentication | None |
| Headers | `Content-Type: application/json` |

**Request:**
```json
{ "email": "priya@example.com", "password": "correct-horse-battery" }
```

**Response `200 OK`:**
```json
{
  "access_token": "eyJhbGciOi...",
  "refresh_token": "eyJhbGciOi...",
  "user": { "id": "b3f...", "full_name": "Priya Sharma", "role": "analyst" }
}
```

**Errors:** `401 UNAUTHENTICATED` (invalid credentials), `423 LOCKED` (account locked, FR-AUTH-05).

### 6.2 `POST /api/v1/auth/refresh`

| Field | Value |
|---|---|
| Authentication | Refresh token in body |

**Request:** `{ "refresh_token": "eyJhbGciOi..." }`
**Response `200 OK`:** `{ "access_token": "eyJhbGciOi..." }`
**Errors:** `401 UNAUTHENTICATED` (expired/invalid refresh token).

### 6.3 `POST /api/v1/auth/logout`

| Field | Value |
|---|---|
| Authentication | Bearer JWT |

**Response `204 No Content`.** **Errors:** `401 UNAUTHENTICATED`.

### 6.4 `GET /api/v1/auth/me`

| Field | Value |
|---|---|
| Authentication | Bearer JWT |

**Response `200 OK`:** `{ "id": "b3f...", "email": "priya@example.com", "full_name": "Priya Sharma", "role": "analyst" }`

## 7. User Administration (Phase 1)

| Method | URL | Auth | Purpose |
|---|---|---|---|
| GET | `/api/v1/users` | `admin` | List users (paginated) |
| POST | `/api/v1/users` | `admin` | Create user (FR-AUTH-06) |
| PATCH | `/api/v1/users/{id}` | `admin` | Update role/name |
| POST | `/api/v1/users/{id}/deactivate` | `admin` | Deactivate account (FR-AUTH-04) |

**Example — `POST /api/v1/users` request:**
```json
{ "email": "karan@example.com", "full_name": "Karan Mehta", "role": "approver", "temporary_password": "TempPass123!" }
```
**Response `201 Created`:** `{ "id": "d91...", "email": "karan@example.com", "role": "approver", "is_active": true }`
**Errors:** `422 VALIDATION_ERROR` (duplicate email, invalid role), `403 FORBIDDEN` (non-admin caller).

## 8. Entity Endpoints (Phase 1)

### 8.1 Shared Pattern

Suppliers, Warehouses, Orders, Shipments, Products, and Inventory each expose:

| Method | URL Pattern | Purpose |
|---|---|---|
| GET | `/api/v1/{entity}` | Paginated, filterable list |
| GET | `/api/v1/{entity}/{id}` | Single entity detail, including current risk score and linked entities |

Common query parameters: `?risk_level=high,medium&search=&page=&page_size=`.

### 8.2 Suppliers (Representative Example)

**`GET /api/v1/suppliers?risk_level=high&page=1&page_size=20`**

| Field | Value |
|---|---|
| Authentication | Bearer JWT (any role) |
| Headers | `Accept: application/json` |

**Response `200 OK`:**
```json
{
  "items": [
    {
      "id": "a1b2...",
      "name": "Acme Components Ltd.",
      "country": "IN",
      "lead_time_days": 21,
      "reliability_history": 0.82,
      "risk": { "delay_probability": 0.78, "impact_score": 0.71, "confidence": 0.86, "risk_category": "high" }
    }
  ],
  "page": 1, "page_size": 20, "total": 4
}
```
**Errors:** `401 UNAUTHENTICATED`, `422 VALIDATION_ERROR` (invalid `risk_level` value).

**`GET /api/v1/suppliers/{id}`**

**Response `200 OK`:**
```json
{
  "id": "a1b2...",
  "name": "Acme Components Ltd.",
  "country": "IN",
  "lead_time_days": 21,
  "reliability_history": 0.82,
  "components": [{ "id": "c1...", "name": "Capacitor 10uF", "component_type": "electronic" }],
  "risk": { "delay_probability": 0.78, "impact_score": 0.71, "confidence": 0.86, "risk_category": "high", "scoring_method": "weighted_formula", "model_version": "hgt-v1", "architecture": "heterogeneous_graph_transformer", "scored_at": "2026-07-10T08:00:00Z" }
}
```
**Errors:** `404 NOT_FOUND`.

### 8.3 Customers

**`GET /api/v1/customers?priority_tier=strategic&page=1&page_size=20`**

| Field | Value |
|---|---|
| Authentication | Bearer JWT (any role) |

**Response `200 OK`:**
```json
{
  "items": [
    { "id": "cu1...", "name": "Nova Retail Group", "priority_tier": "strategic", "is_active": true }
  ],
  "page": 1, "page_size": 20, "total": 6
}
```

**`POST /api/v1/customers`**

| Field | Value |
|---|---|
| Authentication | Bearer JWT (`analyst`, `admin`) |

**Request:** `{ "name": "Nova Retail Group", "priority_tier": "strategic", "contract_terms": { "sla_days": 5, "penalty_pct": 2.5 } }`
**Response `201 Created`:** `{ "id": "cu1...", "name": "Nova Retail Group", "priority_tier": "strategic", "is_active": true }`
**Errors:** `422 VALIDATION_ERROR` (invalid `priority_tier`).

**`GET /api/v1/customers/{id}`** — customer detail incl. linked orders. **`PATCH /api/v1/customers/{id}`** (`admin`) — update tier/contract terms. **Errors:** `404 NOT_FOUND`.
**Delivery Phase:** Phase 1 (FR-CUST-01)

### 8.4 Customer Allocation (OR-Tools Optimization)

**`GET /api/v1/customers/allocations/{product_id}`**

Reads-facing endpoint over the Decision Intelligence → Optimization pipeline (Document 4, Section 20): the Customer Service supplies customer/order inputs, the Decision Intelligence Service assembles the constraint set, and the Optimization Service (`POST /api/v1/optimize/customer-allocation`, Section 12.2) solves it; this endpoint returns the result read-model.

| Field | Value |
|---|---|
| Authentication | Bearer JWT (`analyst`, `approver`, `admin`) |

**Response `200 OK`:**
```json
{
  "product_id": "p1...",
  "available_stock": 320,
  "objective_value": 41250.0,
  "optimal": true,
  "allocations": [
    { "order_id": "o1...", "customer_id": "cu1...", "priority_tier": "strategic", "order_value": 48000, "sla_penalty_pct": 2.5, "allocated_qty": 200 },
    { "order_id": "o2...", "customer_id": "cu2...", "priority_tier": "low", "order_value": 6000, "sla_penalty_pct": 0, "allocated_qty": 0 }
  ],
  "constraints_applied": { "inventory": 320, "supplier_capacity": 500, "warehouse_capacity": 800, "lead_time_days": 14 }
}
```
**Errors:** `404 NOT_FOUND` (no shortage flagged for this product), `422 VALIDATION_ERROR`.
**Delivery Phase:** Phase 2 (FR-CUST-02) — approving a proposed allocation creates an `action_requests` row with `source='optimizer'`, `entity_type='order'`, and a populated `decision_trace`, via the Approval Endpoints (Section 13).

### 8.5 Document Upload (Graph Construction — FR-GC-05)

**`POST /api/v1/documents`**

| Field | Value |
|---|---|
| Authentication | Bearer JWT (`analyst`, `admin`) |
| Headers | `Content-Type: multipart/form-data` |

**Request:** form fields `file` (PDF), `document_type` (`invoice`\|`purchase_order`), `supplier_id` (optional).
**Response `202 Accepted`:**
```json
{ "id": "e5f6...", "parse_status": "pending" }
```
**Errors:** `422 VALIDATION_ERROR` (unsupported file type), `413 PAYLOAD_TOO_LARGE`.

## 9. Prediction Endpoints (Phase 1)

### 9.1 `GET /api/v1/predictions?entity_type=supplier&risk_level=high`

| Field | Value |
|---|---|
| Authentication | Bearer JWT (any role) |

**Response `200 OK`:**
```json
{
  "items": [
    {
      "entity_type": "supplier", "entity_id": "a1b2...",
      "delay_probability": 0.78, "shortage_risk": null, "impact_score": 0.71,
      "confidence": 0.86, "risk_category": "high",
      "scoring_method": "weighted_formula",
      "affected_orders": ["o1...", "o2..."], "model_version": "hgt-v1", "architecture": "heterogeneous_graph_transformer",
      "scored_at": "2026-07-10T08:00:00Z"
    }
  ],
  "page": 1, "page_size": 20, "total": 12
}
```
**Errors:** `422 VALIDATION_ERROR` (invalid `entity_type`).

### 9.2 `GET /api/v1/predictions/{entity_type}/{entity_id}/explanation`

**Response `200 OK`:**
```json
{
  "risk_score_id": "f7...",
  "nodes": [{ "entity_type": "supplier", "entity_id": "a1b2...", "contribution_weight": 0.62 }],
  "edges": [{ "source": "a1b2...", "target": "c1...", "edge_type": "SUPPLIES", "contribution_weight": 0.45 }]
}
```
**Errors:** `404 NOT_FOUND` (no prediction exists yet for this entity).

### 9.3 `GET /api/v1/predictions/{entity_type}/{entity_id}/trend?from=&to=`

**Response `200 OK`:** `{ "entity_id": "a1b2...", "points": [{ "scored_at": "2026-06-01T00:00:00Z", "impact_score": 0.40 }, { "scored_at": "2026-07-10T08:00:00Z", "impact_score": 0.71 }] }`
**Delivery Phase:** Phase 2 (US-TREND-01) — endpoint reads the Phase 1 `risk_scores` table, so no backend change is needed at Phase 2 activation beyond exposing this route.

### 9.4 Evaluation & Architecture Comparison Endpoints (FR-EVAL-01/02, FR-ABL-02)

**`GET /api/v1/models/evaluation-runs?architecture=hgt&metric_name=roc_auc`**

| Field | Value |
|---|---|
| Authentication | Bearer JWT (any role) |

**Response `200 OK`:**
```json
{
  "items": [
    { "model_version": "hgt-v1", "architecture": "heterogeneous_graph_transformer", "metric_name": "roc_auc", "metric_value": 0.847, "dataset_split": "test", "evaluated_at": "2026-09-12T10:00:00Z" }
  ],
  "page": 1, "page_size": 20, "total": 21
}
```

**`GET /api/v1/models/comparison`** — returns the latest `test`-split metrics for each of `graphsage`, `gat`, `heterogeneous_graph_transformer` side by side, backing the ablation comparison (FR-ABL-02).

**Response `200 OK`:**
```json
{
  "architectures": [
    { "architecture": "graphsage", "model_version": "graphsage-v1", "metrics": { "roc_auc": 0.71, "f1": 0.62 } },
    { "architecture": "gat", "model_version": "gat-v1", "metrics": { "roc_auc": 0.79, "f1": 0.69 } },
    { "architecture": "heterogeneous_graph_transformer", "model_version": "hgt-v1", "metrics": { "roc_auc": 0.847, "f1": 0.76 } }
  ]
}
```
**Errors:** `422 VALIDATION_ERROR` (unknown `architecture`/`metric_name` value).
**Delivery Phase:** Phase 1

### 9.5 Model Registry / Governance Endpoints (FR-GOV-01/02)

**`GET /api/v1/models/registry?status=active`**

| Field | Value |
|---|---|
| Authentication | Bearer JWT (any role) |

**Response `200 OK`:**
```json
{
  "items": [
    {
      "model_version": "hgt-v1", "architecture": "heterogeneous_graph_transformer",
      "training_dataset": "supply_chain_snapshot_2026-09-01", "training_timestamp": "2026-09-12T06:00:00Z",
      "experiment_id": "exp-hgt-014", "git_commit": "a1b2c3d", "hyperparameters": { "lr": 0.001, "layers": 3 },
      "parameter_count": 482300, "status": "active"
    }
  ],
  "page": 1, "page_size": 20, "total": 3
}
```

**`GET /api/v1/models/active`** — shortcut for the single currently `active` model version (used by the Model Metadata panel, Document 3, Section 6.19).
**Errors:** `404 NOT_FOUND` (no model currently `active`), `422 VALIDATION_ERROR` (invalid `status` filter).
**Delivery Phase:** Phase 1

## 10. Chatbot Endpoints (Phase 2)

### 10.1 `POST /api/v1/chat/sessions`

| Field | Value |
|---|---|
| Authentication | Bearer JWT (`analyst`, `approver`, `admin`) |

**Response `201 Created`:** `{ "session_id": "s1..." }`

### 10.2 `POST /api/v1/chat/sessions/{session_id}/messages` (or WebSocket `/ws/chat/{session_id}` for streaming, per Document 2 §7)

**Request:** `{ "content": "Why is Supplier X flagged as high risk?" }`
**Response `200 OK` (non-streaming fallback):**
```json
{
  "role": "assistant",
  "content": "Supplier Acme Components has a 78% delay probability, mainly driven by extended lead time and a recent incident...",
  "intent": "explanation",
  "citations": [{ "collection": "incidents", "source_document_id": "doc-1" }]
}
```
**Errors:** `503 SERVICE_UNAVAILABLE` (RAG/LLM temporarily unavailable, Document 4 §9).

## 11. What-If Simulator Endpoints (Phase 2)

### 11.1 `POST /api/v1/simulate`

| Field | Value |
|---|---|
| Authentication | Bearer JWT (`analyst`, `approver`, `admin`) |

**Request:**
```json
{ "entity_type": "supplier", "entity_id": "a1b2...", "feature_overrides": { "lead_time_days": 42 } }
```
**Response `200 OK`:**
```json
{
  "baseline": { "impact_score": 0.71 },
  "simulated": { "impact_score": 0.89 },
  "shifted_entities": [{ "entity_type": "order", "entity_id": "o1...", "baseline_impact": 0.30, "simulated_impact": 0.62 }]
}
```
**Errors:** `422 VALIDATION_ERROR` (feature value out of allowed range, e.g., negative lead time).

## 12. Recommendation & Optimization Endpoints (Phase 2)

### 12.1 `GET /api/v1/recommendations/suppliers/{supplier_id}`

**Response `200 OK`:**
```json
{
  "flagged_supplier_id": "a1b2...",
  "candidates": [
    { "supplier_id": "z9...", "name": "Beta Parts Co.", "similarity_score": 0.94, "impact_score": 0.22 }
  ]
}
```
**Errors:** `404 NOT_FOUND`, `422 VALIDATION_ERROR` (no component type resolvable for supplier).

### 12.2 Optimization Endpoints (OR-Tools, FR-OPT-01/02/03)

All three endpoints below are invoked by the Decision Intelligence Service once it determines a flagged entity's response is a closed-form decision type (FR-DEC-01); none is ever computed by the LLM.

**`POST /api/v1/optimize/safety-stock`**

| Field | Value |
|---|---|
| Authentication | Bearer JWT (`analyst`, `approver`, `admin`) |

**Request:** `{ "product_id": "p1...", "warehouse_id": "w1...", "service_level": 0.95 }`
**Response `200 OK`:**
```json
{ "optimal": true, "recommended_reorder_threshold": 180, "objective_value": 180.0, "rationale": "lead-time and demand-variance constrained optimal solution" }
```

**`POST /api/v1/optimize/po-split`**

**Request:** `{ "product_id": "p1...", "required_qty": 500, "candidate_supplier_ids": ["s1...", "s2..."] }`
**Response `200 OK`:**
```json
{ "optimal": true, "split": [{ "supplier_id": "s1...", "qty": 200 }, { "supplier_id": "s2...", "qty": 300 }], "rationale": "lead-time-constrained optimal split" }
```

**`POST /api/v1/optimize/customer-allocation`**

**Request:** `{ "product_id": "p1...", "constraint_set": { "inventory": 320, "supplier_capacity": 500, "warehouse_capacity": 800, "lead_time_days": 14 } }` — `constraint_set` is assembled by the Decision Intelligence Service (Document 4, Section 20), not supplied ad hoc by the caller.
**Response `200 OK`:** same shape as `GET /api/v1/customers/allocations/{product_id}` (Section 8.4).

**Errors (all three):** `422 VALIDATION_ERROR` (no optimal/feasible solution — surfaced as `{ "optimal": false, "reason": "..." }` in a `200 OK`, not forced into a recommendation, per NFR-20).
**Delivery Phase:** Phase 2 — approving any result creates an `action_requests` row with `source='optimizer'` and a populated `decision_trace` via the Approval Endpoints (Section 13).

## 13. Approval Endpoints (Phase 2)

| Method | URL | Auth | Purpose |
|---|---|---|---|
| GET | `/api/v1/approvals?status=pending` | `analyst`,`approver`,`admin` | List action requests |
| GET | `/api/v1/approvals/{id}` | `analyst`,`approver`,`admin` | Detail incl. evidence/action payload/decision trace |
| GET | `/api/v1/approvals/{id}/decision-trace` | `analyst`,`approver`,`admin` | Decision trace only (FR-DEC-03) |
| POST | `/api/v1/approvals/{id}/approve` | `approver`,`admin` | Approve (FR-MCP-03) |
| POST | `/api/v1/approvals/{id}/reject` | `approver`,`admin` | Reject with reason (FR-MCP-04) |

**`GET /api/v1/approvals/{id}` response `200 OK` (excerpt):**
```json
{
  "id": "ar1...", "source": "optimizer", "status": "pending",
  "action_payload": { "action": "safety_stock_adjustment", "recommended_reorder_threshold": 180 },
  "decision_trace": {
    "risk_intelligence": { "risk_category": "high", "confidence": 0.86, "scoring_method": "weighted_formula" },
    "decision_intelligence": { "routed_to": "optimizer", "policy_checks_passed": true },
    "optimization": { "objective_value": 180.0, "optimal": true },
    "llm": null
  }
}
```

**`POST /api/v1/approvals/{id}/reject` request:**
```json
{ "reason": "Alternative supplier lead time not verified yet." }
```
**Response `200 OK`:** `{ "id": "ar1...", "status": "rejected", "decided_by": "k1...", "decided_at": "2026-07-15T09:00:00Z" }`
**Errors:** `422 VALIDATION_ERROR` (missing reason on reject), `409 CONFLICT` (action already decided).

## 14. Alerts and Thresholds Endpoints (Phase 2)

| Method | URL | Auth | Purpose |
|---|---|---|---|
| GET | `/api/v1/alerts?status=active` | any authenticated role | List alerts |
| POST | `/api/v1/alerts/{id}/acknowledge` | `analyst`,`approver`,`admin` | Acknowledge alert |
| GET | `/api/v1/alert-thresholds` | any authenticated role | List configured thresholds |
| PUT | `/api/v1/alert-thresholds/{entity_type}/{metric}` | `admin` | Set threshold (FR-MCP-06) |

**`PUT /api/v1/alert-thresholds/supplier/delay_probability` request:**
```json
{ "threshold_value": 0.75 }
```
**Response `200 OK`:** `{ "entity_type": "supplier", "metric": "delay_probability", "threshold_value": 0.75, "configured_by": "arjun-id" }`
**Errors:** `422 VALIDATION_ERROR` (value outside 0–1).

## 15. Audit Endpoints

### 15.1 `GET /api/v1/audit?from=&to=&event_type=`

| Field | Value |
|---|---|
| Authentication | Bearer JWT (`admin`, `approver`) |

**Response `200 OK`:**
```json
{
  "items": [
    { "id": "al1...", "actor_id": "u1...", "event_type": "login_success", "created_at": "2026-07-15T08:00:00Z" }
  ],
  "page": 1, "page_size": 20, "total": 340
}
```
**Errors:** `403 FORBIDDEN` (non-admin/approver caller).
**Delivery Phase:** Phase 1 (auth events only); Phase 2 (approval/action `event_type` values appear).

## 16. Rate Limiting

All endpoints are subject to a per-user rate limit (default 100 requests/minute), returning `429 RATE_LIMITED` with a `Retry-After` header when exceeded, per Document 12's rate-limiting control.

## 17. Risks

| ID | Risk | Mitigation | Delivery Phase |
|---|---|---|---|
| API-01 | Six near-identical entity endpoint families (Section 8.1) risk drifting out of sync if changed independently | Shared controller/service base pattern in Document 8; contract tests in Document 13 run against all six | Phase 1 |
| API-02 | Streaming chat endpoint (WebSocket) is harder to test/document than plain REST | Non-streaming fallback documented (Section 10.2) as the contract-testable baseline | Phase 2 |
| API-03 | Approval endpoints are a high-stakes surface (trigger real ERP actions downstream) | `409 CONFLICT` guard against double-decision; full audit trail (Section 15) | Phase 2 |
| API-04 | Optimizer/allocation endpoints (Sections 8.4, 12.2) could return an infeasible result that gets surfaced as if it were a valid recommendation | Infeasible solver results return `{ "optimal": false, ... }` explicitly rather than a partial/forced recommendation (NFR-20) | Phase 2 |
| API-05 | `GET /api/v1/approvals/{id}` returns a `null`/empty `decision_trace` for an approved request, breaking audit coverage | Controller-level contract test asserts `decision_trace` is non-null for every `approved`/`rejected` row (NFR-23) | Phase 2 |

## 18. Future Extension

Bulk endpoints (e.g., `POST /api/v1/suppliers/bulk-import`) and GraphQL-style flexible querying are natural extensions once entity volume grows, per Document 1, Section 15, without changing the REST contract shape documented here.

---

## Document Control

- **Purpose:** Section 1. **Scope:** Section 2. **Assumptions:** Section 3. **Dependencies:** Section 4. **Risks:** Section 17. **Future Extension:** Section 18.
- Baseline for Document 13 (Testing Documentation — API test cases) and frontend implementation (Document 3 screens consume these contracts).
