# Abstract

## Graph Neural Network and Generative AI-Based Supply Chain Risk Prediction System

*Six-Layer Architecture with Retrieval-Augmented Reasoning, Autonomous Execution, an Interactive Chatbot, and Extended Decision-Support Features*

Final Year Project — BE CSE (AIML)

---

Modern supply chains are dense, interconnected networks of suppliers, components, products, factories, warehouses, shipments, and customer orders, in which a disruption at a single node can cascade unpredictably across many others. Traditional risk-monitoring systems analyze these entities as isolated, tabular records, so row-by-row machine learning models fail to capture the multi-hop relational effects that drive real-world cascading failures. Even systems that do produce a risk score typically stop there: the output is a static number requiring expert interpretation, with no way to interrogate or simulate alternatives, and any resulting action still has to be carried out manually in a separate system. This project addresses that gap by proposing a closed-loop, six-layer Graph Neural Network and Generative AI-based Supply Chain Risk Prediction System that senses, explains, and — with human approval — acts on disruption risk.

The system represents the supply chain as a heterogeneous graph of typed nodes (suppliers, components, products, factories, warehouses, shipments, orders) and typed edges (SUPPLIES, USED_IN, SHIPS_TO, and others), built from structured operational records plus a lightweight parsing step that normalizes the small share of unstructured documents, such as invoices or purchase orders, into the same schema. A hybrid GNN–Transformer model — a heterogeneous GNN encoder, evolving from GraphSAGE/GAT to a Heterogeneous Graph Transformer, paired with a Transformer prediction head — is trained on this graph to predict delay probability, shortage risk, and overall disruption impact, while GNNExplainer produces a per-prediction explanation subgraph and reusable entity embeddings. A retrieval-augmented generation (RAG) layer grounds subsequent reasoning in real historical evidence — past incidents, contract clauses, supplier history — retrieved from a vector database, and a Large Language Model translates the risk score, subgraph, and evidence into plain-language explanations and recommended actions. An agentic execution layer, built on Model Context Protocol (MCP) servers and gated by mandatory human-in-the-loop approval, then carries out approved recommendations directly inside ERP or procurement systems.

Beyond this core pipeline, the system exposes an interactive natural-language chatbot and five decision-support features that reuse existing layer outputs without new modeling risk: a what-if scenario simulator that re-scores a perturbed graph copy without retraining, a visual explainability overlay synchronized with chatbot answers, an alternative-supplier recommender built on GNN embedding similarity, a risk trend timeline, and proactive MCP-based alerts. The system is delivered in two phases: a fifteen-week Phase 1 proving the graph-construction-and-prediction core with a working dashboard (targeting AUC-ROC ≥ 0.80, sub-two-second inference), and a Phase 2, targeted for March 2027, layering on RAG, the LLM, the chatbot, and the agentic execution and alerting capabilities. Implemented with PyTorch Geometric, FastAPI, PostgreSQL, React/D3, and Docker, the architecture is service-oriented and additive, so Phase 2 extends the Phase 1 core without redesigning it. Role-based access control, JWT authentication, and immutable audit logging of authentication, approval, and execution events run throughout, ensuring growing autonomy is matched by traceability rather than unmonitored automation. The system aims to show that relationship-aware, evidence-grounded, human-gated AI can shorten the path from detecting a supply chain disruption to understanding and acting on it — closing the loop from prediction to outcome instead of stopping at a static score, while remaining usable by non-technical operations staff.

---

## At a Glance

| | |
|---|---|
| **Architecture** | Six layers — Graph Construction, GNN × Transformer, RAG, LLM, MCP Servers, Frontend Dashboard |
| **Core model** | Heterogeneous GNN (GraphSAGE/GAT → Heterogeneous Graph Transformer) + Transformer prediction head |
| **Extended features** | Interactive chatbot, what-if simulator, explainability overlay, supplier recommender, risk trend timeline, proactive alerts |
| **Delivery** | Phase 1 — 15 weeks (research prototype/MVP); Phase 2 — targeted March 2027 (full closed loop) |
| **Phase 1 target** | AUC-ROC ≥ 0.80 on held-out test set; ≤ 2s inference latency (p95) |
| **Stack** | PyTorch Geometric, FastAPI, PostgreSQL, React + D3, Docker |

---

*Source: `docs/problem_statement.md`, `docs/01_Product_Requirement_Document.md`, `docs/02_Technical_Requirement_Specification.md`, `docs/10_AI_ML_Documentation.md`, `docs/14_Project_Roadmap.md`.*
