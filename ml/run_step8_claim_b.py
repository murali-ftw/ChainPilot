"""
Step 8 — Claim B, dyadic risk reweighting (`docs/10_AI_ML_Documentation.md` §8.5,
`docs/05_Database_Design.md` §6.26). Not a learned model component -- see
`ml/models/dyadic.py`'s module docstring for the full formula and rationale.

===============================================================================================
Device allocation -- MPS for training, CPU for everything else, and why
===============================================================================================
Training (SHARE + Variant A, `rgcn_attn_rung5_a`) runs on **MPS**: this exact architecture's
ops (RGCN basis-decomposed transform, joint-softmax attention, the Variant A bounded-residual
gate) were verified correct (forward+backward, CPU vs MPS, exact match) and faster (~2.5x) in
`reports/layer_2.md` Round 5 and reused unchanged in every Step 7/7b round since -- re-running
an identical correctness/speed check on unchanged code would just reproduce the same result at
real compute cost, so this round cites that prior verification rather than repeating it.

Everything else in this round -- the three dyadic weight queries (`ml/models/dyadic.py`,
SQL + pandas), the `reordering_rate` computation (small pure-Python loops over customer-
supplier pairs, at most a few thousand pairs), and the final `supplier_dyadic_risk` persistence
-- runs on **CPU** by construction: none of it touches a `torch.Tensor`, so there is no device
to choose -- pandas/SQL/pure-Python operations have no MPS-accelerated path, and moving them
through torch tensors just to run them "on a device" would add conversion overhead for zero
benefit. The only genuinely GPU/MPS-relevant step is the encoder forward/backward pass itself.

Run: python -u -m ml.run_step8_claim_b
"""

from __future__ import annotations

import copy
import datetime as dt
import statistics
import time

import torch

from ml.data.db import get_connection
from ml.evaluate import best_f1_threshold, collect_predictions, evaluate_split, log_evaluation_runs
from ml.models.depth import TASKS
from ml.models.dyadic import (
    compute_fulfilment_preference_weight,
    compute_order_volume_share,
    contract_priority_weight,
    dyadic_risk_score,
    reordering_rate,
)
from ml.train import (
    TEST_START,
    TRAIN_CUTOFF,
    SnapshotBundle,
    load_all_snapshot_bundles,
    move_bundles_to_device,
    run_training_job,
    split_bundles,
)

SEEDS = [0, 1, 2, 3, 4]
HIDDEN, NUM_BASES = 128, 10
DEVICE = "mps"
ARCH = "rgcn_attn_rung5_a"
LAMBDA_BOUND = 0.3
RUN_ID = "step8-claimb-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _clone_bundles(bundles: list[SnapshotBundle]) -> list[SnapshotBundle]:
    return [SnapshotBundle(t0=b.t0, snapshot_id=b.snapshot_id, data=b.data.clone(),
                            id_maps=b.id_maps, labels=b.labels) for b in bundles]


def _mask_customer_priority_tier(bundles: list[SnapshotBundle]) -> None:
    """In place: zeroes `Customer.x` (the ENTIRE Customer node feature set is the
    `priority_tier` one-hot, confirmed against `ml/data/features.py::customer_features_asof`
    -- nothing else to selectively mask)."""
    for b in bundles:
        b.data["Customer"].x.zero_()


def _train_eval_log(conn, model_version, train_b, val_b, test_b, seed, purpose):
    t0 = time.monotonic()
    result = run_training_job(
        conn, model_version, ARCH, train_b, val_b,
        num_layers=4, shared_depth=None, hidden=HIDDEN, epochs=100,
        seed=seed, purpose=purpose, num_bases=NUM_BASES, device=DEVICE,
        rung5a_lambda_bound=LAMBDA_BOUND,
    )
    model = result["model"]
    thresholds = {task: best_f1_threshold(*collect_predictions(model, val_b)[task]) for task in TASKS}
    results = evaluate_split(model, test_b, thresholds)
    log_evaluation_runs(conn, model_version, ARCH, results, "test",
                         train_end_t0=TRAIN_CUTOFF, test_start_t0=TEST_START)
    elapsed = time.monotonic() - t0
    aucs = {task: results.get(task, {}).get("roc_auc", {}).get("value", float("nan")) for task in TASKS}
    print(f"  [{elapsed:6.1f}s] {model_version:32s} best_epoch={result['best_epoch']:3d}  " +
          "  ".join(f"{t}={aucs[t]:.4f}" for t in TASKS), flush=True)
    return aucs, elapsed, model


@torch.no_grad()
def _global_risk_by_supplier(model, bundle) -> dict[str, float]:
    model.eval()
    logits, _ = model(bundle.data.x_dict, bundle.data.edge_index_dict)
    probs = torch.sigmoid(logits["impact"]).cpu().tolist()
    node_ids = bundle.data["Supplier"].node_id
    return dict(zip(node_ids, probs))


def _compute_reordering_rate(global_risk: dict[str, float], customer_suppliers: dict[str, list[str]],
                              volume_share: dict[tuple[str, str], float],
                              priority: dict[str, float],
                              fulfilment: dict[tuple[str, str], float]) -> float:
    data = {}
    for cust_id, supplier_ids in customer_suppliers.items():
        pair_risks = {}
        for sup_id in supplier_ids:
            if sup_id not in global_risk:
                continue
            g = global_risk[sup_id]
            d = dyadic_risk_score(
                g,
                volume_share.get((cust_id, sup_id)),
                priority.get(cust_id),
                fulfilment.get((cust_id, sup_id)),
            )
            pair_risks[sup_id] = (g, d)
        if len(pair_risks) >= 2:
            data[cust_id] = pair_risks
    return reordering_rate(data)


def main() -> None:
    overall_start = time.monotonic()
    conn = get_connection()

    print("=" * 100)
    print("Device: MPS for training (correctness+speedup already verified for this exact")
    print("architecture, reports/layer_2.md Round 5 -- see module docstring). CPU for the")
    print("dyadic weight queries and reordering-rate computation (pandas/SQL/pure-Python, no")
    print("tensor ops, no device to choose).")
    print("=" * 100, flush=True)

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM suppliers")
        n_suppliers = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM customers")
        n_customers = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM risk_scores")
        n_risk_scores = cur.fetchone()[0]
    print(f"=== Live dataset check: {n_suppliers} suppliers, {n_customers} customers, "
          f"{n_risk_scores} risk_scores rows ===", flush=True)

    bundles = load_all_snapshot_bundles(conn)
    train_b, val_b, test_b = split_bundles(bundles)
    print(f"train t0s: {[str(b.t0.date()) for b in train_b]}", flush=True)
    print(f"val   t0s: {[str(b.t0.date()) for b in val_b]}", flush=True)
    print(f"test  t0s: {[str(b.t0.date()) for b in test_b]}", flush=True)
    t0_scoring = test_b[-1].t0
    print(f"Scoring t0 (most recent test snapshot): {t0_scoring}", flush=True)

    # ------------------------------------------------------------- dyadic weight tables (CPU)
    print("\n=== Computing dyadic weight tables (order_volume_share, fulfilment_preference_weight) ===", flush=True)
    ovs_df = compute_order_volume_share(conn, t0_scoring)
    fpw_df = compute_fulfilment_preference_weight(conn, t0_scoring)
    volume_share = {(r.customer_id, r.supplier_id): r.order_volume_share for r in ovs_df.itertuples()}
    fulfilment = {(r.customer_id, r.supplier_id): r.fulfilment_preference_weight for r in fpw_df.itertuples()}
    print(f"  order_volume_share: {len(volume_share)} (customer, supplier) pairs")
    print(f"  fulfilment_preference_weight: {len(fulfilment)} (customer, supplier) pairs "
          f"({len(fulfilment) / max(1, len(volume_share)):.1%} coverage of volume-share pairs)")

    import pandas as pd
    priority_df = pd.read_sql("SELECT id AS customer_id, priority_tier FROM customers", conn)
    priority = {r.customer_id: contract_priority_weight(r.priority_tier) for r in priority_df.itertuples()}

    customer_suppliers: dict[str, set[str]] = {}
    for (cust_id, sup_id) in volume_share:
        customer_suppliers.setdefault(cust_id, set()).add(sup_id)
    customer_suppliers = {c: sorted(s) for c, s in customer_suppliers.items()}
    n_multi = sum(1 for s in customer_suppliers.values() if len(s) >= 2)
    print(f"  {len(customer_suppliers)} customers have >=1 supplier relationship; "
          f"{n_multi} have >=2 (reordering_rate is computed over these)", flush=True)

    # ------------------------------------------------------------- masked bundles for the "without" arm
    train_b_masked = _clone_bundles(train_b)
    val_b_masked = _clone_bundles(val_b)
    test_b_masked = _clone_bundles(test_b)
    _mask_customer_priority_tier(train_b_masked)
    _mask_customer_priority_tier(val_b_masked)
    _mask_customer_priority_tier(test_b_masked)

    move_bundles_to_device(train_b, DEVICE)
    move_bundles_to_device(val_b, DEVICE)
    move_bundles_to_device(test_b, DEVICE)
    move_bundles_to_device(train_b_masked, DEVICE)
    move_bundles_to_device(val_b_masked, DEVICE)
    move_bundles_to_device(test_b_masked, DEVICE)

    reorder_rates = {"with": [], "without": []}
    run_elapsed = []
    all_aucs = {"with": {}, "without": {}}
    last_models = {"with": None, "without": None}

    print("\n=== Arm (with_priority_tier): SHARE + Variant A, Customer.priority_tier VISIBLE -- 5 seeds ===", flush=True)
    for seed in SEEDS:
        mv = f"{ARCH}-claimb-with-seed{seed}"
        purpose = f"Step 8 Claim B double-counting test -- WITH priority_tier, seed={seed}"
        aucs, elapsed, model = _train_eval_log(conn, mv, train_b, val_b, test_b, seed, purpose)
        all_aucs["with"][seed] = aucs
        run_elapsed.append((mv, elapsed))
        global_risk = _global_risk_by_supplier(model, test_b[-1])
        rate = _compute_reordering_rate(global_risk, customer_suppliers, volume_share, priority, fulfilment)
        reorder_rates["with"].append(rate)
        print(f"      reordering_rate = {rate:.4f}", flush=True)
        last_models["with"] = model

    print("\n=== Arm (without_priority_tier): SHARE + Variant A, Customer.priority_tier MASKED -- 5 seeds ===", flush=True)
    for seed in SEEDS:
        mv = f"{ARCH}-claimb-without-seed{seed}"
        purpose = f"Step 8 Claim B double-counting test -- WITHOUT priority_tier (masked), seed={seed}"
        aucs, elapsed, model = _train_eval_log(conn, mv, train_b_masked, val_b_masked, test_b_masked, seed, purpose)
        all_aucs["without"][seed] = aucs
        run_elapsed.append((mv, elapsed))
        global_risk = _global_risk_by_supplier(model, test_b_masked[-1])
        rate = _compute_reordering_rate(global_risk, customer_suppliers, volume_share, priority, fulfilment)
        reorder_rates["without"].append(rate)
        print(f"      reordering_rate = {rate:.4f}", flush=True)
        last_models["without"] = model

    total_elapsed = time.monotonic() - overall_start
    print(f"\n=== ALL 10 TRAINING RUNS COMPLETE in {total_elapsed/3600:.2f}h ({total_elapsed:.0f}s) ===\n", flush=True)

    # =============================================================== REPORTING
    print("=" * 100)
    print("AUC spread across 5 seeds, per (arm, task)")
    print("=" * 100)
    for arm in ["with", "without"]:
        for task in TASKS:
            values = [all_aucs[arm][s][task] for s in SEEDS]
            print(f"  {arm:8s} {task:10s} mean={statistics.fmean(values):.4f}  "
                  f"std={statistics.pstdev(values):.4f}  values={[round(v, 4) for v in values]}")

    print("\n" + "=" * 100)
    print("DOUBLE-COUNTING TEST -- Claim B reordering rate, with vs without priority_tier in the encoder")
    print("=" * 100)
    with_vals, without_vals = reorder_rates["with"], reorder_rates["without"]
    mean_with, std_with = statistics.fmean(with_vals), statistics.pstdev(with_vals)
    mean_without, std_without = statistics.fmean(without_vals), statistics.pstdev(without_vals)
    print(f"  WITH priority_tier    : mean={mean_with:.4f}  std={std_with:.4f}  values={[round(v,4) for v in with_vals]}")
    print(f"  WITHOUT priority_tier : mean={mean_without:.4f}  std={std_without:.4f}  values={[round(v,4) for v in without_vals]}")
    ratio = mean_with / mean_without if mean_without > 0 else float("nan")
    print(f"  ratio (with / without) = {ratio:.4f}")
    if 0.7 <= ratio <= 1.3:
        verdict = "RATES CLOSE -- signals appear largely independent, low double-counting risk"
    else:
        verdict = ("RATES DIVERGE -- possible double-counting: the encoder's own risk score already "
                   "reflects some of what contract_priority_weight/priority_tier would add")
    print(f"  VERDICT: {verdict}", flush=True)

    # ------------------------------------------------------------- persist supplier_dyadic_risk
    print("\n" + "=" * 100)
    print("Persisting supplier_dyadic_risk (using EXISTING risk_scores rows, UNMODIFIED)")
    print("=" * 100)
    risk_df = pd.read_sql(
        "SELECT id AS risk_score_id, entity_id AS supplier_id, impact_score FROM risk_scores WHERE entity_type = 'supplier'",
        conn)
    risk_by_supplier = {r.supplier_id: (r.risk_score_id, float(r.impact_score)) for r in risk_df.itertuples()}

    rows = []
    for (cust_id, sup_id), ovs in volume_share.items():
        if sup_id not in risk_by_supplier:
            continue
        risk_score_id, global_risk_val = risk_by_supplier[sup_id]
        cpw = priority.get(cust_id)
        fpw = fulfilment.get((cust_id, sup_id))
        dyadic = dyadic_risk_score(global_risk_val, ovs, cpw, fpw)
        rows.append((sup_id, cust_id, risk_score_id, round(ovs, 4), round(cpw, 4) if cpw is not None else None,
                     round(fpw, 4) if fpw is not None else None, round(dyadic, 4), RUN_ID))

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO supplier_dyadic_risk
                (supplier_id, customer_id, risk_score_id, order_volume_share,
                 contract_priority_weight, fulfilment_preference_weight, dyadic_risk_score,
                 double_counting_test_run_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            rows,
        )
    conn.commit()
    print(f"  Inserted {len(rows)} supplier_dyadic_risk rows, double_counting_test_run_id={RUN_ID}", flush=True)

    print("\n" + "=" * 100)
    print("Per-run elapsed time")
    print("=" * 100)
    for mv, elapsed in run_elapsed:
        print(f"  {mv:38s} {elapsed:7.1f}s")

    conn.close()
    print(f"\nTotal wall-clock: {total_elapsed/3600:.2f}h ({total_elapsed:.0f}s)", flush=True)
    print(f"RUN_ID: {RUN_ID}")


if __name__ == "__main__":
    main()
