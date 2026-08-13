#!/usr/bin/env python3
"""
Phase 1, Steps 2-4 — latent operational state estimation heads on the frozen backbone.

Reads the FROZEN SHARE + Markov representation (`ml/ds_backbone.py`, unchanged) and trains one
small head per confirmed latent state to predict that state from observable graph structure
alone. This is V3's Layer 3: hidden operational *state*, never hidden supplier *identity* --
the latter is closed (`findings/evolution.md` §5) and nothing here reopens it.

**Supervision discipline.** Every target here is privileged generator-internal state, lifted by
`ml/confirm_latent_states.py`. It is used at training and evaluation time only and is NEVER a
model input: features come exclusively from the frozen backbone's forward pass over the emitted
graph. `ml/data/loader.py::verify_no_hidden_state` independently asserts none of these
quantities reach the CSVs the loader reads.

**Backbone is frozen and unmodified.** `get_backbone()` calls `freeze()`, and this module adds a
second, stronger assertion (`assert_backbone_frozen`) that runs AFTER the head is constructed and
again after training -- the gap Phase 0's audit identified in `freeze()` itself, which force-sets
`requires_grad=False` before checking and therefore cannot detect a head re-enabling gradients.

**Depth probe.** Each state is estimated from every encoder depth h^0..h^4 separately, not only
the Markov readout depth. h^0 is the raw input projection, so a state that is estimable at h^0 is
estimable from input features alone and the encoder is adding nothing -- the same control
`reports/layer3_testing.md` used when it probed hidden-parent identity at every depth.

**Reproduction floor.** Every reported AUC is accompanied by the spread produced by changing the
head's init seed alone, on fixed data. A delta smaller than that spread is not evidence.

    python3 ml/latent_state_head.py --variant A --seeds 42,43,44,45,46 --config v1 \
        --init-seeds 0,1,2,3,4 --out out/phase1/heads_A.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch
from torch import nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ml.confirm_latent_states import namespace          # noqa: E402
from ml.ds_backbone import get_backbone, load_world     # noqa: E402
from ml.hypothesis_ranker import roc_auc                # noqa: E402

DEPTHS = [0, 1, 2, 3, 4]
# `ml/models/rgcn_attn_markov_encoder.py`: the Supplier-entity task (impact) reads h^4.
MARKOV_SUPPLIER_DEPTH = 4


# ---------------------------------------------------------------------------
# privileged targets
# ---------------------------------------------------------------------------

def latent_targets(variant: str, seed: int, config: str, t0s) -> dict:
    """Per-supplier latent state at each snapshot t0, lifted from the generator.

    Returns `{state: {"kind": ..., "values": {t0_iso: {sup_id: float}}}}`. Only states
    confirmed present on THIS variant are returned -- e.g. `recovery_capability` requires
    Mechanism E and is simply absent from a Variant A run rather than silently zero-filled.
    """
    ns = namespace(variant, seed, config)
    suppliers, sup_by_id = ns["suppliers"], ns["sup_by_id"]
    stress, RESILIENCE, IDIO = ns["stress"], ns["RESILIENCE"], ns["IDIO"]
    visible = ns["VISIBLE_SUP"]

    out: dict = {}

    # -- Supply Stress: continuous, time-varying, the causal driver of everything -------
    sv: dict = {}
    for t0 in t0s:
        tt = t0.to_pydatetime() if hasattr(t0, "to_pydatetime") else t0
        sv[t0.isoformat()] = {s["id"]: stress(s["id"], s["base_rel"], tt)
                              for s in suppliers if s["id"] in visible}
    out["supply_stress"] = {"kind": "continuous_time_varying", "values": sv}

    # -- Supplier Reliability: IDIO outage active at t0. Binary, time-varying ----------
    iv: dict = {}
    for t0 in t0s:
        tt = t0.to_pydatetime() if hasattr(t0, "to_pydatetime") else t0
        row = {}
        for s in suppliers:
            if s["id"] not in visible:
                continue
            ev = IDIO.get(s["id"])
            row[s["id"]] = 1.0 if (ev and ev[0] <= tt <= ev[2]) else 0.0
        iv[t0.isoformat()] = row
    out["supplier_reliability"] = {"kind": "binary_time_varying", "values": iv}

    # -- Mitigation Level: recorded during simulation, requires Mechanism E ------------
    # Provenance differs from every other target here: MITIGATION_HISTORY is written by the
    # generator AS IT RUNS (db/generate_dataset.py:1203-1210), because agent_observe()
    # destructively drains agent_pending and asserts clock monotonicity, so the value cannot
    # be recomputed afterwards the way stress() can.
    #
    # The recorder is keyed by the WEEKLY walk timestamp; snapshots are monthly. Each t0 is
    # therefore matched to the most recent recorded week at or before it -- proper as-of
    # semantics, and never a future read.
    MH = ns.get("MITIGATION_HISTORY", {})
    if MH:
        by_sup: dict = {}
        for (sid, wk), val in MH.items():
            by_sup.setdefault(sid, []).append((wk, val))
        for sid in by_sup:
            by_sup[sid].sort(key=lambda p: p[0])

        mv: dict = {}
        for t0 in t0s:
            tt = t0.to_pydatetime() if hasattr(t0, "to_pydatetime") else t0
            row = {}
            for sid, series in by_sup.items():
                if sid not in visible:
                    continue
                latest = None
                for wk, val in series:
                    if wk <= tt:
                        latest = val
                    else:
                        break
                if latest is not None:      # no record yet at this t0 -> supplier omitted
                    row[sid] = latest
            mv[t0.isoformat()] = row
        out["mitigation_level"] = {"kind": "continuous_time_varying", "values": mv}

    # -- Recovery Capability: RESILIENCE. Static; requires Mechanism E -----------------
    if RESILIENCE:
        rv = {s["id"]: RESILIENCE[s["id"]] for s in suppliers
              if s["id"] in visible and s["id"] in RESILIENCE}
        out["recovery_capability"] = {
            "kind": "continuous_static",
            "values": {t0.isoformat(): rv for t0 in t0s},
        }
    return out


# ---------------------------------------------------------------------------
# frozen-backbone features
# ---------------------------------------------------------------------------

@torch.no_grad()
def depth_embeddings(model, bundles) -> dict:
    """`{depth: (X, sup_ids, t0s)}` -- Supplier embeddings at every encoder depth.

    Runs under `no_grad` on a frozen model, so nothing here can touch SHARE's parameters.
    """
    per_depth = {d: [] for d in DEPTHS}
    ids, times = [], []
    for b in bundles:
        _logits, layers = model(b.data.x_dict, b.data.edge_index_dict)
        node_ids = b.data["Supplier"].node_id
        for d in DEPTHS:
            per_depth[d].append(layers[d]["Supplier"].detach().cpu().numpy())
        ids.extend(list(node_ids))
        times.extend([b.t0.isoformat()] * len(node_ids))
    return {d: (np.concatenate(per_depth[d], axis=0), ids, times) for d in DEPTHS}


def align(X, ids, times, target: dict):
    """Match feature rows to privileged targets by (supplier id, t0); drop unmatched."""
    vals, keep = [], []
    for i, (sid, t) in enumerate(zip(ids, times)):
        row = target["values"].get(t)
        if row is not None and sid in row:
            vals.append(row[sid])
            keep.append(i)
    if not keep:
        return None, None
    return X[np.array(keep)], np.asarray(vals, dtype=float)


def binarise(y_train, y_eval, kind):
    """Median split for continuous states -- the same binarisation
    `db/phase2_coverage_recheck.py:218,236` used to produce the 0.599 resilience precedent,
    so the numbers are comparable. Binary states pass through untouched.

    The threshold is taken from TRAIN only and applied to eval, so no test-set statistic
    leaks into the label definition."""
    if kind.startswith("binary"):
        return y_train, y_eval, None
    thr = float(np.median(y_train))
    return (y_train > thr).astype(float), (y_eval > thr).astype(float), thr


# ---------------------------------------------------------------------------
# the estimator head
# ---------------------------------------------------------------------------

class LatentStateHead(nn.Module):
    """Deliberately small: hidden->32->1. The question is whether the frozen representation
    CARRIES the state, not how much capacity can be stacked on top of it."""

    def __init__(self, in_dim: int, hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def assert_backbone_frozen(model, head=None, optimizer=None) -> None:
    """The assertion `ml/ds_backbone.py::freeze` cannot make.

    `freeze()` sets `requires_grad=False` on every parameter and only then checks, so its
    check is a post-condition on its own mutation and can never fire (Phase 0 audit, check 2).
    This runs AFTER head construction and AFTER training, and additionally verifies no backbone
    parameter reached the optimizer -- the actual failure mode that would let a head's gradients
    into SHARE.
    """
    live = [n for n, p in model.named_parameters() if p.requires_grad]
    if live:
        raise RuntimeError(f"backbone parameter requires grad: {live[:5]}")
    if optimizer is not None:
        backbone = {id(p) for p in model.parameters()}
        leaked = [i for g in optimizer.param_groups for i, p in enumerate(g["params"])
                  if id(p) in backbone]
        if leaked:
            raise RuntimeError(f"backbone parameters reached the optimizer: {len(leaked)}")
    if head is not None:
        shared = {id(p) for p in model.parameters()} & {id(p) for p in head.parameters()}
        if shared:
            raise RuntimeError("head shares parameter objects with the backbone")


def train_head(Xtr, ytr, Xte, yte, init_seed: int, epochs: int = 120,
               lr: float = 1e-3, model=None) -> float | None:
    """Train one head at one init seed; return test AUC (None if a class is degenerate)."""
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return None
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)

    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-8
    xtr = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    xte = torch.tensor((Xte - mu) / sd, dtype=torch.float32)
    ttr = torch.tensor(ytr, dtype=torch.float32)

    head = LatentStateHead(xtr.shape[1])
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    if model is not None:
        assert_backbone_frozen(model, head=head, optimizer=opt)

    pos = float(ttr.sum())
    pw = torch.tensor(max(1.0, (len(ttr) - pos) / max(1.0, pos)), dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)

    head.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(head(xtr), ttr)
        loss.backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        p = torch.sigmoid(head(xte)).numpy()
    if model is not None:
        assert_backbone_frozen(model, head=head, optimizer=opt)
    return roc_auc(p, yte.astype(bool))


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run(variant: str, dseeds: list[int], config: str, init_seeds: list[int],
        csv_root: str, device: str = "cpu") -> dict:
    results: dict = {"variant": variant, "config": config, "dataset_seeds": dseeds,
                     "init_seeds": init_seeds, "per_seed": {}, "states_present": []}

    for dseed in dseeds:
        csv_dir = os.path.join(csv_root, f"v{variant}_seed{dseed}")
        print(f"\n=== variant {variant} seed {dseed} ({os.path.basename(csv_root)}) ===",
              flush=True)
        t_start = time.time()

        model, meta = get_backbone(csv_dir, variant, dseed, mseed=0, device=device)
        assert_backbone_frozen(model)
        tr, va, te, _sup_ids = load_world(csv_dir, device)

        targets = latent_targets(variant, dseed, config,
                                 [b.t0 for b in tr] + [b.t0 for b in va] + [b.t0 for b in te])
        if not results["states_present"]:
            results["states_present"] = sorted(targets)
        print(f"  states present: {sorted(targets)}", flush=True)

        emb_tr = depth_embeddings(model, tr)
        emb_te = depth_embeddings(model, te)

        seed_out: dict = {}
        for state, tgt in sorted(targets.items()):
            per_depth = {}
            for d in DEPTHS:
                Xtr, ytr = align(*emb_tr[d], tgt)
                Xte, yte = align(*emb_te[d], tgt)
                if Xtr is None or Xte is None:
                    continue
                ytr_b, yte_b, thr = binarise(ytr, yte, tgt["kind"])
                aucs = [train_head(Xtr, ytr_b, Xte, yte_b, s, model=model) for s in init_seeds]
                aucs = [a for a in aucs if a is not None]
                if not aucs:
                    continue
                per_depth[d] = {
                    "auc_mean": statistics.fmean(aucs),
                    "auc_all": aucs,
                    "init_seed_spread": max(aucs) - min(aucs),
                    "n_train": int(len(ytr_b)), "n_test": int(len(yte_b)),
                    "pos_train": int(ytr_b.sum()), "pos_test": int(yte_b.sum()),
                    "median_threshold": thr,
                }
                print(f"  {state:<22} h^{d}  AUC {per_depth[d]['auc_mean']:.4f}  "
                      f"floor(init spread) {per_depth[d]['init_seed_spread']:.4f}  "
                      f"pos {per_depth[d]['pos_test']:,}/{per_depth[d]['n_test']:,}",
                      flush=True)
            if per_depth:
                seed_out[state] = per_depth
        results["per_seed"][str(dseed)] = seed_out
        print(f"  ({time.time() - t_start:.0f}s)", flush=True)

    return results


def summarise(results: dict) -> dict:
    """Aggregate across dataset seeds: mean AUC, sign consistency, and the floor."""
    summary: dict = {}
    for state in results["states_present"]:
        per_depth = {}
        for d in DEPTHS:
            vals, spreads, pos, n = [], [], [], []
            for dseed in results["dataset_seeds"]:
                cell = results["per_seed"].get(str(dseed), {}).get(state, {}).get(d)
                if cell:
                    vals.append(cell["auc_mean"])
                    spreads.append(cell["init_seed_spread"])
                    pos.append(cell["pos_test"])
                    n.append(cell["n_test"])
            if not vals:
                continue
            init_floor = max(spreads)
            # `docs/14_Project_Roadmap.md` §4 risk 4: dataset-seed variance dominates total
            # AUC variance (68-99%), so an init-seed-only floor measures the SMALLER source
            # and would overstate significance. Report both and gate on the larger.
            dseed_floor = (max(vals) - min(vals)) if len(vals) > 1 else float("nan")
            floor = max([init_floor] + ([dseed_floor] if len(vals) > 1 else []))
            mean = statistics.fmean(vals)
            per_depth[d] = {
                "auc_mean": mean,
                "auc_per_dataset_seed": vals,
                "n_dataset_seeds": len(vals),
                "init_seed_floor": init_floor,
                "dataset_seed_floor": dseed_floor,
                "reproduction_floor": floor,
                "above_chance_by": mean - 0.5,
                "clears_floor": (mean - 0.5) > floor,
                "sign_consistent": all(v > 0.5 for v in vals) or all(v < 0.5 for v in vals),
                "pos_test_total": sum(pos), "n_test_total": sum(n),
            }
        if per_depth:
            best = max(per_depth, key=lambda k: per_depth[k]["auc_mean"])
            # h^0 is the raw input projection: signal present there is available from input
            # features alone and is NOT contributed by the encoder. Quote the lift explicitly
            # so a state that "works" at h^0 cannot be reported as a representation finding.
            h0 = per_depth.get(0, {}).get("auc_mean")
            lift = (per_depth[best]["auc_mean"] - h0) if h0 is not None else None
            summary[state] = {"per_depth": per_depth, "best_depth": best,
                              "h0_auc": h0,
                              "lift_over_h0": lift,
                              "lift_exceeds_floor": (lift is not None
                                                     and lift > per_depth[best]["reproduction_floor"]),
                              "markov_depth": MARKOV_SUPPLIER_DEPTH,
                              "markov_depth_result": per_depth.get(MARKOV_SUPPLIER_DEPTH)}
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="A")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--config", default="v1")
    ap.add_argument("--init-seeds", default="0,1,2,3,4")
    ap.add_argument("--csv-root", default=os.path.join(REPO, "db", "csv_v1scale"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dseeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    iseeds = [int(s) for s in args.init_seeds.split(",") if s.strip()]

    res = run(args.variant, dseeds, args.config, iseeds, args.csv_root)
    res["summary"] = summarise(res)

    print("\n" + "=" * 104)
    print(f"PHASE 1 SUMMARY — variant {args.variant}, config {args.config}, "
          f"{len(dseeds)} dataset seeds x {len(iseeds)} init seeds")
    print("reference point: Mechanism E resilience recoverable at AUC 0.599 "
          "(docs/phase2_coverage_recheck.md) — a reference, not a target")
    print("=" * 104)
    hdr = (f"{'state':<22} {'depth':>5} {'AUC':>8} {'init fl':>8} {'dset fl':>8} "
           f"{'above .5':>9} {'clears':>7} {'sign':>5} {'pos/test':>17}")
    print(hdr); print("-" * len(hdr))
    for state, s in sorted(res["summary"].items()):
        for d in DEPTHS:
            c = s["per_depth"].get(d)
            if not c:
                continue
            mark = " <- markov" if d == MARKOV_SUPPLIER_DEPTH else ""
            print(f"{state:<22} {'h^'+str(d):>5} {c['auc_mean']:>8.4f} "
                  f"{c['init_seed_floor']:>8.4f} {c['dataset_seed_floor']:>8.4f} "
                  f"{c['above_chance_by']:>+9.4f} "
                  f"{('YES' if c['clears_floor'] else 'no'):>7} "
                  f"{('yes' if c['sign_consistent'] else 'NO'):>5} "
                  f"{c['pos_test_total']:>7,}/{c['n_test_total']:<9,}{mark}")
        h0, lift = s["h0_auc"], s["lift_over_h0"]
        if lift is not None:
            print(f"{'':22} best h^{s['best_depth']}  AUC {s['per_depth'][s['best_depth']]['auc_mean']:.4f}  "
                  f"vs h^0 {h0:.4f}  ->  encoder lift {lift:+.4f}  "
                  f"({'exceeds' if s['lift_exceeds_floor'] else 'DOES NOT exceed'} its floor)")
        print("-" * len(hdr))

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2, sort_keys=True, default=str)
        print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
