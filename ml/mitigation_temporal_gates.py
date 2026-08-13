#!/usr/bin/env python3
"""
Gates 2 and 3 — does Mitigation Level gain anything from a TEMPORAL channel?

Gate 0.5 established a static baseline: the frozen SHARE + Markov representation at a single
snapshot estimates `mitigation_level` at AUC 0.9855 (h^1, Variant E, 5x5 seeds). These gates ask
whether ordered per-snapshot structure adds to that.

  GATE 2 -- deterministic delta. dh(t) = h(t) - h(t-1) from consecutive snapshot embeddings.
            Zero new trainable parameters. Concatenated into the SAME probe architecture and
            evaluation methodology Gate 0.5 used, so the lift is attributable to the feature,
            not to a different model or a different metric.

  GATE 3 -- learned temporal estimator. A small GRU over the ordered per-supplier snapshot
            embeddings. Discrete on purpose: snapshots are monthly and evenly spaced, so a
            continuous-time model would add adjoint solvers and stability tuning to fit a shape
            that is not observable at this resolution.

**The comparison is made on identical rows.** dh is undefined at the first snapshot of a split,
so those rows are dropped -- and the static baseline is RECOMPUTED on the same reduced row set.
Comparing a temporal model on n-k rows against Gate 0.5's baseline on n rows would confound the
feature with the sample.

**Ordered snapshots are constructed post hoc, not a native SHARE capability.** `ml/train.py` and
`ml/data/loader.py` forward every snapshot independently; there is no recurrence anywhere in the
existing pipeline. The sequence here is assembled by collecting SHARE's independent per-t0 outputs
in order, and `ml/ds_backbone.py::load_world` asserts the Supplier node ordering is stable across
snapshots, which is what makes that assembly valid.

    python3 ml/mitigation_temporal_gates.py --variant E --seeds 42,43,44,45,46 --config v1 --depth 1
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import numpy as np
import torch
from torch import nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ml.ds_backbone import get_backbone, load_world                       # noqa: E402
from ml.hypothesis_ranker import roc_auc                                  # noqa: E402
from ml.latent_state_head import (LatentStateHead, assert_backbone_frozen,  # noqa: E402
                                  latent_targets)

STATE = "mitigation_level"


@torch.no_grad()
def ordered_embeddings(model, bundles, depth: int):
    """[n_snapshots, n_suppliers, hidden] plus the supplier ids and t0s, in order."""
    mats, ids, t0s = [], None, []
    for b in bundles:
        _logits, layers = model(b.data.x_dict, b.data.edge_index_dict)
        mats.append(layers[depth]["Supplier"].detach().cpu().numpy())
        if ids is None:
            ids = list(b.data["Supplier"].node_id)
        t0s.append(b.t0)
    return np.stack(mats, axis=0), ids, t0s


def build_rows(H, ids, t0s, target):
    """Rows for one split. Returns (h, dh, y, n_dropped).

    The first snapshot of the split has no predecessor INSIDE the split, so dh is undefined
    there and every row at that snapshot is dropped."""
    h_rows, dh_rows, y_rows = [], [], []
    dropped = 0
    for si, t0 in enumerate(t0s):
        row = target["values"].get(t0.isoformat())
        if row is None:
            continue
        for ni, sid in enumerate(ids):
            if sid not in row:
                continue
            if si == 0:
                dropped += 1
                continue
            h_rows.append(H[si, ni])
            dh_rows.append(H[si, ni] - H[si - 1, ni])
            y_rows.append(row[sid])
    if not h_rows:
        return None, None, None, dropped
    return (np.stack(h_rows), np.stack(dh_rows), np.asarray(y_rows, dtype=float), dropped)


def fit_probe(Xtr, ytr, Xte, yte, seed, epochs=120, lr=1e-3):
    """Gate 0.5's probe, unchanged."""
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return None
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-8
    xtr = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    xte = torch.tensor((Xte - mu) / sd, dtype=torch.float32)
    ttr = torch.tensor(ytr, dtype=torch.float32)
    head = LatentStateHead(xtr.shape[1])
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    pos = float(ttr.sum())
    pw = torch.tensor(max(1.0, (len(ttr) - pos) / max(1.0, pos)), dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    head.train()
    for _ in range(epochs):
        opt.zero_grad(); lossf(head(xtr), ttr).backward(); opt.step()
    head.eval()
    with torch.no_grad():
        return roc_auc(torch.sigmoid(head(xte)).numpy(), yte.astype(bool))


class GRUEstimator(nn.Module):
    """Gate 3. Small, discrete, single-layer; per-timestep output so every snapshot is scored."""

    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=1, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):                      # x: [n_suppliers, n_steps, in_dim]
        h, _ = self.gru(x)
        return self.out(h).squeeze(-1)         # [n_suppliers, n_steps]


def fit_gru(Htr, Ytr, Mtr, Hte, Yte, Mte, seed, epochs=200, lr=1e-3):
    """Sequence model over ordered snapshots. M is the valid-row mask."""
    torch.manual_seed(seed); np.random.seed(seed)
    mu = Htr.reshape(-1, Htr.shape[-1]).mean(0)
    sd = Htr.reshape(-1, Htr.shape[-1]).std(0) + 1e-8
    xtr = torch.tensor((Htr - mu) / sd, dtype=torch.float32)
    xte = torch.tensor((Hte - mu) / sd, dtype=torch.float32)
    ytr = torch.tensor(Ytr, dtype=torch.float32)
    mtr = torch.tensor(Mtr, dtype=torch.float32)

    model = GRUEstimator(xtr.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    pos = float((ytr * mtr).sum()); tot = float(mtr.sum())
    pw = torch.tensor(max(1.0, (tot - pos) / max(1.0, pos)), dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = (lossf(model(xtr), ytr) * mtr).sum() / mtr.sum()
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        p = model(xte).numpy()
    m = Mte.astype(bool)
    return roc_auc(p[m], Yte[m].astype(bool))


def seq_tensors(H, ids, t0s, target, thr):
    """[n_suppliers, n_steps, hidden] with binarised targets and a validity mask.
    Step 0 is masked out, matching Gate 2's dropped rows exactly."""
    n_s, n_n = H.shape[0], H.shape[1]
    X = np.transpose(H, (1, 0, 2))
    Y = np.zeros((n_n, n_s)); M = np.zeros((n_n, n_s))
    for si, t0 in enumerate(t0s):
        row = target["values"].get(t0.isoformat())
        if row is None or si == 0:
            continue
        for ni, sid in enumerate(ids):
            if sid in row:
                Y[ni, si] = 1.0 if row[sid] > thr else 0.0
                M[ni, si] = 1.0
    return X, Y, M


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="E")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--config", default="v1")
    ap.add_argument("--init-seeds", default="0,1,2,3,4")
    ap.add_argument("--depth", type=int, default=1, help="depth fixed by Gate 0.5")
    ap.add_argument("--csv-root", default=os.path.join(REPO, "db", "csv_v1scale"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dseeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    iseeds = [int(s) for s in args.init_seeds.split(",") if s.strip()]
    per_seed: dict = {}

    for dseed in dseeds:
        csv_dir = os.path.join(args.csv_root, f"v{args.variant}_seed{dseed}")
        print(f"\n=== variant {args.variant} seed {dseed} (depth h^{args.depth}) ===", flush=True)
        model, _ = get_backbone(csv_dir, args.variant, dseed, mseed=0, device="cpu", verbose=False)
        assert_backbone_frozen(model)
        tr, va, te, _ = load_world(csv_dir, "cpu")
        targets = latent_targets(args.variant, dseed, args.config,
                                 [b.t0 for b in tr] + [b.t0 for b in va] + [b.t0 for b in te])
        tgt = targets.get(STATE)
        if tgt is None:
            print("  mitigation_level absent on this variant — skipping", flush=True)
            continue

        Htr, ids_tr, t0_tr = ordered_embeddings(model, tr, args.depth)
        Hte, ids_te, t0_te = ordered_embeddings(model, te, args.depth)

        htr, dhtr, ytr, drop_tr = build_rows(Htr, ids_tr, t0_tr, tgt)
        hte, dhte, yte, drop_te = build_rows(Hte, ids_te, t0_te, tgt)
        if htr is None or hte is None:
            continue

        # median split, threshold from TRAIN only -- Gate 0.5's binarisation, unchanged
        thr = float(np.median(ytr))
        ytr_b, yte_b = (ytr > thr).astype(float), (yte > thr).astype(float)

        static_tr, static_te = htr, hte
        temporal_tr = np.hstack([htr, dhtr])
        temporal_te = np.hstack([hte, dhte])

        a_static = [fit_probe(static_tr, ytr_b, static_te, yte_b, s) for s in iseeds]
        a_temporal = [fit_probe(temporal_tr, ytr_b, temporal_te, yte_b, s) for s in iseeds]
        a_static = [a for a in a_static if a is not None]
        a_temporal = [a for a in a_temporal if a is not None]

        # Gate 3
        X_tr, Y_tr, M_tr = seq_tensors(Htr, ids_tr, t0_tr, tgt, thr)
        X_te, Y_te, M_te = seq_tensors(Hte, ids_te, t0_te, tgt, thr)
        a_gru = [fit_gru(X_tr, Y_tr, M_tr, X_te, Y_te, M_te, s) for s in iseeds]
        a_gru = [a for a in a_gru if a is not None]

        # Paired lifts: the same init seed drives both arms, on identical rows and the same
        # frozen backbone, so lift_i isolates the FEATURE SET. The floor for a paired delta is
        # the variability of the delta -- gating it against the spread of the absolute AUC
        # would charge the lift for between-world differences it does not contain.
        paired_dh = [t - s for t, s in zip(a_temporal, a_static)]
        paired_gru_dh = [g - t for g, t in zip(a_gru, a_temporal)] if a_gru else []

        cell = {
            "static_auc_mean": statistics.fmean(a_static),
            "static_init_spread": max(a_static) - min(a_static),
            "static_auc_all": a_static,
            "temporal_auc_mean": statistics.fmean(a_temporal),
            "temporal_init_spread": max(a_temporal) - min(a_temporal),
            "temporal_auc_all": a_temporal,
            "gru_auc_mean": statistics.fmean(a_gru) if a_gru else None,
            "gru_init_spread": (max(a_gru) - min(a_gru)) if a_gru else None,
            "gru_auc_all": a_gru,
            "paired_lift_delta_h_all": paired_dh,
            "paired_lift_delta_h_init_spread": max(paired_dh) - min(paired_dh),
            "paired_lift_gru_over_dh_all": paired_gru_dh,
            "paired_lift_gru_over_dh_init_spread": ((max(paired_gru_dh) - min(paired_gru_dh))
                                                    if paired_gru_dh else None),
            "lift_delta_h": statistics.fmean(a_temporal) - statistics.fmean(a_static),
            "lift_gru_over_static": (statistics.fmean(a_gru) - statistics.fmean(a_static))
                                    if a_gru else None,
            "lift_gru_over_delta_h": (statistics.fmean(a_gru) - statistics.fmean(a_temporal))
                                     if a_gru else None,
            "n_train": int(len(ytr_b)), "n_test": int(len(yte_b)),
            "pos_test": int(yte_b.sum()),
            "dropped_first_snapshot_train": drop_tr,
            "dropped_first_snapshot_test": drop_te,
        }
        per_seed[str(dseed)] = cell
        print(f"  static h^{args.depth:<1}      AUC {cell['static_auc_mean']:.4f}  "
              f"(init spread {cell['static_init_spread']:.4f})", flush=True)
        print(f"  + delta-h        AUC {cell['temporal_auc_mean']:.4f}  "
              f"(init spread {cell['temporal_init_spread']:.4f})  "
              f"lift {cell['lift_delta_h']:+.4f}", flush=True)
        if cell["gru_auc_mean"] is not None:
            print(f"  GRU (seq)        AUC {cell['gru_auc_mean']:.4f}  "
                  f"(init spread {cell['gru_init_spread']:.4f})  "
                  f"vs static {cell['lift_gru_over_static']:+.4f}  "
                  f"vs delta-h {cell['lift_gru_over_delta_h']:+.4f}", flush=True)
        print(f"  rows: train {cell['n_train']:,} (dropped {drop_tr:,}), "
              f"test {cell['n_test']:,} (dropped {drop_te:,}), "
              f"pos {cell['pos_test']:,}", flush=True)

    # ---- aggregate: two floors, gate on the larger --------------------------------
    def agg(key, spread_key):
        vals = [c[key] for c in per_seed.values() if c.get(key) is not None]
        spreads = [c[spread_key] for c in per_seed.values() if c.get(spread_key) is not None]
        if not vals:
            return None
        return {"mean": statistics.fmean(vals), "per_seed": vals,
                "init_floor": max(spreads),
                "dataset_floor": (max(vals) - min(vals)) if len(vals) > 1 else float("nan"),
                "floor": max([max(spreads)] + ([max(vals) - min(vals)] if len(vals) > 1 else []))}

    static = agg("static_auc_mean", "static_init_spread")
    temporal = agg("temporal_auc_mean", "temporal_init_spread")
    gru = agg("gru_auc_mean", "gru_init_spread")

    lifts_d = [c["lift_delta_h"] for c in per_seed.values()]
    lifts_g = [c["lift_gru_over_delta_h"] for c in per_seed.values()
               if c.get("lift_gru_over_delta_h") is not None]

    def lift_block(per_seed_lifts, init_spread_key, absolute_floor):
        """Two readings of the same lift.

        `paired` gates the lift against the variability of the lift itself (init-seed spread
        within a world, and between-world spread of the lift) -- the appropriate test for a
        paired comparison. `absolute` gates it against the spread of the absolute AUC, which
        is stricter and charges the lift for between-world differences it does not contain.
        Both are reported; the paired reading is the one the gate turns on, and the absolute
        reading is stated so the stricter verdict is visible rather than hidden.
        """
        init_spreads = [c[init_spread_key] for c in per_seed.values()
                        if c.get(init_spread_key) is not None]
        # NOTE the parenthesisation: `[a] if cond else [b] + [c]` parses as
        # `[a] if cond else ([b] + [c])`, which silently drops the between-seed term whenever
        # init spreads exist. Build the candidate list explicitly instead.
        candidates = list(init_spreads)
        if len(per_seed_lifts) > 1:
            candidates.append(max(per_seed_lifts) - min(per_seed_lifts))
        paired_floor = max(candidates) if candidates else 0.0
        mean = statistics.fmean(per_seed_lifts)
        return {
            "lift_mean": mean, "lift_per_seed": per_seed_lifts,
            "paired_init_floor": max(init_spreads) if init_spreads else None,
            "paired_between_seed_floor": ((max(per_seed_lifts) - min(per_seed_lifts))
                                          if len(per_seed_lifts) > 1 else None),
            "paired_floor": paired_floor,
            "clears_paired_floor": mean > paired_floor,
            "absolute_auc_floor": absolute_floor,
            "clears_absolute_floor": mean > absolute_floor,
            "sign_consistent": all(v > 0 for v in per_seed_lifts) or all(v < 0 for v in per_seed_lifts),
        }

    summary = {
        "static": static, "temporal_delta_h": temporal, "gru": gru,
        "gate2": lift_block(lifts_d, "paired_lift_delta_h_init_spread", temporal["floor"]),
        "gate3": (lift_block(lifts_g, "paired_lift_gru_over_dh_init_spread", gru["floor"])
                  if lifts_g else None),
    }
    if summary["gate3"]:
        summary["gate3"]["beats_gate2"] = summary["gate3"]["lift_mean"] > 0

    print("\n" + "=" * 96)
    print(f"GATES 2 & 3 — mitigation temporal channel, variant {args.variant}, "
          f"h^{args.depth}, {len(per_seed)} dataset seeds x {len(iseeds)} init seeds")
    print("=" * 96)
    for name, blk in (("static (Gate 0.5 basis)", static), ("+ delta-h (Gate 2)", temporal),
                      ("GRU (Gate 3)", gru)):
        if blk:
            print(f"  {name:<26} AUC {blk['mean']:.4f}  init floor {blk['init_floor']:.4f}  "
                  f"dataset floor {blk['dataset_floor']:.4f}")
    for name, blk in (("GATE 2  (delta-h over static)", summary["gate2"]),
                      ("GATE 3  (GRU over delta-h)", summary["gate3"])):
        if not blk:
            continue
        print(f"\n  {name}")
        print(f"          lift {blk['lift_mean']:+.4f}   per-seed "
              f"{[round(v, 4) for v in blk['lift_per_seed']]}")
        print(f"          paired floor   {blk['paired_floor']:.4f} "
              f"(init {blk['paired_init_floor']:.4f}, between-seed "
              f"{blk['paired_between_seed_floor']:.4f})  -> "
              f"{'CLEARS' if blk['clears_paired_floor'] else 'MISSES'}")
        print(f"          absolute floor {blk['absolute_auc_floor']:.4f} (stricter)        -> "
              f"{'CLEARS' if blk['clears_absolute_floor'] else 'MISSES'}")
        print(f"          sign-consistent across seeds: "
              f"{'yes' if blk['sign_consistent'] else 'NO'}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"variant": args.variant, "depth": args.depth,
                       "per_seed": per_seed, "summary": summary}, f, indent=2, default=str)
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
