#!/usr/bin/env python3
"""
Phase 1b — a delta-prediction head supervised directly on generator-produced true effects.

Built **because Phase 1 was a clean miss**, which is the condition the build prompt sets for
this component existing at all. The naive approach -- edit the graph, re-run the frozen
forward pass, subtract -- reached sign agreement of **0.524** against a chance level of 0.500
over 2,695 affected entities, was never sign-consistent across all five dataset seeds, went
*below* chance on `add_dual_source` (0.465), and produced deltas roughly **8x too small**
(magnitude ratio 0.13) while saying nothing at all on 36% of truly-affected entities. An
associational predictor handed an edited input does not answer the intervention question here.

**What changes, and what does not.** SHARE and the Markov Blanket depth readout are still
frozen and still unmodified; this head sits on top of their outputs. The difference from
Phase 1 is the *training signal*: instead of hoping a forward pass on an edited graph
implies the effect, this head is fitted directly on
`(factual embedding, edited embedding, intervention encoding, true delta)` quadruples, where
the true delta comes from the generator's own causal recomputation. That is an interventional
training signal, which is precisely what Phase 1 established the model lacks.

**Input features per (entity, intervention).** Deliberately small and interpretable, because
the question is whether the *signal* is learnable, not whether a large head can memorise:

    z_v            frozen Markov readout embedding of the entity, factual        [128]
    z~_v           the same under the edited graph                               [128]
    z~_v - z_v     the shift the edit produced, given explicitly                 [128]
    naive_delta    what Phase 1 predicted for this entity                        [1]
    rho_v          reach features: hop distance to the intervened set, whether   [5]
                   it is within the task's Markov depth, how many intervened
                   nodes are within that depth, |T(a)|, and membership
    kind           one-hot over the three intervention types                     [3]

`rho_v` matters more than its size suggests: the generator's coupling term reaches only
*direct* co-parents, so hop distance is close to a sufficient statistic for whether any true
effect exists at all, and giving it explicitly separates "can the effect be predicted" from
"can the affected set be identified".

**Split discipline.** Leave-one-dataset-seed-out, and additionally no *target supplier* may
appear in both train and test folds. §9.8 of `reports/layer3_testing.md` established that this
benchmark's structure is distinguishable largely by entity identity; a split that shared
targets would let the head memorise which suppliers matter and post a meaningless held-out
number.

**Zero-inflation.** Most entities have exactly zero true effect. The head therefore emits a
*gate* and a *magnitude*, and the loss scores them separately -- otherwise a regressor
minimises its objective almost perfectly by predicting zero everywhere, which is the trap
§10's `unknown` class was designed around.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from ml.counterfactual_edit import edit_graph, naive_counterfactual  # noqa: E402
from ml.counterfactual_ground_truth import (  # noqa: E402
    CausalWorld, apply_intervention, generator_namespace, sample_interventions)
from ml.ds_backbone import get_backbone, predict  # noqa: E402
from ml.models.depth import TASK_ENTITY_TYPE  # noqa: E402

KINDS = ("add_dual_source", "remove_supplier", "substitute_supplier")
EPS = 1e-9
PRED_EPS = 1e-4


class DeltaHead(nn.Module):
    """Two outputs: a gate logit (is there any effect) and a signed magnitude."""

    def __init__(self, d_in: int, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, 64), nn.ReLU())
        self.gate = nn.Linear(64, 1)
        self.mag = nn.Linear(64, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.gate(h).squeeze(-1), self.mag(h).squeeze(-1)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


@torch.no_grad()
def _readout(model, data, task: str) -> np.ndarray:
    """Markov readout embedding for `task`'s entity type, from the frozen encoder.

    Reads the *same* fixed depth the prediction uses -- `delay -> h^1`, `shortage -> h^3`,
    `impact -> h^4`. No depth adaptivity is introduced anywhere.
    """
    from ml.models.rgcn_attn_markov_encoder import MARKOV_READOUT_DEPTH
    model.eval()
    layers = model.encoder(data.x_dict, data.edge_index_dict)
    et = TASK_ENTITY_TYPE[task]
    return layers[MARKOV_READOUT_DEPTH[task]][et].float().cpu().numpy()


def reach_features(world: CausalWorld, sup_ids: list[str], touched: set,
                   coparents: dict) -> np.ndarray:
    """`[N, 5]` -- distance-like features on the co-parent graph, which is the only channel
    the generator's coupling actually travels along on Variant 0."""
    idx = {s: i for i, s in enumerate(sup_ids)}
    n = len(sup_ids)
    dist = np.full(n, 6.0)
    frontier = {s for s in touched if s in idx}
    seen = set(frontier)
    for s in frontier:
        dist[idx[s]] = 0.0
    d = 0
    while frontier and d < 5:
        d += 1
        nxt = set()
        for s in frontier:
            for p in coparents.get(s, ()):
                if p in idx and p not in seen:
                    seen.add(p)
                    dist[idx[p]] = float(d)
                    nxt.add(p)
        frontier = nxt
    return np.stack([
        dist,
        (dist <= 1).astype(float),
        (dist == 0).astype(float),
        np.full(n, float(len(touched))),
        np.array([1.0 if s in touched else 0.0 for s in sup_ids]),
    ], axis=1)


def build_dataset(dseed: int, variant: str, config: str, csv_root: str, n_per_kind: int,
                  mseed: int, sample_seed: int, task: str) -> dict:
    """Assemble `(X, y, meta)` for one world."""
    csv_dir = os.path.join(csv_root, f"v{variant}_seed{dseed}")
    model, meta = get_backbone(csv_dir, variant, dseed, mseed, device="cpu", verbose=False)
    test = meta["test_bundles"]
    sup_ids = meta["supplier_ids"]
    sup_index = {s: i for i, s in enumerate(sup_ids)}

    ns = generator_namespace(variant, dseed, config)
    world = CausalWorld(ns)
    n_all = len(world.T0S)
    off = int(round(0.4 * n_all)) + int(round(0.2 * n_all))
    p0 = world.p_impact(world.base_coparents)
    specs = sample_interventions(world, n_per_kind, sample_seed)

    base_emb = [_readout(model, b.data, task) for b in test]
    base_pred = [predict(model, b)[task] for b in test]

    X, Y, TGT, KND = [], [], [], []
    for spec in specs:
        kind = spec["kind"]
        cop, owner, touched = apply_intervention(world, kind, spec)
        p1 = world.p_impact(cop, owner)
        rho = reach_features(world, sup_ids, touched, world.base_coparents)
        khot = np.zeros(len(KINDS)); khot[KINDS.index(kind)] = 1.0

        for j, bundle in enumerate(test):
            ti = off + j
            edited = edit_graph(bundle.data, kind, spec, sup_index)
            if edited is None:
                continue
            z, zt = base_emb[j], _readout(model, edited, task)
            if zt.shape != z.shape:
                continue
            naive = naive_counterfactual(model, bundle, kind, spec, sup_index, task=task,
                                         baseline=base_pred[j])
            if naive is None:
                continue
            for si, s in enumerate(sup_ids):
                k = (s, ti)
                if k not in p0:
                    continue
                v1 = p1.get(k)
                if v1 is None:
                    continue
                X.append(np.concatenate([z[si], zt[si], zt[si] - z[si],
                                         [naive[si]], rho[si], khot]))
                Y.append(v1 - p0[k])
                TGT.append(s)
                KND.append(kind)
    return {"X": np.asarray(X, dtype=np.float32), "y": np.asarray(Y, dtype=np.float32),
            "target": np.asarray(TGT), "kind": np.asarray(KND), "dseed": dseed}


def _attribute_overlap(csv_root: str, variant: str, d_a: int, d_b: int) -> float:
    """Fraction of shared supplier ids whose observable attributes actually agree.

    Near-chance means the shared ids are a naming artifact, not a persistent entity. This is
    the check that distinguishes "the same supplier appears in both worlds" (real leakage)
    from "the same UUID labels two unrelated suppliers" (not leakage).
    """
    import pandas as pd
    a = pd.read_csv(os.path.join(csv_root, f"v{variant}_seed{d_a}", "suppliers.csv.gz"))
    b = pd.read_csv(os.path.join(csv_root, f"v{variant}_seed{d_b}", "suppliers.csv.gz"))
    m = a.merge(b, on="id", suffixes=("_a", "_b"))
    if m.empty:
        return 0.0
    cols = ["country", "lead_time_days", "capacity_score", "reliability_history"]
    return float(np.mean([(m[f"{c}_a"] == m[f"{c}_b"]).mean() for c in cols]))


def train_head(Xtr, ytr, Xva, yva, epochs=300, lr=1e-3, seed=0, max_pos_weight=50.0):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-8] = 1.0
    xtr = torch.tensor((Xtr - mu) / sd)
    xva = torch.tensor((Xva - mu) / sd)
    ttr = torch.tensor(ytr)
    tva = torch.tensor(yva)
    gtr = (ttr.abs() > EPS).float()
    gva = (tva.abs() > EPS).float()
    pw = float(np.clip((len(gtr) - gtr.sum().item()) / max(gtr.sum().item(), 1.0),
                       1.0, max_pos_weight))
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw))
    model = DeltaHead(Xtr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best, best_state = float("inf"), None
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        g, m = model(xtr)
        loss = bce(g, gtr)
        pos = gtr > 0
        if pos.any():
            # Huber on the signed magnitude, on truly-affected rows only -- the gate is
            # responsible for the zeros, so the regressor is not dragged toward zero.
            loss = loss + nn.functional.smooth_l1_loss(m[pos], ttr[pos], beta=0.05)
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            gv, mv = model(xva)
            vl = bce(gv, gva).item()
            pv = gva > 0
            if pv.any():
                vl += nn.functional.smooth_l1_loss(mv[pv], tva[pv], beta=0.05).item()
        if vl < best - 1e-7:
            best = vl
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model, mu, sd, pw


@torch.no_grad()
def predict_delta(model, mu, sd, X):
    model.eval()
    g, m = model(torch.tensor((X - mu) / sd))
    return (torch.sigmoid(g) * m).numpy(), torch.sigmoid(g).numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="0")
    ap.add_argument("--config", default="v1")
    ap.add_argument("--csv-root", default=os.path.join(REPO, "db", "csv_v1scale"))
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--interventions", type=int, default=30)
    ap.add_argument("--sample-seed", type=int, default=20260813)
    ap.add_argument("--task", default="impact")
    ap.add_argument("--head-seeds", default="0,1,2,3,4",
                    help="head-init seeds; the spread across them is this metric's floor")
    ap.add_argument("--out", default=os.path.join(REPO, "out", "cf_phase1b.json"))
    a = ap.parse_args()

    dseeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    hseeds = [int(s) for s in a.head_seeds.split(",") if s.strip()]

    data = {}
    for d in dseeds:
        data[d] = build_dataset(d, a.variant, a.config, a.csv_root, a.interventions,
                                a.model_seed, a.sample_seed, a.task)
        n_aff = int((np.abs(data[d]["y"]) > EPS).sum())
        print(f"  built d{d}: {len(data[d]['y']):,} rows, {n_aff:,} affected", flush=True)

    folds = []
    csv_root, variant = a.csv_root, a.variant
    for i, test_d in enumerate(dseeds):
        val_d = dseeds[(i + 1) % len(dseeds)]
        tr_d = [d for d in dseeds if d not in (test_d, val_d)]
        Xtr = np.concatenate([data[d]["X"] for d in tr_d])
        ytr = np.concatenate([data[d]["y"] for d in tr_d])
        # Asserting id-disjointness across worlds FAILS here, and the reason is worth
        # recording rather than relaxing away. The generator derives primary keys as UUIDv5
        # from stable keys (docs/08_Backend_Design.md §3), and a supplier's stable key is its
        # index -- so supplier ids are **100% shared across dataset seeds**. The entity behind
        # the id is not: measured on seeds 42 vs 43, `country` matches on 19.5% of ids (chance
        # is 16.7% for six countries), `lead_time_days` on 2.2%, and `capacity_score` and
        # `reliability_history` on 0.0%. The identifier persists; the business does not.
        #
        # So the memorisation risk this check exists to exclude is genuinely absent -- there
        # is no stable entity to memorise -- but the check has to be on the property that
        # actually holds. It is therefore on attribute divergence, not id disjointness.
        tr_tgt = set(np.concatenate([data[d]["target"] for d in tr_d]).tolist())
        te_tgt = set(data[test_d]["target"].tolist())
        shared = tr_tgt & te_tgt
        if shared:
            same_attr = _attribute_overlap(csv_root, variant, tr_d[0], test_d)
            assert same_attr < 0.25, (
                f"ids shared across worlds AND attributes agree on {same_attr:.1%} of them -- "
                f"entities may be genuinely paired, which would be real leakage")

        for hs in hseeds:
            head, mu, sd, pw = train_head(Xtr, ytr, data[val_d]["X"], data[val_d]["y"],
                                          seed=hs)
            dhat, gate = predict_delta(head, mu, sd, data[test_d]["X"])
            yte = data[test_d]["y"]
            kte = data[test_d]["kind"]
            aff = np.abs(yte) > EPS
            rec = {"test_seed": test_d, "val_seed": val_d, "head_seed": hs,
                   "n_train": int(len(ytr)), "n_test": int(len(yte)),
                   "n_affected": int(aff.sum()), "pos_weight": pw,
                   "params": head.parameter_count()}
            if aff.sum():
                rec["sign_agreement"] = float((np.sign(yte[aff]) == np.sign(dhat[aff])).mean())
                rec["magnitude_ratio"] = float(np.abs(dhat[aff]).mean()
                                               / max(np.abs(yte[aff]).mean(), 1e-12))
                rec["silent_fraction"] = float((np.abs(dhat[aff]) < PRED_EPS).mean())
                # Naive baseline on the SAME rows, so the comparison is paired.
                naive = data[test_d]["X"][:, 3 * 128]
                rec["naive_sign_agreement"] = float(
                    (np.sign(yte[aff]) == np.sign(naive[aff])).mean())
                rec["per_kind"] = {
                    k: float((np.sign(yte[aff & (kte == k)])
                              == np.sign(dhat[aff & (kte == k)])).mean())
                    for k in KINDS if (aff & (kte == k)).sum()}
            folds.append(rec)
            print(f"  fold test=d{test_d} head={hs}: "
                  f"sign={rec.get('sign_agreement')} "
                  f"naive={rec.get('naive_sign_agreement')} "
                  f"magratio={rec.get('magnitude_ratio')}", flush=True)

    blob = {"config": vars(a), "folds": folds}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(blob, fh, indent=1, default=float)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
