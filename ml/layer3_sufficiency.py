#!/usr/bin/env python3
"""
Layer 3, STEP 1 — downstream usefulness / sufficiency of the estimated latent state.

**What this step does NOT establish, stated up front and repeated in the report.** Nothing
here is evidence that Z is a *correct* or *faithful* representation of the true latent state.
A downstream-informative Z can be a confounded proxy, a partial shadow, or a rescaling of some
other observable. Latent-state correctness is STEP 2's question (identifiability /
recoverability), not this one. A positive result here must not be read as evidence of
identifiability or causal correctness.

**The comparison.** For each state, with the frozen backbone and Model A's head:

    P(Y|Z)    -- downstream task from the ESTIMATED state alone
    P(Y|X,Z)  -- downstream task from the observed supplier features PLUS the estimated state
    P(Y|X)    -- observed features alone, reported as context for reading the other two

`Y` is `impact`, which is the only one of the three benchmark tasks whose label is carried by
the **Supplier** entity (`ml/models/depth.py::TASK_ENTITY_TYPE`) and therefore the only one
that can be aligned row-for-row with a per-supplier latent state. `X` is the raw observable
supplier feature vector the loader builds (`data["Supplier"].x`) -- the same 14 numbers h^0
projects, i.e. genuinely "observed features", not a representation derived from them.

**Three-way split, so no arm is scored on its own training data.**

    tr (40%)  -> fits the Z head (Model A's head, unmodified)
    va (20%)  -> fits the downstream P(Y|.) head, on Z values that are OUT OF SAMPLE for it
    te (40%)  -> scores everything

Training the downstream head on `tr` would feed it in-sample Z, whose error distribution is
not the one it meets at test time. That would be a leak in the only direction that flatters
this step, so it is not done.

**Model A's head is reused, never modified.** `LatentStateHead`, `train_head`'s protocol,
`latent_targets`, `align`, `binarise` and `depth_embeddings` are all imported from
`ml/latent_state_head.py`. `fit_probs()` below is `train_head` with the return value changed
from "AUC" to "probabilities" and nothing else touched; `--verify-model-a` proves that by
re-deriving Phase 1's per-seed AUC from these very probabilities and diffing against
`out/phase1/heads_{A,E}.json` to 1e-9.

**Gate.** `P(Y|Z)` alone must clear its own reproduction floor above chance, where the floor is
`max(init-seed spread, dataset-seed spread)` -- Phase 1's definition, gated on the larger source
per `docs/14_Project_Roadmap.md` §4 risk 4. The `P(Y|X,Z)` - `P(Y|Z)` gap is REPORTED, never
gated on: it measures how incomplete Z is, which is diagnostic, not disqualifying.

    python3 ml/layer3_sufficiency.py --variant A --states supply_stress \
        --seeds 42,43,44,45,46 --init-seeds 0,1,2,3,4 --out out/layer3/step1_A.json
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

from ml.ds_backbone import get_backbone, load_world              # noqa: E402
from ml.hypothesis_ranker import roc_auc                         # noqa: E402
from ml.latent_state_head import (                               # noqa: E402
    LatentStateHead, align, assert_backbone_frozen, binarise, depth_embeddings,
    latent_targets,
)

TASK = "impact"          # the only Supplier-entity task; see module docstring


# ---------------------------------------------------------------------------
# Model A's head, returning probabilities instead of a scalar AUC
# ---------------------------------------------------------------------------

def fit_probs(Xtr, ytr, evals, init_seed: int, epochs: int = 120, lr: float = 1e-3,
              model=None):
    """`ml/latent_state_head.py::train_head`, byte-for-byte in every training decision.

    Same standardisation (train mean/std), same architecture, same Adam(lr=1e-3,
    weight_decay=1e-4), same 120 full-batch epochs, same `pos_weight`, same seeding of both
    torch and numpy. The ONLY change is that it returns sigmoid probabilities for a list of
    evaluation matrices rather than a single AUC -- because a calibration and confidence
    pipeline cannot be built on a scalar. `--verify-model-a` checks the equivalence
    numerically rather than asking the reader to take this paragraph on trust.
    """
    if len(np.unique(ytr)) < 2:
        return None
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)

    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-8
    xtr = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
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
    out = []
    with torch.no_grad():
        for Xe in evals:
            xe = torch.tensor((Xe - mu) / sd, dtype=torch.float32)
            out.append(torch.sigmoid(head(xe)).numpy())
    if model is not None:
        assert_backbone_frozen(model, head=head, optimizer=opt)
    return out


# ---------------------------------------------------------------------------
# observables and the downstream label
# ---------------------------------------------------------------------------

def observed_rows(bundles) -> tuple:
    """`(X_obs, sup_ids, t0s)` -- the raw Supplier input features, in bundle/node order.

    This is exactly what the encoder is handed at h^0's input, i.e. the observed feature
    vector, with no message passing and no learned transform applied.
    """
    X, ids, times = [], [], []
    for b in bundles:
        X.append(b.data["Supplier"].x.detach().cpu().numpy())
        node_ids = b.data["Supplier"].node_id
        ids.extend(list(node_ids))
        times.extend([b.t0.isoformat()] * len(node_ids))
    return np.concatenate(X, axis=0), ids, times


def downstream_labels(bundles) -> dict:
    """`{(sup_id, t0_iso): y}` for the `impact` task, read from the bundles' own label
    tensors -- the same tensors `ml/evaluate.py` scores the backbone on."""
    out = {}
    for b in bundles:
        idx, y = b.labels[TASK]
        node_ids = b.data["Supplier"].node_id
        t = b.t0.isoformat()
        for i, yy in zip(idx.detach().cpu().numpy(), y.detach().cpu().numpy()):
            out[(node_ids[int(i)], t)] = float(yy)
    return out


def stack_arm(z: np.ndarray, X: np.ndarray, arm: str) -> np.ndarray:
    if arm == "Z":
        return z.reshape(-1, 1)
    if arm == "X":
        return X
    return np.concatenate([X, z.reshape(-1, 1)], axis=1)


ARMS = ("Z", "X", "XZ")


# ---------------------------------------------------------------------------
# one (variant, dataset seed, state)
# ---------------------------------------------------------------------------

def run_state(model, tr, va, te, targets, state: str, depth: int, init_seeds: list[int],
              verify: dict | None) -> dict | None:
    tgt = targets.get(state)
    if tgt is None:
        return None

    emb = {sp: depth_embeddings(model, bl) for sp, bl in (("tr", tr), ("va", va), ("te", te))}
    Ztr = align(*emb["tr"][depth], tgt)
    Zva = align(*emb["va"][depth], tgt)
    Zte = align(*emb["te"][depth], tgt)
    if Ztr[0] is None or Zva[0] is None or Zte[0] is None:
        return None
    Xz_tr, y_tr = Ztr
    Xz_va, _ = Zva
    Xz_te, y_te = Zte
    # Median threshold from TRAIN only -- Phase 1's rule, so the Z head being re-fit here is
    # solving the identical problem Phase 1 scored.
    ytr_b, yte_b, thr = binarise(y_tr, y_te, tgt["kind"])

    # Rows carrying a downstream label, in the same (bundle, node) order the embeddings use.
    lab_va, lab_te = downstream_labels(va), downstream_labels(te)
    Xo_va, ids_va, t_va = observed_rows(va)
    Xo_te, ids_te, t_te = observed_rows(te)

    def keep_rows(ids, times, lab, tgt):
        """Indices that BOTH have a latent target (so a Z value exists for them) and a
        downstream label. Order matches `align`'s, which walks the same list."""
        zi, keep_z, keep_y, ys = 0, [], [], []
        for i, (sid, t) in enumerate(zip(ids, times)):
            row = tgt["values"].get(t)
            if row is None or sid not in row:
                continue                        # not a Z row; `align` skipped it too
            if (sid, t) in lab:
                keep_z.append(zi)
                keep_y.append(i)
                ys.append(lab[(sid, t)])
            zi += 1
        return np.array(keep_z, dtype=int), np.array(keep_y, dtype=int), np.asarray(ys, float)

    kz_va, ky_va, Y_va = keep_rows(ids_va, t_va, lab_va, tgt)
    kz_te, ky_te, Y_te = keep_rows(ids_te, t_te, lab_te, tgt)
    if len(kz_va) == 0 or len(kz_te) == 0 or Y_va.sum() == 0 or Y_te.sum() == 0:
        return None
    Xo_va_k, Xo_te_k = Xo_va[ky_va], Xo_te[ky_te]

    per_arm = {a: [] for a in ARMS}
    z_auc = []
    for s in init_seeds:
        got = fit_probs(Xz_tr, ytr_b, [Xz_va, Xz_te], s, model=model)
        if got is None:
            continue
        z_va_all, z_te_all = got
        z_auc.append(roc_auc(z_te_all, yte_b.astype(bool)))       # Model A's own number
        z_va, z_te = z_va_all[kz_va], z_te_all[kz_te]
        for arm in ARMS:
            Ftr, Fte = stack_arm(z_va, Xo_va_k, arm), stack_arm(z_te, Xo_te_k, arm)
            pr = fit_probs(Ftr, Y_va, [Fte], s, model=model)
            per_arm[arm].append(roc_auc(pr[0], Y_te.astype(bool)) if pr else None)

    z_auc = [a for a in z_auc if a is not None]
    out = {
        "depth": depth, "median_threshold": thr,
        "z_head_auc_mean": statistics.fmean(z_auc) if z_auc else None,
        "z_head_auc_all": z_auc,
        "n_va": int(len(Y_va)), "pos_va": int(Y_va.sum()),
        "n_te": int(len(Y_te)), "pos_te": int(Y_te.sum()),
    }
    for arm in ARMS:
        vals = [v for v in per_arm[arm] if v is not None]
        out[arm] = {"auc_mean": statistics.fmean(vals) if vals else None,
                    "auc_all": vals,
                    "init_seed_spread": (max(vals) - min(vals)) if len(vals) > 1 else None}
    if verify is not None:
        verify.append_check(state, depth, z_auc)
    return out


class ModelAVerifier:
    """Re-derives Phase 1's per-seed Z-head AUC from `fit_probs` and diffs it against the
    recorded `out/phase1/heads_{A,E}.json`. If `fit_probs` had drifted from `train_head` in
    any training decision, these would not agree to 1e-9."""

    def __init__(self, path: str, dseed: int):
        self.recorded = json.load(open(path))["per_seed"][str(dseed)]
        self.dseed = dseed
        self.checks = []

    def append_check(self, state, depth, aucs):
        """Compared PER INIT SEED, not on the mean -- a mean over a different number of init
        seeds would differ for reasons that have nothing to do with the head."""
        cell = self.recorded.get(state, {}).get(str(depth))
        if not cell:
            return
        want = cell["auc_all"]
        for i, got in enumerate(aucs):
            if i >= len(want):
                break
            self.checks.append({"state": state, "depth": depth, "seed": self.dseed,
                                "init_seed": i, "phase1_auc": want[i], "refit_auc": got,
                                "abs_diff": abs(got - want[i]),
                                "match": abs(got - want[i]) < 1e-9})


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run(variant: str, states: list[str], dseeds: list[int], config: str,
        init_seeds: list[int], csv_root: str, depths: dict, verify_path: str = "") -> dict:
    res = {"variant": variant, "config": config, "task": TASK, "dataset_seeds": dseeds,
           "init_seeds": init_seeds, "depths": depths, "per_seed": {}, "model_a_checks": []}

    for dseed in dseeds:
        csv_dir = os.path.join(csv_root, f"v{variant}_seed{dseed}")
        print(f"\n=== variant {variant} seed {dseed} ===", flush=True)
        t_start = time.time()
        model, _meta = get_backbone(csv_dir, variant, dseed, mseed=0, device="cpu")
        assert_backbone_frozen(model)
        tr, va, te, _ = load_world(csv_dir, "cpu")
        targets = latent_targets(variant, dseed, config,
                                 [b.t0 for b in tr] + [b.t0 for b in va] + [b.t0 for b in te])
        ver = ModelAVerifier(verify_path, dseed) if verify_path else None

        seed_out = {}
        for state in states:
            if state not in targets:
                print(f"  {state}: not present on variant {variant}", flush=True)
                continue
            cell = run_state(model, tr, va, te, targets, state, depths[state], init_seeds, ver)
            if cell is None:
                print(f"  {state}: no usable rows", flush=True)
                continue
            seed_out[state] = cell
            print(f"  {state:<22} h^{cell['depth']}  Zhead {cell['z_head_auc_mean']:.4f}   "
                  f"P(Y|Z) {cell['Z']['auc_mean']:.4f}   P(Y|X) {cell['X']['auc_mean']:.4f}   "
                  f"P(Y|X,Z) {cell['XZ']['auc_mean']:.4f}   "
                  f"impact pos {cell['pos_te']:,}/{cell['n_te']:,}", flush=True)
        res["per_seed"][str(dseed)] = seed_out
        if ver:
            res["model_a_checks"].extend(ver.checks)
        print(f"  ({time.time() - t_start:.0f}s)", flush=True)
    return res


def summarise(res: dict) -> dict:
    summary = {}
    states = sorted({s for d in res["per_seed"].values() for s in d})
    for state in states:
        cells = [res["per_seed"][str(d)][state] for d in res["dataset_seeds"]
                 if state in res["per_seed"].get(str(d), {})]
        if not cells:
            continue
        entry = {"n_dataset_seeds": len(cells), "depth": cells[0]["depth"],
                 "z_head_auc_mean": statistics.fmean([c["z_head_auc_mean"] for c in cells]),
                 "pos_te_total": sum(c["pos_te"] for c in cells),
                 "n_te_total": sum(c["n_te"] for c in cells),
                 "pos_va_total": sum(c["pos_va"] for c in cells),
                 "n_va_total": sum(c["n_va"] for c in cells)}
        for arm in ARMS:
            vals = [c[arm]["auc_mean"] for c in cells if c[arm]["auc_mean"] is not None]
            spreads = [c[arm]["init_seed_spread"] for c in cells
                       if c[arm]["init_seed_spread"] is not None]
            if not vals:
                continue
            init_floor = max(spreads) if spreads else float("nan")
            dseed_floor = (max(vals) - min(vals)) if len(vals) > 1 else float("nan")
            floor = max([f for f in (init_floor, dseed_floor) if f == f] or [float("nan")])
            mean = statistics.fmean(vals)
            entry[arm] = {
                "auc_mean": mean, "auc_per_dataset_seed": vals,
                "init_seed_floor": init_floor, "dataset_seed_floor": dseed_floor,
                "reproduction_floor": floor,
                "above_chance_by": mean - 0.5,
                "clears_floor": (mean - 0.5) > floor,
                "sign_consistent": all(v > 0.5 for v in vals) or all(v < 0.5 for v in vals),
            }
        if "Z" in entry and "XZ" in entry:
            gap = entry["XZ"]["auc_mean"] - entry["Z"]["auc_mean"]
            entry["gap_XZ_minus_Z"] = gap
            entry["gap_exceeds_Z_floor"] = gap > entry["Z"]["reproduction_floor"]
        # THE GATE, and only this: P(Y|Z) above chance by more than its own floor,
        # sign-consistent across dataset seeds. The XZ-Z gap is diagnostic and is NOT gated.
        entry["gate_pass"] = bool(entry.get("Z", {}).get("clears_floor")
                                  and entry.get("Z", {}).get("sign_consistent"))
        summary[state] = entry
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="A")
    ap.add_argument("--states", default="supply_stress")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--config", default="v1")
    ap.add_argument("--init-seeds", default="0,1,2,3,4")
    ap.add_argument("--csv-root", default=os.path.join(REPO, "db", "csv_v1scale"))
    ap.add_argument("--phase1", default="", help="out/phase1/heads_X.json, for --verify-model-a")
    ap.add_argument("--verify-model-a", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    dseeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    iseeds = [int(s) for s in a.init_seeds.split(",") if s.strip()]
    states = [s.strip() for s in a.states.split(",") if s.strip()]

    p1 = a.phase1 or os.path.join(REPO, "out", "phase1", f"heads_{a.variant}.json")
    depths = {}
    blob = json.load(open(p1))
    for st in states:
        if st in blob["summary"]:
            depths[st] = int(blob["summary"][st]["best_depth"])
    missing = [s for s in states if s not in depths]
    if missing:
        print(f"no Phase 1 best depth recorded for {missing} on variant {a.variant}")
        return 1

    res = run(a.variant, states, dseeds, a.config, iseeds, a.csv_root, depths,
              verify_path=(p1 if a.verify_model_a else ""))
    res["summary"] = summarise(res)

    print("\n" + "=" * 118)
    print(f"STEP 1 — downstream sufficiency, variant {a.variant}, Y = {TASK}, "
          f"{len(dseeds)} dataset seeds x {len(iseeds)} init seeds")
    print("=" * 118)
    hdr = (f"{'state':<22}{'arm':>6}{'AUC':>9}{'init fl':>9}{'dset fl':>9}{'floor':>9}"
           f"{'above .5':>10}{'clears':>8}{'sign':>6}")
    print(hdr); print("-" * len(hdr))
    for state, s in sorted(res["summary"].items()):
        for arm in ARMS:
            c = s.get(arm)
            if not c:
                continue
            print(f"{state:<22}{arm:>6}{c['auc_mean']:>9.4f}{c['init_seed_floor']:>9.4f}"
                  f"{c['dataset_seed_floor']:>9.4f}{c['reproduction_floor']:>9.4f}"
                  f"{c['above_chance_by']:>+10.4f}"
                  f"{('YES' if c['clears_floor'] else 'no'):>8}"
                  f"{('yes' if c['sign_consistent'] else 'NO'):>6}")
        print(f"{'':22}gap P(Y|X,Z) - P(Y|Z) = {s.get('gap_XZ_minus_Z', float('nan')):+.4f}"
              f"   (reported, not gated)   Z-head AUC {s['z_head_auc_mean']:.4f}"
              f"   impact pos {s['pos_te_total']:,}/{s['n_te_total']:,}")
        print(f"{'':22}GATE (P(Y|Z) clears its floor, sign-consistent): "
              f"{'PASS' if s['gate_pass'] else 'STOP'}")
        print("-" * len(hdr))

    if res["model_a_checks"]:
        bad = [c for c in res["model_a_checks"] if not c["match"]]
        print(f"\nModel A equivalence: {len(res['model_a_checks'])} checks, "
              f"{len(bad)} mismatches"
              + ("" if not bad else f"  WORST {max(c['abs_diff'] for c in bad):.3e}"))

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=str)
        print(f"written to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
