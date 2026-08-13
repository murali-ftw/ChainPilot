"""
Multi-label ground truth for §10's hypothesis ranking — one label vector per supplier *pair*.

Every category here is a mechanism `db/generate_dataset.py` already tracks. Nothing is an
invented taxonomy over unsupervised clusters, which is the failure mode the brief for this
section rules out first: an unsupervised category has no truth to be calibrated against, so
its "confidence" could never be checked, only asserted.

**Disclosure, in the same terms `reports/layer3_testing.md` §9.1 used.** These labels are read
privileged, out of the generator's namespace (`ml/extract_mechanism_state.py`), and are
**training-time and evaluation-time only**. They are never a model input — not to SHARE, and
not to the ranking head, whose feature vector is built in `ml/hypothesis_features.py` from
emitted CSV columns alone. Unlike Stage 1, no privileged value reaches a gradient of SHARE
itself; the ranking head is a separate model that sits on top of frozen observable data.

The categories, and why each one is what it is:

* **`shared_upstream`** — co-membership in a Mechanism B group of Type **A** or **B**. Type C
  is excluded on purpose: it is the decoy, structurally identical and downstream-inert, so
  labelling it positive would score a model for finding a thing that does not exist. V1's
  legacy `H_POLYMER` pool is folded in here, because it *is* the shared-hidden-upstream
  scenario (4 suppliers, 6 pairs — negligible, but mislabelling them "unknown" would be
  wrong). **This is the class §9.8 predicts cannot work**, and the report says so before
  the number is produced.

* **`regional_logistics`** — co-membership in one of the base world's shared event pools,
  `H_PORT` / `H_TRUCK` / `H_CUSTOMS`. §9.7.1 measured these at 50% population coverage and a
  363:1 pair-count advantage over real Type A/B pairs. Two of the three are *defined* by
  `country` and the third by the `sea` flag, so this class is close to observable and is
  expected to be the one that works.

* **`shared_sourcing`** — the two suppliers are co-parents on a component, i.e. an edge of the
  `component_suppliers` junction table, whose `COPARENT_COUPLING` term is a live causal
  bleed-through on Variants B and D (0.35). **This is a disclosed substitution for the brief's
  `supplier_switching` category**, which is untestable here: `CS_REWIRES` is empty on every
  B/D variant-seed (§10.1), so that class has zero positives by construction and training on
  it would be training on an empty label. `shared_sourcing` is the nearest mechanism of the
  same *kind* — a sourcing-graph relation between the two named suppliers — that does have
  positives. It is also fully **observable** (`component_suppliers.csv.gz` is emitted), which
  makes it a deliberate **positive control**: a class whose cause is directly readable should
  come back near-perfectly ranked and calibrated, and if it does not, the pipeline is broken
  rather than the world being uninformative. `ml/run_hypothesis_module.py --ablate-coparent`
  re-runs it with that feature withheld, which converts it into a genuine behavioural test.

* **`unknown`** — none of the above. A real category, not a leftover: `IDIO` gives 25% of
  suppliers an idiosyncratic outage at a uniformly random start, and two of those overlapping
  is exactly the coincidental co-degradation a reviewer needs the module to be willing to call
  coincidence. It is the complement of the other three by construction and is trained as its
  own head rather than derived, so its calibration can be read directly.

A pair may legitimately carry several labels at once (a Type A pair that also shares
`H_PORT`), so this is genuinely multi-label; `label_matrix` never forces a single class and
`composition` reports the overlap.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

# Order is fixed and load-bearing: it is the column order of every label matrix, score
# matrix and per-class table in §10.
CLASSES = ("shared_upstream", "regional_logistics", "shared_sourcing", "unknown")
MECHANISM_CLASSES = CLASSES[:-1]

# The brief's fourth category, retained by name so the report can state precisely what was
# dropped and why rather than quietly omitting it.
DROPPED_CLASSES = ("supplier_switching",)


def load_mechanism_state(path: str, csv_dir: str,
                         cross_check_dir: str | None = None) -> dict:
    """Read one variant-seed's privileged state, asserting it describes *these* CSVs.

    Two checks, both because the alternative is a silently mislabelled ground truth:

    1. The generator's supplier insertion order must equal `suppliers.csv.gz` row order. Every
       index in this module is a position in that list, so a mismatch would scramble every
       label while training happily -- the same class of bug as the snapshot-cache collision
       in §6 and the `k % 3` bug in Phase 0.
    2. When `cross_check_dir` is given, `hp_groups` must match the file §9 already used
       (`out/hidden_mid/`) group for group, so §10's ground truth is provably the same object
       §2 and §9 scored against and not a re-derivation that drifted.
    """
    state = json.load(open(path))
    emitted = pd.read_csv(os.path.join(csv_dir, "suppliers.csv.gz"),
                          usecols=["id"])["id"].tolist()
    if state["supplier_ids"] != emitted:
        raise ValueError(f"{path}: supplier order differs from {csv_dir}/suppliers.csv.gz "
                         f"({len(state['supplier_ids'])} vs {len(emitted)} ids)")
    if cross_check_dir:
        ref_path = os.path.join(cross_check_dir, f"{state['variant']}_{state['seed']}.json")
        if os.path.exists(ref_path):
            ref = [(g["type"], sorted(g["members"]))
                   for g in json.load(open(ref_path))["hp_groups"]]
            got = [(g["type"], sorted(g["members"])) for g in state["hp_groups"]]
            if sorted(ref) != sorted(got):
                raise ValueError(f"{path}: hp_groups disagree with {ref_path} -- §10's ground "
                                 f"truth is not the object §9 scored against")
    return state


def membership_vectors(state: dict) -> dict[str, np.ndarray]:
    """`{mechanism_name: int array of length N}` giving each supplier's group id, or -1.

    One integer per supplier per mechanism is enough because a supplier joins at most one
    Mechanism B group (`HP_OF` in the generator) and the base pools are disjoint by
    construction at these sizes; co-membership is then a cheap `a == b >= 0` test instead of
    an N^2 set intersection.
    """
    idx = {s: i for i, s in enumerate(state["supplier_ids"])}
    n = len(idx)
    out: dict[str, np.ndarray] = {}

    hp = np.full(n, -1, dtype=np.int32)
    hp_type = np.full(n, -1, dtype=np.int32)
    type_code = {"A": 0, "B": 1, "C": 2}
    for g in state["hp_groups"]:
        for m in g["members"]:
            if m in idx:
                hp[idx[m]] = g["index"]
                hp_type[idx[m]] = type_code[g["type"]]
    out["hp_group"] = hp
    out["hp_type"] = hp_type

    for j, (name, members) in enumerate(sorted(state["base_pools"].items())):
        v = np.full(n, -1, dtype=np.int32)
        for m in members:
            if m in idx:
                v[idx[m]] = j
        out[name] = v
    return out


def label_matrix(state: dict, pairs: np.ndarray) -> np.ndarray:
    """`[P, len(CLASSES)]` uint8 multi-label matrix for the given `[P, 2]` index pairs."""
    mv = membership_vectors(state)
    a, b = pairs[:, 0], pairs[:, 1]

    hp, hpt = mv["hp_group"], mv["hp_type"]
    same_group = (hp[a] == hp[b]) & (hp[a] >= 0)
    # Type C is the decoy and must never count as a positive; the generator makes A/B/C
    # structurally identical, so the only thing separating them is this line.
    upstream = same_group & np.isin(hpt[a], (0, 1))
    poly = mv["H_POLYMER"]
    upstream |= (poly[a] == poly[b]) & (poly[a] >= 0)

    regional = np.zeros(len(pairs), dtype=bool)
    for name in ("H_PORT", "H_TRUCK", "H_CUSTOMS"):
        v = mv[name]
        regional |= (v[a] == v[b]) & (v[a] >= 0)

    idx = {s: i for i, s in enumerate(state["supplier_ids"])}
    edges = set()
    for src, dsts in state["coparents"].items():
        if src not in idx:
            continue
        for dst in dsts:
            if dst in idx:
                edges.add((min(idx[src], idx[dst]), max(idx[src], idx[dst])))
    sourcing = np.fromiter(
        ((min(int(x), int(y)), max(int(x), int(y))) in edges for x, y in zip(a, b)),
        dtype=bool, count=len(pairs))

    y = np.zeros((len(pairs), len(CLASSES)), dtype=np.uint8)
    y[:, 0] = upstream
    y[:, 1] = regional
    y[:, 2] = sourcing
    y[:, 3] = ~(upstream | regional | sourcing)
    return y


def pair_subtypes(state: dict, pairs: np.ndarray) -> dict[str, np.ndarray]:
    """Diagnostic masks that the headline classes deliberately collapse.

    `shared_upstream` merges Type A and Type B co-membership, but they are not the same
    object: on **Variant B**, `HP_ALPHA` is 0, so a Type A group has *no downstream effect
    whatsoever* and is behaviourally indistinguishable from the Type C decoy. Only on Variant
    D (`HP_ALPHA = 0.35`) does Type A couple. Reporting `shared_upstream` without this split
    would let Variant B's inert Type A pairs drag down a number that Type B pairs might carry,
    and would hide which of the two is responsible.
    """
    mv = membership_vectors(state)
    a, b = pairs[:, 0], pairs[:, 1]
    hp, hpt = mv["hp_group"], mv["hp_type"]
    same = (hp[a] == hp[b]) & (hp[a] >= 0)
    return {
        "type_a_pair": same & (hpt[a] == 0),
        "type_b_pair": same & (hpt[a] == 1),
        "type_c_pair": same & (hpt[a] == 2),   # decoy: must stay at chance
    }


def truth_ceiling(state: dict, usable: np.ndarray) -> dict:
    """How many true pairs of each class could be evaluated **at all**, before any detector.

    A pair whose suppliers have no recorded `on_time_rate_90d` trajectory has no behavioural
    pattern to explain, so it can never enter a candidate set however the detector is tuned.
    That makes this the hard cap on every positive count in §10, and it is severe: at mid
    scale roughly 87% of true Type A/B pairs have at least one member with no usable history.
    Reporting a null on `shared_upstream` without this number attached would be reporting a
    null on a sample size the reader cannot see.
    """
    mv = membership_vectors(state)
    hp, hpt = mv["hp_group"], mv["hp_type"]
    out = {"n_usable_suppliers": int(usable.sum()), "n_suppliers": int(len(usable))}
    total = both = 0
    for gi in np.unique(hp[hp >= 0]):
        idxs = np.where(hp == gi)[0]
        if hpt[idxs[0]] not in (0, 1):
            continue
        m = len(idxs)
        total += m * (m - 1) // 2
        k = int(usable[idxs].sum())
        both += k * (k - 1) // 2
    out["shared_upstream_pairs_total"] = int(total)
    out["shared_upstream_pairs_both_usable"] = int(both)
    out["shared_upstream_evaluable_frac"] = float(both / total) if total else 0.0
    idx = {s: i for i, s in enumerate(state["supplier_ids"])}
    edges = {(min(idx[a], idx[b]), max(idx[a], idx[b]))
             for a, ds in state["coparents"].items() if a in idx
             for b in ds if b in idx}
    out["shared_sourcing_pairs_total"] = len(edges)
    out["shared_sourcing_pairs_both_usable"] = sum(
        1 for a, b in edges if usable[a] and usable[b])
    return out


def composition(y: np.ndarray, subtypes: dict[str, np.ndarray] | None = None) -> dict:
    """Label counts, per-class positives and the multi-label overlap matrix."""
    out = {
        "n_pairs": int(len(y)),
        "positives": {c: int(y[:, i].sum()) for i, c in enumerate(CLASSES)},
        "prevalence": {c: float(y[:, i].mean()) for i, c in enumerate(CLASSES)},
        "n_labels_per_pair": {str(k): int(v) for k, v in
                              zip(*np.unique(y[:, :3].sum(axis=1), return_counts=True))},
        "overlap": {f"{CLASSES[i]}&{CLASSES[j]}": int((y[:, i] & y[:, j]).sum())
                    for i in range(3) for j in range(i + 1, 3)},
    }
    if subtypes:
        out["subtypes"] = {k: int(v.sum()) for k, v in subtypes.items()}
    return out
