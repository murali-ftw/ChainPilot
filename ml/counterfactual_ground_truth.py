#!/usr/bin/env python3
"""
Interventional ground truth from the generator's own causal model.

`db/generate_dataset.py` is a deterministic simulator whose causal core is explicit:

    own_stress(s,t) = base + shared-factor events + idiosyncratic outage
    stress(s,t)     = own_stress(s,t)
                      + recv_atten(s) * COPARENT_COUPLING * SUM_{p in coparents[s]} own_stress(p,t)
                      + recv_atten(s) * hp_coupling(s,t)          [Mechanism D]
                      + upstream_stress(s,t)                      [Mechanism J]
    p_delay(sh)     = min(0.80, 0.025 + 0.38 * stress(sup, base_rel, dispatch) * absorption(sup))

and the **impact** label for supplier `s` at `t0` is exactly "at least one of `s`'s in-flight
shipments transitions to `delayed` inside `(t0, t0+horizon]`". So the true impact risk is
available in closed form, with no Monte Carlo noise at all:

    p_impact(s,t0) = 1 - PROD_{sh in InFlight(s,t0), eta(sh) in horizon} ( 1 - p_delay(sh) )

**On Variant 0 the co-parent graph is the only structural lever on stress**, and it is live:
`MECHS = ()` zeroes `hp_coupling`, `upstream_stress`, and sets `recv_atten = absorption = 1`,
leaving `stress = own_stress + 0.35 * SUM own_stress(coparents)`. All three interventions this
module supports act through that graph, which is what makes them causally real rather than
cosmetic.

---

**Why the ground truth is computed this way rather than by re-simulating -- measured, not
assumed.**

The obvious approach is to re-run the generator with the intervention applied and diff the
emitted labels. The simulation consumes draws from one shared `random` stream *sequentially*:

    delayed = random.random() < p_delay
    late_by = lognormvariate(...) if delayed else timedelta(0)      # branch-dependent draw
    actual  = eta + late_by      if delayed else eta - randint(...) # branch-dependent draw

so the moment an intervention flips one shipment's outcome, draw counts diverge and every
later shipment gets different randomness. `--demo-rng-divergence` quantifies the consequence by
inserting **one extra draw** -- the smallest possible perturbation, standing in for what any
intervention does -- and diffing the emitted labels. Measured on Variant 0 / v1 preset / seed 42:

| quantity | effect of ONE extra random draw |
|---|---|
| shipments differing in status or eta | **100.0%** (39,184 of 39,184) |
| `delay` labels flipped | **797 of 5,527 = 14.4%**, i.e. **179% of the 444 true positives** |
| `shortage` labels flipped | 0 |
| `impact` labels flipped | 0 |

So the answer is task-dependent, and worth stating precisely rather than sweeping into a
blanket claim. For **delay**, re-simulation is unusable: a single draw's worth of divergence
flips nearly twice as many labels as there are positives, so any intervention's realised effect
would be buried. For **impact** and **shortage** the realised labels are *robust* to complete
schedule divergence, because both are coarse aggregates -- impact is "at least one in-flight
shipment is delayed in the horizon", which stays true under reshuffling for any supplier with
several shipments.

This module still uses the closed-form route for all three, for two reasons that survive that
finding. It is **exact** rather than robust-by-saturation, and it yields a **continuous**
probability delta instead of a binary label flip -- which matters enormously here, because the
true effects are small (a few points of risk on a handful of suppliers) and a binary target at
that effect size would have almost no statistical power. Holding the realised schedule fixed
(same shipments, same dispatch times, same eta) and recomputing only the causal quantity the
intervention changes is common random numbers in its exact form.

**The estimand this defines, stated plainly.** It is the effect of the structural change on
delay risk *holding the replenishment schedule fixed* -- a partial effect through the stress
channel. It excludes the second-order path where a structural change alters which supplier is
chosen for future replenishment (`random.choice(prod_bom_sup[...])`) and hence the schedule
itself. For `add_dual_source` that second path is genuinely absent from the first-order
question being asked; for `substitute_supplier` and `remove_supplier` this module reassigns
shipment ownership explicitly, which captures the dominant part of it. The residual is a
stated limitation, not a hidden one.

    python3 ml/counterfactual_ground_truth.py --variant 0 --seed 42 --config v1 \
        --interventions 40 --out out/cf_truth/v0_seed42.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import sys
import types

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(REPO, "db", "generate_dataset.py")

INTERVENTIONS = ("add_dual_source", "remove_supplier", "substitute_supplier")


# --------------------------------------------------------------------------- generator access

def generator_namespace(variant: str, seed: int, config: str) -> dict:
    """Execute the generator in-process with `write()` stubbed; return its namespace.

    Same mechanism as `ml/extract_hidden_state.py` and `ml/extract_mechanism_state.py`, with
    the same hard-won caution: a `SystemExit` raised by `resolve_config` happens *before* the
    world is built and must not be swallowed, while one raised by the closing validation suite
    happens *after* and leaves the namespace perfectly readable.
    """
    src = open(GEN).read()
    src = src.replace("def write(name, header, rows):",
                      "def write(name, header, rows):\n    return", 1)
    src = src.replace(
        'with open(os.path.join(OUT, "resolved_config.json"), "w") as _f:\n'
        '    json.dump(_manifest, _f, indent=2, sort_keys=True)',
        "MANIFEST_OUT = _manifest", 1)
    argv = ["generate_dataset.py", "--variant", variant, "--config", config, "--seed", str(seed)]
    saved, sys.argv = sys.argv, argv
    mod = types.ModuleType("__cf_truth__")
    mod.__file__ = GEN
    sys.modules["__cf_truth__"] = mod
    ns = mod.__dict__
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                exec(compile(src, GEN, "exec"), ns)
            except SystemExit:
                pass
    finally:
        sys.argv = saved
    if not ns.get("suppliers"):
        raise RuntimeError(f"generator produced no suppliers for variant {variant} seed {seed}")
    return ns


# --------------------------------------------------------------------------- causal recomputation

class CausalWorld:
    """The generator's causal model, re-evaluable under a modified co-parent graph.

    Holds the realised schedule (shipments, dispatch times, eta) fixed and recomputes
    `stress -> p_delay -> p_impact` under whatever structure it is handed.
    """

    def __init__(self, ns: dict):
        self.ns = ns
        self.suppliers = ns["suppliers"]
        self.sup_by_id = ns["sup_by_id"]
        self.shipments = ns["shipments"]
        self.T0S = list(ns["T0S"])
        self.horizon_days = int(ns["HORIZON"])
        self.coparent_coupling = float(ns.get("COPARENT_COUPLING", 0.0))
        self.own_stress = ns["own_stress"]
        self.recv_atten = ns["recv_atten"]
        self.absorption = ns["absorption"]
        self.hp_coupling = ns["hp_coupling"]
        self.upstream_stress = ns["upstream_stress"]
        self.visible = set(ns.get("VISIBLE_SUP", set()) or {s["id"] for s in self.suppliers})
        self.base_coparents = {k: set(v) for k, v in (ns.get("coparents", {}) or {}).items()}
        # Realised, in-flight-at-t0 shipment sets, computed once from the FACTUAL world and
        # then held fixed -- this is the common-random-numbers pin.
        self._trans_by_sid = {}
        for (sid, stat, prev, at, rec) in ns["transitions"]:
            self._trans_by_sid.setdefault(sid, []).append((stat, at, rec))
        self._asof = ns["asof_status"]
        self._inflight = self._build_inflight()

    def _build_inflight(self) -> dict:
        """`{t0_index: {supplier_id: [shipment, ...]}}` -- shipments that are in flight as of
        `t0` and whose delay transition, were it to occur, would land inside the horizon.

        Mirrors the generator's own impact-label construction exactly (`sup_hit`), including
        that it reads the recorded `asof_status` rather than true status.
        """
        from datetime import timedelta
        out = {}
        for ti, t0 in enumerate(self.T0S):
            hz = t0 + timedelta(days=self.horizon_days)
            per_sup = {}
            for sh in self.shipments:
                sup = sh.get("supplier_id")
                if not sup or sh["created_at"] > t0:
                    continue
                if self._asof(sh["id"], t0) not in ("scheduled", "in_transit"):
                    continue
                # The generator emits the `delayed` transition at eta + 6h; only shipments
                # whose eta+6h falls inside the horizon can produce a positive.
                if not (t0 < sh["eta"] + timedelta(hours=6) <= hz):
                    continue
                per_sup.setdefault(sup, []).append(sh)
            out[ti] = per_sup
        return out

    def stress(self, sup_id: str, t, coparents: dict) -> float:
        """The generator's `stress()`, with the co-parent graph supplied rather than global."""
        base_rel = self.sup_by_id[sup_id]["base_rel"]
        x = self.own_stress(sup_id, base_rel, t)
        rx = self.recv_atten(sup_id)
        for p in coparents.get(sup_id, ()):
            x += rx * self.coparent_coupling * self.own_stress(
                p, self.sup_by_id[p]["base_rel"], t)
        x += rx * self.hp_coupling(sup_id, t)
        x += self.upstream_stress(sup_id, t)
        return min(0.95, x)

    def p_delay(self, sh: dict, sup_id: str, coparents: dict) -> float:
        st = self.stress(sup_id, sh["dispatched_at"], coparents)
        return min(0.80, 0.025 + 0.38 * st * self.absorption(sup_id))

    def p_impact(self, coparents: dict, owner: dict | None = None) -> dict:
        """`{(supplier_id, t0_index): probability}` under the given structure.

        `owner` optionally remaps `shipment_id -> supplier_id`, which is how
        `substitute_supplier` and `remove_supplier` express a change of who ships.
        """
        out = {}
        for ti, per_sup in self._inflight.items():
            # Re-bucket by (possibly remapped) owner.
            buckets = {}
            for sup, ships in per_sup.items():
                for sh in ships:
                    o = (owner or {}).get(sh["id"], sup)
                    if o is None:
                        continue                      # shipment removed with its supplier
                    buckets.setdefault(o, []).append(sh)
            for sup, ships in buckets.items():
                if sup not in self.visible:
                    continue
                q = 1.0
                for sh in ships:
                    q *= (1.0 - self.p_delay(sh, sup, coparents))
                out[(sup, ti)] = 1.0 - q
        return out


# --------------------------------------------------------------------------- interventions

def apply_intervention(world: CausalWorld, kind: str, spec: dict
                       ) -> tuple[dict, dict, set]:
    """Return `(coparents_after, owner_remap, touched_suppliers)`.

    `touched_suppliers` is the set the intervention *structurally* acts on, used to separate
    directly-affected from bystander entities in the evaluation -- the same reachability
    partition discipline the retrieval work used, and necessary here because a large majority
    of suppliers are untouched and would otherwise dominate any pooled metric.
    """
    cop = {k: set(v) for k, v in world.base_coparents.items()}
    owner: dict = {}
    touched: set = set()

    if kind == "add_dual_source":
        # A new secondary source for a component: the incumbent primary and the new
        # secondary become co-parents, so each picks up 0.35x the other's own stress.
        a, b = spec["primary"], spec["secondary"]
        cop.setdefault(a, set()).add(b)
        cop.setdefault(b, set()).add(a)
        touched = {a, b}

    elif kind == "remove_supplier":
        s = spec["supplier"]
        for p in cop.get(s, set()):
            cop.get(p, set()).discard(s)
            touched.add(p)
        cop.pop(s, None)
        for sh in world.shipments:
            if sh.get("supplier_id") == s:
                owner[sh["id"]] = None            # its shipments cease to exist
        touched.add(s)

    elif kind == "substitute_supplier":
        old, new = spec["old"], spec["new"]
        # The replacement inherits the incumbent's co-parent relationships.
        partners = set(cop.get(old, set()))
        for p in partners:
            cop.get(p, set()).discard(old)
            cop.get(p, set()).add(new)
            cop.setdefault(new, set()).add(p)
            touched.add(p)
        cop.pop(old, None)
        for sh in world.shipments:
            if sh.get("supplier_id") == old:
                owner[sh["id"]] = new
        touched |= {old, new}

    else:
        raise ValueError(f"unknown intervention: {kind}")

    return cop, owner, touched


def sample_interventions(world: CausalWorld, n_per_kind: int, rng_seed: int) -> list[dict]:
    """Sample interventions, biased toward suppliers that actually ship.

    A uniform draw over all suppliers would mostly pick entities with no in-flight shipments
    in any horizon, for which the true effect is exactly zero by construction and which would
    pad the corpus with uninformative nulls.
    """
    rng = random.Random(rng_seed)
    active = sorted({sh["supplier_id"] for sh in world.shipments if sh.get("supplier_id")}
                    & world.visible)
    coparented = sorted(s for s in active if world.base_coparents.get(s))
    out = []
    for _ in range(n_per_kind):
        if len(active) >= 2:
            a, b = rng.sample(active, 2)
            out.append({"kind": "add_dual_source", "primary": a, "secondary": b})
    for _ in range(n_per_kind):
        pool = coparented or active
        if pool:
            out.append({"kind": "remove_supplier", "supplier": rng.choice(pool)})
    for _ in range(n_per_kind):
        if len(active) >= 2:
            old, new = rng.sample(active, 2)
            out.append({"kind": "substitute_supplier", "old": old, "new": new})
    return out


# --------------------------------------------------------------------------- RNG divergence demo

def demo_rng_divergence(variant: str, seed: int, config: str) -> dict:
    """Measure, rather than assert, why re-simulation cannot supply this ground truth.

    Re-runs the generator twice with the identical seed but one extra draw consumed from the
    shared stream before the simulation -- the minimum possible perturbation, standing in for
    what any real intervention does -- and counts how many suppliers' realised impact labels
    change. If a one-draw shift moves a large fraction of the world, then a structural
    intervention's realised diff is dominated by divergence rather than by the intervention.
    """
    src = open(GEN).read()
    src = src.replace("def write(name, header, rows):",
                      "def write(name, header, rows):\n    return", 1)
    src = src.replace(
        'with open(os.path.join(OUT, "resolved_config.json"), "w") as _f:\n'
        '    json.dump(_manifest, _f, indent=2, sort_keys=True)',
        "MANIFEST_OUT = _manifest", 1)

    def run(perturb: bool) -> set:
        s2 = src
        if perturb:
            # One extra draw immediately before the shipment simulation begins.
            s2 = s2.replace("agent_pending = {}     # sup_id ->",
                            "random.random()\nagent_pending = {}     # sup_id ->", 1)
        argv = ["generate_dataset.py", "--variant", variant, "--config", config,
                "--seed", str(seed)]
        saved, sys.argv = sys.argv, argv
        mod = types.ModuleType("__cf_rngdemo__")
        mod.__file__ = GEN
        sys.modules["__cf_rngdemo__"] = mod
        ns = mod.__dict__
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                try:
                    exec(compile(s2, GEN, "exec"), ns)
                except SystemExit:
                    pass
        finally:
            sys.argv = saved
        return {(r[3], r[1]) for r in ns["label_rows"] if r[4] == "impact" and r[5] == "true"}

    base = run(False)
    pert = run(True)
    changed = base.symmetric_difference(pert)
    return {"variant": variant, "seed": seed,
            "impact_positives_base": len(base), "impact_positives_perturbed": len(pert),
            "labels_changed": len(changed),
            "changed_fraction_of_positives": (len(changed) / max(1, len(base)))}


# --------------------------------------------------------------------------- driver

def build_truth(variant: str, seed: int, config: str, n_per_kind: int,
                sample_seed: int) -> dict:
    ns = generator_namespace(variant, seed, config)
    world = CausalWorld(ns)
    base_cop = world.base_coparents
    p0 = world.p_impact(base_cop)

    specs = sample_interventions(world, n_per_kind, sample_seed)
    records = []
    for spec in specs:
        cop, owner, touched = apply_intervention(world, spec["kind"], spec)
        p1 = world.p_impact(cop, owner)
        deltas = {}
        for key, v0 in p0.items():
            v1 = p1.get(key)
            if v1 is None:
                continue                       # entity ceased to exist under the intervention
            d = v1 - v0
            if d != 0.0:
                deltas[f"{key[0]}|{key[1]}"] = d
        records.append({"spec": spec, "touched": sorted(touched),
                        "n_nonzero": len(deltas), "deltas": deltas})
    return {
        "variant": variant, "seed": seed, "config": config,
        "coparent_coupling": world.coparent_coupling,
        "n_suppliers": len(world.suppliers), "n_visible": len(world.visible),
        "n_shipments": len(world.shipments), "n_t0": len(world.T0S),
        "horizon_days": world.horizon_days,
        "n_coparent_edges": sum(len(v) for v in base_cop.values()) // 2,
        "baseline_p_impact": {f"{k[0]}|{k[1]}": v for k, v in p0.items()},
        "interventions": records,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default="v1")
    ap.add_argument("--interventions", type=int, default=40,
                    help="how many of EACH of the three kinds")
    ap.add_argument("--sample-seed", type=int, default=20260813)
    ap.add_argument("--demo-rng-divergence", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    if a.demo_rng_divergence:
        d = demo_rng_divergence(a.variant, a.seed, a.config)
        print(json.dumps(d, indent=1))
        return 0

    blob = build_truth(a.variant, a.seed, a.config, a.interventions, a.sample_seed)
    nz = [r["n_nonzero"] for r in blob["interventions"]]
    print(f"v{a.variant} s{a.seed}: {len(blob['interventions'])} interventions, "
          f"{blob['n_coparent_edges']} coparent edges, "
          f"nonzero-effect entities per intervention: "
          f"min={min(nz)} median={int(np.median(nz))} max={max(nz)}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(blob, fh)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
