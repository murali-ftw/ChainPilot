#!/usr/bin/env python3
"""
HADES V3 Layer 4 — Structural Causal Model, extracted from the generator's own equations.

===============================================================================================
VALIDITY CAVEAT — READ BEFORE CITING ANY RESULT PRODUCED WITH THIS MODULE
===============================================================================================
This SCM is correct *because* HADES's benchmark world is synthetic and its structural equations
are known and coded in `db/generate_dataset.py`. The equations below were transcribed from that
source, not learned from data. That is what makes them exact, and it is also the entire limit of
the claim:

  * "validated on this benchmark via generator-extracted equations"  -- achievable, and what the
    accompanying report measures.
  * "ready for real industrial deployment"                            -- NOT established by that
    validation, and must not be asserted on its basis.

In any real deployment no such generator exists. The true causal structure would have to be
hand-specified by domain experts or learned by causal discovery from real operational data -- a
substantially harder problem this project has not attempted (`docs/HADES-v3_init.md` §7, §10;
`docs/14_Project_Roadmap.md` §3.2 step 3). Every report citing this module's correctness must
carry this caveat.
===============================================================================================

**What is extracted, and from where.** Every constant and every functional form below is cited to
a line in `db/generate_dataset.py`. Nothing is fitted.

    own_stress(s,t)  = min(0.95, (1-base_rel_s)*0.5 + SUM ramp(event, t) over the event
                       families s belongs to)                                          :827-857
    ramp(e,t)        = mag * clamp01((t-s0)/(pk-s0) if t<=pk else 1-(t-pk)/(s1-pk))     :839-856
    stress(s,t)      = min(0.95, own_stress(s,t)
                            + recv_atten(s) * COPARENT_COUPLING * SUM_partners own_stress
                            + recv_atten(s) * HP_ALPHA * mean_{co-members} own_stress
                            + upstream_stress(s,t))                                     :860-880
    upstream_stress  = SUM over chain hops of (prod of downstream attenuations)
                       * own_stress(up, t)                                              :634-650
    absorption(s)    = max(0, 1 - resilience_lambda * RESILIENCE[s])                     :536-547
    attenuation_of(s)= piecewise-linear in RESILIENCE through (0,atten_low),
                       (0.5,atten_medium), (1,atten_high)                                :560-566
    p_delay          = min(0.80, 0.025 + 0.38 * st * absorption(s))                      :1085
    st(port)         = st + 0.10 for a SEA carrier dispatched inside a PORT_EVENT window :1080-1082
    st(factory)      = st + 0.35 inside the FACTORY_OUTAGE window                        :1078
    sourcing_stress  = max over BOM-reachable suppliers of stress(s,t)                   :961-971

`mitigation_level` enters the world causally too, and is represented here for completeness even
though `reports/phase1_mitigation_level.md` found it observable rather than latent:

    replenish_trigger(t) = 1.55 + 0.45 * max_over_BOM_suppliers mitigation_level          :1221-1225
    replenish_qty        = thr * U(2.0,2.8) * (1 + 0.35 * mitigation_level(chosen sup))   :1233-1235

The stochastic draws (`random.random() < p_delay`, the lognormal lateness, `U(2.0,2.8)`) are the
generator's noise terms. This SCM reproduces the *structural* part -- the probability and the
mechanism -- not the individual Bernoulli outcomes, which are not identifiable from any model.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


@dataclass
class SCMParams:
    """Constants transcribed from `db/generate_dataset.py`. None are fitted."""
    stress_cap: float = 0.95            # :857, :880
    base_rel_coeff: float = 0.5         # :838   x = (1 - base_rel) * 0.5
    p_delay_intercept: float = 0.025    # :1085
    p_delay_slope: float = 0.38         # :1085
    p_delay_cap: float = 0.80           # :1085
    port_bump: float = 0.10             # :1082
    factory_bump: float = 0.35          # :1079
    default_stress: float = 0.08        # :971 sourcing_stress default
    trigger_base: float = 1.55          # :1221
    trigger_mit_coeff: float = 0.45     # :1225
    qty_mit_coeff: float = 0.35         # :1235
    # per-world, read off the resolved config / namespace
    resilience_lambda: float = 1.3      # CFG.resilience_lambda, used by absorption() :547
    coparent_coupling: float = 0.35     # COPARENT_COUPLING :408
    hp_alpha: float = 0.0               # HP_ALPHA :804 (0 unless Mechanism D)
    atten_low: float = 0.90
    atten_medium: float = 0.55
    atten_high: float = 0.15


@dataclass
class WorldStructure:
    """The world's structural facts the equations range over.

    These are the SCM's exogenous inputs -- the graph and the event calendar. They are read
    from the generator's namespace, which is legitimate: an SCM is a statement about structure,
    and structure is exactly what a deployment would have to supply from domain knowledge.
    What must NOT leak in is the latent *state* (stress, resilience), which is why those are
    passed separately at evaluation time.
    """
    base_rel: dict = field(default_factory=dict)        # sup_id -> base reliability
    events: list = field(default_factory=list)          # shared hidden-factor pools
    hp_b_events: list = field(default_factory=list)     # Mechanism B Type B
    shock_events: list = field(default_factory=list)    # Mechanism H
    idio: dict = field(default_factory=dict)            # sup_id -> (s0,pk,s1,mag) | None
    coparents: dict = field(default_factory=dict)       # sup_id -> [partner ids]
    hp_of: dict = field(default_factory=dict)           # sup_id -> group index
    hp_groups: list = field(default_factory=list)       # [{members, type}]
    sup_chain: dict = field(default_factory=dict)       # head -> [upstream ids]
    sup_atten: dict = field(default_factory=dict)       # sup_id -> per-hop coefficient
    f_on: bool = False                                  # Mechanism F active
    port_events: list = field(default_factory=list)
    factory_outage: tuple | None = None


class SupplyChainSCM:
    """The extracted structural causal model.

    Deliberately a standalone re-implementation: it never calls into
    `db/generate_dataset.py`'s functions. That is the point -- a wrapper would validate
    nothing, since it would agree with the generator by construction.
    """

    def __init__(self, params: SCMParams, structure: WorldStructure):
        self.p, self.w = params, structure

    # -- event ramp -------------------------------------------------------------------
    @staticmethod
    def ramp(t, s0, pk, s1, mag) -> float:
        """`db/generate_dataset.py:841-842` -- linear rise to peak, linear decay after."""
        if not (s0 <= t <= s1):
            return 0.0
        frac = (t - s0) / (pk - s0) if t <= pk else 1 - (t - pk) / (s1 - pk)
        return mag * clamp01(frac)

    # -- own_stress -------------------------------------------------------------------
    def own_stress(self, sup_id: str, t) -> float:
        """`:827-857`. Summation order matches the generator's: EVENTS, HP_B_EVENTS,
        SHOCK_EVENTS, then IDIO -- float addition is not associative, and exact agreement
        is the fidelity test."""
        w, p = self.w, self.p
        x = (1 - w.base_rel[sup_id]) * p.base_rel_coeff
        for members, s0, pk, s1, mag in w.events:
            if sup_id in members:
                x += self.ramp(t, s0, pk, s1, mag)
        for members, s0, pk, s1, mag in w.hp_b_events:
            if sup_id in members:
                x += self.ramp(t, s0, pk, s1, mag)
        for members, s0, pk, s1, mag in w.shock_events:
            if sup_id in members:
                x += self.ramp(t, s0, pk, s1, mag)
        ev = w.idio.get(sup_id)
        if ev:
            s0, pk, s1, mag = ev
            x += self.ramp(t, s0, pk, s1, mag)
        return min(p.stress_cap, x)

    # -- transmission -----------------------------------------------------------------
    def recv_atten(self, sup_id: str) -> float:
        """`:585-592`. Attenuation on an inbound edge is the DOWNSTREAM node's coefficient."""
        return self.w.sup_atten.get(sup_id, 1.0) if self.w.f_on else 1.0

    def hp_coupling(self, sup_id: str, t) -> float:
        """`:812-824`. Type A groups only; reads co-members' OWN stress, never coupled."""
        gi = self.w.hp_of.get(sup_id)
        if gi is None or not self.p.hp_alpha:
            return 0.0
        grp = self.w.hp_groups[gi]
        if grp.get("type") != "A":
            return 0.0
        peers = [m for m in grp["members"] if m != sup_id]
        if not peers:
            return 0.0
        return self.p.hp_alpha * sum(self.own_stress(m, t) for m in peers) / len(peers)

    def upstream_stress(self, sup_id: str, t) -> float:
        """`:634-650`. Bounded walk up the hidden chain, attenuated once per hop."""
        chain = self.w.sup_chain.get(sup_id)
        if not chain:
            return 0.0
        total, carry, downstream = 0.0, 1.0, sup_id
        for up in chain:
            carry *= self.w.sup_atten[downstream]
            total += carry * self.own_stress(up, t)
            downstream = up
        return total

    def stress(self, sup_id: str, t) -> float:
        """`:860-880`. The causal driver of every downstream outcome."""
        x = self.own_stress(sup_id, t)
        rx = self.recv_atten(sup_id)
        for partner in self.w.coparents.get(sup_id, ()):
            x += rx * self.p.coparent_coupling * self.own_stress(partner, t)
        x += rx * self.hp_coupling(sup_id, t)
        x += self.upstream_stress(sup_id, t)
        return min(self.p.stress_cap, x)

    # -- stress -> outcome ------------------------------------------------------------
    def absorption(self, sup_id: str, resilience: dict | None) -> float:
        """`:536-547`. 1.0 when Mechanism E is off."""
        if not resilience:
            return 1.0
        return max(0.0, 1.0 - self.p.resilience_lambda * resilience[sup_id])

    def p_delay(self, stress_value: float, sup_id: str | None = None,
                resilience: dict | None = None, sea_carrier: bool = False,
                dispatch=None, factory_outage_hit: bool = False) -> float:
        """`:1078-1085`. The stress -> delay conversion, including the two direct
        stress bumps the generator applies at shipment construction.

        Takes `stress_value` as an ARGUMENT rather than computing it, so the same equation
        can be driven by a true generator state or by a Layer 3 estimate -- which is exactly
        the comparison `docs/14_Project_Roadmap.md` §3.2 step 3 asks for.
        """
        p, st = self.p, stress_value
        if factory_outage_hit:
            st = min(p.stress_cap, st + p.factory_bump)
        if sea_carrier and dispatch is not None:
            for _m, s0, _pk, s1, _mag in self.w.port_events:
                if s0 <= dispatch <= s1:
                    st = min(p.stress_cap, st + p.port_bump)
        absorb = self.absorption(sup_id, resilience) if sup_id else 1.0
        return min(p.p_delay_cap, p.p_delay_intercept + p.p_delay_slope * st * absorb)

    def sourcing_stress(self, sup_ids, t) -> float:
        """`:961-971`. A product is only as good as its worst BOM supplier."""
        vals = [self.stress(s, t) for s in sup_ids]
        return max(vals) if vals else self.p.default_stress

    # -- mitigation's causal role (represented; see module docstring) -----------------
    def replenish_trigger(self, mitigation: float) -> float:
        """`:1221-1225`."""
        return self.p.trigger_base + self.p.trigger_mit_coeff * mitigation

    def replenish_qty_multiplier(self, mitigation: float) -> float:
        """`:1235`."""
        return 1.0 + self.p.qty_mit_coeff * mitigation


def from_namespace(ns: dict) -> tuple[SCMParams, WorldStructure]:
    """Build the SCM's parameters and structure from a generator namespace.

    Reads only STRUCTURE (graph, event calendar, coefficients) -- never latent state, which
    callers pass explicitly so true-vs-estimated can be compared.
    """
    cfg = ns["CFG"]
    params = SCMParams(
        resilience_lambda=cfg.resilience_lambda,
        coparent_coupling=ns.get("COPARENT_COUPLING", 0.0),
        hp_alpha=ns.get("HP_ALPHA", 0.0),
        atten_low=cfg.atten_low, atten_medium=cfg.atten_medium, atten_high=cfg.atten_high,
    )
    structure = WorldStructure(
        base_rel={s["id"]: s["base_rel"] for s in ns["suppliers"]},
        events=list(ns.get("EVENTS", [])),
        hp_b_events=list(ns.get("HP_B_EVENTS", [])),
        shock_events=list(ns.get("SHOCK_EVENTS", [])),
        idio=dict(ns.get("IDIO", {})),
        coparents={k: list(v) for k, v in ns.get("coparents", {}).items()},
        hp_of=dict(ns.get("HP_OF", {})),
        hp_groups=[{"members": list(g["members"]), "type": g["type"]}
                   for g in ns.get("HP_GROUPS", [])],
        sup_chain={k: list(v) for k, v in ns.get("SUP_CHAIN", {}).items()},
        sup_atten=dict(ns.get("SUP_ATTEN", {})),
        f_on=bool(ns.get("F_ON", False)),
        port_events=list(ns.get("PORT_EVENTS", [])),
        factory_outage=ns.get("FACTORY_OUTAGE"),
    )
    return params, structure
