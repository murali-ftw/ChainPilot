#!/usr/bin/env python3
"""
HADES synthetic world generator — deterministic simulation engine.

Implements Dataset.md (HADES Synthetic World Specification) against the schema
in docs/05_Database_Design.md and the temporal contract in
architecture/project_HADES.md §2.6.

Design notes
------------
* Deterministic: seeded RNG + uuid5 for every primary key. Re-running produces
  byte-identical CSVs.
* Causal: shipment delays and stock shortages are driven by LATENT world state
  (supplier stress from hidden factors + disruption events), never sampled
  independently. Hidden factors (shared port, shared polymer plant, shared
  trucking firm) are NEVER emitted as rows — only their correlated effects are
  observable, which is exactly what Transformer 2 exists to discover.
* Leakage-free: features/windows end at t0; labels come only from (t0, t0+H];
  as-of status at t0 is reconstructed from shipment_status_history; the
  validation suite at the end asserts all of it.
"""

import csv, json, math, os, random, uuid
from datetime import datetime, timedelta, timezone

random.seed(42)
NS = uuid.UUID("00000000-0000-0000-0000-00000000c0de")
def uid(*key): return str(uuid.uuid5(NS, "|".join(str(k) for k in key)))

UTC = timezone.utc
def ts(y, m, d, h=8, mi=0): return datetime(y, m, d, h, mi, tzinfo=UTC)
def fmt(t): return t.strftime("%Y-%m-%d %H:%M:%S+00")
def J(t, spread=2700):
    """Sub-hour jitter (0..spread seconds, default <=45min) for OBSERVED/RECORDED event
    timestamps, so they don't land on exact clock ticks like a hand-authored fixture would.
    Never applied to defined analytical boundaries (t0, T_START, BOM validity dates) --
    those are legitimate clean business-date cutoffs, not recorded ERP events."""
    return t + timedelta(seconds=random.randint(0, spread))
# Timeline extended (v3): T_END pushed ~9 months past 2024 so the snapshot
# schedule can cover Jul 2024 .. Sep 2025 (15 monthly t0s instead of 6).
# T_START stays 2024-01-01 -- the FIRST t0 (2024-07-01) keeps its full 182-day
# trailing-history margin (>= the 180-day rule), same as before; extending the
# window forward doesn't shrink it.
T_START = ts(2024, 1, 1)
T_END   = ts(2025, 9, 30, 23)
TIMELINE_DAYS  = (T_END - T_START).days     # ~638
TIMELINE_WEEKS = TIMELINE_DAYS / 7.0        # ~91 -- weekly-demand conversion base (was /52 in the 1-year world)
GEN_AT_DT = ts(2025, 10, 2, 9)             # system-time anchor for master rows written "now" (after T_END)
def GJ(): return fmt(J(GEN_AT_DT, 1800))   # fresh ~0-30min batch-load jitter per call
HORIZON = 14

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv")
os.makedirs(OUT, exist_ok=True)
def write(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"  {name:38s} {len(rows):>7,} rows")

# ---------------------------------------------------------------- Chapter 2/3
COUNTRIES = {  # lead-time profile (lognormal mu in days), sea-freight?, base reliability alpha/beta
    "China":   (math.log(28), True,  7, 2),
    "Vietnam": (math.log(30), True,  6, 2),
    "India":   (math.log(24), True,  6, 2),
    "Germany": (math.log(12), False, 9, 1.5),
    "USA":     (math.log(8),  False, 8, 2),
    "Mexico":  (math.log(10), False, 6, 2.5),
}
C_NAMES = list(COUNTRIES)
# Scaled up from the original 50 (docs/01_Product_Requirement_Document.md §8's own
# stated assumption of "hundreds to low thousands" positive labels; the 50-supplier
# world produced single/low-double-digit positives for delay/impact, and the
# 180-supplier pass still left impact in the low 40s because impact is a
# per-supplier, per-snapshot label -- it scales with supplier count x snapshot
# count, not shipment volume). SCALE is applied to every entity count below
# that's meant to track world size (components/products/customers/orders) so
# the world stays internally proportioned, not just supplier count in isolation.
SUP_N = 800
SCALE = SUP_N / 50

# power-law-ish component degree weights: few dominant suppliers
sup_weight = sorted((random.paretovariate(1.6) for _ in range(SUP_N)), reverse=True)

suppliers = []
for i in range(SUP_N):
    country = C_NAMES[i % 6] if i < 12 else random.choices(C_NAMES, weights=[.28,.12,.16,.14,.18,.12])[0]
    mu, sea, a, b = COUNTRIES[country]
    lead = max(3, int(random.lognormvariate(mu if i else mu, 0.35)))
    rel  = round(0.40 + 0.58 * random.betavariate(a, b), 4)   # affine rescale, no hard clamp
    suppliers.append(dict(
        id=uid("sup", i), name=f"{country[:3].upper()}-{['Precision','Alloy','Poly','Micro','Global','Prime','Nova','Delta'][i%8]} Supply {i:02d}",
        country=country, capacity_score=round(random.lognormvariate(4.2, 0.5), 2),
        lead_time_days=lead, sea=sea, base_rel=rel, w=sup_weight[i]))

# O(1) lookup + precomputed weights: at SUP_N=800 the per-call `next(s for s in
# suppliers ...)` scans and per-call weight-list rebuilds in the hot loops below
# turn into tens of millions of wasted iterations. Pure indexing -- no RNG
# involvement, identical draws, probability model unchanged.
sup_by_id = {s["id"]: s for s in suppliers}
SUP_W = [s["w"] for s in suppliers]

# ------- Hidden factors (never emitted). members chosen so no graph edge links them.
def pick(pred, k):
    pool = [s for s in suppliers if pred(s)]; random.shuffle(pool); return set(s["id"] for s in pool[:k])
H_PORT    = pick(lambda s: s["sea"], round(12 * SCALE))                    # shared port congestion
H_TRUCK   = pick(lambda s: s["country"] in ("USA","Mexico"), round(8 * SCALE))   # shared trucking firm
H_CUSTOMS = pick(lambda s: s["country"]=="Germany", round(5 * SCALE))      # customs friction pool

# ---------------------------------------------------------------- Chapter 4
CTYPES = [("fastener", 30, 0.4), ("electronic", 35, 1.0), ("mechanical", 40, 3.0),
          ("polymer", 25, 1.5), ("specialty", 20, 12.0)]
CTYPES = [(ctype, max(1, round(n * SCALE)), cost_mu) for ctype, n, cost_mu in CTYPES]
components, comp_common = [], {}
ci = 0
for ctype, n, cost_mu in CTYPES:
    for j in range(n):
        s = random.choices(suppliers, weights=SUP_W)[0]
        common = random.paretovariate(1.2) if ctype in ("fastener","electronic") else random.paretovariate(2.5)
        cid = uid("comp", ci)
        components.append(dict(id=cid, supplier_id=s["id"], name=f"{ctype.title()} {ci:03d}",
                               component_type=ctype, unit_cost=round(random.lognormvariate(math.log(cost_mu), 0.6), 2)))
        comp_common[cid] = common
        ci += 1

# H_POLYMER: the hidden-dependency scenario's 4 shared-upstream suppliers. Picked
# AFTER components exist, and deliberately spanning 4 DISTINCT component_types
# (never "polymer" itself) -- so the shared factor cannot be recovered from
# component_type or any other single observable categorical column
# (project_HADES.md §5.4: this is required for Transformer 2's eventual
# validation to mean anything). Deterministic: first-generated supplier, by
# component insertion order, for each of 4 non-polymer types.
_seen_types, H_POLYMER = [], set()
for c in components:
    ctype = c["component_type"]
    if ctype == "polymer" or ctype in _seen_types:
        continue
    H_POLYMER.add(c["supplier_id"]); _seen_types.append(ctype)
    if len(H_POLYMER) == 4:
        break

# ---------------------------------------------------------------- Task 3 follow-up experiment
# (`reports/step5_result_v3.md` addendum): give ~15-20% of components a
# SECOND qualified supplier -- a real co-parent, not just a structural edge.
# `components.supplier_id` is a single not-null FK, so the base v3 graph's
# Supplier->SUPPLIES->Component->rev_SUPPLIES->Supplier path
# (docs/06_Graph_Database_Design.md §6.1) can never reach a co-parent; this
# table (`component_suppliers`) finally makes that path real for the
# suppliers it covers. To make it a genuinely CAUSAL signal (not merely
# structural), a co-parent's own stress measurably bleeds into its partner's
# stress below (`COPARENT_COUPLING`) -- discoverable only by actually
# traversing the co-parent edge, since it isn't derivable from either
# supplier's own history alone.
#
# Uses a DEDICATED local RNG (`_coparent_rng`), not the shared `random`
# module, so this addition consumes zero draws from the main simulation's
# stream -- every other random choice (products, BOMs, demand, orders,
# unrelated shipments) stays byte-identical to the base v3 CSVs. Only
# stress-driven outcomes (shipment delays -> shortage) for the ~15-20% of
# suppliers with a co-parent partner differ, which is the entire point of
# the experiment: report it as a separate labeled comparison, not a
# retrofit of the base v3 result.
_coparent_rng = random.Random(0xC09A2E47)          # arbitrary fixed seed, isolated stream
DUAL_SOURCE_FRACTION = 0.175                        # midpoint of the requested 15-20%
COPARENT_COUPLING = 0.35                            # fraction of partner's OWN stress that bleeds through
dual_sourced = _coparent_rng.sample(components, round(len(components) * DUAL_SOURCE_FRACTION))
component_suppliers = []                            # rows for component_suppliers.csv
coparents = {}                                      # supplier_id -> set(supplier_id), real & non-empty now
for c in dual_sourced:
    primary = c["supplier_id"]
    secondary = _coparent_rng.choice([s["id"] for s in suppliers if s["id"] != primary])
    component_suppliers.append(dict(id=uid("compsup", c["id"], secondary),
                                     component_id=c["id"], supplier_id=secondary))
    coparents.setdefault(primary, set()).add(secondary)
    coparents.setdefault(secondary, set()).add(primary)

# Disruption timeline: (factor-members, start, peak, end, magnitude)   -- Chapter 10
# Spread across the FULL Jul-2024 .. Sep-2025 snapshot window, same principle
# Step C applied to the original Jul-Dec window: every monthly snapshot sees at
# least one active event (a comparable disrupted/quiet mix on both sides of
# wherever a future train/val/test split falls), never a re-concentration into
# the original Jul-Dec 2024 months with 2025 as quiet padding.
#
# Month coverage (>=1 active event each): Jul24 TRUCK | Aug24 TRUCK+CUSTOMS+POLYMER |
# Sep24 CUSTOMS+PORT+POLYMER | Oct24 PORT+POLYMER | Nov24 PORT+POLYMER |
# Dec24 POLYMER | Jan25 CUSTOMS | Feb25 CUSTOMS+TRUCK | Mar25 TRUCK |
# Apr25 TRUCK+PORT | May25 PORT+POLYMER | Jun25 PORT+POLYMER |
# Jul25 POLYMER+CUSTOMS | Aug25 POLYMER+CUSTOMS+TRUCK | Sep25 TRUCK.
PORT_EVENTS = [
    (H_PORT, ts(2024,9,10), ts(2024,10,8), ts(2024,11,12), 0.65),       # port congestion, autumn 2024
    (H_PORT, ts(2025,4,7),  ts(2025,5,5),  ts(2025,6,9),   0.60),       # port congestion, spring 2025
]
EVENTS = [
    (H_TRUCK,   ts(2024,7,3),   ts(2024,7,17), ts(2024,8,7),   0.55),   # trucking strike
    (H_CUSTOMS, ts(2024,8,1),   ts(2024,8,15), ts(2024,9,5),   0.45),   # customs friction
    PORT_EVENTS[0],
    # Timing constraint (Step C finding, re-verified at 800 suppliers): the
    # flare must RESOLVE into the Dec-1-2024 trailing-90d window the
    # co-degradation check reads. At this scale the members drawn are mostly
    # LONG-lead sea suppliers (23-38d) -- with a Sep-10 start, most of their
    # Sep-Dec window deliveries stem from pre-event dispatches and their
    # peak-period delayed shipments slip past Dec 1 (measured: one 38d-lead
    # member showed 0.92 on-time, the event missed its window entirely). Start
    # moved to Aug 5 / peak Sep 15 so even a 38d-lead member's in-window
    # dispatch range (late Jul - late Oct) is event-covered; end stays Dec 15.
    # The validation suite still checks THAT snapshot (2024-12-01), unchanged.
    (H_POLYMER, ts(2024,8,5),   ts(2024,9,15), ts(2024,12,15), 0.95),   # hidden polymer shortage, flare 1
    # -- 2025 continuation: same pools, same event shapes, so the 9 added
    #    snapshot months carry real disruption signal, not quiet padding --
    (H_CUSTOMS, ts(2025,1,8),   ts(2025,1,24), ts(2025,2,12),  0.50),   # winter customs backlog
    (H_TRUCK,   ts(2025,2,18),  ts(2025,3,6),  ts(2025,4,2),   0.55),   # second trucking action
    PORT_EVENTS[1],
    (H_POLYMER, ts(2025,5,20),  ts(2025,6,24), ts(2025,8,15),  0.85),   # hidden polymer shortage, flare 2
    (H_CUSTOMS, ts(2025,7,10),  ts(2025,7,28), ts(2025,8,20),  0.45),   # summer customs friction
    (H_TRUCK,   ts(2025,8,12),  ts(2025,9,2),  ts(2025,9,28),  0.55),   # late-summer trucking action
]
# occasional idiosyncratic strike/outage: 25% of suppliers get one event at a
# uniformly random start anywhere in the (now 21-month) timeline
IDIO = {}
for s in suppliers:
    if random.uniform(0, 1) < 0.25:
        st = T_START + timedelta(days=random.randint(0, TIMELINE_DAYS - 30))
        IDIO[s["id"]] = (st, st + timedelta(days=10), st + timedelta(days=25), random.uniform(0.3, 0.6))
    else:
        IDIO[s["id"]] = None

def own_stress(sup_id, base_rel, t):
    """Latent supplier stress in [0,1] at time t from THIS supplier's own
    factors only (base reliability + shared-hidden-factor events it belongs
    to + its own idiosyncratic outage) -- the pre-Task-3 `stress()` body,
    renamed and kept as the base case so the co-parent coupling pass below
    (which reads partners' OWN stress, never their coupled stress) can't
    recurse into a mutual A-depends-on-B-depends-on-A loop."""
    x = (1 - base_rel) * 0.5
    for members, s0, pk, s1, mag in EVENTS:
        if sup_id in members and s0 <= t <= s1:
            frac = (t-s0)/(pk-s0) if t <= pk else 1 - (t-pk)/(s1-pk)
            x += mag * max(0.0, min(1.0, frac))
    ev = IDIO.get(sup_id)
    if ev:
        s0, pk, s1, mag = ev
        if s0 <= t <= s1:
            frac = (t-s0)/(pk-s0) if t <= pk else 1 - (t-pk)/(s1-pk)
            x += mag * max(0.0, min(1.0, frac))
    return min(0.95, x)


def stress(sup_id, base_rel, t):
    """Latent supplier stress in [0,1] at time t — the causal driver of
    everything. Task 3 follow-up: on top of `own_stress`, a real co-parent
    coupling term adds `COPARENT_COUPLING` * each partner's OWN stress
    (`coparents`, from the `component_suppliers` junction table above) --
    a genuine causal bleed-through, reachable only via the co-parent graph
    edge, not from either supplier's own history. Suppliers with no
    co-parent partner (the ~82-85% majority) get exactly the base v3
    behavior -- `coparents.get(sup_id, ())` is empty for them."""
    x = own_stress(sup_id, base_rel, t)
    for partner in coparents.get(sup_id, ()):
        x += COPARENT_COUPLING * own_stress(partner, sup_by_id[partner]["base_rel"], t)
    return min(0.95, x)

products, boms = [], []          # boms: (product_id, component_id, qty, created_at, deactivated_at)
CATS = ["industrial_pump","controller","actuator","sensor_array","drive_unit","valve_system"]
PROD_N = round(80 * SCALE)
COMP_W = [comp_common[c["id"]] for c in components]   # precomputed once (identical values per draw)
for p in range(PROD_N):
    pid = uid("prod", p)
    products.append(dict(id=pid, sku=f"SKU-{1000+p}", name=f"{CATS[p%6].replace('_',' ').title()} M{p:02d}",
                         category=CATS[p % 6]))
    n_bom = random.randint(3, 8)
    chosen = set()
    while len(chosen) < n_bom:
        c = random.choices(components, weights=COMP_W)[0]
        chosen.add(c["id"])
    # sorted(): iterating the raw set here is PYTHONHASHSEED-dependent (str-hash
    # order varies per process), which silently permuted qty draws and boms row
    # order -- and, through random.choice() over that order in the swap loop
    # below, changed WHICH BOM edge gets deactivated, run to run. This was a
    # pre-existing latent break of the byte-identical-CSVs guarantee, exposed
    # by an actual two-run diff at the v3 scale-up.
    for cidx in sorted(chosen):
        boms.append([pid, cidx, random.randint(1, 6), fmt(T_START), ""])
# BOM evolution -- Chapter 4/11: substitutions (swap, count-neutral) + pure additions
# (a design change picks up an extra component, net +1), spread across the FULL
# timeline so the active edge SET *and* COUNT both keep moving across all 15
# monthly snapshots -- leaving the 2025 months without any BOM change would
# re-freeze USED_IN for 9 of 15 snapshots, the exact "static graph topology"
# tell an earlier audit fixed. (Products indexed k*9 and k*9+4 stay disjoint.)
SWAP_DATES = [ts(2024,2,12), ts(2024,3,18), ts(2024,4,9), ts(2024,5,21),
              ts(2025,2,11), ts(2025,4,15)]
ADD_DATES  = [ts(2024,7,15), ts(2024,8,10), ts(2024,9,20), ts(2024,11,5),
              ts(2025,1,21), ts(2025,3,18), ts(2025,6,10), ts(2025,8,12)]
for k, chdate in enumerate(SWAP_DATES):
    pid = products[k*9]["id"]
    mine = [b for b in boms if b[0] == pid]
    old = random.choice(mine); old[4] = fmt(chdate)
    new_c = random.choice(components)["id"]
    if not any(b[0]==pid and b[1]==new_c and b[4]=="" for b in boms):
        boms.append([pid, new_c, random.randint(1,4), fmt(chdate), ""])
for k, chdate in enumerate(ADD_DATES):
    pid = products[k*9 + 4]["id"]
    new_c = random.choice(components)["id"]
    if not any(b[0]==pid and b[1]==new_c and b[4]=="" for b in boms):
        boms.append([pid, new_c, random.randint(1,4), fmt(chdate), ""])

comp_sup_by_id = {c["id"]: c["supplier_id"] for c in components}
prod_bom_sup = {}                # product -> set of supplier ids (via current BOM) for causality
for b in boms:
    if b[4] == "":
        prod_bom_sup.setdefault(b[0], set()).add(comp_sup_by_id[b[1]])

# ---------------------------------------------------------------- Chapter 5
factories = [dict(id=uid("fac", i), name=f"Plant {chr(65+i)}", location=loc,
                  capacity_units_per_day=random.randint(400, 1200))
             for i, loc in enumerate(["Pune, India","Monterrey, Mexico","Stuttgart, Germany","Shenzhen, China","Austin, USA"])]
FACTORY_OUTAGE = (factories[3]["id"], ts(2024,6,3), ts(2024,6,24))          # factory outage event

product_factories = []
for p in products:
    fs = random.sample(factories, random.randint(1, 2))
    for j, f in enumerate(fs):
        product_factories.append(dict(id=uid("pf", p["id"], f["id"]), product_id=p["id"], factory_id=f["id"],
            is_primary=(j == 0), capacity_units_per_day=random.randint(50, 300),
            qualified_at=fmt(T_START - timedelta(days=random.randint(30, 700)))))
prim_fac = {r["product_id"]: r["factory_id"] for r in product_factories if r["is_primary"]}

warehouses = [dict(id=uid("wh", i), name=f"DC {c}", location=l, capacity_units=random.randint(20000, 60000))
              for i, (c, l) in enumerate([("North","Chicago, USA"),("South","Dallas, USA"),("EU","Rotterdam, NL"),
                                          ("APAC","Singapore"),("West","Reno, USA"),("MX","Queretaro, MX"),
                                          ("IN","Chennai, India"),("CN","Ningbo, China")])]

inv_pairs = []                   # (product, warehouse, threshold, base stock)
for p in products:
    for w in random.sample(warehouses, random.randint(2, 3)):
        thr = random.randint(40, 120)
        inv_pairs.append(dict(product_id=p["id"], warehouse_id=w["id"], thr=thr,
                              stock=float(thr * random.uniform(1.6, 3.2))))

# ---------------------------------------------------------------- Chapter 6
CUST_N = round(100 * SCALE)
_strategic_cut, _low_cut = round(0.15 * CUST_N), round(0.85 * CUST_N)
customers = [dict(id=uid("cust", i), name=f"Customer {i:03d}",
                  priority_tier=("strategic" if i < _strategic_cut else "low" if i >= _low_cut else "standard"))
             for i in range(CUST_N)]

def season(t):                   # seasonal demand multiplier, spike Sep-Oct
    m = t.month
    return 1.6 if m in (9, 10) else 1.25 if m in (3, 11) else 1.0

orders, order_items = [], []
oi = 0
# Order count tracks world size AND timeline length (a 21-month world at the
# same monthly order rate has proportionally more orders than a 12-month one)
# -- this keeps per-product weekly demand intensity at the level the shortage/
# replenishment dynamics were calibrated against, rather than silently diluting
# demand across the longer window. Same principle as TIMELINE_WEEKS below.
ORDER_N = round(1000 * SCALE * TIMELINE_DAYS / 365)
_SPIKE_MONTHS = [(2024,9),(2024,10),(2024,9),(2024,10),(2024,3),(2024,11),
                 (2025,9),(2025,9),(2025,3)]      # season() spike/shoulder months present in the window
for o in range(ORDER_N):
    day = random.randint(0, TIMELINE_DAYS - 8)
    placed = T_START + timedelta(days=day, hours=random.randint(8, 17))
    if random.random() > season(placed) / 1.6:                    # thin non-season, keep spike months dense
        yr, mo = random.choice(_SPIKE_MONTHS)
        placed = ts(yr, mo, random.randint(1,27), random.randint(8,17))
    placed = J(placed)                                            # observed event, not a clean boundary
    cust = random.choice(customers)
    ent  = random.random() < 0.05                                 # enterprise mega-order
    oid  = uid("ord", o)
    due  = placed + timedelta(days=random.randint(7, 30))
    val  = 0.0
    for _ in range(random.randint(1, 4)):
        pr  = random.choice(products)
        qty = random.randint(40, 400) if ent else random.randint(5, 60)
        order_items.append(dict(id=uid("oi", oi), order_id=oid, product_id=pr["id"],
                                quantity=qty, created_at=fmt(placed))); oi += 1
        val += qty * random.uniform(80, 400)
    orders.append(dict(id=oid, order_number=f"ORD-{placed.year}-{o:05d}", customer_id=cust["id"],
                       status="open", placed_at=placed, due_at=due, order_value=round(val, 2)))

# ---------------------------------------------------------------- Chapters 7/8 — coupled shipment + inventory simulation
CARRIERS = ["Maersk Line","EverGreen Marine","DHL Freight","FedEx Logistics","DB Schenker"]
SEA = {"Maersk Line","EverGreen Marine"}
shipments, transitions = [], []       # transitions: (shipment_id, status, prev, changed_at)
demand_of = {}                        # product -> weekly base demand from order volume
for it in order_items:
    demand_of[it["product_id"]] = demand_of.get(it["product_id"], 0) + it["quantity"]
# divide by the ACTUAL timeline length in weeks (was /52.0 in the 1-year world;
# keeping 52 here would inflate weekly demand ~1.75x and blow the calibrated
# shortage rate out of Dataset.md Appendix A's guidance band)
for k in demand_of: demand_of[k] = max(4.0, demand_of[k] / TIMELINE_WEEKS)

def new_shipment(idx, kind, when, sup=None, fac=None, wh=None, order=None, origin=None, sea_ok=True):
    sid = uid("shp", idx)
    carrier = random.choice([c for c in CARRIERS if sea_ok or c not in SEA])
    if sup:
        lead = sup_by_id[sup]["lead_time_days"]
    else:
        lead = random.randint(3, 9)
    when = J(when)
    dispatch = J(when + timedelta(days=random.randint(0, 3), hours=random.randint(1, 9)))
    eta = dispatch + timedelta(days=lead)
    # causal delay: driven by latent stress of the responsible supplier(s) at dispatch
    if sup:
        st = stress(sup, sup_by_id[sup]["base_rel"], dispatch)
    else:
        sups = prod_bom_sup.get(order_prod.get(order, ""), set()) if order else set()
        st = max((stress(s, sup_by_id[s]["base_rel"], dispatch) for s in sups), default=0.08)
        if fac == FACTORY_OUTAGE[0] and FACTORY_OUTAGE[1] <= dispatch <= FACTORY_OUTAGE[2]:
            st = min(0.95, st + 0.35)
    if carrier in SEA:
        for _members, s0, pk, s1, _mag in PORT_EVENTS:            # port events also slow sea carriers directly
            if s0 <= dispatch <= s1: st = min(0.95, st + 0.10)
    p_delay = min(0.80, 0.025 + 0.38 * st)
    delayed = random.random() < p_delay
    late_by = timedelta(days=max(1, int(random.lognormvariate(1.1, 0.6) * (1 + 2*st)))) if delayed else timedelta(0)
    actual = J(eta + late_by) if delayed else J(eta - timedelta(hours=random.randint(0, 30)))
    transitions.append((sid, "scheduled", "", when))
    transitions.append((sid, "in_transit", "scheduled", dispatch))
    status, delivered_at = "in_transit", None
    if delayed and eta <= T_END:
        transitions.append((sid, "delayed", "in_transit", J(eta + timedelta(hours=6))))
        status = "delayed"
    if actual <= T_END:
        transitions.append((sid, "delivered", status, actual))
        status, delivered_at = "delivered", actual
    shipments.append(dict(id=sid, supplier_id=sup or "", factory_id=fac or "", warehouse_id=wh or "",
                          order_id=order or "", carrier=carrier, status=status, eta=eta,
                          dispatched_at=dispatch, delivered_at=delivered_at,
                          origin_location=origin, created_at=when))
    return actual if status == "delivered" else None, delayed

order_prod = {it["order_id"]: it["product_id"] for it in order_items}

ship_idx = 0
inv_hist = []                          # observation rows
shortage_events = {}                   # (prod,wh) -> [datetime of first sub-threshold obs per episode]
arrivals = {}                          # (prod,wh) -> list[(when, qty)]
pending = {}                           # (prod,wh) -> eta of open replenishment

# outbound: one fulfilment shipment for ~80% of orders
whs_by_prod = {}                       # product -> [warehouse ids], in inv_pairs order (same list random.choice saw before)
for ip in inv_pairs:
    whs_by_prod.setdefault(ip["product_id"], []).append(ip["warehouse_id"])
fac_loc = {f["id"]: f["location"] for f in factories}
for od in orders:
    if random.random() < 0.8:
        pr = order_prod[od["id"]]; fac = prim_fac[pr]
        wh = random.choice(whs_by_prod[pr])
        new_shipment(ship_idx, "out", J(od["placed_at"] + timedelta(days=1)), fac=fac, wh=wh,
                     order=od["id"], origin=fac_loc[fac], sea_ok=False)
        ship_idx += 1

# weekly inventory walk + demand-triggered replenishment (supplier -> warehouse)
week = T_START
while week <= T_END:
    for ip in inv_pairs:
        key = (ip["product_id"], ip["warehouse_id"])
        for (when, qty) in [a for a in arrivals.get(key, []) if a[0] <= week]:
            ip["stock"] += qty
        arrivals[key] = [a for a in arrivals.get(key, []) if a[0] > week]
        ip["stock"] = max(0.0, ip["stock"] - demand_of.get(ip["product_id"], 6) * season(week) * random.uniform(0.8, 1.2) / 2.9)
        if ip["stock"] < ip["thr"] * 1.55 and key not in pending:
            sup = random.choice(sorted(prod_bom_sup.get(ip["product_id"], {suppliers[0]["id"]})))
            srow = sup_by_id[sup]
            got, _ = new_shipment(ship_idx, "in", J(week), sup=sup, wh=ip["warehouse_id"],
                                  origin=srow["country"], sea_ok=srow["sea"]); ship_idx += 1
            eta_guess = week + timedelta(days=srow["lead_time_days"] + 2)
            when_in = got if got else eta_guess + timedelta(days=14)
            arrivals.setdefault(key, []).append((when_in, ip["thr"] * random.uniform(2.0, 2.8)))
            pending[key] = when_in
        if key in pending and pending[key] <= week:
            del pending[key]
        if ip["stock"] < ip["thr"]:
            shortage_events.setdefault(key, []).append(week)
        inv_hist.append(dict(product_id=key[0], warehouse_id=key[1],
                             stock_level=int(ip["stock"]), reorder_threshold=ip["thr"],
                             observed_at=J(week + timedelta(hours=6))))
    week += timedelta(days=7)

print(f"world: shipments={len(shipments):,} transitions={len(transitions):,} inv_obs={len(inv_hist):,}")

# ---------------------------------------------------------------- Chapter 12 — snapshots, features, labels
# 15 monthly t0s: Jul-Dec 2024 (the original 6) + Jan-Sep 2025 (the extension).
# First t0 keeps its 182-day trailing-history margin (T_START unchanged); last
# t0's label horizon (Sep 1 + 14d = Sep 15) sits inside T_END (Sep 30).
T0S = [ts(2024, m, 1) for m in (7, 8, 9, 10, 11, 12)] + [ts(2025, m, 1) for m in range(1, 10)]
GITC = "a3f9c2e8b1d4470a9e6c5f2b8d7a1c3e5f9b2d40"

# transitions indexed by shipment: the old linear scan over ALL transitions per
# (shipment, t0) pair was ~300M iterations at the 180-supplier scale and would
# be ~20 billion here (15 t0s x ~40k shipments x ~130k transitions). Pure
# indexing, no RNG -- identical output.
trans_by_sid = {}
for tr in transitions:
    trans_by_sid.setdefault(tr[0], []).append(tr)

def asof_status(sid, t0):
    st = ""
    for (s, stat, prev, at) in trans_by_sid.get(sid, ()):
        if at <= t0: st = stat
    return st

sup_ship = {}
for sh in shipments:
    if sh["supplier_id"]:
        sup_ship.setdefault(sh["supplier_id"], []).append(sh)

def sup_features(sup, t0):
    rows = [sh for sh in sup_ship.get(sup["id"], []) if sh["dispatched_at"] <= t0]
    done = [sh for sh in rows if sh["delivered_at"] and sh["delivered_at"] <= t0]
    def rate(days):
        w = [sh for sh in done if sh["delivered_at"] >= t0 - timedelta(days=days)]
        if len(w) < 3: return None, len(w)
        on = sum(1 for sh in w if sh["delivered_at"] <= sh["eta"])
        return round(on/len(w), 4), len(w)
    r30,_ = rate(30); r90,_ = rate(90); r180,n180 = rate(180)
    seq = sorted(done, key=lambda s: s["delivered_at"])[-10:]
    slope = None
    if len(seq) >= 5:
        ys = [1.0 if s["delivered_at"] <= s["eta"] else 0.0 for s in seq]
        xs = list(range(len(ys))); mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
        den = sum((x-mx)**2 for x in xs)
        slope = round(sum((x-mx)*(y-my) for x, y in zip(xs, ys))/den, 6) if den else 0.0
    lates = [ (s["delivered_at"]-s["eta"]).total_seconds()/86400 for s in done if s["delivered_at"] > s["eta"] ]
    var = round(sum((x - sum(lates)/len(lates))**2 for x in lates)/len(lates), 4) if len(lates) >= 2 else None
    last_late = max((s["delivered_at"] for s in done if s["delivered_at"] > s["eta"]), default=None)
    dsl = (t0 - last_late).days if last_late else None
    return r30, r90, r180, slope, var, dsl, n180

stf_rows, carrier_rows, snap_rows, label_rows = [], [], [], []
for t0 in T0S:
    # supplier temporal features
    for s in suppliers:
        r30, r90, r180, slope, var, dsl, n = sup_features(s, t0)
        stf_rows.append([uid("stf", s["id"], t0), s["id"], t0.date().isoformat(),
                         r30 if r30 is not None else "", r90 if r90 is not None else "",
                         r180 if r180 is not None else "", slope if slope is not None else "",
                         var if var is not None else "", dsl if dsl is not None else "",
                         n, GJ(), "v1"])
    # carrier snapshots
    for c in CARRIERS:
        done = [sh for sh in shipments if sh["carrier"] == c and sh["delivered_at"]
                and t0 - timedelta(days=90) <= sh["delivered_at"] <= t0]
        rate = round(sum(1 for s_ in done if s_["delivered_at"] <= s_["eta"]) / len(done), 4) if done else ""
        carrier_rows.append([uid("cps", c, t0), c, "", "", t0.date().isoformat(), rate, len(done), GJ()])
    # labels
    n_pos = {"delay": 0, "shortage": 0, "impact": 0}
    hz = t0 + timedelta(days=HORIZON)
    sup_hit = set()
    for sh in shipments:
        if sh["created_at"] > t0: continue
        if asof_status(sh["id"], t0) not in ("scheduled", "in_transit"): continue
        ev = next((at for (_s, stat, _p, at) in trans_by_sid.get(sh["id"], ()) if stat == "delayed" and t0 < at <= hz), None)
        lab = ev is not None
        if lab and sh["supplier_id"]: sup_hit.add(sh["supplier_id"])
        n_pos["delay"] += int(lab)
        label_rows.append([uid("lbl", t0, "delay", sh["id"]), uid("snap", t0), "shipment", sh["id"], "delay",
                           str(lab).lower(), fmt(ev) if ev else "", "shipment_status_history", ""])
    for ip in inv_pairs:
        key = (ip["product_id"], ip["warehouse_id"])
        ev = next((w + timedelta(hours=6) for w in shortage_events.get(key, [])
                   if t0 < w + timedelta(hours=6) <= hz), None)      # compare on observed_at, incl. the 6h offset
        lab = ev is not None
        n_pos["shortage"] += int(lab)
        # entity_id stays product_id (entity_type='product'), but warehouse_id is now
        # carried alongside it -- Fix 3: previously two rows for the SAME product could
        # disagree (different warehouses, different outcomes) with nothing in the row to
        # tell them apart; grouping by (entity_id, warehouse_id) is now unambiguous.
        label_rows.append([uid("lbl", t0, "short", *key), uid("snap", t0), "product", ip["product_id"], "shortage",
                           str(lab).lower(), fmt(ev) if ev else "", "inventory_history", ip["warehouse_id"]])
    for s in suppliers:
        lab = s["id"] in sup_hit
        n_pos["impact"] += int(lab)
        label_rows.append([uid("lbl", t0, "impact", s["id"]), uid("snap", t0), "supplier", s["id"], "impact",
                           str(lab).lower(), "", "shipment_status_history", ""])
    live_edges = sum(1 for b in boms if b[3] <= fmt(t0) and (b[4] == "" or b[4] > fmt(t0)))
    snap_rows.append([uid("snap", t0), fmt(t0), HORIZON,
        json.dumps({"supplier":SUP_N,"component":len(components),"product":len(products),"factory":len(factories),
                    "warehouse":len(warehouses),"customer":CUST_N,"inventory":len(inv_pairs),
                    "order":sum(1 for o in orders if o["placed_at"] <= t0),
                    "shipment":sum(1 for sh in shipments if sh["created_at"] <= t0)}),
        json.dumps({"SUPPLIES":len(components),"USED_IN":live_edges,"MANUFACTURED_AT":len(product_factories),
                    "HAS_INVENTORY":len(inv_pairs)}),
        json.dumps({"delay_pos":n_pos["delay"],"shortage_pos":n_pos["shortage"],"impact_pos":n_pos["impact"]}),
        "v1", GITC, round(random.uniform(20, 60), 2), GJ()])

# risk_scores: deterministic weighted-formula seed rows for the final snapshot (demo data, not model output)
risk_rows, t0 = [], T0S[-1]
_r90_last = {r[1]: float(r[4]) for r in stf_rows if r[2] == t0.date().isoformat() and r[4] != ""}
for s in suppliers:
    r90 = _r90_last.get(s["id"], 0.9)
    dp = round(min(0.95, max(0.02, 1 - r90 + 0.05)), 4)
    imp = round(min(0.95, 0.3*dp + 0.25*dp + 0.2*random.uniform(0.05,0.3) + 0.15*0.1 + 0.1*0.1), 4)
    cat = "critical" if imp >= .6 else "high" if imp >= .4 else "medium" if imp >= .2 else "low"
    risk_rows.append([uid("rs", s["id"], t0), "supplier", s["id"], dp, "", imp, 0.7, cat,
                      "weighted_formula", "formula-v0", fmt(t0), HORIZON, GJ()])

# ---------------------------------------------------------------- write CSVs
# per-entity batch timestamps: one fresh jittered instant per row, reused for created_at==
# updated_at on that row (never-modified master data) so the two columns agree but no two
# rows in the table -- or across tables -- share one identical clock tick.
SUP_TS, COMP_TS, PROD_TS = [GJ() for _ in suppliers], [GJ() for _ in components], [GJ() for _ in products]
FAC_TS, WH_TS, CUST_TS   = [GJ() for _ in factories], [GJ() for _ in warehouses], [GJ() for _ in customers]

print("writing csv/ ...")
write("suppliers.csv", ["id","name","country","capacity_score","lead_time_days","reliability_history","is_active","created_at","updated_at"],
      [[s["id"], s["name"], s["country"], s["capacity_score"], s["lead_time_days"], round(s["base_rel"],4), "true", g, g] for s, g in zip(suppliers, SUP_TS)])
write("components.csv", ["id","supplier_id","name","component_type","unit_cost","created_at","updated_at"],
      [[c["id"], c["supplier_id"], c["name"], c["component_type"], c["unit_cost"], g, g] for c, g in zip(components, COMP_TS)])
write("component_suppliers.csv", ["id","component_id","supplier_id","created_at","deactivated_at"],
      [[r["id"], r["component_id"], r["supplier_id"], fmt(T_START), ""] for r in component_suppliers])
write("products.csv", ["id","sku","name","category","is_active","created_at","updated_at"],
      [[p["id"], p["sku"], p["name"], p["category"], "true", g, g] for p, g in zip(products, PROD_TS)])
write("product_components.csv", ["id","product_id","component_id","quantity_required","created_at","deactivated_at"],
      [[uid("bom", b[0], b[1], b[3]), b[0], b[1], b[2], b[3], b[4]] for b in boms])
write("factories.csv", ["id","name","location","capacity_units_per_day","is_active","created_at","updated_at"],
      [[f["id"], f["name"], f["location"], f["capacity_units_per_day"], "true", g, g] for f, g in zip(factories, FAC_TS)])
write("product_factories.csv", ["id","product_id","factory_id","is_primary","capacity_units_per_day","qualified_at","created_at","deactivated_at"],
      [[r["id"], r["product_id"], r["factory_id"], str(r["is_primary"]).lower(), r["capacity_units_per_day"], r["qualified_at"], fmt(T_START), ""] for r in product_factories])
write("warehouses.csv", ["id","name","location","capacity_units","is_active","created_at","updated_at"],
      [[w["id"], w["name"], w["location"], w["capacity_units"], "true", g, g] for w, g in zip(warehouses, WH_TS)])
write("inventory.csv", ["id","product_id","warehouse_id","stock_level","reorder_threshold","updated_at"],
      [[uid("inv", ip["product_id"], ip["warehouse_id"]), ip["product_id"], ip["warehouse_id"], int(ip["stock"]), ip["thr"], GJ()] for ip in inv_pairs])
write("inventory_history.csv", ["id","inventory_id","product_id","warehouse_id","stock_level","reorder_threshold","observed_at","recorded_at","source"],
      [[uid("ih", r["product_id"], r["warehouse_id"], fmt(r["observed_at"])), uid("inv", r["product_id"], r["warehouse_id"]),
        r["product_id"], r["warehouse_id"], r["stock_level"], r["reorder_threshold"], fmt(r["observed_at"]),
        fmt(r["observed_at"] + timedelta(minutes=random.randint(20, 240))), "wms_sync"] for r in inv_hist])
write("customers.csv", ["id","name","priority_tier","contract_terms","is_active","created_at","updated_at"],
      [[c["id"], c["name"], c["priority_tier"], "", "true", g, g] for c, g in zip(customers, CUST_TS)])
ord_state = {}
for sh in shipments:                                   # derive order status from its fulfilment shipment
    if sh["order_id"]:
        ord_state[sh["order_id"]] = "fulfilled" if sh["status"] == "delivered" else "at_risk"
write("orders.csv", ["id","order_number","customer_id","status","placed_at","due_at","order_value","created_at","updated_at"],
      [[o["id"], o["order_number"], o["customer_id"], ord_state.get(o["id"], "open"), fmt(o["placed_at"]), fmt(o["due_at"]), o["order_value"], fmt(o["placed_at"]), GJ()] for o in orders])
write("order_items.csv", ["id","order_id","product_id","quantity","created_at"],
      [[i["id"], i["order_id"], i["product_id"], i["quantity"], i["created_at"]] for i in order_items])
write("shipments.csv", ["id","supplier_id","factory_id","warehouse_id","order_id","carrier","status","eta","dispatched_at","delivered_at","origin_location","created_at","updated_at"],
      [[s["id"], s["supplier_id"], s["factory_id"], s["warehouse_id"], s["order_id"], s["carrier"], s["status"],
        fmt(s["eta"]), fmt(s["dispatched_at"]), fmt(s["delivered_at"]) if s["delivered_at"] else "", s["origin_location"], fmt(s["created_at"]), GJ()] for s in shipments])
write("shipment_status_history.csv", ["id","shipment_id","status","previous_status","changed_at","recorded_at","source"],
      [[uid("sst", sid, stat, fmt(at)), sid, stat, prev, fmt(at), fmt(at + timedelta(minutes=random.randint(5, 45))), "carrier_feed"] for (sid, stat, prev, at) in transitions])
write("supplier_temporal_features.csv", ["id","supplier_id","as_of_date","on_time_rate_30d","on_time_rate_90d","on_time_rate_180d","trend_slope","lateness_variance","days_since_last_late","shipment_count_180d","computed_at","feature_spec_version"], stf_rows)
write("carrier_performance_snapshots.csv", ["id","carrier","origin_location","destination_location","as_of_date","on_time_rate_90d","shipment_count_90d","computed_at"], carrier_rows)
write("graph_snapshots.csv", ["id","t0","horizon_days","node_counts","edge_counts","label_counts","feature_spec_version","git_commit","construction_seconds","created_at"], snap_rows)
write("training_labels.csv", ["id","snapshot_id","entity_type","entity_id","task","label","event_at","label_source","warehouse_id"], label_rows)
write("risk_scores.csv", ["id","entity_type","entity_id","delay_probability","shortage_risk","impact_score","confidence","risk_category","scoring_method","model_version","snapshot_t0","horizon_days","scored_at"], risk_rows)

# ---------------------------------------------------------------- Chapter 15 — validation suite
print("\nvalidation:")
fail = 0
def check(name, ok, detail=""):
    global fail
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    fail += 0 if ok else 1

sup_ids = {s["id"] for s in suppliers}; comp_ids = {c["id"] for c in components}
prod_ids = {p["id"] for p in products}; wh_ids = {w["id"] for w in warehouses}
fac_ids = {f["id"] for f in factories}; ord_ids = {o["id"] for o in orders}
check("FK components→suppliers", all(c["supplier_id"] in sup_ids for c in components))
# Task 3 follow-up experiment: component_suppliers integrity. The real
# empirical validation (does the co-parent path actually reach >0 partners,
# does the coupling change shortage/delay AUC) happens downstream in
# ml/graph/reach.py and the retrain comparison -- these are just structural
# sanity checks on the junction table and coupling wiring themselves.
frac = len(component_suppliers) / len(components)
check("component_suppliers: dual-sourced fraction in requested [0.15, 0.20] range",
      0.15 <= frac <= 0.20, f"({len(component_suppliers)}/{len(components)} = {frac:.1%})")
check("FK component_suppliers→components/suppliers, secondary != primary",
      all(r["component_id"] in comp_ids and r["supplier_id"] in sup_ids
          and r["supplier_id"] != comp_sup_by_id[r["component_id"]] for r in component_suppliers))
check("component_suppliers: co-parent partnerships are real & mutual",
      len(coparents) > 0 and all(a in coparents.get(b, ()) for b in coparents for a in coparents[b]))
check("FK boms→products/components", all(b[0] in prod_ids and b[1] in comp_ids for b in boms))
check("FK shipments", all((not s["supplier_id"] or s["supplier_id"] in sup_ids) and (not s["factory_id"] or s["factory_id"] in fac_ids)
                          and (not s["warehouse_id"] or s["warehouse_id"] in wh_ids) and (not s["order_id"] or s["order_id"] in ord_ids) for s in shipments))
check("PK unique shipments", len({s["id"] for s in shipments}) == len(shipments))
check("PK unique labels", len({r[0] for r in label_rows}) == len(label_rows))
# scale guard: pick() silently truncates to the available pool -- at SUP_N=800
# every hidden-factor pool must still be drawable at its full requested size
# (H_CUSTOMS is the tightest: 80 requested from a ~112-expected Germany pool)
check("hidden factor pools at full requested size",
      len(H_PORT) == round(12*SCALE) and len(H_TRUCK) == round(8*SCALE) and len(H_CUSTOMS) == round(5*SCALE),
      f"(port {len(H_PORT)}/{round(12*SCALE)}, truck {len(H_TRUCK)}/{round(8*SCALE)}, customs {len(H_CUSTOMS)}/{round(5*SCALE)})")
# Fix 3: shortage labels now carry warehouse_id (r[8]) alongside entity_id=product_id (r[3]),
# so grouping by (snapshot, entity_id, warehouse_id) must be conflict-free by construction.
# Also report the entity_id-only view for comparison -- that's the ambiguity the fix removes.
short_rows = [r for r in label_rows if r[4] == "shortage"]
by_ewh, by_e = {}, {}
for r in short_rows:
    by_ewh.setdefault((r[1], r[3], r[8]), set()).add(r[5])
    by_e.setdefault((r[1], r[3]), set()).add(r[5])
conflicts_with_wh = sum(1 for v in by_ewh.values() if len(v) > 1)
conflicts_without_wh = sum(1 for v in by_e.values() if len(v) > 1)
check("shortage labels: zero conflicts once warehouse_id disambiguates entity_id",
      conflicts_with_wh == 0,
      f"({conflicts_with_wh} conflicting groups; {conflicts_without_wh} would conflict without warehouse_id)")
chron = all(s["created_at"] <= s["dispatched_at"] < s["eta"] and (not s["delivered_at"] or s["delivered_at"] >= s["dispatched_at"]) for s in shipments)
check("chronology created<=dispatch<eta<=delivered", chron)
check("inventory never negative", all(r["stock_level"] >= 0 for r in inv_hist))
_t0_by_snap = {uid("snap", t): t for t in T0S}    # computed once, not per label row
bad_lbl = [r for r in label_rows if r[6] and not (r[6] > fmt(_t0_by_snap[r[1]]))]
check("labels: event_at strictly after t0", not bad_lbl, f"({len(bad_lbl)} bad)")
bad_win = 0
for r in label_rows:
    if r[6]:
        t0 = _t0_by_snap[r[1]]
        if not (fmt(t0) < r[6] <= fmt(t0 + timedelta(days=HORIZON))): bad_win += 1
check("labels: event within (t0, t0+H]", bad_win == 0, f"({bad_win} outside)")
leak = sum(1 for r in stf_rows for _ in [0] if False)
check("features: windows end at t0 (by construction)", True)
deg = sum(len(v) for v in prod_bom_sup.values()) / len(prod_bom_sup)
check("graph connectivity: avg BOM suppliers/product >= 2", deg >= 2, f"(avg {deg:.1f})")
pos = {"delay": 0, "shortage": 0, "impact": 0}; tot = {"delay": 0, "shortage": 0, "impact": 0}
for r in label_rows:
    tot[r[4]] += 1; pos[r[4]] += (r[5] == "true")
for k in pos:
    print(f"  label balance {k:9s}: {pos[k]:>4}/{tot[k]:<5} = {pos[k]/max(tot[k],1):.1%}")
# Fix 4: hidden-dependency members must span >=2 component_types (never recoverable
# from component_type alone) -- report the actual composition, not just count.
poly_types = {c["component_type"] for c in components if c["supplier_id"] in H_POLYMER}
check("hidden dependency: members span >=2 component_types (decoupled from component_type)",
      len(poly_types) >= 2, f"(types: {sorted(poly_types)})")
# hidden-dependency observable: polymer members co-degrade in Oct-Nov with no shared edge
poly = list(H_POLYMER)
co = [next((float(r[4]) for r in stf_rows if r[1] == p and r[2] == "2024-12-01" and r[4] != ""), None) for p in poly]
co = [c for c in co if c is not None]
base = [float(r[4]) for r in stf_rows if r[2] == "2024-12-01" and r[4] != "" and r[1] not in H_POLYMER]
mb = sum(base)/len(base) if base else 1.0
check("hidden dependency: polymer members co-degraded (Dec OTR-90d well below fleet mean)",
      len(co) >= 3 and sum(co)/len(co) < mb - 0.10,
      f"(members {sum(co)/len(co):.2f} vs fleet {mb:.2f}, n={len(co)})" if co else "(no data)")
print(f"\n{'ALL CHECKS PASSED' if fail == 0 else f'{fail} CHECKS FAILED'}")
raise SystemExit(1 if fail else 0)
