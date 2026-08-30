#!/usr/bin/env python3
"""
V3 schema enrichment — the node and edge types the V2 world simulated but never emitted.

`reports/new_task_depth.md` closed out the two structural explanations for `delay` and
`shortage` failing to clear their reproduction floors: readout DEPTH (h¹/h²/h³/h⁴ all tested,
none cleared) and readout WEIGHTING (a per-task probe refit on the same frozen representation,
also failed). What is left is the hypothesis that the emitted graph simply does not contain the
entities that carry the signal. This module emits them.

**The cardinal constraint: the underlying world must not change.** If enrichment perturbed the
simulation, the old-vs-new comparison would confound "new schema" with "different world" and
would answer nothing. Two rules enforce that, and `db/verify_enriched.py` asserts both:

1. **No RNG draw is taken from the simulation's stream.** Everything here runs AFTER the
   simulation is complete and draws from a *separate* `random.Random(seed ^ 0x5EED)`. The one
   touch inside `db/generate_dataset.py`'s weekly walk (`REPLENISH_LINKS.append`) consumes no
   draw and changes no branch.
2. **Every pre-existing table stays byte-identical.** The enrichment is purely additive: new
   files, never edits to old ones. Verified by gzip-level diff against a pre-change build.

**Nothing here emits hidden state.** `H_PORT`, `H_TRUCK` and `H_CUSTOMS` are latent supplier
groups, and `ml/data/loader.py::verify_no_hidden_state` exists to catch them reaching a CSV.
The brief asks for the scalar hidden factors to become `DEPENDS_ON` edges; emitting the true
group membership would be exactly the leak this project's whole design guards against, and
would inflate every number below for free. So the coupling edges are built from the
**observable** infrastructure a shipment demonstrably shares -- its origin port, its trucking
corridor, its customs jurisdiction, all derivable from `shipments.origin_location` and the
carrier's mode, which are already emitted columns. The structure the brief wants (shipments
coupled through shared infrastructure) is reproduced; the latent membership is not handed over.
`db/verify_enriched.py` measures how much of the true hidden grouping the observable proxy
actually recovers, and that number is reported rather than assumed.

**Temporal integrity is the existing rule, not a new one.** Every as-of row here is computed
only from deliveries whose REPORTED time (`delivered_rec`, Mechanism G's clock) is at or before
the as-of instant, which is the same backfill-honesty rule `sup_features()` already applies, and
each emitted observation carries its own `recorded_at` jitter so the loader's `recorded_at <= t0`
masks work on these tables exactly as they do on `inventory_history`.
"""
from __future__ import annotations

import math
import random
from datetime import timedelta

# ---------------------------------------------------------------------------
# geography
# ---------------------------------------------------------------------------
# Approximate (lat, lon) for every location string the generator emits, so lane distance is a
# real quantity rather than a hash. Country-only origins (a supplier's `country`) resolve to
# that country's main export gateway, which is what a country-level origin means operationally.
COORDS = {
    "Austin, USA": (30.27, -97.74), "Shenzhen, China": (22.54, 114.06),
    "Stuttgart, Germany": (48.78, 9.18), "Pune, India": (18.52, 73.86),
    "Monterrey, Mexico": (25.69, -100.32), "Chicago, USA": (41.88, -87.63),
    "Dallas, USA": (32.78, -96.80), "Rotterdam, NL": (51.92, 4.48),
    "Singapore": (1.35, 103.82), "Reno, USA": (39.53, -119.81),
    "Queretaro, MX": (20.59, -100.39), "Chennai, India": (13.08, 80.27),
    "Ningbo, China": (29.87, 121.55), "Vietnam": (10.82, 106.63),
    "India": (19.08, 72.88), "USA": (33.75, -118.19), "Mexico": (19.43, -99.13),
    "China": (31.23, 121.47), "Germany": (53.55, 9.99),
}
# Locations that are genuinely maritime gateways. A route only acquires sea ports when it is
# actually carried by a sea carrier, so this set gates port assignment, not lane creation.
SEA_GATEWAYS = {"Shenzhen, China", "Ningbo, China", "Singapore", "Rotterdam, NL",
                "Chennai, India", "Vietnam", "India", "USA", "China", "Germany"}
# Corridor transit hubs: the intermediate port a long sea lane passes through. Keyed by the
# (origin region, destination region) pair, so it is a property of the lane, not of a shipment.
TRANSIT_HUBS = {("APAC", "EMEA"): "Singapore", ("EMEA", "APAC"): "Singapore",
                ("APAC", "AMER"): "Singapore", ("AMER", "APAC"): "Singapore",
                ("EMEA", "AMER"): "Rotterdam, NL", ("AMER", "EMEA"): "Rotterdam, NL"}
REGION = {"USA": "AMER", "Mexico": "AMER", "MX": "AMER", "China": "APAC", "India": "APAC",
          "Vietnam": "APAC", "Singapore": "APAC", "Germany": "EMEA", "NL": "EMEA"}


def country_of(loc: str) -> str:
    return loc.split(",")[-1].strip() if "," in loc else loc.strip()


def region_of(loc: str) -> str:
    return REGION.get(country_of(loc), "AMER")


def haversine(a: str, b: str) -> float:
    """Great-circle km between two emitted location strings; 0.0 if either is unknown."""
    if a not in COORDS or b not in COORDS:
        return 0.0
    (la1, lo1), (la2, lo2) = COORDS[a], COORDS[b]
    p1, p2 = math.radians(la1), math.radians(la2)
    dp, dl = p2 - p1, math.radians(lo2 - lo1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371.0 * 2 * math.asin(min(1.0, math.sqrt(h)))


# ---------------------------------------------------------------------------
# as-of statistics, computed only from REPORTED deliveries
# ---------------------------------------------------------------------------

def _ontime_stats(rows, as_of, days):
    """`(on_time_rate, mean_transit_days, transit_var, n)` over deliveries REPORTED at or
    before `as_of` and delivered within the trailing `days` window.

    Mechanism G's clock is respected the same way `sup_features()` respects it: a delivery
    enters the window when it was reported (`delivered_rec`), never when it happened. Windows
    with fewer than 3 observations return `None` rather than a rate computed from noise --
    the same minimum `sup_features()` uses, so these columns behave like the existing ones.
    """
    lo = as_of - timedelta(days=days)
    w = [r for r in rows
         if r["delivered_at"] and (r["delivered_rec"] or r["delivered_at"]) <= as_of
         and r["delivered_at"] >= lo]
    if len(w) < 3:
        return None, None, None, len(w)
    on = sum(1 for r in w if r["delivered_at"] <= r["eta"])
    tr = [(r["delivered_at"] - r["dispatched_at"]).total_seconds() / 86400 for r in w]
    mu = sum(tr) / len(tr)
    var = sum((x - mu) ** 2 for x in tr) / len(tr)
    return round(on / len(w), 4), round(mu, 4), round(var, 4), len(w)


def _slope(rows, as_of, n=10):
    """Least-squares slope of the last `n` reported on-time outcomes. Same shape as
    `sup_features()`'s `trend_slope`, so a Carrier's trend column means what a Supplier's does."""
    done = sorted((r for r in rows
                   if r["delivered_at"] and (r["delivered_rec"] or r["delivered_at"]) <= as_of),
                  key=lambda r: r["delivered_at"])[-n:]
    if len(done) < 5:
        return None
    ys = [1.0 if r["delivered_at"] <= r["eta"] else 0.0 for r in done]
    xs = list(range(len(ys)))
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den, 6) if den else 0.0


def _days_since_late(rows, as_of):
    last = max((r["delivered_at"] for r in rows
                if r["delivered_at"] and (r["delivered_rec"] or r["delivered_at"]) <= as_of
                and r["delivered_at"] > r["eta"]), default=None)
    return (as_of - last).days if last else None


# ---------------------------------------------------------------------------
# the emitter
# ---------------------------------------------------------------------------

def emit(*, write, uid, fmt, rng_seed, shipments, warehouses, factories, suppliers,
         customers, orders, order_items, sup_by_id, replenish_links, carriers, sea_carriers,
         t_start, t_end, report_delay, delivered_rec):
    """Emit every V3 table and return a summary dict for the manifest.

    `write`, `uid` and `fmt` are the generator's own helpers, passed in rather than imported,
    so this module never reaches back into the generator's module state. `report_delay` is the
    generator's Mechanism-G delay sampler, but it is called on THIS module's RNG-free path
    only for `recorded_at` jitter -- see `rng` below.
    """
    # A private stream. The simulation's `random` module state is never touched, which is what
    # keeps every pre-existing table byte-identical.
    rng = random.Random(rng_seed ^ 0x5EED)
    wh_by_id = {w["id"]: w for w in warehouses}

    # THE clock. `delivered_rec[sid]` is the `recorded_at` this run actually WROTE into
    # `shipment_status_history.csv` for that shipment's `delivered` transition -- not the
    # simulation's in-memory `rec`, which on a variant without Mechanism G equals the true
    # event time and would therefore make every delivery visible up to 45 minutes early.
    # Every window below is computed against this and nothing else.
    # Rebound to a local copy, never mutated in place: the generator keeps running its own
    # validation suite after this call and must see exactly the objects it built.
    shipments = [dict(sh, delivered_rec=(delivered_rec.get(sh["id"]) if sh["delivered_at"] else None))
                 for sh in shipments]
    fac_by_id = {f["id"]: f for f in factories}

    def dest_loc(sh):
        return wh_by_id[sh["warehouse_id"]]["location"] if sh.get("warehouse_id") else ""

    # -- weekly as-of schedule, shared by every temporal table here ---------
    weeks = []
    w = t_start
    while w <= t_end:
        weeks.append(w)
        w += timedelta(days=7)

    def rec_at(when):
        """`recorded_at` for an observation made at `when`: reporting jitter on this module's
        own stream, so the loader's `recorded_at <= t0` mask has something real to bite on."""
        return when + timedelta(minutes=rng.randint(20, 240)) + report_delay()

    # ================================================================= Carrier
    car_mode = {c: ("sea" if c in sea_carriers else "land") for c in carriers}
    ships_by_carrier: dict = {}
    for sh in shipments:
        ships_by_carrier.setdefault(sh["carrier"], []).append(sh)

    car_id = {c: uid("car", c) for c in carriers}
    car_rows = []
    for c in carriers:
        rows = ships_by_carrier.get(c, [])
        lanes = {(sh["origin_location"], dest_loc(sh)) for sh in rows}
        car_rows.append([car_id[c], c, car_mode[c], len(rows), len(lanes),
                         round(sum(1 for sh in rows if sh["supplier_id"]) / max(1, len(rows)), 4),
                         fmt(t_start), fmt(t_start)])
    write("carriers.csv",
          ["id", "name", "mode", "shipment_count_total", "lane_count", "inbound_share",
           "created_at", "updated_at"], car_rows)

    ctf = []
    for c in carriers:
        rows = ships_by_carrier.get(c, [])
        for wk in weeks:
            r30, m30, v30, n30 = _ontime_stats(rows, wk, 30)
            r90, _, _, _ = _ontime_stats(rows, wk, 90)
            r180, _, _, n180 = _ontime_stats(rows, wk, 180)
            active = sum(1 for sh in rows
                         if sh["dispatched_at"] <= wk
                         and (not sh["delivered_at"] or (sh["delivered_rec"] or sh["delivered_at"]) > wk))
            ctf.append([uid("ctf", c, fmt(wk)), car_id[c], fmt(wk),
                        r30 if r30 is not None else "", r90 if r90 is not None else "",
                        r180 if r180 is not None else "",
                        _slope(rows, wk) if _slope(rows, wk) is not None else "",
                        v30 if v30 is not None else "",
                        _days_since_late(rows, wk) if _days_since_late(rows, wk) is not None else "",
                        n180, active, fmt(rec_at(wk)), "carrier_feed"])
    write("carrier_temporal_features.csv",
          ["id", "carrier_id", "as_of_date", "on_time_rate_30d", "on_time_rate_90d",
           "on_time_rate_180d", "trend_slope", "transit_variance_30d", "days_since_last_late",
           "shipment_count_180d", "active_shipments", "recorded_at", "source"], ctf)

    # ================================================================= Port
    # A port exists where freight actually enters or leaves the network: every distinct origin
    # and destination location, plus the corridor transit hubs. Nothing latent decides this.
    locs = sorted({sh["origin_location"] for sh in shipments if sh["origin_location"]}
                  | {dest_loc(sh) for sh in shipments if sh.get("warehouse_id")}
                  | {f["location"] for f in factories} | set(TRANSIT_HUBS.values()))
    port_id = {l: uid("port", l) for l in locs}
    write("ports.csv", ["id", "name", "location", "country", "region", "port_type",
                        "is_sea_gateway", "created_at"],
          [[port_id[l], f"PORT {l}", l, country_of(l), region_of(l),
            "sea" if l in SEA_GATEWAYS else "inland", str(l in SEA_GATEWAYS).lower(),
            fmt(t_start)] for l in locs])

    # ================================================================= Route (lane)
    # A lane is (origin, destination, mode). Mode comes from the carrier that moved the
    # shipment, so a sea lane and a land lane between the same pair are different routes --
    # which is the distinction port congestion actually acts on.
    lane_ships: dict = {}
    for sh in shipments:
        if not sh.get("warehouse_id"):
            continue
        key = (sh["origin_location"], dest_loc(sh), car_mode[sh["carrier"]])
        lane_ships.setdefault(key, []).append(sh)
    lanes = sorted(lane_ships)
    route_id = {k: uid("route", *k) for k in lanes}

    rt_rows, rp_rows = [], []
    for k in lanes:
        o, d, mode = k
        rows = lane_ships[k]
        km = haversine(o, d)
        typical = sorted((r["eta"] - r["dispatched_at"]).total_seconds() / 86400 for r in rows)
        med = typical[len(typical) // 2] if typical else 0.0
        rt_rows.append([route_id[k], o, d, mode, round(med, 3), round(km, 1),
                        len(rows), str(region_of(o) != region_of(d)).lower(), fmt(t_start)])
        # PASSES_THROUGH: origin port, destination port, and for a cross-region sea lane the
        # corridor hub in between. `sequence` makes the ordering explicit rather than implied.
        seq = [(o, "origin"), (d, "destination")]
        hub = TRANSIT_HUBS.get((region_of(o), region_of(d)))
        if mode == "sea" and hub and hub not in (o, d):
            seq.insert(1, (hub, "transit"))
        for i, (loc, role) in enumerate(seq):
            if loc in port_id:
                rp_rows.append([uid("rp", route_id[k], loc, role), route_id[k], port_id[loc],
                                role, i, fmt(t_start)])
    write("routes.csv", ["id", "origin_location", "destination_location", "mode",
                         "typical_transit_days", "distance_km", "shipment_count_total",
                         "is_cross_region", "created_at"], rt_rows)
    write("route_ports.csv", ["id", "route_id", "port_id", "role", "sequence", "created_at"],
          rp_rows)

    # -- MOVES_ON / HANDLED_BY ---------------------------------------------
    sr_rows = []
    for sh in shipments:
        if not sh.get("warehouse_id"):
            continue
        k = (sh["origin_location"], dest_loc(sh), car_mode[sh["carrier"]])
        sr_rows.append([uid("sr", sh["id"]), sh["id"], route_id[k], car_id[sh["carrier"]],
                        fmt(sh["created_at"])])
    write("shipment_routes.csv",
          ["id", "shipment_id", "route_id", "carrier_id", "created_at"], sr_rows)

    # -- route condition, weekly, reported-clock only -----------------------
    rc_rows = []
    for k in lanes:
        rows = lane_ships[k]
        for wk in weeks:
            r30, m30, v30, n30 = _ontime_stats(rows, wk, 30)
            r90, m90, _, _ = _ontime_stats(rows, wk, 90)
            if n30 == 0 and r90 is None:
                continue
            # condition_index: how much slower than the lane's own typical transit the last
            # 30 reported days ran. 0 = nominal, positive = degraded. Purely observational.
            typ = next((r[4] for r in rt_rows if r[0] == route_id[k]), 0.0) or 1.0
            cond = round(((m30 / typ) - 1.0), 4) if m30 else ""
            rc_rows.append([uid("rc", route_id[k], fmt(wk)), route_id[k], fmt(wk),
                            r30 if r30 is not None else "", r90 if r90 is not None else "",
                            m30 if m30 is not None else "", v30 if v30 is not None else "",
                            n30, cond, fmt(rec_at(wk)), "lane_telemetry"])
    write("route_conditions.csv",
          ["id", "route_id", "as_of_date", "on_time_rate_30d", "on_time_rate_90d",
           "transit_days_mean_30d", "transit_days_var_30d", "volume_30d", "condition_index",
           "recorded_at", "source"], rc_rows)

    # -- port congestion, weekly -------------------------------------------
    # A port's traffic is every shipment on every route that passes through it, so a transit
    # hub aggregates the corridors crossing it -- which is the mechanism by which one congested
    # port degrades otherwise unrelated shipments.
    port_ships: dict = {}
    for row in rp_rows:
        rid, pid = row[1], row[2]
        k = next(kk for kk in lanes if route_id[kk] == rid)
        port_ships.setdefault(pid, []).extend(lane_ships[k])
    pc_rows = []
    for pid, rows in sorted(port_ships.items()):
        for wk in weeks:
            r30, m30, v30, n30 = _ontime_stats(rows, wk, 30)
            if n30 < 3:
                continue
            arrivals = sum(1 for r in rows
                           if r["delivered_at"] and (r["delivered_rec"] or r["delivered_at"]) <= wk
                           and r["delivered_at"] >= wk - timedelta(days=30))
            late = [(r["delivered_at"] - r["eta"]).total_seconds() / 86400 for r in rows
                    if r["delivered_at"] and (r["delivered_rec"] or r["delivered_at"]) <= wk
                    and r["delivered_at"] >= wk - timedelta(days=30)
                    and r["delivered_at"] > r["eta"]]
            dwell = round(sum(late) / len(late), 4) if late else 0.0
            pc_rows.append([uid("pc", pid, fmt(wk)), pid, fmt(wk),
                            round(1.0 - r30, 4), dwell, arrivals, round(len(late) / max(1, arrivals), 4),
                            m30 if m30 is not None else "", fmt(rec_at(wk)), "port_authority"])
    write("port_congestion_history.csv",
          ["id", "port_id", "as_of_date", "congestion_index", "dwell_days_mean",
           "arrivals_30d", "late_rate_30d", "transit_days_mean_30d", "recorded_at", "source"],
          pc_rows)

    # ================================================================= DEPENDS_ON
    # The brief's "convert H_PORT / H_TRUCK / H_CUSTOMS into DEPENDS_ON edges". Built from the
    # OBSERVABLE checkpoint two lanes share, never from latent membership -- see the module
    # docstring. Emitted at ROUTE level, not shipment level: 112 lanes give ~10^3 coupling
    # edges, whereas 39k shipments would give ~10^8, and a shipment inherits its couplings in
    # one extra hop through MOVES_ON. Same structure, four orders of magnitude cheaper.
    def checkpoints(k):
        o, d, mode = k
        out = set()
        if mode == "sea":
            out.add(("port", o))                       # shared origin port
            hub = TRANSIT_HUBS.get((region_of(o), region_of(d)))
            if hub:
                out.add(("port", hub))                 # shared corridor hub
        if mode == "land" and country_of(o) in ("USA", "Mexico") and country_of(d) in ("USA", "Mexico", "MX"):
            out.add(("trucking", "NAFTA-CORRIDOR"))    # shared trucking corridor
        if country_of(o) == "Germany":
            out.add(("customs", "DE-CUSTOMS"))         # shared customs jurisdiction
        return out

    by_cp: dict = {}
    for k in lanes:
        for cp in checkpoints(k):
            by_cp.setdefault(cp, []).append(k)
    dep_rows = []
    for (channel, keyname), members in sorted(by_cp.items()):
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                dep_rows.append([uid("rdep", route_id[a], route_id[b], channel), route_id[a],
                                 route_id[b], channel, f"{channel}:{keyname}", len(members),
                                 fmt(t_start)])
    write("route_dependencies.csv",
          ["id", "route_id", "depends_on_route_id", "channel", "shared_key", "group_size",
           "created_at"], dep_rows)

    # ================================================================= REPLENISHED_BY
    # The edge shortage was missing: a (product, warehouse) stock position linked to the
    # inbound shipment currently resupplying it. `ordered_at` opens the link and the reported
    # delivery closes it, so the loader can pick the CURRENT one as of any t0 without ever
    # reading a future row.
    sh_by_id = {sh["id"]: sh for sh in shipments}
    pr_rows = []
    for r in replenish_links:
        sh = sh_by_id.get(r["shipment_id"])
        if sh is None:
            continue
        closed = (sh["delivered_rec"] or sh["delivered_at"]) if sh["delivered_at"] else None
        pr_rows.append([uid("prep", r["product_id"], r["warehouse_id"], r["shipment_id"]),
                        r["product_id"], r["warehouse_id"], r["shipment_id"],
                        fmt(r["ordered_at"]), fmt(sh["eta"]),
                        fmt(closed) if closed else "", fmt(rec_at(r["ordered_at"])), "wms_po"])
    write("product_replenishment.csv",
          ["id", "product_id", "warehouse_id", "shipment_id", "ordered_at", "eta",
           "closed_at", "recorded_at", "source"], pr_rows)

    # ================================================================= Warehouse role
    # The brief asks Warehouse to be SPLIT by role. It is emitted as a per-warehouse role
    # assignment plus per-role traffic features rather than as two node types, because this
    # world does not support the split: see `db/verify_enriched.py`, which measures the
    # inbound share of every warehouse and reports the spread. The role distinction is instead
    # carried where an RGCN can actually use it -- as two distinct RELATIONS (`REPLENISHES`
    # vs `DELIVERS_TO`) plus these features.
    wh_ships: dict = {}
    for sh in shipments:
        if sh.get("warehouse_id"):
            wh_ships.setdefault(sh["warehouse_id"], []).append(sh)
    wr_rows = []
    for wid, rows in sorted(wh_ships.items()):
        for wk in weeks:
            seen = [r for r in rows if r["created_at"] <= wk]
            if not seen:
                continue
            inb = [r for r in seen if r["supplier_id"]]
            out = [r for r in seen if not r["supplier_id"]]
            w30 = [r for r in seen if r["created_at"] >= wk - timedelta(days=30)]
            i30 = sum(1 for r in w30 if r["supplier_id"])
            open_in = sum(1 for r in inb if r["dispatched_at"] <= wk
                          and (not r["delivered_at"] or (r["delivered_rec"] or r["delivered_at"]) > wk))
            open_out = sum(1 for r in out if r["dispatched_at"] <= wk
                           and (not r["delivered_at"] or (r["delivered_rec"] or r["delivered_at"]) > wk))
            share = round(len(inb) / len(seen), 4)
            wr_rows.append([uid("whr", wid, fmt(wk)), wid, fmt(wk),
                            "replenishment" if share >= 0.5 else "fulfillment", share,
                            i30, len(w30) - i30, open_in, open_out,
                            fmt(rec_at(wk)), "wms_sync"])
    write("warehouse_role_features.csv",
          ["id", "warehouse_id", "as_of_date", "role", "inbound_share", "inbound_30d",
           "outbound_30d", "open_inbound", "open_outbound", "recorded_at", "source"], wr_rows)

    # ================================================================= FULFILLS
    # Warehouse -> Customer, via the order its outbound shipment fulfils. Purely a join over
    # rows already emitted, carried as its own relation so the path from a customer to the
    # stock position serving it is one hop instead of three.
    ord_cust = {o["id"]: o["customer_id"] for o in orders}
    ord_placed = {o["id"]: o["placed_at"] for o in orders}
    pairs: dict = {}
    for sh in shipments:
        if sh.get("order_id") and sh.get("warehouse_id"):
            cust = ord_cust.get(sh["order_id"])
            if cust:
                k = (sh["warehouse_id"], cust)
                p = pairs.setdefault(k, [0, sh["created_at"]])
                p[0] += 1
                p[1] = min(p[1], sh["created_at"])
    write("warehouse_customers.csv",
          ["id", "warehouse_id", "customer_id", "order_count", "first_seen_at", "created_at"],
          [[uid("whc", k[0], k[1]), k[0], k[1], v[0], fmt(v[1]), fmt(v[1])]
           for k, v in sorted(pairs.items())])

    return {"carriers": len(car_rows), "ports": len(locs), "routes": len(rt_rows),
            "route_ports": len(rp_rows), "route_dependencies": len(dep_rows),
            "shipment_routes": len(sr_rows), "product_replenishment": len(pr_rows),
            "warehouse_customers": len(pairs), "carrier_temporal_features": len(ctf),
            "route_conditions": len(rc_rows), "port_congestion_history": len(pc_rows),
            "warehouse_role_features": len(wr_rows)}
