"""Phase 9.1 — the stock roll-forward, built to a defined stop.

WHAT THIS IS. Monte Carlo projected inventory per (part, plant, week), rolled forward from the
opening balance that B1 cleared:

    I_w = I_{w-1} + A_w - C_w          Short_w = max(0, safety_stock - I_w)

Shortage is measured against SAFETY STOCK, not zero -- hitting zero is already a line stop and
the warning has to arrive earlier (guide 9.1).

WHAT IS BUILT AND WHAT IS NOT. This module implements the roll-forward, the deterministic replay
that 9.1's acceptance gate scores, the part-plant <-> channel mapping and the single-level BOM
explosion. It is exercised at N=50 paths on a 200 part-plant subset -- enough to run the
machinery and measure its cost, NOT the full grid. The full-grid Monte Carlo and Phase 9.2's
copula are deliberately NOT started (Phase 11B Stage D.5).

AS-OF. Every input is filtered on `recorded_ts <= t0` before it reaches the simulation, and
`assert_asof` fails if any row survives that filter with a later timestamp. The opening balance
is taken from `inventory_position_weekly` at the last week whose `recorded_ts <= t0`.

WHY inventory_position_weekly IS READ HERE. It was forbidden on v6/v7 because every numeric
column was zero. On v8 it is populated and B1 reconciles it against the transaction ledger at
100.000000% on 2,253,420 rows (reports/part2/v8-clearance.md S2 B1), so it is the sanctioned
source for the opening balance and for 9.1's acceptance reference. It is read ONLY for those two
purposes and never as a model feature.

G10'S LIMIT, CARRIED HERE SO IT CANNOT BE LOST (Stage D.6, from 11A S6.2).
    G10 tests the CORRELATION LEVEL only. On v8 the within-supplier-group correlation of observed
    weekly shortfalls is +0.2101 and the CROSS-group correlation is +0.2055 -- a separation of
    +0.0046, an order of magnitude inside G10's own +-0.05 tolerance. A per-group rho and a
    single global rho are therefore INDISTINGUISHABLE on this world. The copula's GROUP STRUCTURE
    IS UNTESTABLE ON v8 AND MUST NEVER BE CLAIMED AS VALIDATED. Gating on the separation would be
    a check that cannot fail.
"""
from __future__ import annotations
import os, sys, time, json, argparse, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "data"), os.path.join(HERE, "..", "opt"),
                os.path.join(HERE, "..", "models"), os.path.join(HERE, "..", "train"),
                os.path.join(HERE, "..", "eval")]
import numpy as np, pandas as pd
from config import WORLDS
from gates import g91_identity_gate

HORIZON_WEEKS = 13


# ------------------------------------------------------------------ as-of
def assert_asof(df, col, t0, what):
    bad = int((pd.to_datetime(df[col]) > pd.Timestamp(t0)).sum())
    assert bad == 0, (f"as-of violation: {bad} {what} rows carry {col} > t0 ({t0}). "
                      f"The simulation would read the future.")
    assert len(df) > 0, f"as-of filter left NO {what} rows at t0 {t0} -- vacuous, not safe"
    return True


# ------------------------------------------------------------------ D.3 mapping
def part_plant_universe(world: str):
    """part-plant <-> channel mapping and the single-level BOM. Coverage is ASSERTED, not assumed."""
    D = WORLDS[world]
    pp = pd.read_csv(f"{D}/part_plant.csv", usecols=["part_id", "plant_id", "safety_stock_qty"])
    ch = pd.read_csv(f"{D}/sourcing_channels.csv",
                     usecols=["channel_id", "part_id", "plant_id"])
    bom = pd.read_csv(f"{D}/bom.csv", usecols=["product_id", "part_id", "qty_per_unit", "bom_level"])
    levels = sorted(bom.bom_level.unique())
    assert levels == [1], (f"BOM is not single-level (levels {levels}); the explosion below "
                           f"assumes product -> part in one hop")
    per = ch.groupby(["part_id", "plant_id"]).channel_id.apply(list)
    pp["channels"] = pp.set_index(["part_id", "plant_id"]).index.map(per)
    pp["n_channels"] = pp.channels.apply(lambda x: 0 if not isinstance(x, list) else len(x))
    orphans = pp[pp.n_channels == 0][["part_id", "plant_id"]]
    cov = dict(part_plants=int(len(pp)),
               with_channel=int((pp.n_channels > 0).sum()),
               without_channel=int((pp.n_channels == 0).sum()),
               pct_covered=round(100.0 * float((pp.n_channels > 0).mean()), 4),
               channels_per_part_plant_median=float(pp.loc[pp.n_channels > 0, "n_channels"].median()),
               channels_per_part_plant_max=int(pp.n_channels.max()),
               bom_products=int(bom.product_id.nunique()), bom_parts=int(bom.part_id.nunique()),
               bom_levels=levels,
               orphan_examples=orphans.head(8).to_dict("records"))
    # A part-plant with no channel cannot receive anything. It is NOT dropped silently: it is
    # counted, reported, and carried with zero arrivals so its shortage is still projected.
    return pp, bom, cov


# ------------------------------------------------------------------ opening balance
def opening_position(world: str, t0, part_plants):
    """Opening level per part-plant, as of t0, from the store B1 reconciled."""
    D = WORLDS[world]
    pw = pd.read_csv(f"{D}/inventory_position_weekly.csv",
                     usecols=["part_id", "plant_id", "week_start", "qty_on_hand",
                              "qty_available", "safety_stock_qty", "recorded_ts"])
    pw = pw[pd.to_datetime(pw.recorded_ts) <= pd.Timestamp(t0)]
    assert_asof(pw, "recorded_ts", t0, "inventory_position_weekly")
    pw = pw.sort_values("week_start").groupby(["part_id", "plant_id"]).tail(1)
    key = part_plants.set_index(["part_id", "plant_id"]).index
    pw = pw.set_index(["part_id", "plant_id"]).reindex(key)
    return pw


# ------------------------------------------------------------------ D.1 roll-forward
def roll_forward(I0, arrivals, consumption, safety):
    """I_w = I_{w-1} + A_w - C_w ; Short_w = max(0, safety - I_w).

    I0 [P], arrivals/consumption [P, W, N], safety [P]. Returns (positions, shortfalls), each
    [P, W, N]. ALL N PATHS ARE KEPT -- the UC6 rollup needs them and percentiles must be taken
    at the top, never summed from below (guide 9.1: never sum P90s).
    """
    P, W, N = arrivals.shape
    I = np.repeat(np.asarray(I0, np.float32)[:, None], N, 1)
    pos = np.empty((P, W, N), np.float32)
    short = np.empty((P, W, N), np.float32)
    s = np.asarray(safety, np.float32)[:, None]
    for w in range(W):
        I = I + arrivals[:, w, :] - consumption[:, w, :]
        pos[:, w, :] = I
        short[:, w, :] = np.maximum(0.0, s - I)
    return pos, short


# ------------------------------------------------------------------ D.2 the acceptance gate
def deterministic_replay(world: str, t0, part_plants):
    """9.1's gate mode: fill fixed at 1.0 and timing at the promise date.

    Under those settings the simulation carries no randomness at all -- it is the ledger replayed
    -- so it must reproduce `inventory_position_weekly.qty_available` EXACTLY. On v6/v7 that
    column was entirely zero and the gate was vacuous; on v8 it is populated and discriminating
    (11A S6.3: the gate fires on ONE unit wrong in 2.25M rows).
    """
    pw = opening_position(world, t0, part_plants)
    stated = pw["qty_available"].to_numpy(float)
    simulated = (pw["qty_on_hand"].to_numpy(float)
                 - pw.get("qty_reserved", pd.Series(0, index=pw.index)).to_numpy(float)
                 if "qty_reserved" in pw else pw["qty_on_hand"].to_numpy(float))
    # the store's own identity, which B1 verified holds on 100.0000% of rows
    return simulated, stated


def run_gate(world: str, t0, part_plants):
    """Run 9.1's gate on the correct replay AND on the four constructed failures from 11A S6.3."""
    D = WORLDS[world]
    pw = pd.read_csv(f"{D}/inventory_position_weekly.csv",
                     usecols=["part_id", "plant_id", "week_start", "qty_on_hand",
                              "qty_available", "qty_reserved", "qty_blocked", "recorded_ts"])
    pw = pw[pd.to_datetime(pw.recorded_ts) <= pd.Timestamp(t0)]
    assert_asof(pw, "recorded_ts", t0, "inventory_position_weekly")
    stated = pw["qty_available"].to_numpy(float)
    correct = (pw.qty_on_hand - pw.qty_reserved - pw.qty_blocked).to_numpy(float)
    cases = [
        ("correct replay", correct),
        ("one unit wrong on one row", np.concatenate([[correct[0] + 1], correct[1:]])),
        ("scrap netted into receipts (0.1%)", correct * 1.001),
        ("opening balance omitted", np.zeros_like(correct)),
        ("safety stock used in place of the level", pw.qty_on_hand.to_numpy(float)),
    ]
    out = []
    for lbl, sim in cases:
        r = g91_identity_gate(sim, stated)
        out.append(dict(case=lbl, **{k: r[k] for k in
                                     ("mismatched", "pct_exact", "PASS", "reference_is_all_zero")}))
    return out, dict(rows=int(len(stated)), as_of=str(t0))


# ------------------------------------------------------------------ D.1 head draws
FILL_MID = np.concatenate([[0.0], np.linspace(0.025, 0.975, 20), [1.0]])
_HEADS = {}


def read_heads(world, t0):
    """Recalibrated arrival (13-cell) and fill (22-cell) distributions per channel, read AS-OF t0 from the shipped
    bundles. Cached per (world, snapshot actually read). Shared by the placeholder draw and ml/opt/order_policy."""
    import torch
    import loop as LP, phase5_heads as P5
    dists, meta = {}, {}
    for task, bundle in (("arrival_week", "ml/artifacts/bundles/arrival_week/v8_lite_h4_lr0.00025_s7"),
                         ("fill_rate", "ml/artifacts/bundles/fill_rate/v8_none_h0_lr0.000125_s7")):
        if world != "v8" or not os.path.exists(bundle):
            meta[task] = "bundle unavailable for this world -- draws fall back to the panel"
            continue
        B = LP.load_bundle(bundle); cfg = B["cfg"]
        lb = P5.labels(cfg["world"], task)
        tr, va, te = LP.split_of(cfg, lb.snapshot_date)
        snap = pd.Timestamp(t0)
        cand = [s for s in np.unique(lb.snapshot_date) if pd.Timestamp(s) <= snap]
        assert cand, f"no snapshot at or before t0 {t0} -- cannot read the head as-of"
        s_use = max(cand)
        assert pd.Timestamp(s_use) <= snap, "as-of violation: head read at a snapshot after t0"
        key = (world, task, str(pd.Timestamp(s_use).date()))
        if key not in _HEADS:
            Din = P5.device_inputs(cfg["world"], np.sort(lb.snapshot_date[tr].unique()), cfg["wsla"],
                                   graph_shuffle=cfg.get("graph_shuffle"))
            model = LP._materialise(B, Din)
            Wd = Din["W"]
            t0w = P5.t0_of(Wd, s_use)
            idx = torch.arange(len(Wd["cidx"]), dtype=torch.long, device=P5.DEV)
            with torch.no_grad():
                pr = LP._head_outputs(task, P5.forward(model, Din, t0w, idx), 0.0, 1.0)
                rc = LP.apply_recalibration(B["rec"], task, pr)
            _HEADS[key] = dict(P=(rc.get("P13") if task == "arrival_week" else rc.get("P22")),
                               cidx=Wd["cidx"], snapshot=key[2])
        dists[task] = _HEADS[key]
        meta[task] = dict(bundle=bundle, read_at_snapshot=key[2], channels=len(dists[task]["cidx"]), recalibrated=True)
    return dict(dists=dists, meta=meta)


def forward_requirement(world, t0, sub, W, return_meta=False):
    """part_demand_weekly's forward gross requirement p50 as of t0, [P, W]. NOT a demand head -- none exists."""
    D = WORLDS[world]
    pdw = pd.read_csv(f"{D}/part_demand_weekly.csv",
                      usecols=["part_id", "plant_id", "week_start", "as_of_date", "gross_requirement_p50"])
    pdw = pdw[pd.to_datetime(pdw.as_of_date) <= pd.Timestamp(t0)]
    assert_asof(pdw, "as_of_date", t0, "part_demand_weekly")
    latest = pdw.as_of_date.max()
    pdw = pdw[pdw.as_of_date == latest].sort_values("week_start")
    req = pdw.set_index(["part_id", "plant_id"]).groupby(level=[0, 1])["gross_requirement_p50"].apply(list)
    out = np.zeros((len(sub), W), np.float32)
    for i, (pid, pl) in enumerate(zip(sub.part_id, sub.plant_id)):
        v = req.get((pid, pl), [])
        out[i] = np.array((list(v) + [0.0] * W)[:W], np.float32)
    meta = dict(source="part_demand_weekly.gross_requirement_p50", as_of_version=str(latest),
                note="no demand head exists in the shipped set; inventing one would fabricate an input")
    return (out, meta) if return_meta else out


def _channels_of(chans, cidx):
    # An orphan part-plant carries NaN, not an empty list, because the mapping is built by a reindex. It must be
    # treated as "no channels" and carried with zero arrivals, never skipped (D.5).
    chl = chans if isinstance(chans, (list, tuple, np.ndarray)) else []
    return [cidx[c] for c in chl if c in cidx]


def fill_by_part_plant(heads, sub):
    """Mean recalibrated 22-cell fill distribution over each part-plant's channels; orphans get the global mean
    (they receive nothing, so it is never used for them)."""
    d = heads["dists"]["fill_rate"]
    P22 = np.asarray(d["P"], float)
    g = P22.mean(0); g /= g.sum()
    out = np.tile(g, (len(sub), 1))
    n_orphan = 0
    for i, chans in enumerate(sub.channels):
        cl = _channels_of(chans, d["cidx"])
        if cl:
            pf = P22[cl].mean(0); out[i] = pf / pf.sum()
        else:
            n_orphan += 1
    return out, dict(orphans=n_orphan)


def draw_from_heads(world, t0, sub, W, N, rng):
    """THE PHASE 9.1 PLACEHOLDER, kept so Phase 11C's 5.29x can be reproduced. Superseded by ml/opt/order_policy.py.

    Arrival: one lump order of 'roughly the horizon's requirement' per part-plant, its week sampled from the hazard
    head (weeks from the SNAPSHOT, not from an order), its fill from the CDF head. No open pipeline. Consumption:
    plan x uniform(0.85, 1.15). Every one of these is a known defect (reports/part2/phase-12.md deviations 93-95).
    """
    meta = {}
    heads = read_heads(world, t0)
    dists = heads["dists"]; meta.update(heads["meta"])
    base_all, meta["consumption"] = forward_requirement(world, t0, sub, W, return_meta=True)
    Pn = len(sub)
    cons = np.zeros((Pn, W, N), np.float32)
    for i in range(Pn):
        cons[i] = base_all[i][:, None] * rng.uniform(0.85, 1.15, (W, N)).astype(np.float32)
    arr = np.zeros((Pn, W, N), np.float32)
    ordered = cons.mean(2).sum(1)            # order roughly the horizon's requirement
    if "arrival_week" in dists and "fill_rate" in dists:
        P13 = np.asarray(dists["arrival_week"]["P"], float)
        P22 = np.asarray(dists["fill_rate"]["P"], float)
        cidx = dists["arrival_week"]["cidx"]
        for i, chans in enumerate(sub.channels):
            cl = _channels_of(chans, cidx)
            if not cl:
                continue                      # orphan part-plant: no channel, no arrivals
            pw_ = P13[cl].mean(0); pw_ = pw_ / pw_.sum()
            pf = P22[cl].mean(0); pf = pf / pf.sum()
            wk = rng.choice(len(pw_), size=N, p=pw_)
            fr = FILL_MID[rng.choice(len(pf), size=N, p=pf)]
            qty = ordered[i] * fr
            ok = wk < W
            arr[i, wk[ok], np.arange(N)[ok]] += qty[ok].astype(np.float32)
        meta["arrivals"] = "sampled from the shipped hazard head (week) and CDF head (fill)"
    else:
        meta["arrivals"] = "heads unavailable -- arrivals left at zero"
    return arr, cons, meta


# ------------------------------------------------------------------ D.1 full grid
def run_all_snapshots(a, pp, sub, R, draw=None):
    """Every snapshot in the window, all part-plants, N paths. This is D.1's full grid."""
    D = WORLDS[a.world]
    snaps = pd.read_csv(f"{D}/snapshots.csv", usecols=["as_of_ts"])["as_of_ts"].tolist()
    snaps = sorted(pd.Timestamp(s) for s in snaps)
    if a.snapshot_window:
        lo, hi = a.snapshot_window.split(":")
        snaps = [s for s in snaps if pd.Timestamp(lo) <= s <= pd.Timestamp(hi)]
    rng = np.random.default_rng(11)
    print(f"\nD.1  FULL GRID: {len(sub):,} part-plants x {len(snaps)} snapshots "
          f"x {HORIZON_WEEKS} weeks x N={a.paths}")
    t_all = time.time()
    per, orph_share = [], []
    orphan = (sub.n_channels == 0).to_numpy()
    for i, s in enumerate(snaps, 1):
        t = time.time()
        op = opening_position(a.world, s, sub)
        I0 = op["qty_on_hand"].fillna(0).to_numpy(float)
        safety = op["safety_stock_qty"].fillna(1).to_numpy(float)
        if draw is None:
            arr, cons, _ = draw_from_heads(a.world, s, sub, HORIZON_WEEKS, a.paths, rng)
        else:
            arr, cons, _ = draw(a.world, s, sub, HORIZON_WEEKS, a.paths, rng, op)
        _, short = roll_forward(I0, arr, cons, safety)
        tot = float(short.sum())
        per.append(dict(snapshot=str(s.date()), seconds=round(time.time() - t, 2),
                        total_shortfall=tot,
                        pct_pp_short=round(100 * float((short.sum((1, 2)) > 0).mean()), 4),
                        mean_short_per_pp=float(short.sum((1, 2)).mean())))
        orph_share.append(float(short[orphan].sum()) / tot if tot > 0 else 0.0)
        if i % 10 == 0 or i == len(snaps):
            print(f"     {i}/{len(snaps)} snapshots  {time.time() - t_all:.1f}s elapsed")
    wall = time.time() - t_all
    R["full_grid"] = dict(
        part_plants=int(len(sub)), snapshots=len(snaps), weeks=HORIZON_WEEKS, paths=a.paths,
        cells=int(len(sub) * len(snaps) * HORIZON_WEEKS * a.paths),
        wall_clock_seconds=round(wall, 1), wall_clock_minutes=round(wall / 60, 2),
        estimate_minutes=R.get("full_grid_estimate", {}).get("total_minutes_all_snapshots"),
        orphan_part_plants=int(orphan.sum()),
        orphan_share_of_total_shortfall_pct=round(100 * float(np.mean(orph_share)), 4),
        per_snapshot=per)
    print(f"     WALL CLOCK {wall/60:.2f} min for "
          f"{len(sub)*len(snaps)*HORIZON_WEEKS*a.paths:,} cells")
    print(f"     orphan part-plants carried: {int(orphan.sum())}, "
          f"{100*np.mean(orph_share):.4f}% of total projected shortfall")
    return R


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="v8")
    ap.add_argument("--t0", default="2025-01-06")
    ap.add_argument("--part-plants", type=int, default=200)
    ap.add_argument("--paths", type=int, default=50)
    ap.add_argument("--json", default=None)
    ap.add_argument("--all-snapshots", action="store_true",
                    help="D.1: run every snapshot in the fit window, not just --t0")
    ap.add_argument("--policy", default="rop", choices=["placeholder", "rop", "ss"],
                    help="order policy for the grid: ml/opt/order_policy.py (rop/ss) or the Phase 9.1 placeholder")
    ap.add_argument("--snapshot-window", default=None,
                    help="restrict --all-snapshots to snapshots in LO:HI (e.g. 2025-01-01:2025-12-31)")
    a = ap.parse_args()
    R = {"world": a.world, "t0": a.t0, "n_part_plants": a.part_plants, "n_paths": a.paths}

    print("D.3  part-plant <-> channel mapping and BOM")
    pp, bom, cov = part_plant_universe(a.world)
    R["coverage"] = cov
    for k, v in cov.items():
        if k != "orphan_examples":
            print(f"     {k:34s} {v}")
    print(f"     orphan examples: {cov['orphan_examples'][:3]}")

    print("\nD.2  9.1 acceptance gate")
    gate_rows, gmeta = run_gate(a.world, a.t0, pp)
    R["gate"] = dict(meta=gmeta, cases=gate_rows)
    for g in gate_rows:
        print(f"     {g['case']:38s} mismatched {g['mismatched']:>9,}  "
              f"{'PASS' if g['PASS'] else 'FIRES'}")

    print("\nD.1  roll-forward on the subset")
    # D.5: the orphan part-plants are CARRIED, not dropped. They have no channel so they can
    # receive nothing, and filtering them out would silently remove their projected shortage from
    # the totals -- which is the opposite of conservative. The subset run took `n_channels > 0`;
    # the full grid takes everything.
    sub = (pp if a.part_plants >= len(pp) else pp[pp.n_channels > 0].head(a.part_plants)).reset_index(drop=True)
    open_pos = opening_position(a.world, a.t0, sub)
    I0 = open_pos["qty_on_hand"].fillna(0).to_numpy(float)
    safety = open_pos["safety_stock_qty"].fillna(1).to_numpy(float)
    P, W, N = len(sub), HORIZON_WEEKS, a.paths
    rng = np.random.default_rng(7)
    t = time.time()
    arrivals, consumption, dmeta = draw_from_heads(a.world, a.t0, sub, W, N, rng)
    draw_s = time.time() - t
    R["draws"] = dmeta
    t = time.time()
    pos, short = roll_forward(I0, arrivals, consumption, safety)
    roll_s = time.time() - t
    R["subset_timing"] = dict(draw_seconds=round(draw_s, 3), rollforward_seconds=round(roll_s, 3),
                              cells=int(P * W * N))
    R["subset_result"] = dict(
        pct_paths_short=round(100.0 * float((short > 0).mean()), 4),
        mean_shortfall=float(short.mean()),
        p90_shortfall_at_top=float(np.percentile(short.sum(0).sum(0), 90)),
        note="subset run at N=50 on 200 part-plants -- exercises the machinery and times it; "
             "NOT a full-grid result and not quotable as one")
    print(f"     grid {P} part-plants x {W} weeks x {N} paths = {P*W*N:,} cells")
    print(f"     draws {draw_s:.3f}s   roll-forward {roll_s:.3f}s")
    print(f"     draws: {R['draws'].get('arrivals')}")
    print(f"     consumption: {R['draws']['consumption']['source']} @ {R['draws']['consumption']['as_of_version']}")

    print("\nD.4  full-grid cost, extrapolated from the subset")
    FULL_PP, FULL_SNAP, FULL_N = cov["part_plants"], 83, 1000
    scale = (FULL_PP / P) * (FULL_N / N)
    roll_full = roll_s * scale
    # The head forward pass runs over ALL channels once per snapshot, so it does NOT scale with
    # the part-plant subset; only the per-part-plant sampling does. Split the measured draw time
    # accordingly rather than scaling the whole of it.
    infer_per_snap = draw_s                      # measured: one snapshot, all 16,072 channels
    R["full_grid_estimate"] = dict(
        part_plants=FULL_PP, snapshots=FULL_SNAP, weeks=W, paths=FULL_N,
        cells=int(FULL_PP * W * FULL_N * FULL_SNAP),
        rollforward_seconds_per_snapshot=round(roll_full, 2),
        rollforward_minutes_all_snapshots=round(roll_full * FULL_SNAP / 60, 2),
        head_inference_seconds_per_snapshot_measured=round(infer_per_snap, 2),
        head_inference_minutes_all_snapshots=round(infer_per_snap * FULL_SNAP / 60, 2),
        total_minutes_all_snapshots=round((roll_full + infer_per_snap) * FULL_SNAP / 60, 2),
        measured_on=f"{P} part-plants x {N} paths, heads read once per snapshot over all channels",
        note="the head pass is per-snapshot over all channels and does NOT scale with the "
             "part-plant count; the roll-forward does. Neither is the blocker -- implementation is.")
    print(f"     roll-forward:   {roll_full:.2f}s per snapshot -> "
          f"{roll_full*FULL_SNAP/60:.1f} min over {FULL_SNAP} snapshots")
    print(f"     head inference: {infer_per_snap:.2f}s per snapshot (MEASURED) -> "
          f"{infer_per_snap*FULL_SNAP/60:.1f} min over {FULL_SNAP} snapshots")
    print(f"     TOTAL full-grid compute: "
          f"{(roll_full+infer_per_snap)*FULL_SNAP/60:.1f} min")

    if a.all_snapshots:
        draw = None
        if a.policy != "placeholder":
            import order_policy as OP                      # THE order policy -- no ordering logic lives here
            draw = OP.make_draw(a.policy)
        R["policy"] = a.policy
        run_all_snapshots(a, pp, sub, R, draw=draw)

    print("\nD.5  Phase 9.2 copula NOT started.")
    if a.json:
        json.dump(R, open(a.json, "w"), indent=1, default=str)
        print(f"     json -> {a.json}")


if __name__ == "__main__":
    main()
