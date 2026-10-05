"""Phase 23B Stage 1 -- leak audit of predict-the-rescue (B1a LightGBM, B1b neural) and of the shortage simulation's inputs.

(a) INPUTS, read from code and stored configs (never from memory):
    B1a  the stored feature list in ml/artifacts/phase17/b1a_lgbm.json (written by ml/train/phase17_b1.py run_lgbm from
         phase7_fit.World.part_plant_features: per part-plant mean and min of every live panel column at row t0, plus graph counts),
    B1b  the stored rescue_week bundle configs + phase5_heads.device_inputs (every live panel column and its missing-indicator,
         a 52-week window ending at row t0), and
    SIM  ml/sim/montecarlo.read_heads (the two neural heads it runs, read from the source text).
(b) Intersection with Phase 22's nine LEAKING columns (ml/data/clean_panel.LEAKING), BY NAME, and the FUTURE-POISON TEST on the
    columns B1a actually reads: Phase 22's scan code is imported unchanged (phase22_leakscan.Sources / rebuild / poison, H_week);
    for each sampled snapshot week every later-visible source row is poisoned and B1a's OWN part-plant features (mean / min) at row t
    are recomputed. A feature LEAKS if any part-plant's value moves.
(c) The simulation's inputs, LEAKY / CLEAN per the Phase 22 restatement.
GATE G1 and its constructed failing case: the NL feature list must pass; the NL list with `lead_time_actual_days_mean` re-added
(a deliberately leaked arm) must be FLAGGED by name AND by poison.

AUDIT INSTRUMENT: reads the generator's _sim.npz through phase22_leakscan only; nothing here is a model input.

  python ml/eval/phase23b_audit.py      -> ml/artifacts/phase23b/stage1_audit.json
"""
from __future__ import annotations
import os, sys, json, time, inspect
import phase12_common as C
import numpy as np, pandas as pd
import config
import phase21_paths as PP
import phase22_leakscan as LS
from clean_panel import LEAKING

OUT = "phase23b"
SNAP_FROM, SNAP_TO = "2019-01-01", "2025-12-31"


def source_column(feature):
    """The panel column a feature derives from (None for graph counts and the week offset)."""
    f = feature[4:] if feature.startswith("obs:") else feature
    for suf in ("_mean", "_min"):
        if f.endswith(suf) and f[: -len(suf)] in LS.COLS:
            return f[: -len(suf)]
    return f if f in LS.COLS else None


def name_flags(features):
    return sorted({source_column(f) for f in features if source_column(f) in LEAKING})


def b1a_inputs():
    j = json.load(open(os.path.join(C.ART, "phase17", "b1a_lgbm.json")))
    return dict(source="ml/artifacts/phase17/b1a_lgbm.json (stored by phase17_b1.run_lgbm; stamp %s)" % j["stamp"]["code_version"],
                features=j["features"], n_features=j["n_features"])


def b1b_inputs():
    out = {}
    meta = json.load(open(os.path.join(config.CACHE, "v8", "meta.json")))
    names = list(meta["cols"]) + ["obs:" + c for c in meta["nullable"]]
    for s in (7, 17):
        p = os.path.join(C.BUND, "rescue_week", f"v8_mp_h1_lr0.000125_s{s}", "config.json")
        cfg = json.load(open(p))
        out[f"s{s}"] = dict(config=os.path.relpath(p, C.REPO), stamp=cfg["stamps"]["code_version"], arch=cfg["arch"], depth=cfg["depth"],
                            row_input=cfg.get("row_input"))
    return dict(source="rescue_week bundle configs + phase5_heads.device_inputs (panel columns + missing indicators, 52-week window "
                       "ending at row t0; cache/v8/meta.json)", features=names, n_features=len(names), bundles=out)


def sim_inputs():
    import montecarlo as MC
    src = inspect.getsource(MC.read_heads)
    heads = [("arrival_week", "ml/artifacts/bundles/arrival_week/v8_lite_h4_lr0.00025_s7"),
             ("fill_rate", "ml/artifacts/bundles/fill_rate/v8_none_h0_lr0.000125_s{fill_seed}  (fill seeds 7, 17, 27, 37, 47)")]
    for needle in ("bundles/arrival_week/v8_lite_h4_lr0.00025_s7", "bundles/fill_rate/v8_none_h0_lr0.000125_s{fill_seed}"):
        assert needle in src, f"read_heads no longer names {needle}: re-read the simulation's inputs"
    meta = json.load(open(os.path.join(config.CACHE, "v8", "meta.json")))
    names = list(meta["cols"]) + ["obs:" + c for c in meta["nullable"]]
    return dict(source="ml/sim/montecarlo.read_heads (called by ml/opt/order_policy.py@9e2d59d make_draw, used by phase14_sim / phase15_sim)",
                heads=[dict(task=t, bundle=b, model="neural (phase5_heads.HeadNet via loop._materialise), recalibrated",
                            reads=names) for t, b in heads],
                other_inputs=["po_lines / grn_lines / part_plant / sourcing_channels (lead pmf, policy parameters)",
                              "inventory_position_weekly (opening position and open pipeline as of t0: the sanctioned simulation path, "
                              "evaluation only)", "part_demand_weekly (forward requirement as of t0: the plan, not a model)"],
                capacity_head="none -- the simulation reads no capacity prediction")


def pp_index(world):
    ch = pd.read_csv(os.path.join(config.WORLDS[world], "sourcing_channels.csv"), usecols=["part_id", "plant_id"])
    pk = (ch.part_id + "|" + ch.plant_id).to_numpy()
    keys = sorted(set(pk)); m = {k: i for i, k in enumerate(keys)}
    return np.array([m[k] for k in pk]), len(keys)


def pp_agg(x, ppc, n_pp):
    s = pd.DataFrame({"pp": ppc, "v": x}).groupby("pp").v
    return s.mean().reindex(range(n_pp)).to_numpy(), s.min().reindex(range(n_pp)).to_numpy()


def changed(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float((~((np.isnan(a) & np.isnan(b)) | (np.abs(np.nan_to_num(a, nan=-999) - np.nan_to_num(b, nan=-999)) <= 1e-9))).mean())


def poison_test(features, weeks, S, R, ppc, n_pp, rng):
    """Share of part-plant values that move under H_week poison, per feature (panel-derived features only)."""
    cols = sorted({source_column(f) for f in features if source_column(f) is not None})
    res = {f: [] for f in features if source_column(f) is not None}
    for t in weeks:
        Rp = LS.rebuild(S, **LS.poison(S, int(t), False, rng))
        for c in cols:
            ma, mi = pp_agg(R[c][:, t], ppc, n_pp); pa, pi_ = pp_agg(Rp[c][:, t], ppc, n_pp)
            for f in res:
                if source_column(f) != c:
                    continue
                if f.startswith("obs:"):
                    res[f].append(changed(np.isfinite(R[c][:, t]), np.isfinite(Rp[c][:, t])))
                elif f.endswith("_min"):
                    res[f].append(changed(mi, pi_))
                else:
                    res[f].append(changed(ma, pa))
    return {f: dict(max_share_changed=float(max(v)), mean_share_changed=float(np.mean(v)), leaks=bool(max(v) > 0)) for f, v in res.items()}


def main():
    PP.register()
    st = C.require_clean()
    os.makedirs(os.path.join(C.ART, OUT), exist_ok=True)
    t0 = time.time()
    A = b1a_inputs(); Bn = b1b_inputs(); Sm = sim_inputs()
    S = LS.Sources("v8")
    R = LS.rebuild(S)
    rep = LS.reproduce(S, R, "v8")
    print("reproduction of the stored panel:", {c: round(v["value_agrees"], 4) for c, v in rep.items()}, flush=True)
    snaps = pd.to_datetime(pd.read_csv(os.path.join(config.WORLDS["v8"], "snapshots.csv")).as_of_ts)
    snaps = snaps[(snaps >= SNAP_FROM) & (snaps <= SNAP_TO)]
    weeks_all = ((snaps - S.W0).dt.days // 7).to_numpy()
    weeks = weeks_all[np.linspace(0, len(weeks_all) - 1, 6).astype(int)]
    ppc, n_pp = pp_index("v8")
    rng = np.random.default_rng(23)
    nl = [f for f in A["features"] if source_column(f) not in LEAKING]
    leaked_arm = nl + ["lead_time_actual_days_mean"]
    t1 = time.time()
    p_b1a = poison_test(A["features"], weeks, S, R, ppc, n_pp, rng)
    print(f"B1a poison test: {time.time() - t1:.0f}s", flush=True)
    rng2 = np.random.default_rng(23)
    p_case = poison_test(leaked_arm, weeks[:2], S, R, ppc, n_pp, rng2)
    g1 = dict(nl_passes_by_name=not name_flags(nl), nl_passes_by_poison=not any(p_b1a[f]["leaks"] for f in nl if f in p_b1a),
              leaked_arm_flagged_by_name=name_flags(leaked_arm) == ["lead_time_actual_days"],
              leaked_arm_flagged_by_poison=bool(p_case["lead_time_actual_days_mean"]["leaks"]))
    g1["gate_valid"] = all(g1.values())
    assert g1["leaked_arm_flagged_by_name"] and g1["leaked_arm_flagged_by_poison"], f"G1's constructed failing case did not fire: {g1}"
    out = dict(stamp=st, machine="Windows 11, RTX 3050 Ti 4 GB, CPU only (HADES_DEVICE=cpu)", concurrency=1,
               leaking_columns_phase22=LEAKING, scan_code="ml/eval/phase22_leakscan.py (imported unchanged)",
               panel_reproduction=rep, poison_weeks=[str((S.W0 + pd.Timedelta(days=7 * int(t))).date()) for t in weeks],
               B1a=dict(**A, leaking_by_name=name_flags(A["features"]),
                        leaking_features=[f for f in A["features"] if source_column(f) in LEAKING],
                        poison=p_b1a, leaking_by_poison=sorted({source_column(f) for f, v in p_b1a.items() if v["leaks"]}),
                        clean_features_by_poison=[f for f, v in p_b1a.items() if not v["leaks"]]),
               B1b=dict(**Bn, leaking_by_name=name_flags(Bn["features"])),
               SIM=dict(**Sm, leaking_by_name=name_flags(Sm["heads"][0]["reads"]),
                        verdict="LEAKY: both heads (arrival s7, fill s{7..47}) read the panel incl. all nine LEAKING columns; Phase 22 "
                                "restated both on clean inputs (arrival lateness AUC -0.0117, fill P(full) AUC -0.021, UC2b @5% -0.058)"),
               G1=g1, seconds=time.time() - t0)
    C.dump(out, f"{OUT}/stage1_audit.json")
    print(json.dumps(dict(B1a_leaking_by_name=out["B1a"]["leaking_by_name"], B1a_leaking_by_poison=out["B1a"]["leaking_by_poison"],
                          B1b_leaking_by_name=out["B1b"]["leaking_by_name"], SIM=out["SIM"]["leaking_by_name"], G1=g1,
                          seconds=round(out["seconds"])), indent=1))


if __name__ == "__main__":
    main()
