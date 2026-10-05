"""Phase 24 Stage 2 -- the shortage simulation restated on CLEAN model inputs. New file; a thin wrapper.

The harness is imported UNCHANGED: ml/sim/montecarlo.py (opening_position, roll_forward, read_heads), ml/opt/order_policy.py
pinned at 9e2d59d (deviation 122; loaded from git exactly as phase14_sim.b2_order_policy does, but written under
ml/artifacts/phase24/ so no stored cache file is rewritten), phase14_sim (snapshots, store_labels, W, N, SEEDS) and the
driver loop of ml/sim/phase15_sim.py (its UC5 rows; the UC8 universes are not needed here).

The ONLY change is which two neural heads `montecarlo.read_heads` loads. Its source is taken with inspect, the two
hard-coded bundle path literals are replaced, and the result is executed in montecarlo's own namespace. Every other
character of the function is asserted unchanged. Arms:
  leaky  the original paths (no substitution): must REPRODUCE the stored simulation (gate G2), else STOP
  clean  phase22/bundles/arrival_week/v8clean_lite_h4_lr0.00025_s7 and phase22/bundles/fill_rate/v8clean_none_h0_lr0.000125_s{s}
Every bundle is checked before use (gate G1): task / world / seed / arch / depth / lr, directory == its identity module's
bundle_name, and checkpoint.pt SHA-1 == the value pinned by phase24_inputs.py.

  python ml/eval/phase24_sim_clean.py run --arm leaky      (writes ml/artifacts/phase24/sim/leaky/)
  python ml/eval/phase24_sim_clean.py run --arm clean
  python ml/eval/phase24_sim_clean.py score                 (gates, Q1-Q4, block intervals -> phase24/stage2_sim.json)

inventory_position_weekly: read by montecarlo.opening_position / order_policy@9e2d59d (as-of t0, the sanctioned simulation
path, unchanged) and by phase14_sim.store_labels / phase13_s1.store_2025 (evaluation references, unchanged). Never a feature.
part_demand_weekly: read by montecarlo.forward_requirement inside the unchanged harness (as-of t0), as in every prior run.
"""
from __future__ import annotations
import os, sys, json, time, hashlib, inspect, subprocess, importlib.util, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "sim"), os.path.join(HERE, "..", "opt"),
                os.path.join(HERE, "..", "data"), os.path.join(HERE, "..", "train")]
import numpy as np, pandas as pd
import phase12_common as C

ART24 = os.path.join(C.ART, "phase24")
SIM24 = os.path.join(ART24, "sim")
STORED_SIM = os.path.join(C.ART, "phase15_sim")
SEEDS = (7, 17, 27, 37, 47)
B2_COMMIT = "9e2d59d"
LEAKY_A = '"ml/artifacts/bundles/arrival_week/v8_lite_h4_lr0.00025_s7"'
LEAKY_F = 'f"ml/artifacts/bundles/fill_rate/v8_none_h0_lr0.000125_s{fill_seed}"'
CLEAN_A = '"ml/artifacts/phase22/bundles/arrival_week/v8clean_lite_h4_lr0.00025_s7"'
CLEAN_F = 'f"ml/artifacts/phase22/bundles/fill_rate/v8clean_none_h0_lr0.000125_s{fill_seed}"'
EXPECT = {"arrival_week": dict(arch="lite", depth=4, lr=0.00025), "fill_rate": dict(arch="none", depth=0, lr=0.000125)}
BOOT, BOOT_SEED = 1000, 2024


# ================================================================== G1 identity
class IdentityMismatch(Exception):
    pass


def slot_path(arm, task, seed):
    lit = {("leaky", "arrival_week"): LEAKY_A, ("leaky", "fill_rate"): LEAKY_F,
           ("clean", "arrival_week"): CLEAN_A, ("clean", "fill_rate"): CLEAN_F}[(arm, task)]
    return eval(lit, {"fill_seed": seed})


def _sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_slot(arm, task, seed, path):
    """RAISE unless `path` is the bundle this (arm, task, seed) slot must load."""
    import artifact_identity as AI, phase19_identity as PI
    pins = json.load(open(os.path.join(ART24, "inputs.json")))
    key = f"{task}|s{seed}"
    pin = pins["leaky_bundles" if arm == "leaky" else "clean_bundles"][key]
    world, namer = ("v8", AI.bundle_name) if arm == "leaky" else ("v8clean", PI.bundle_name)
    p = os.path.join(C.REPO, path)
    if not os.path.exists(os.path.join(p, "config.json")):
        raise IdentityMismatch(f"{arm} {key}: no bundle at {path}")
    cfg = json.load(open(os.path.join(p, "config.json")))
    want = dict(task=task, world=world, seed=seed, **EXPECT[task])
    bad = {k: (cfg.get(k), v) for k, v in want.items() if cfg.get(k) != v}
    if bad:
        raise IdentityMismatch(f"{arm} {key}: {path} config differs {bad}")
    if os.path.basename(p) != namer(cfg):
        raise IdentityMismatch(f"{arm} {key}: directory is not {namer.__module__}.bundle_name ({namer(cfg)})")
    if _sha1(os.path.join(p, "checkpoint.pt")) != pin["checkpoint_sha1"]:
        raise IdentityMismatch(f"{arm} {key}: checkpoint.pt SHA-1 differs from the Stage 1 pin")
    return dict(path=path, checkpoint_sha1=pin["checkpoint_sha1"])


def g1_failing_cases():
    out = {}
    for name, args in (("fill_s17_in_fill_s7_slot", ("clean", "fill_rate", 7, slot_path("clean", "fill_rate", 17))),
                       ("leaky_arrival_in_clean_arrival_slot", ("clean", "arrival_week", 7, slot_path("leaky", "arrival_week", 7)))):
        try:
            verify_slot(*args); out[name] = "DID NOT RAISE"
        except IdentityMismatch as e:
            out[name] = f"RAISED: {e}"
    out["all_fire"] = all(v.startswith("RAISED") for v in out.values())
    return out


# ================================================================== the swap
def install(arm):
    """Returns the read_heads source actually installed and the substitution record."""
    import montecarlo as MC
    src = inspect.getsource(MC.read_heads)
    rec = dict(arm=arm, original_sha1=hashlib.sha1(src.encode()).hexdigest())
    if arm == "leaky":
        rec.update(substituted=False); return rec
    assert src.count(LEAKY_A) == 1 and src.count(LEAKY_F) == 1, "read_heads no longer carries the two expected literals"
    new = src.replace(LEAKY_A, CLEAN_A).replace(LEAKY_F, CLEAN_F)
    # every character outside the two literals is unchanged
    assert new.replace(CLEAN_A, LEAKY_A).replace(CLEAN_F, LEAKY_F) == src
    ns = MC.__dict__
    exec(compile(new, MC.__file__, "exec"), ns)
    assert MC.read_heads.__code__.co_filename == MC.__file__
    rec.update(substituted=True, installed_sha1=hashlib.sha1(new.encode()).hexdigest(),
               replaced={LEAKY_A: CLEAN_A, LEAKY_F: CLEAN_F})
    return rec


def order_policy_b2():
    """phase14_sim.b2_order_policy, unchanged in effect: the 9e2d59d source from git, written under phase24/."""
    src = subprocess.check_output(["git", "-C", C.REPO, "show", f"{B2_COMMIT}:ml/opt/order_policy.py"], text=True)
    path = os.path.join(ART24, f"order_policy_{B2_COMMIT}.py")
    open(path, "w").write(src)
    spec = importlib.util.spec_from_file_location("order_policy_b2", path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m, hashlib.sha1(src.encode()).hexdigest()


# ================================================================== run (phase15_sim.py's driver, UC5 rows)
def run(arm):
    st = C.require_clean()
    if arm == "clean":
        import phase22_train as T22
        T22.register_clean()                       # in-process: v8clean reads v8's CSVs, its own cache dir
    import montecarlo as MC
    from phase14_sim import snapshots, store_labels, W, N, SEEDS as S14
    assert tuple(S14) == SEEDS
    ident = {f"{t}|s{s}": verify_slot(arm, t, s, slot_path(arm, t, s)) for t, ss in (("arrival_week", (7,)), ("fill_rate", SEEDS)) for s in ss}
    swap = install(arm)
    OPb, op_sha = order_policy_b2()
    stored = json.load(open(os.path.join(C.ART, "phase12_b2_validate.json")))
    stored_s = json.load(open(os.path.join(C.ART, "phase12_b2_seeds.json")))
    ref = {7: {p["snapshot"]: p["below_ss"] for p in stored["policy_rop"]["per_snapshot"]}}
    for s in (17, 27, 37, 47):
        ref[s] = {p["snapshot"]: p["below_ss"] for p in stored_s[f"policy_rop_fill_s{s}"]["per_snapshot"]}
    pp, _, _ = MC.part_plant_universe("v8")
    sub = pp.reset_index(drop=True)
    labels = store_labels()
    parts, plant = sub.part_id.to_numpy(), sub.plant_id.to_numpy()
    out = os.path.join(SIM24, arm); os.makedirs(out, exist_ok=True)
    import phase5_heads as P5
    log = dict(stamp=st, arm=arm, identity=ident, swap=swap, order_policy_sha1=op_sha, device=str(P5.DEV), gate=[], seconds={})
    t_all = time.time()
    for fold, (lo, hi) in (("test", ("2025-01-01", "2025-12-31")), ("val", ("2024-01-01", "2024-12-31"))):
        for seed in SEEDS:
            t1 = time.time()
            rng = np.random.default_rng(11)
            draw = OPb.make_draw("rop", fill_seed=seed)
            A = {k: [] for k in ("p", "obs", "pre", "acted", "pp", "week", "snap")}
            for k, t0 in enumerate(snapshots(lo, hi)):
                op = MC.opening_position("v8", t0, sub)
                arr, cons, _ = draw("v8", t0, sub, W, N, rng, op)
                I0 = op.qty_on_hand.fillna(0).to_numpy(float); ss = op.safety_stock_qty.fillna(1).to_numpy(float)
                pos, _ = MC.roll_forward(I0, arr, cons, ss)
                frac = float((pos < ss[:, None, None]).mean())
                if fold == "test":
                    g = dict(seed=seed, t0=str(t0.date()), regenerated=frac, stored=ref[seed][str(t0.date())],
                             exact=frac == ref[seed][str(t0.date())])
                    log["gate"].append(g)
                    if arm == "leaky":
                        assert g["exact"], f"G2 STOP: leaky wrapper does not reproduce seed {seed} {t0.date()}: {frac} != {g['stored']}"
                p_short = (pos < ss[:, None, None]).mean(2)
                weeks = [t0 + pd.Timedelta(days=7 * (w + 1)) for w in range(W)]
                weeks = [w - pd.Timedelta(days=w.weekday()) for w in weeks]
                idx = pd.MultiIndex.from_arrays([np.repeat(parts, W), np.repeat(plant, W), np.tile(weeks, len(sub))])
                lab = labels.reindex(idx); ok = lab.obs_below.notna().to_numpy()
                A["p"].append(p_short.reshape(-1)[ok]); A["obs"].append(lab.obs_below.to_numpy()[ok].astype(bool))
                A["pre"].append(lab.pre_below.to_numpy()[ok].astype(bool)); A["acted"].append(lab.acted.to_numpy()[ok].astype(bool))
                A["pp"].append(np.repeat(np.arange(len(sub)), W)[ok]); A["week"].append(np.tile(np.array(weeks, "datetime64[D]"), len(sub))[ok])
                A["snap"].append(np.full(int(ok.sum()), k, np.int8))
                del pos, arr, cons
                print(f"   {arm} {fold} seed {seed} {t0.date()} below {frac:.5f}", flush=True)
            np.savez_compressed(os.path.join(out, f"{fold}_s{seed}.npz"), **{k: np.concatenate(v) for k, v in A.items()})
            log["seconds"][f"{fold}_s{seed}"] = time.time() - t1
    # the heads actually read: every cached key must name this arm's bundles
    log["heads_read"] = sorted({k[3] for k in MC._HEADS})
    want = {slot_path(arm, "arrival_week", 7)} | {slot_path(arm, "fill_rate", s) for s in SEEDS}
    log["heads_read_are_this_arm"] = set(log["heads_read"]) == want
    assert log["heads_read_are_this_arm"], f"heads read {log['heads_read']} != {sorted(want)}"
    log["gate_exact"] = sum(g["exact"] for g in log["gate"]); log["n_gate"] = len(log["gate"])
    log["seconds_total"] = time.time() - t_all
    json.dump(log, open(os.path.join(out, "run_log.json"), "w"), indent=1, default=str)
    print(f"{arm}: gate {log['gate_exact']}/{log['n_gate']} exact, {log['seconds_total'] / 60:.1f} min, device {log['device']}")


# ================================================================== score
KEYS = ("p", "obs", "pre", "acted", "pp", "week")


def arrays_equal(a, b):
    return all(np.array_equal(a[k], b[k]) for k in KEYS)


def load_arm(arm, seeds=SEEDS):
    if tuple(sorted(seeds)) != SEEDS:
        raise ValueError(f"G4: scorer requires all five seeds {SEEDS}; got {tuple(seeds)}")
    return {(f, s): dict(np.load(os.path.join(SIM24, arm, f"{f}_s{s}.npz"))) for f in ("val", "test") for s in seeds}


def prec_at_min_recall_tau(sv, yv, r=0.20):
    """phase17_b1.score.prec_at_min_recall, the threshold part (nested there, so reproduced here line for line)."""
    import phase15 as P15
    ss, yy, tp, n = P15.sorted_cum(sv, yv)
    last = np.r_[np.flatnonzero(np.diff(ss) != 0), len(ss) - 1]
    rec, prec = tp[last] / yy.sum(), tp[last] / n[last]
    ok = rec >= r
    i = last[np.flatnonzero(ok)[np.argmax(prec[ok])]]
    return float(ss[i]), float(tp[i] / n[i]), float(tp[i] / yy.sum())


def thresholds(Z):
    from phase14_score import fit_tau
    th = {}
    for s in SEEDS:
        v = Z[("val", s)]
        th[s] = dict(tau_a=fit_tau(v["p"], v["obs"].astype(int), "f1")[0], tau_q3=prec_at_min_recall_tau(v["p"], v["acted"].astype(int))[0])
    return th


def per_snapshot_sums(Z, th, fracs):
    """[seed, snapshot] sums for the block bootstrap."""
    out = {}
    for s in SEEDS:
        t = Z[("test", s)]; k = t["snap"].astype(int); K = 9
        a3 = t["p"] >= th[s]["tau_q3"]; y = t["acted"]
        rescued = t["pre"] & ~t["obs"]; flagged = rescued & (t["p"] >= th[s]["tau_a"])
        out[s] = dict(alerts=np.bincount(k, a3, K), tp=np.bincount(k, a3 & y, K), pos=np.bincount(k, y, K),
                      rescued=np.bincount(k, rescued, K), flagged=np.bincount(k, flagged, K), frac=np.asarray(fracs[s], float))
    return out


def quantities(S, refp, idx=None):
    q = {}
    for s, d in S.items():
        g = (lambda x: x[idx].sum()) if idx is not None else (lambda x: x.sum())
        q[s] = dict(q1=(d["frac"][idx].mean() if idx is not None else d["frac"].mean()) / refp,
                    q2_missed=1 - g(d["flagged"]) / g(d["rescued"]), q3_precision=g(d["tp"]) / g(d["alerts"]),
                    q3_recall=g(d["tp"]) / g(d["pos"]))
    return {k: float(np.mean([q[s][k] for s in S])) for k in ("q1", "q2_missed", "q3_precision", "q3_recall")}


def score():
    import phase15 as P15, phase13_s1 as S1
    from phase20_decisions import reachable_and_class
    st = C.require_clean()
    R = dict(stamp=st)
    stored = {(f, s): dict(np.load(os.path.join(STORED_SIM, f"{f}_s{s}.npz"))) for f in ("val", "test") for s in SEEDS}
    L, Cl = load_arm("leaky"), load_arm("clean")
    logs = {a: json.load(open(os.path.join(SIM24, a, "run_log.json"))) for a in ("leaky", "clean")}
    # ---- G4 failing case
    try:
        load_arm("leaky", SEEDS[:4]); R["G4_failing_case"] = "DID NOT RAISE"
    except ValueError as e:
        R["G4_failing_case"] = f"RAISED: {e}"
    # ---- G1 failing cases
    R["G1"] = dict(leaky_identity=logs["leaky"]["identity"], clean_identity=logs["clean"]["identity"],
                   heads_read=dict(leaky=logs["leaky"]["heads_read"], clean=logs["clean"]["heads_read"]),
                   failing_cases=g1_failing_cases())
    # ---- G2 reproduction
    rows_equal = {f"{f}_s{s}": arrays_equal(L[(f, s)], stored[(f, s)]) for f in ("val", "test") for s in SEEDS}
    pert = {k: v.copy() for k, v in L[("test", 7)].items()}; pert["p"][0] += 1e-6
    clean_equal = {f"{f}_s{s}": arrays_equal(Cl[(f, s)], stored[(f, s)]) for f in ("val", "test") for s in SEEDS}
    R["G2"] = dict(leaky_gate_exact=f"{logs['leaky']['gate_exact']}/{logs['leaky']['n_gate']}", leaky_rows_equal_stored=rows_equal,
                   failing_case_perturbed_reports_not_equal=not arrays_equal(pert, stored[("test", 7)]),
                   clean_rows_equal_stored=clean_equal, clean_differs_from_stored=not any(clean_equal.values()),
                   clean_gate_exact=f"{logs['clean']['gate_exact']}/{logs['clean']['n_gate']}",
                   devices=dict(leaky=logs["leaky"]["device"], clean=logs["clean"]["device"]))
    # ---- shared pieces
    pw = S1.store_2025(); refp = S1.reference(pw, +1.0)
    fr = {a: {s: [g["regenerated"] for g in logs[a]["gate"] if g["seed"] == s] for s in SEEDS} for a in ("leaky", "clean")}
    th = {a: thresholds(Z) for a, Z in (("leaky", L), ("clean", Cl))}
    sums = {a: per_snapshot_sums(Z, th[a], fr[a]) for a, Z in (("leaky", L), ("clean", Cl))}
    out = {}
    for a, Z in (("leaky", L), ("clean", Cl)):
        q1 = S1.ratios({s: float(np.mean(fr[a][s])) for s in SEEDS}, refp)
        moved = []
        for s in SEEDS:
            t = Z[("test", s)]; rescued = t["pre"] & ~t["obs"]; pred = t["p"] >= th[a][s]["tau_a"]
            moved.append(dict(seed=s, tau_a=th[a][s]["tau_a"], rescued_weeks=int(rescued.sum()), flagged=int((rescued & pred).sum()),
                              missed=int((rescued & ~pred).sum()), missed_share=float((rescued & ~pred).sum() / rescued.sum())))
        q3 = []
        for s in SEEDS:
            v, t = Z[("val", s)], Z[("test", s)]
            tau, vp, vr = prec_at_min_recall_tau(v["p"], v["acted"].astype(int))
            a_ = P15.apply(t["p"], t["acted"].astype(int), tau)
            q3.append(dict(seed=s, tau=tau, val_precision=vp, val_recall=vr, test_precision=a_["precision"], test_recall=a_["recall"],
                           test_coverage=a_["coverage"]))
        uc5 = P15.analyse([(Z[("val", s)]["p"], Z[("val", s)]["obs"].astype(int), Z[("test", s)]["p"], Z[("test", s)]["obs"].astype(int)) for s in SEEDS])
        rescue_an = P15.analyse([(Z[("val", s)]["p"], Z[("val", s)]["acted"].astype(int), Z[("test", s)]["p"], Z[("test", s)]["acted"].astype(int)) for s in SEEDS])
        peaks = [m["peak_precision"] for m in uc5["stage_b"]["test_per_seed"]]
        out[a] = dict(Q1=q1, Q2=dict(per_seed=moved, missed_share=P15.band([m["missed_share"] for m in moved])),
                      Q3=dict(per_seed=q3, test_precision=P15.band([x["test_precision"] for x in q3]),
                              test_recall=P15.band([x["test_recall"] for x in q3]),
                              base_rate_test=float(np.mean([Z[("test", s)]["acted"].mean() for s in SEEDS])),
                              coverage_curve=rescue_an["curve"], phase15_class=reachable_and_class(rescue_an)),
                      Q4=dict(max_test_precision=float(max(peaks)), per_seed_peak=peaks, uc5a_curve=uc5["curve"],
                              stage_b=uc5["stage_b"]["verdict"], phase15_class=reachable_and_class(uc5)))
        out[a]["UC5a_analyse"] = uc5
    R["arms"] = {a: {k: v for k, v in d.items() if k != "UC5a_analyse"} for a, d in out.items()}
    # ---- reproduction of the published numbers (leaky arm)
    p13 = json.load(open(os.path.join(C.ART, "phase13_s1.json")))["headline"]["ratios"]
    p14 = json.load(open(os.path.join(C.ART, "phase14_sim.json")))["uc5"]["cells_moved_by_transfers"]
    p15 = json.load(open(os.path.join(C.ART, "phase15.json")))["uc"]["UC5a"]["policy_rop_sim|raw"]
    lq = out["leaky"]
    R["reproduction"] = dict(
        Q1_per_seed_max_abs_diff=max(abs(lq["Q1"]["per_seed"][str(s)] - p13["per_seed"][str(s)]) for s in SEEDS),
        Q1_published=p13["mean"], Q1_here=lq["Q1"]["mean"],
        Q2_counts_equal=all(m["flagged"] == q["rescued_and_sim_positive_FP_to_TP"] and m["missed"] == q["rescued_and_sim_negative_TN_to_FN"]
                            and m["rescued_weeks"] == q["rescued_weeks"] for m, q in zip(lq["Q2"]["per_seed"], p14)),
        Q3_published="0.619 [0.619, 0.620] at test recall 0.194 (QUOTED, phase-17.md §2.1; phase17/b1_score.json absent here)",
        Q3_here=lq["Q3"]["test_precision"], Q3_recall_here=lq["Q3"]["test_recall"],
        Q3_matches_to_3dp=[round(x, 3) for x in lq["Q3"]["test_precision"]] == [0.619, 0.619, 0.62],
        Q4_curve_equal=json.dumps(out["leaky"]["UC5a_analyse"]["curve"], sort_keys=True) == json.dumps(p15["curve"], sort_keys=True),
        Q4_peaks_equal=[m["peak_precision"] for m in p15["stage_b"]["test_per_seed"]] == lq["Q4"]["per_seed_peak"],
        Q4_stage_b_equal=p15["stage_b"]["verdict"] == lq["Q4"]["stage_b"])
    rp = R["reproduction"]
    R["G2"]["PASS"] = (logs["leaky"]["gate_exact"] == logs["leaky"]["n_gate"] == 45 and all(rows_equal.values())
                       and rp["Q1_per_seed_max_abs_diff"] < 1e-12 and rp["Q2_counts_equal"] and rp["Q3_matches_to_3dp"]
                       and rp["Q4_curve_equal"] and rp["Q4_peaks_equal"] and rp["Q4_stage_b_equal"]
                       and R["G2"]["failing_case_perturbed_reports_not_equal"] and R["G2"]["clean_differs_from_stored"])
    # ---- snapshot-block bootstrap, paired (same resampled snapshots for both arms)
    rng = np.random.default_rng(BOOT_SEED); K = 9
    pt = {a: quantities(sums[a], refp) for a in ("leaky", "clean")}
    bs = {a: [] for a in ("leaky", "clean", "diff")}
    for _ in range(BOOT):
        idx = rng.integers(0, K, K)
        ql, qc = quantities(sums["leaky"], refp, idx), quantities(sums["clean"], refp, idx)
        bs["leaky"].append(ql); bs["clean"].append(qc); bs["diff"].append({k: qc[k] - ql[k] for k in ql})
    ci = lambda L_, k: [float(np.percentile([x[k] for x in L_], 2.5)), float(np.percentile([x[k] for x in L_], 97.5))]
    R["block"] = {k: dict(leaky=pt["leaky"][k], leaky_ci=ci(bs["leaky"], k), clean=pt["clean"][k], clean_ci=ci(bs["clean"], k),
                          diff=pt["clean"][k] - pt["leaky"][k], diff_ci=ci(bs["diff"], k)) for k in pt["leaky"]}
    R["block_note"] = (f"{BOOT} resamples of the 9 whole test snapshots (seed {BOOT_SEED}), paired; each fill seed at its own "
                       "validation threshold; quantities averaged over the five seeds. Q1's reference is held fixed.")
    # per-snapshot spread of Q3 precision (5-seed mean)
    R["per_snapshot_q3_precision"] = {a: [float(np.mean([sums[a][s]["tp"][k] / max(sums[a][s]["alerts"][k], 1) for s in SEEDS])) for k in range(K)]
                                      for a in ("leaky", "clean")}
    os.makedirs(ART24, exist_ok=True)
    print(json.dumps({k: R[k] for k in ("G2", "reproduction", "block")}, indent=1, default=str)[:5000])
    print(C.dump(R, "phase24/stage2_sim.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "score", "g1"])
    ap.add_argument("--arm", choices=["leaky", "clean"])
    a = ap.parse_args()
    if a.mode == "run":
        run(a.arm)
    elif a.mode == "g1":
        print(json.dumps(g1_failing_cases(), indent=1))
    else:
        score()
