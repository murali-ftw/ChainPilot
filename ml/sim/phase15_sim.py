"""Phase 15 -- regenerate the Phase 14 simulation paths once more and STORE per-row scores (Phase 14 kept only counts).

Same pinned code (ml/opt/order_policy.py @ 9e2d59d, loaded from git -- deviation 122) and the same 45/45 identity gate
against Phase 12 B2's stored per-snapshot below-SS fractions. Stored per (fold, fill seed):
  UC5   rows = part-plant-weeks: score p_short = P_sim(level < SS); labels obs (a), pre (b: PRIVILEGED), acted; keys pp, week
  UC8r  recipient rows = projected-short part-plant-weeks (p_short >= 0.5): score p_short; label = transfer-in that week
  UC8d  donor rows = (projected-short week, same-part candidate with positive simulated median surplus):
        score = -(donor's simulated shortage chance after giving min(deficit, surplus)); label = INFERRED donor truth
        (a same-week transfer-out of the part at the candidate; deviation 119)
inventory_position_weekly: montecarlo.opening_position and order_policy@9e2d59d (as-of t0), and store_labels (evaluation).
"""
from __future__ import annotations
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "eval"), os.path.join(HERE, "..", "opt"),
                os.path.join(HERE, "..", "data")]
import numpy as np, pandas as pd
import phase12_common as C
import montecarlo as MC
import phase13_s2 as S2
from phase14_sim import b2_order_policy, snapshots, store_labels, W, N, SEEDS


def main():
    st = C.require_clean()
    OPb = b2_order_policy()
    pp, _, _ = MC.part_plant_universe("v8")
    sub = pp.reset_index(drop=True)
    stored = json.load(open(os.path.join(C.ART, "phase12_b2_validate.json")))
    stored_s = json.load(open(os.path.join(C.ART, "phase12_b2_seeds.json")))
    ref = {7: {p["snapshot"]: p["below_ss"] for p in stored["policy_rop"]["per_snapshot"]}}
    for s in (17, 27, 37, 47):
        ref[s] = {p["snapshot"]: p["below_ss"] for p in stored_s[f"policy_rop_fill_s{s}"]["per_snapshot"]}
    labels = store_labels(); truth = S2.Truth()
    parts, plant = sub.part_id.to_numpy(), sub.plant_id.to_numpy()
    by_part = pd.Series(np.arange(len(sub))).groupby(parts).apply(list).to_dict()
    gate = []
    out_dir = os.path.join(C.ART, "phase15_sim"); os.makedirs(out_dir, exist_ok=True)
    for fold, (lo, hi) in (("test", ("2025-01-01", "2025-12-31")), ("val", ("2024-01-01", "2024-12-31"))):
        for seed in SEEDS:
            rng = np.random.default_rng(11)
            draw = OPb.make_draw("rop", fill_seed=seed)
            A = {k: [] for k in ("p", "obs", "pre", "acted", "pp", "week", "rp", "ry", "dp", "dy")}
            for t0 in snapshots(lo, hi):
                t = time.time()
                op = MC.opening_position("v8", t0, sub)
                arr, cons, _ = draw("v8", t0, sub, W, N, rng, op)
                I0 = op.qty_on_hand.fillna(0).to_numpy(float); ss = op.safety_stock_qty.fillna(1).to_numpy(float)
                pos, _ = MC.roll_forward(I0, arr, cons, ss)
                frac = float((pos < ss[:, None, None]).mean())
                if fold == "test":
                    ok = frac == ref[seed][str(t0.date())]
                    gate.append(dict(seed=seed, t0=str(t0.date()), regenerated=frac, stored=ref[seed][str(t0.date())], exact=ok))
                    assert ok, f"IDENTITY GATE: seed {seed} {t0.date()} {frac} != {ref[seed][str(t0.date())]}"
                p_short = (pos < ss[:, None, None]).mean(2)
                weeks = [t0 + pd.Timedelta(days=7 * (w + 1)) for w in range(W)]
                weeks = [w - pd.Timedelta(days=w.weekday()) for w in weeks]
                idx = pd.MultiIndex.from_arrays([np.repeat(parts, W), np.repeat(plant, W), np.tile(weeks, len(sub))])
                lab = labels.reindex(idx); ok = lab.obs_below.notna().to_numpy()
                A["p"].append(p_short.reshape(-1)[ok]); A["obs"].append(lab.obs_below.to_numpy()[ok].astype(bool))
                A["pre"].append(lab.pre_below.to_numpy()[ok].astype(bool)); A["acted"].append(lab.acted.to_numpy()[ok].astype(bool))
                A["pp"].append(np.repeat(np.arange(len(sub)), W)[ok]); A["week"].append(np.tile(np.array(weeks, "datetime64[D]"), len(sub))[ok])
                # UC8 universes, S2's conventions (SS filled with 0, short = p >= 0.5, median-based surplus)
                ss0 = op.safety_stock_qty.fillna(0).to_numpy(float)
                ps0 = (pos < ss0[:, None, None]).mean(2); med = np.median(pos, 2)
                for i, w in np.argwhere(ps0 >= S2.SHORT_P):
                    wk = weeks[w]
                    A["rp"].append(ps0[i, w]); A["ry"].append((parts[i], plant[i], wk) in truth.tin)
                    deficit = max(ss0[i] - med[i, w], 1.0)
                    for j in by_part[parts[i]]:
                        if j == i or med[j, w] - ss0[j] <= 0:
                            continue
                        q = min(deficit, med[j, w] - ss0[j])
                        A["dp"].append(-float((pos[j, w] - q < ss0[j]).mean()))
                        A["dy"].append((parts[i], plant[i], wk) in truth.tin and (parts[i], plant[j], wk) in truth.tout)
                del pos, arr, cons
                print(f"   {fold} seed {seed} {t0.date()} below {frac:.5f} ({time.time() - t:.0f}s)", flush=True)
            np.savez_compressed(os.path.join(out_dir, f"{fold}_s{seed}.npz"),
                                **{k: (np.concatenate(v) if k in ("p", "obs", "pre", "acted", "pp", "week") else np.asarray(v))
                                   for k, v in A.items()})
    R = dict(stamp=st, b2_commit="9e2d59d", gate=gate, gate_all_exact=all(g["exact"] for g in gate), n_gate=len(gate))
    print(C.dump(R, "phase15_sim_gate.json"))


if __name__ == "__main__":
    main()
