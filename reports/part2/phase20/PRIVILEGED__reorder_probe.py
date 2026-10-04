"""Phase 20 Stage 4b -- PRIVILEGED reorder-trigger probe. Torch-free (LightGBM process). NEVER a feature.

What a real ERP would hold if the client supplied an opening stock balance and a reorder point: each channel's inventory
position (on hand + on order) against its reorder point, at the end of the snapshot week. v8 keeps these inside the
simulator only, so they are captured by exec'ing generator_v8.py's source with ONE read-only hook line at the top of the
weekly loop (it copies arrays; it draws no random number) and a stop right after `_sim.npz` is written. The file on disk
is untouched (its self-hash, read by the generator from __file__, is unchanged). The capture is accepted only if EVERY
array of the re-run's `_sim.npz` equals the stored world's -- proof the hook did not perturb the simulation.

Probe columns (per channel, end of week t0):  PRIVILEGED__ip_minus_rop_wk = (on hand + on order - reorder point) / planner
forecast; PRIVILEGED__ip_over_rop; PRIVILEGED__dhat (planner forecast, units / week); PRIVILEGED__weeks_to_review.
Arm: BASE (Phase 7's LightGBM-flat arrival arm, unchanged) + these 4 columns, 5 seeds.

Caveat carried with every number: the simulator's internal position is exact and instantly current; a real ERP's book
position lags and carries recording error, so the probe is an UPPER bound on what the client data could be worth.

  python reports/part2/phase20/PRIVILEGED__reorder_probe.py capture --scratch <dir>
  python reports/part2/phase20/PRIVILEGED__reorder_probe.py fit
"""
from __future__ import annotations
import lightgbm as lgb
import os, sys, json, time, hashlib, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ML = os.path.join(REPO, "ml")
sys.path[:0] = [os.path.join(ML, "baselines"), ML, os.path.join(ML, "train"), os.path.join(ML, "data"), os.path.join(ML, "eval")]
import numpy as np, pandas as pd
import phase7_fit as P7
import folds as FO
import phase12_common as C
assert "torch" not in sys.modules

GEN = os.path.join(REPO, "db", "gen_v8", "generator_v8.py")
STORED_SIM = os.path.join(REPO, "db", "gen_v8", "seed_1001", "_sim.npz")
FEAT = os.path.join(HERE, "PRIVILEGED__probe_features_v8s1001.npz")
OUT = os.path.join(HERE, "preds")
LOG = os.path.join(HERE, "PRIVILEGED__reorder_probe_fit.json")
W0 = pd.Timestamp("2016-01-04")
LOOP = "for t in range(T):\n    perf_ord *= _alloc_decay; perf_rec *= _alloc_decay\n"
SAVED = "print('  simulation state saved ->', OUT / '_sim.npz')\n"
COLS = ["PRIVILEGED__ip_minus_rop_wk", "PRIVILEGED__ip_over_rop", "PRIVILEGED__dhat", "PRIVILEGED__weeks_to_review"]


class _Stop(Exception):
    pass


def capture(scratch):
    src = open(GEN, "rb").read()
    assert hashlib.sha1(src).hexdigest()[:12] == "71de78afa645"
    text = src.decode()
    assert text.count(LOOP) == 1 and text.count(SAVED) == 1, "hook anchors not unique"
    text = text.replace(LOOP, "for t in range(T):\n    _P20_HOOK(t, on_hand, on_order, rop, dhat)\n    perf_ord *= _alloc_decay; perf_rec *= _alloc_decay\n")
    text = text.replace(SAVED, SAVED + "raise _P20_STOP()\n")
    cap = {}

    def hook(t, on_hand, on_order, rop, dhat):          # end-of-week (t - 1) state; copies only, no RNG
        cap[t - 1] = (on_hand.astype(np.int64).copy(), on_order.astype(np.int64).copy(), rop.astype(np.int64).copy(), dhat.astype(float).copy())
    ns = {"__name__": "__main__", "__file__": GEN, "__builtins__": __builtins__, "_P20_HOOK": hook, "_P20_STOP": _Stop}
    argv = sys.argv; sys.argv = [GEN, "--seed", "1001", "--out", scratch]
    cwd = os.getcwd(); os.chdir(REPO); t = time.time()
    try:
        exec(compile(text, GEN, "exec"), ns)
        raise AssertionError("the stop sentinel never fired")
    except _Stop:
        pass
    finally:
        os.chdir(cwd); sys.argv = argv
    # ---- acceptance: the re-run's _sim.npz must equal the stored one, array by array
    a, b = np.load(os.path.join(scratch, "seed_1001", "_sim.npz")), np.load(STORED_SIM)
    assert sorted(a.files) == sorted(b.files)
    bad = [k for k in a.files if a[k].shape != b[k].shape or not np.array_equal(a[k], b[k])]
    assert not bad, f"the hooked run differs from the stored world on {bad} -- probe REJECTED"
    snaps = np.arange(26, ns["T"] - 14, 6)
    keep = sorted(k for k in cap if k in set(snaps.tolist()))
    on_hand = np.stack([cap[k][0] for k in keep]); on_order = np.stack([cap[k][1] for k in keep])
    rop = np.stack([cap[k][2] for k in keep]); dhat = np.stack([cap[k][3] for k in keep])
    np.savez_compressed(FEAT, weeks=np.array(keep), on_hand=on_hand, on_order=on_order, rop=rop, dhat=dhat,
                        REVIEW=np.asarray(ns["REVIEW"]), review_weeks=int(ns["P"]["review_weeks"]), CHID=np.asarray(ns["CHID"]))
    rep = dict(sim_arrays_equal_stored=len(a.files), snapshot_weeks_captured=len(keep), seconds=round(time.time() - t, 1),
               generator_self_hash="71de78afa645", hook="one line at the top of the weekly loop; copies arrays; no RNG draw")
    json.dump(rep, open(os.path.join(HERE, "PRIVILEGED__probe_capture.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


def probe_columns(lb):
    z = np.load(FEAT)
    wk = {int(w): i for i, w in enumerate(z["weeks"])}
    cidx = {c: i for i, c in enumerate(z["CHID"].astype(str))}
    t0 = ((lb.snapshot_date - W0).dt.days // 7).to_numpy(np.int64)
    si = np.array([wk[int(t)] for t in t0]); ci = lb.key.map(cidx).to_numpy(np.int64)
    ip = (z["on_hand"][si, ci] + z["on_order"][si, ci]).astype(float); rop = z["rop"][si, ci].astype(float)
    dh = z["dhat"][si, ci]
    rw = int(z["review_weeks"]); nxt = (z["REVIEW"][ci] - (t0 + 1)) % rw
    return pd.DataFrame({COLS[0]: (ip - rop) / np.maximum(dh, 1e-3), COLS[1]: ip / np.maximum(rop, 1.0), COLS[2]: dh,
                         COLS[3]: nxt.astype(float)}).astype(np.float32)


def fit():
    os.makedirs(OUT, exist_ok=True); P7.OUT = OUT
    log = json.load(open(LOG)) if os.path.exists(LOG) else {}
    log.setdefault("_stamps", []).append(C.stamp())
    lb = P7.labels("v8", "arrival_week"); tr, va, te = FO.fixed_split(lb.snapshot_date)
    Wd = P7.World("v8")
    X = pd.concat([Wd.channel_features(lb, with_ids=False, flat=True), probe_columns(lb)], axis=1).astype(np.float32)
    y = lb.label_value.to_numpy(float); ev = ~lb.label_censored.to_numpy(bool)
    for s in P7.SEEDS:
        name = f"PRIVILEGED__reorder_probe_s{s}"
        if all(os.path.exists(os.path.join(OUT, f"v8_arrival_week_{name}_{f}.npz")) for f in ("val", "test")):
            continue
        m = P7.lgbm_fit("l2", X, y, tr, va, s, rows_tr=tr & ev, rows_va=va & ev)
        P7.emit("v8", "arrival_week", name, lb, va, te, lambda o: (m.predict(X.iloc[o]), None), log, dict(best_iter=int(m.best_iteration_ or 400), seed=s))
        json.dump(log, open(LOG, "w"), indent=1)
        print(f"  probe s{s} done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["capture", "fit"])
    ap.add_argument("--scratch", default=None)
    a = ap.parse_args()
    capture(a.scratch) if a.mode == "capture" else fit()
    assert "torch" not in sys.modules
