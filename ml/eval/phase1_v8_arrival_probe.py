"""Phase 1 §4.3 — the pre-registered arrival investigation, run whatever the number.

The brief pre-registers the suspicion that a large jump in arrival C-index is BOTH the expected
correct result AND exactly what leakage looks like, and requires the investigation to be reported
either way, including when the number is modest. It is modest here, and this runs anyway.

Four checks, each able to fail:

  1. WINDOW BOUND     the encoder's slice ends at t0 and never later, for every test snapshot.
  2. FUTURE-BLIND     corrupting the panel AFTER t0 must leave the prediction at t0 BIT-IDENTICAL.
                      This is the strongest available test that recorded_ts, not event_ts, gated
                      the features: if any future week reached the model, this moves.
  3. NO FEATURE IS A FUNCTION OF THE LABEL   correlation of every panel channel at t0 against the
                      label, over the evaluation rows. A near-unit correlation is the tell.
  4. AS-OF ASSERTION ARMED   the Phase 11 po_line assertion still fires when line-level features
                      are built, and the shipped path carries no line-level column.

    python ml/eval/phase1_v8_arrival_probe.py --bundle <path> [--json out.json]
"""
from __future__ import annotations
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "data"),
                os.path.join(HERE, "..", "models"), os.path.join(HERE, "..", "train")]
import numpy as np, pandas as pd, torch
import loop as LP
import phase5_heads as P5
import temporal_share as TS

DEV = P5.DEV
WIN = P5.WIN


def probe(bundle_path, n_snapshots=4, seed=0):
    R = {"bundle": bundle_path}
    B = LP.load_bundle(bundle_path); cfg = B["cfg"]; task = cfg["task"]
    assert task == "arrival_week", "this probe is arrival-specific"
    lb = P5.labels(cfg["world"], task, row_features=bool(cfg.get("row_features")))
    tr, va, te = LP.split_of(cfg, lb.snapshot_date)
    D = P5.device_inputs(cfg["world"], np.sort(lb.snapshot_date[tr].unique()), cfg["wsla"])
    model = LP._materialise(B, D)
    order = P5.ordered(lb, te); dates = lb.snapshot_date.values[order]
    snaps = np.unique(dates)
    rng = np.random.default_rng(seed)

    # ---- 1. window bound -------------------------------------------------
    W = D["W"]
    bounds = []
    for s in snaps:
        t0 = P5.t0_of(W, s)
        lo, hi = t0 - WIN + 1, t0 + 1
        week_of_t0 = W["w0"] + pd.Timedelta(weeks=t0)
        bounds.append(dict(snapshot=str(pd.Timestamp(s).date()), t0=int(t0),
                           slice_lo=int(lo), slice_hi_exclusive=int(hi),
                           last_week_read=str(week_of_t0.date()),
                           last_week_le_snapshot=bool(week_of_t0 <= pd.Timestamp(s))))
    R["check1_window_bound"] = dict(
        n_snapshots=len(bounds),
        all_slices_end_at_t0=all(b["slice_hi_exclusive"] == b["t0"] + 1 for b in bounds),
        all_last_week_le_snapshot=all(b["last_week_le_snapshot"] for b in bounds),
        detail=bounds[:3])

    # ---- 2a. DEVICE CONTROL. On MPS two identical forwards are not bit-identical
    #          (deviation 13: reproduction is allclose, not equality). The criterion for
    #          check 2 therefore has to be "within the device's own run-to-run noise",
    #          and that noise must be MEASURED here, not assumed.
    s0 = snaps[0]; t00 = P5.t0_of(W, s0); ii0 = order[dates == s0]
    idx0 = torch.from_numpy(lb.key.iloc[ii0].map(W["cidx"]).to_numpy(np.int64)).to(DEV)
    reps = []
    with torch.no_grad():
        for _ in range(4):
            reps.append(LP._head_outputs(task, P5.forward(model, D, t00, idx0), 0.0, 1.0)["P"].copy())
    device_noise = max(float(np.abs(reps[0] - r).max()) for r in reps[1:])
    R["check2a_device_control"] = dict(
        note="identical inputs, no perturbation, 4 repeated forwards",
        max_abs_change=device_noise,
        bit_identical=bool(all(np.array_equal(reps[0], r) for r in reps[1:])))

    # ---- 2. future-blindness: corrupt weeks > t0, predictions must not move
    #        beyond the device noise just measured.
    fut = []
    for s in snaps[:n_snapshots]:
        t0 = P5.t0_of(W, s)
        ii = order[dates == s]
        idx = torch.from_numpy(lb.key.iloc[ii].map(W["cidx"]).to_numpy(np.int64)).to(DEV)
        with torch.no_grad():
            base = LP._head_outputs(task, P5.forward(model, D, t0, idx), 0.0, 1.0)["P"].copy()
            # this snapshot's OWN noise floor: an unperturbed repeat, same inputs
            rep = LP._head_outputs(task, P5.forward(model, D, t0, idx), 0.0, 1.0)["P"].copy()
        noise_here = float(np.abs(base - rep).max())
        Xsave = D["X"][:, t0 + 1:].clone()
        dtsave = D["dt"][:, t0 + 1:].clone()
        try:
            # overwrite EVERY week after t0 with noise of the same shape
            D["X"][:, t0 + 1:] = torch.from_numpy(
                rng.normal(0, 3, tuple(D["X"][:, t0 + 1:].shape)).astype(np.float32)).to(DEV)
            D["dt"][:, t0 + 1:] = torch.from_numpy(
                rng.normal(0, 3, tuple(D["dt"][:, t0 + 1:].shape)).astype(np.float32)).to(DEV)
            with torch.no_grad():
                pert = LP._head_outputs(task, P5.forward(model, D, t0, idx), 0.0, 1.0)["P"].copy()
        finally:
            D["X"][:, t0 + 1:] = Xsave
            D["dt"][:, t0 + 1:] = dtsave
        chg = float(np.abs(base - pert).max())
        fut.append(dict(snapshot=str(pd.Timestamp(s).date()), n_rows=int(len(ii)),
                        weeks_corrupted=int(D["X"].shape[1] - t0 - 1),
                        max_abs_change=chg,
                        own_noise_floor=noise_here,
                        within_own_noise=bool(chg <= noise_here),
                        bit_identical=bool(np.array_equal(base, pert))))
    worst = max(f["max_abs_change"] for f in fut)
    R["check2_future_blind"] = dict(
        all_bit_identical=all(f["bit_identical"] for f in fut),
        max_abs_change_over_snapshots=worst,
        device_noise_floor=device_noise,
        # each snapshot is compared against ITS OWN unperturbed repeat -- the floor varies by
        # snapshot, so a single global floor would be the wrong comparison
        within_device_noise=all(f["within_own_noise"] for f in fut),
        verdict=("no future week reached the model: the change under corruption is at or below "
                 "the noise two IDENTICAL forwards produce on this device"
                 if all(f["within_own_noise"] for f in fut) else
                 "A FUTURE WEEK MOVED THE PREDICTION beyond device noise -- investigate"),
        detail=fut)

    # ---- 2b. the SAME perturbation applied INSIDE the window must move it
    #        (otherwise check 2 is a gate that cannot fail)
    s = snaps[0]; t0 = P5.t0_of(W, s)
    ii = order[dates == s]
    idx = torch.from_numpy(lb.key.iloc[ii].map(W["cidx"]).to_numpy(np.int64)).to(DEV)
    with torch.no_grad():
        base = LP._head_outputs(task, P5.forward(model, D, t0, idx), 0.0, 1.0)["P"].copy()
    Xsave = D["X"][:, t0 - 3:t0 + 1].clone()
    try:
        D["X"][:, t0 - 3:t0 + 1] = torch.from_numpy(
            rng.normal(0, 3, tuple(D["X"][:, t0 - 3:t0 + 1].shape)).astype(np.float32)).to(DEV)
        with torch.no_grad():
            inw = LP._head_outputs(task, P5.forward(model, D, t0, idx), 0.0, 1.0)["P"].copy()
    finally:
        D["X"][:, t0 - 3:t0 + 1] = Xsave
    inw_change = float(np.abs(base - inw).max())
    R["check2b_falsification"] = dict(
        note="the same corruption INSIDE the window must change the prediction, "
             "or check 2 is a gate that cannot fail",
        max_abs_change=inw_change,
        device_noise_floor=device_noise,
        ratio_to_noise=(inw_change / device_noise if device_noise else None),
        moves=bool(inw_change > device_noise * 100))

    # ---- 3. no feature is a function of the label ------------------------
    cols = list(D["W"]["meta"]["cols"]) + ["obs:" + c for c in D["W"]["meta"]["nullable"]]
    rows_all, y_all = [], []
    for s in snaps:
        t0 = P5.t0_of(W, s)
        ii = order[dates == s]
        ci = lb.key.iloc[ii].map(W["cidx"]).to_numpy(np.int64)
        rows_all.append(D["X"][:, t0].detach().cpu().numpy()[ci])
        y_all.append(lb.label_value.iloc[ii].to_numpy(float))
    Xt0 = np.concatenate(rows_all); Y = np.concatenate(y_all)
    corr = {}
    for j, c in enumerate(cols):
        v = Xt0[:, j]
        corr[c] = 0.0 if np.std(v) < 1e-12 else float(abs(np.corrcoef(v, Y)[0, 1]))
    top = sorted(corr.items(), key=lambda kv: -kv[1])[:6]
    R["check3_feature_label_corr"] = dict(
        n_rows=int(len(Y)), n_channels=len(cols),
        max_abs_corr=float(max(corr.values())),
        strongest=[{"feature": k, "abs_corr": round(v, 5)} for k, v in top],
        verdict=("SUSPICIOUS: a feature is close to a function of the label"
                 if max(corr.values()) > 0.9 else
                 "no feature approaches a function of the label"))

    # ---- 4. the as-of assertion is armed ---------------------------------
    shipped = TS.labels_for(cfg["world"], "arrival_week")
    pilot = TS.labels_for(cfg["world"], "arrival_week", row_features=True)
    fired = False
    try:
        assert (pilot.line_recorded_ts <= pilot.snapshot_date).all()
    except AssertionError:
        fired = True
    R["check4_asof_assertion"] = dict(
        shipped_path_line_level_columns=[c for c in ("line_age_weeks", "line_recorded_ts",
                                                     "created_ts", "recorded_ts")
                                         if c in shipped.columns],
        pilot_path_fires=fired,
        pilot_violating_rows=int((pilot.line_recorded_ts > pilot.snapshot_date).sum()),
        pilot_rows=int(len(pilot)))
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    R = probe(a.bundle)
    print(json.dumps(R, indent=1, default=str))
    if a.json:
        json.dump(R, open(a.json, "w"), indent=1, default=str)
    ok = (R["check1_window_bound"]["all_slices_end_at_t0"]
          and R["check1_window_bound"]["all_last_week_le_snapshot"]
          and R["check2_future_blind"]["within_device_noise"]
          and R["check2b_falsification"]["moves"]
          and R["check3_feature_label_corr"]["max_abs_corr"] <= 0.9
          and not R["check4_asof_assertion"]["shipped_path_line_level_columns"]
          and R["check4_asof_assertion"]["pilot_path_fires"])
    print(f"\n  INVESTIGATION: {'CLEAN' if ok else 'SOMETHING MOVED -- read the detail above'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
