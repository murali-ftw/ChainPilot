"""Phase 22 Stage 1c verification -- the clean replacements against the stored columns. AUDIT ONLY (reads _sim.npz via
phase22_leakscan.Sources to know which rows the leak reaches).

A stored row (channel c, week t) is NO-LEAK for a column when nothing it depends on is hidden at the end of week t:
  lead / ratio / on-time   every line of c ordered (visible) by t has its receipt visible by t, and the line that set the
                           stored value is the line the clean series keyed last (same last line)
  fill / rolls / ack-gap   every line of c ordered by t has its outcome (receipt) visible by t
  load_ratio               every line of c's supplier created in t's month and visible by t ... and no line of that month
                           becomes visible after t (the month's total is complete by t)
On NO-LEAK rows the clean value must EQUAL the stored one (lead within the CSV's whole-day rounding, 0.5 d; ratio within
0.5 d / contracted; others within 1e-6). On leak rows the difference is reported.

  python ml/eval/phase22_clean_check.py     -> ml/artifacts/phase22/clean_check_v8.json
"""
from __future__ import annotations
import os, json
import phase12_common as C
import numpy as np, pandas as pd
import config
import phase21_paths as PP
import phase22_leakscan as LS
import clean_panel as CP
from cache import load_panel


def main():
    PP.register()
    st = C.require_clean()
    S = LS.Sources("v8")
    panel, miss, active, meta = load_panel(os.path.join(config.CACHE, "v8"))
    CS_ = CP.CsvSources("v8", meta["T"])
    clean = CP.clean_columns(CS_)
    T, NCH = S.T, S.NCH
    rng = np.random.default_rng(3)
    weeks = np.sort(rng.choice(np.arange(160, T - 20), 40, replace=False))
    # pending lines per (channel, week): ordered by t, outcome not visible by t
    IN = (S.vw_ord >= 0) & (S.vw_ord < T)
    out = {}
    pend = np.zeros((NCH, len(weeks)), bool)
    for i, t in enumerate(weeks):
        m = IN & (S.vw_ord <= t) & (S.vw_rec > t)
        pend[np.unique(S.pch[m]), i] = True
    # load: the supplier's month is complete by t (no line created in t's month visible after t)
    mk = S.MONTHKEY
    cm = mk[np.clip(S.pt, 0, T - 1)]
    sup_c = S.CS[S.pch]
    load_ok = np.zeros((NCH, len(weeks)), bool)
    for i, t in enumerate(weeks):
        later = (cm == mk[t]) & (S.vw_ord > t)
        bad_sup = np.unique(sup_c[later])
        ok_sup = np.ones(len(S.DECL[0]), bool); ok_sup[bad_sup] = False
        load_ok[:, i] = ok_sup[S.CS]
    for c in CP.LEAKING:
        j = meta["cols"].index(c)
        P = np.asarray(panel[:, weeks, j], float)
        V = clean[c][:, weeks]
        fin = np.isfinite(V) & (np.asarray(miss[:, weeks, meta["nullable"].index(c)]) > 0)
        noleak = (load_ok if c == "load_ratio" else ~pend) & fin
        tol = 0.5 if c == "lead_time_actual_days" else (0.5 / S.contracted[:, None] if c == "lead_time_ratio" else 1e-6)
        tol = np.broadcast_to(tol, P.shape)
        d = np.abs(P - V)
        out[c] = dict(no_leak_rows=int(noleak.sum()), equal_on_no_leak=float((d[noleak] <= tol[noleak] + 1e-9).mean()) if noleak.any() else None,
                      leak_rows=int((~noleak & fin).sum()), mean_abs_diff_on_leak_rows=float(d[~noleak & fin].mean()) if (~noleak & fin).any() else None,
                      mean_abs_diff_on_no_leak_rows=float(d[noleak].mean()) if noleak.any() else None)
    res = dict(stamp=st, weeks_sampled=len(weeks), columns=out,
               note="lead: the stored value is the LAST-ORDERED line's float lead, the clean one the LAST-RECEIVED line's whole-day lead; "
                    "on rows with no pending line they are the same line unless receipts arrive out of order")
    C.dump(res, "phase22/clean_check_v8.json")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
