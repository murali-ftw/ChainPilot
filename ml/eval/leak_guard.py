"""Phase 23AC T3 -- the standing leak guard: a registry of every feature a model reads, and per-feature as-of tests.

Pre-registration: reports/part2/phase-23ac-preregistration.md (T3, and the Gates table). Standard: docs/standards/leak_guard.md.

Why a standing guard. The Phase 0-1 check (max |corr(feature, label)| = 0.319) passed while nine panel columns carried each
line's EVENTUAL outcome (reports/part2/phase22/stage1_leak.md). A forward-filled eventual outcome correlates only weakly with
any one label, but it EQUALS a still-pending line's outcome exactly. So the guard tests values, not correlations:

  registry          built by code (`build_registry`): the panel columns (cache meta.json of v8 and v8clean), phase7_fit
                    World.STATIC / FLAT (AST of phase7_fit.py; it is never imported, it pulls in lightgbm), the family
                    modules' COLS (fwd_load, fwd_season, cadence, pulse, stock_asof), grpstats ASTATS and the assembled
                    arrival / fill / ack block columns, phase22_rows ACK_COLS and its L4 block, and the order-time product's
                    flag feature list (ml/artifacts/phase22/serve/order_time_v8clean/product.json). Source tables are read
                    from each module's own `*.csv` string literals.
  future poison     raw panel: ml/eval/phase22_leakscan.py's exact rebuild from the world's _sim.npz (AUDIT ONLY), every
                    source row visible after the row's own week t replaced with noise; clean panel: clean_panel.clean_columns
                    under the same poisoning of the emitted CSVs. FAIL if any channel's row-t value moves (NaN-safe).
  label alignment   at week t, PENDING lines = ordered (visible week <= t) with outcome not visible by t. Candidates per
                    channel: each pending line's eventual lead, lead / contracted, eventual fill min(delivered / ordered, 1)
                    and 1 - fill; the quantity-weighted eventual fill of the channel's lines ordered in week t (if any is
                    pending) and 1 - it; the supplier's eventual whole-month ordered quantity / declared capacity for the
                    month of t (if that month is still in progress: a line of it not yet visible, or its capacity not yet
                    recorded). Share = rows whose value equals a candidate within the column's tolerance / active rows (rows
                    with a pending line or a month in progress). FAIL if share >= ALIGN_FAIL = 0.20 (pre-registered).
                    Candidates equal to 0 or 1 are not matched (ALIGN_ATOMS; the reason and the measurement are at the
                    constant); the literal share with them is recorded beside it.
  family modules    each module's falsify() must fire on its constructed offender; fwd_load and cadence are also recomputed
                    at 2 sampled t0 with every row recorded after t0 given random values (recorded_ts kept) and must not move.
                    grpstats / phase22_rows / self-exclusion: covered by the Phase 21 / 22 tests (recorded by reference).
  known answer      on every world the nine clean_panel.LEAKING columns must FAIL on the raw panel and their nine clean
                    replacements must PASS both tests; otherwise GuardInvalid (exit 2).

Exit codes (main): 0 = every FAIL is a KNOWN_FAIL (the nine LEAKING columns on the RAW panel, which no clean model reads);
1 = a FAIL outside KNOWN_FAIL (a FINDING) or an UNCOVERED feature; 2 = the known answer did not reproduce (guard invalid).

  HADES_DEVICE=cpu ./venv/bin/python ml/eval/leak_guard.py [--worlds v8,v8w1002] [--weeks 12]
      -> ml/artifacts/phase23ac/t3/{registry.json, guard_{world}.json, guard.log}
         reports/part2/phase23ac/t3/{registry.csv, guard.csv}
"""
from __future__ import annotations
import os, sys, ast, json, csv, copy, time, argparse, re
HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(ML, ".."))
for _p in (os.path.join(ML, "data"), ML, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import phase12_common as C          # noqa: E402  (path setup; no torch, no lightgbm)
import numpy as np, pandas as pd    # noqa: E402
import config                        # noqa: E402
import phase21_paths as PP           # noqa: E402
import clean_panel as CP             # noqa: E402

ALIGN_FAIL = 0.20                    # pre-registered (phase-23ac-preregistration.md, T3); never tuned
ALIGN_TOL = {"lead_time_actual_days": 0.006, "lead_time_ratio": 6e-5, "load_ratio": 6e-5}
ALIGN_TOL_DEFAULT = 1e-6
# Implementation choice the pre-registration is silent on (to be recorded as a deviation): candidates equal to 0 or 1 are
# not matched. Fill outcomes pile up on exactly 1 (full delivery) and 0 (nothing delivered), so 1 - fill = 0 matches every
# zero-valued column and fill = 1 every column at 1. Measured on v8, 3 weeks (smoke, before any full run): with atoms
# matched, qty_ordered / qty_received / is_active_week / revision_count align at 0.27-0.50 and the clean fill_rate /
# ack_gap_ratio / otd_rate_last13 at 0.21-0.37; with atoms excluded every clean column is <= 0.0014 and every non-leaking
# raw column <= 0.0002, while raw lead_time_actual_days / lead_time_ratio / load_ratio still align at 0.25-1.0.
# The literal share (atoms included) is recorded beside the verdict share in every output.
ALIGN_ATOMS = (0.0, 1.0)
POISON_EPS = 1e-9
KNOWN_FAIL = frozenset(("raw", c) for c in CP.LEAKING)     # the known raw leak; no clean model reads the raw panel
OUT_ART = os.path.join(config.ARTIFACTS, "phase23ac", "t3")
OUT_REP = os.path.join(REPO, "reports", "part2", "phase23ac", "t3")
PRODUCT = os.path.join(config.ARTIFACTS, "phase22", "serve", "order_time_v8clean", "product.json")
PHASE7_FIT = os.path.join(ML, "baselines", "phase7_fit.py")
FAMILY_MODULES = {          # module -> file; each exposes COLS (stock_asof, pulse, cadence, fwd_load also falsify())
    "fwd_load": "ml/data/fwd_load.py", "fwd_season": "ml/data/fwd_season.py", "cadence": "ml/data/cadence.py",
    "pulse": "ml/data/pulse.py", "stock_asof": "ml/data/stock_asof.py"}
READERS = {                 # model families that read each feature family (from the phase reports; extended by code below)
    "panel_raw": ["phase7_fit LightGBM (row t0, raw worlds; Phases 7-21)", "temporal_share neural (window to t0; Phases 1-21)"],
    "panel_clean": ["phase22 clean LightGBM / neural arms (v8clean, v8w1002clean)"],
    "static": ["phase7_fit LightGBM (STATIC)"], "flat": ["phase7_fit LightGBM (flat arms)"],
    "fwd_load": ["phase18 fwd_load arms", "ml/serve/features.py (Phase 20)"],
    "fwd_season": ["phase19 fill blend", "phase22 consolidated fill (season)"],
    "cadence": ["phase19 cadence arms", "phase22 consolidated fill (cadence)"],
    "pulse": ["phase18 pulse arms"], "stock_asof": ["phase20 stock_asof arm"],
    "grpstats_astats": ["phase21 grpstats arrival / fill arms"], "grpstats_arrival": ["phase21 grpstats arrival arms"],
    "grpstats_fill": ["phase21 grpstats fill arms"], "grpstats_ack": ["phase21 grpstats ack block"],
    "phase22_ack": ["phase22 fill arms (ack family)"], "phase22_l4": ["phase22 fill arms (L4 family)"],
    "row_own": []}
ROW_FEATURES = {"log1p_qty_ordered": ("po_lines", "the row's own line quantity, known at creation (the row IS that line)")}
NOT_FEATURE_TABLES = {"training_labels", "snapshots"}


class GuardInvalid(AssertionError):
    """The known answer did not reproduce: the guard cannot be trusted. Exit 2."""


def log(msg, fh=None):
    print(msg, flush=True)
    if fh is not None:
        fh.write(msg + "\n"); fh.flush()


# ----------------------------------------------------------------------------------------------------------------- registry
def csv_tables(path, cls=None):
    """Every `<table>.csv` string literal in a module (optionally inside one class), minus label / snapshot files."""
    tree = ast.parse(open(path).read())
    if cls is not None:
        tree = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls)
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            out.update(m for m in re.findall(r"([A-Za-z_]+)\.csv", n.value) if m not in NOT_FEATURE_TABLES)
    return sorted(out)


def world_static_flat(path=PHASE7_FIT):
    """phase7_fit.World.__init__'s `self.STATIC = [...]` / `self.FLAT = [...]`, by AST (the module imports lightgbm)."""
    tree = ast.parse(open(path).read())
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "World")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    got = {}
    for n in ast.walk(init):
        if isinstance(n, ast.Assign):
            for tg in n.targets:
                if isinstance(tg, ast.Attribute) and tg.attr in ("STATIC", "FLAT"):
                    got[tg.attr] = list(ast.literal_eval(n.value))
    assert set(got) == {"STATIC", "FLAT"}, f"phase7_fit.World: STATIC / FLAT not found by AST ({sorted(got)})"
    return got["STATIC"], got["FLAT"]


def _grpstats_block_cols():
    """Assembled grpstats block column names, by calling the assemble functions on a one-row dummy input."""
    import grpstats as GS
    nA, nK = len(GS.ASTATS), len(GS.KSTATS)
    Z = {"A_L4": np.ones((1, nA)), "A_L2": np.ones((1, nA)), "A_L1": np.ones((1, nA)),
         "A_L5c": np.ones((1, 12, nA)), "A_L3c": np.ones((1, 12, nA)), "month": np.zeros(1, np.int8),
         "H_L4": np.ones((1, 22)), "H_L2": np.ones((1, 22)), "H_L1": np.ones((1, 22)), "H_L5c": np.ones((1, 12, 22)),
         "F_L4": np.ones((1, 3)), "F_L2": np.ones((1, 3)), "F_L1": np.ones((1, 3)),
         "F_L5c": np.ones((1, 12, 3)), "F_L3c": np.ones((1, 12, 3)), "K_L4": np.ones((1, nK)), "K_L2": np.ones((1, nK))}
    arr = list(GS.assemble_arrival(Z, 100, "L5").columns)
    fil = list(GS.assemble_fill(Z, 100, "L5").columns)
    ack = list(GS.assemble_ack(Z).columns)
    import phase22_rows as R22
    l4 = list(GS.assemble_fill(Z, R22.K_FILL, "L4").columns)
    return GS, arr, fil, ack, R22, l4


def build_registry(product_path=PRODUCT):
    """-> list of {column, family, source_table, horizon, readers, test_kind, nullable}. Built from the builders' own code."""
    import importlib
    reg = []

    def add(column, family, source, horizon, test_kind, nullable=None, extra_readers=()):
        reg.append(dict(column=column, family=family, source_table=source, horizon=horizon,
                        readers=list(READERS.get(family, [])) + list(extra_readers), test_kind=test_kind, nullable=nullable))

    # --- panel columns, raw and clean (cache meta.json)
    clean_tables = ", ".join(csv_tables(os.path.join(ML, "data", "clean_panel.py")))
    for world, fam in (("v8", "panel_raw"), ("v8clean", "panel_clean")):
        meta = json.load(open(os.path.join(config.CACHE, world, "meta.json")))
        repl = set(meta.get("replaced_columns", []))
        for c in meta["cols"]:
            if fam == "panel_clean" and c in repl:
                src, tk = clean_tables, "poison_clean + label_alignment; self-exclusion: ml/tests/test_phase22_clean_panel.py"
            else:
                src = "channel_performance_weekly"
                tk = ("poison_raw + label_alignment" if fam == "panel_raw"
                      else "inherits panel_raw (bytes asserted identical)")
            add(c, fam, src, "end of the row's own week t (H_week)", tk, nullable=c in meta["nullable"])
    # --- phase7_fit World STATIC / FLAT (time-invariant masters)
    static, flat = world_static_flat()
    masters = ", ".join(csv_tables(PHASE7_FIT, cls="World"))
    for c in static:
        add(c, "static", f"{masters} masters (time-invariant)", "none (no time dimension)", "time-invariant masters: N/A")
    for c in flat:
        add(c, "flat", f"{masters} masters (time-invariant)", "none (no time dimension)", "time-invariant masters: N/A")
    # --- family modules' COLS (imported; pure numpy / pandas)
    for mod, rel in FAMILY_MODULES.items():
        M = importlib.import_module(mod)
        src = ", ".join(csv_tables(os.path.join(REPO, rel)))
        if mod == "fwd_season":
            src = ", ".join(csv_tables(os.path.join(REPO, FAMILY_MODULES["fwd_load"])))
            tk = "inherits fwd_load (network mean of fwd_load cells)"
        elif mod in ("fwd_load", "cadence"):
            tk = "falsify() + poisoned recompute at sampled t0"
        else:
            tk = "falsify()"
        for c in M.COLS:
            add(c, mod, src, "snapshot t0 (recorded_ts <= t0, asserted)", tk)
    # --- grpstats and phase22_rows
    GS, arr, fil, ack, R22, l4 = _grpstats_block_cols()
    gsrc = ", ".join(csv_tables(os.path.join(ML, "data", "grpstats.py")))
    by_ref = "covered by ml/tests/test_phase21_grpstats.py (future poison + self-exclusion)"
    for c in GS.ASTATS:
        add(c, "grpstats_astats", gsrc, "snapshot t0 (recorded_ts <= t0, asserted)", by_ref)
    for fam, cols in (("grpstats_arrival", arr), ("grpstats_fill", fil), ("grpstats_ack", ack)):
        for c in cols:
            add(c, fam, gsrc, "snapshot t0 (recorded_ts <= t0, asserted)", by_ref)
    for c in R22.ACK_COLS:
        add(c, "phase22_ack", gsrc, "snapshot t0", by_ref + "; built by grpstats.Builder.fill (ml/data/phase22_rows.py)")
    for c in l4:
        add(c, "phase22_l4", gsrc, "snapshot t0", by_ref + "; built by grpstats.Builder.fill (ml/data/phase22_rows.py)")
    # --- the order-time product's flag features: attach the reader; anything unregistered is UNCOVERED
    if not os.path.exists(product_path):
        raise FileNotFoundError(f"order-time product not found: {product_path} (the registry must cover its flag features)")
    prod = json.load(open(product_path))
    feats = prod["flag"]["models"][0]["features"]
    reader = f"order_time flag (ml/serve/order_time.py, world {prod.get('world', '?')})"
    pref = "panel_clean" if "clean" in str(prod.get("world", "")) else "panel_raw"
    for f in feats:
        hits = [e for e in reg if e["column"] == f and (not e["family"].startswith("panel") or e["family"] == pref)]
        hits = [e for e in hits if e["family"] not in ("grpstats_astats",)] or hits
        if hits:
            for e in hits:
                if reader not in e["readers"]:
                    e["readers"].append(reader)
        elif f in ROW_FEATURES:
            add(f, "row_own", ROW_FEATURES[f][0], "the row's own creation time", "row's own order-time value: N/A",
                extra_readers=[reader])
        else:
            add(f, "UNREGISTERED", "?", "?", "UNCOVERED: no feature family registers this column", extra_readers=[reader])
    return reg


def write_registry(reg):
    os.makedirs(OUT_ART, exist_ok=True); os.makedirs(OUT_REP, exist_ok=True)
    json.dump(reg, open(os.path.join(OUT_ART, "registry.json"), "w"), indent=1)
    with open(os.path.join(OUT_REP, "registry.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["column", "family", "source_table", "horizon", "readers", "test_kind", "nullable"])
        for e in reg:
            w.writerow([e["column"], e["family"], e["source_table"], e["horizon"], "; ".join(e["readers"]), e["test_kind"],
                        "" if e["nullable"] is None else e["nullable"]])


# ------------------------------------------------------------------------------------------------------- feature-list check
def check_feature_list(columns, panel):
    """T1's leak gate. panel 'raw': flag every column in clean_panel.LEAKING; 'clean': the clean cache replaced them, none."""
    if panel not in ("raw", "clean"):
        raise ValueError(f"panel must be 'raw' or 'clean', got {panel!r}")
    if panel == "clean":
        return []
    return sorted(set(columns) & set(CP.LEAKING))


# ------------------------------------------------------------------------------------------------------------ future poison
def _row_change(a, b):
    a_, b_ = np.nan_to_num(a, nan=-999.0), np.nan_to_num(b, nan=-999.0)
    return np.abs(a_ - b_) > POISON_EPS


def poison_raw(S, R, weeks, rng, cols=None):
    """-> {col: [share of channels whose row-t value moved, per week]} for the generator rebuild (phase22_leakscan)."""
    import phase22_leakscan as LS
    cols = list(R) if cols is None else cols
    out = {c: [] for c in cols}
    for t in weeks:
        Rp = LS.rebuild(S, **LS.poison(S, int(t), False, rng))
        for c in cols:
            out[c].append(float(_row_change(R[c][:, t], Rp[c][:, t]).mean()))
    return out


def poisoned_csv(S, t, rng):
    """clean_panel.CsvSources with every row visible after week t replaced by noise (reimplements the Phase 22 test's
    poisoned(): order quantity if the order is visible later; receipt quantity and lead if the receipt is; capacity
    declarations recorded later)."""
    P = copy.copy(S)
    for nm in ("qty", "recv", "lead", "decl_q"):
        setattr(P, nm, getattr(S, nm).copy())
    o = S.vw_ord > t
    P.qty[o] = rng.integers(1, 5000, o.sum())
    g = np.maximum(S.vw_rec, S.vw_ord) > t
    P.recv[g] = rng.integers(0, 5000, g.sum()); P.lead[g] = rng.uniform(3, 200, g.sum())
    d = S.decl_vw > t
    P.decl_q[d] = rng.uniform(60, 5e4, d.sum())
    return P


def poison_clean(CS, base, weeks, rng):
    out = {c: [] for c in base}
    for t in weeks:
        Pc = CP.clean_columns(poisoned_csv(CS, int(t), rng))
        for c in base:
            out[c].append(float(_row_change(base[c][:, t], Pc[c][:, t]).mean()))
    return out


# ---------------------------------------------------------------------------------------------------------- label alignment
def alignment_candidates(t, L):
    """Outcome-derived candidates at week t. L: dict of per-line arrays chan, vw_ord, vw_out, lead, fill, qty; per-channel
    contracted; optional per-channel sup_ratio [NCH, T] and sup_pending [NCH, T] (bool). -> (cand_chan, cand_val, active)."""
    chan, vw_ord, vw_out = L["chan"], L["vw_ord"], L["vw_out"]
    NCH = len(L["contracted"])
    pend = (vw_ord >= 0) & (vw_ord <= t) & (vw_out > t)
    p = np.flatnonzero(pend)
    cc = [chan[p]] * 4
    cv = [L["lead"][p], L["lead"][p] / L["contracted"][chan[p]], L["fill"][p], 1.0 - L["fill"][p]]
    # quantity-weighted eventual fill of the lines ordered in week t, where any of them is pending
    wk = np.flatnonzero(vw_ord == t)
    if len(wk):
        num = np.bincount(chan[wk], weights=L["fill"][wk] * L["qty"][wk], minlength=NCH)
        den = np.bincount(chan[wk], weights=L["qty"][wk].astype(float), minlength=NCH)
        anyp = np.bincount(chan[wk], weights=pend[wk].astype(float), minlength=NCH) > 0
        ok = np.flatnonzero(anyp & (den > 0))
        wf = num[ok] / den[ok]
        cc += [ok, ok]; cv += [wf, 1.0 - wf]
    active = np.bincount(chan[p], minlength=NCH) > 0
    if L.get("sup_ratio") is not None:
        sp = L["sup_pending"][:, t]
        ok = np.flatnonzero(sp & np.isfinite(L["sup_ratio"][:, t]))
        cc.append(ok); cv.append(L["sup_ratio"][ok, t])
        active = active | sp
    return np.concatenate(cc).astype(np.int64), np.concatenate(cv).astype(float), active


def aligned_rows(values, cand_chan, cand_val, tol, atoms=ALIGN_ATOMS):
    """-> bool [NCH]: the channel's value equals one of its candidates within tol (NaN values never align). A candidate
    equal to an atom (0 or 1) never counts: it is shared by a large share of all outcomes and identifies none of them."""
    v = np.asarray(values, float)
    keep = np.isfinite(cand_val)
    for a in atoms:
        keep &= np.abs(cand_val - a) > tol
    cc, cv = cand_chan[keep], cand_val[keep]
    hit = np.isfinite(v[cc]) & (np.abs(v[cc] - cv) <= tol)
    return np.bincount(cc[hit], minlength=len(v)) > 0


def alignment_share(columns, weeks, L, tol=None):
    """columns: {col: [NCH, T]}. -> {col: dict(share, share_literal, aligned, active, per_week)}, pooled over the weeks.
    `share` (atoms excluded) decides the verdict; `share_literal` (every candidate, atoms included) is recorded beside it."""
    tol = tol or {}
    acc = {c: [0, 0, [], 0] for c in columns}
    for t in weeks:
        cc, cv, active = alignment_candidates(int(t), L)
        for c, X in columns.items():
            tc = tol.get(c, ALIGN_TOL_DEFAULT)
            al = aligned_rows(X[:, int(t)], cc, cv, tc) & active
            lit = aligned_rows(X[:, int(t)], cc, cv, tc, atoms=()) & active
            acc[c][0] += int(al.sum()); acc[c][1] += int(active.sum()); acc[c][3] += int(lit.sum())
            acc[c][2].append(float(al.sum() / max(active.sum(), 1)))
    return {c: dict(share=a / max(n, 1), share_literal=li / max(n, 1), aligned=a, active=n, per_week=pw)
            for c, (a, n, pw, li) in acc.items()}


def line_arrays(S, CS):
    """Per-line arrays for alignment from the leakscan Sources (eventual pl / pd / pq; generator order) and the CSV sources
    (first receipt and the zero-close signal: a line's outcome is visible at its first receipt or, for a line closed at
    zero, at the rejection / qty -> 0 revision, never before its order)."""
    assert len(S.pq) == len(CS.qty) and S.NCH == CS.NCH, "leakscan Sources and clean_panel.CsvSources disagree on lines"
    assert (S.vw_ord == CS.vw_ord).all(), "order-visible weeks disagree between the two sources"
    vw_out = np.maximum(np.minimum(CS.vw_rec, CS.vw_zero), CS.vw_ord)
    fill = np.where(S.pq > 0, np.minimum(S.pd / np.maximum(S.pq, 1), 1.0), 1.0)
    # supplier month in progress at t: a line of the month (event week) not yet visible by t, or the month's capacity
    # recorded after the end of week t
    T, NS = S.T, S.DECL.shape[1]
    m_line = S.MONTHKEY[np.clip(S.pt, 0, T - 1)]
    sup_line = S.CS[S.pch]
    month_ordered = np.zeros(S.DECL.shape); np.add.at(month_ordered, (m_line, sup_line), S.pq)
    ratio_ms = month_ordered / S.DECL
    last_vis = np.full(S.DECL.shape, -1, np.int64)
    np.maximum.at(last_vis, (m_line, sup_line), S.vw_ord)
    week_end = S.W0 + pd.to_timedelta(7 * (np.arange(T) + 1), unit="D")
    mk = S.MONTHKEY
    decl_late = S.decl_rec[mk] > week_end.to_numpy()[:, None]            # [T, NS]
    pend_s = (last_vis[mk] > np.arange(T)[:, None]) | decl_late          # [T, NS]
    sup_ratio = ratio_ms[mk][:, S.CS].T                                  # [NCH, T]
    sup_pending = pend_s[:, S.CS].T
    return dict(chan=S.pch, vw_ord=S.vw_ord, vw_out=vw_out, lead=S.pl.astype(float), fill=fill, qty=S.pq.astype(float),
                contracted=S.contracted.astype(float), sup_ratio=sup_ratio, sup_pending=sup_pending)


# ------------------------------------------------------------------------------------------------------- family modules
def _poison_frames(S, t0, rng, keys):
    """Copy of fwd_load-style sources with every row recorded after t0 given random values (recorded_ts kept)."""
    P = dict(S); t0 = pd.Timestamp(t0)
    for k in keys:
        d = S[k].copy(); late = (d.recorded_ts > t0).to_numpy(); n = int(late.sum())
        if k == "pdw":
            d.loc[late, "gross_requirement_p50"] = rng.uniform(0, 1e4, n)
            d.loc[late, "week_start"] = t0 + pd.to_timedelta(7 * rng.integers(1, 14, n), unit="D")
        elif k == "pol":
            d.loc[late, "qty_ordered"] = rng.integers(1, 5000, n)
            d.loc[late, "created_ts"] = t0 - pd.to_timedelta(rng.integers(0, 365, n), unit="D")
            d.loc[late, "ci"] = rng.integers(0, len(S["ch"]), n)
        elif k == "grn":
            d.loc[late, "qty_received"] = rng.integers(0, 5000, n)
            d.loc[late, "event_ts"] = t0 - pd.to_timedelta(rng.integers(0, 365, n), unit="D")
            d.loc[late, "is_final_receipt"] = rng.integers(0, 2, n).astype(bool)
            d.loc[late, "ci"] = rng.integers(0, len(S["ch"]), n)
        P[k] = d
    return P


def family_checks(world, rng, n_t0=2, fh=None):
    """-> {module: dict(falsify, poison, verdict)}."""
    import fwd_load as FL, cadence as CA, pulse as PU, stock_asof as SA
    out = {}
    for name, M in (("fwd_load", FL), ("cadence", CA), ("pulse", PU), ("stock_asof", SA)):
        try:
            r = M.falsify(world)
            fired = True if r is None else all(str(v).startswith("fired") for k, v in r.items()
                                               if k not in ("t0", "clean_rows_asserted", "clean_vector"))
        except AssertionError as e:
            fired, r = False, f"falsify assertion: {e}"
        out[name] = dict(falsify="fired" if fired else "DID NOT FIRE")
    snaps = FL.snapshot_dates(world)
    t0s = [pd.Timestamp(snaps[i]) for i in np.linspace(len(snaps) // 4, 3 * len(snaps) // 4, n_t0).astype(int)]
    S = FL.load_sources(world)
    res = []
    for t0 in t0s:
        F, _ = FL.snapshot_features(S, t0)
        Fp, _ = FL.snapshot_features(_poison_frames(S, t0, rng, ("pdw", "pol", "grn")), t0)
        res.append(dict(t0=str(t0.date()), changed=int((~np.isclose(F, Fp, equal_nan=True, rtol=0, atol=0)).sum())))
    out["fwd_load"]["poison"] = res
    SC = CA.load_sources(world)
    res = []
    for t0 in t0s:
        F, _ = CA.snapshot_cadence(SC, t0)
        P = _poison_frames(SC, t0, rng, ("pol", "grn"))
        P["pol"] = P["pol"].sort_values(["ci", "created_ts"], kind="stable").reset_index(drop=True)
        Fp, _ = CA.snapshot_cadence(P, t0)
        res.append(dict(t0=str(t0.date()), changed=int((~np.isclose(F, Fp, equal_nan=True, rtol=0, atol=0)).sum())))
    out["cadence"]["poison"] = res
    for name, r in out.items():
        ok = r["falsify"] == "fired" and all(x["changed"] == 0 for x in r.get("poison", []))
        r["verdict"] = "PASS" if ok else "FAIL"
        log(f"  family {name}: falsify {r['falsify']}; poison {r.get('poison', 'n/a')} -> {r['verdict']}", fh)
    return out


# ------------------------------------------------------------------------------------------------- verdicts and exit codes
def column_verdict(poison_shares, align):
    pf = max(poison_shares) > 0 if poison_shares else False
    af = align["share"] >= ALIGN_FAIL
    failed = [n for n, f in (("poison", pf), ("label_alignment", af)) if f]
    return ("FAIL" if failed else "PASS"), failed


def known_answer(results):
    """results: {world: {"rows": [row, ...]}}. The nine LEAKING must FAIL on raw; their clean replacements must PASS both."""
    bad = []
    for world, R in results.items():
        rows = {(r["panel"], r["column"]): r for r in R["rows"] if r.get("panel") in ("raw", "clean")}
        for c in CP.LEAKING:
            r = rows.get(("raw", c))
            if r is None or r["verdict"] != "FAIL":
                bad.append(f"{world}: raw {c} did not FAIL ({None if r is None else r['verdict']})")
            r = rows.get(("clean", c))
            if r is None or r["verdict"] != "PASS":
                bad.append(f"{world}: clean {c} did not PASS ({None if r is None else r['failed_tests']})")
    if bad:
        raise GuardInvalid("known answer not reproduced: " + "; ".join(bad))
    return True


def findings(results):
    out = []
    for world, R in results.items():
        for r in R["rows"]:
            if r["verdict"] in ("FAIL", "UNCOVERED") and (r.get("panel"), r["column"]) not in KNOWN_FAIL:
                out.append(f"{world}: {r.get('panel') or r.get('family')} {r['column']} {r['verdict']} {r.get('failed_tests', '')}")
    return out


def exit_code(results):
    """2 if the known answer is invalid; 1 if any FAIL / UNCOVERED outside KNOWN_FAIL; else 0."""
    try:
        known_answer(results)
    except GuardInvalid:
        return 2
    return 1 if findings(results) else 0


# ------------------------------------------------------------------------------------------------------------------ run
def sample_weeks(world, S_W0, n):
    snaps = pd.to_datetime(pd.read_csv(os.path.join(config.WORLDS[world], "snapshots.csv")).as_of_ts)
    snaps = snaps[(snaps >= "2019-01-01") & (snaps <= "2025-12-31")]
    weeks = ((snaps - S_W0).dt.days // 7).to_numpy()
    return [int(w) for w in weeks[np.linspace(0, len(weeks) - 1, n).astype(int)]]


def run_world(world, n_weeks, reg, fh=None, seed=2303):
    import phase22_leakscan as LS
    t = time.time(); rng = np.random.default_rng(seed)
    S = LS.Sources(world)
    R = LS.rebuild(S)
    meta = json.load(open(os.path.join(config.CACHE, world, "meta.json")))
    assert meta["T"] == S.T, f"cache T {meta['T']} != _sim T {S.T}"
    CS = CP.CsvSources(world, meta["T"])
    Cc = CP.clean_columns(CS)
    weeks = sample_weeks(world, S.W0, n_weeks)
    log(f"[{world}] weeks {[str((S.W0 + pd.Timedelta(days=7 * w)).date()) for w in weeks]}", fh)
    # the clean cache's non-replaced columns must be byte-identical to the raw cache's
    from cache import load_panel
    raw_p, _, _, rmeta = load_panel(os.path.join(config.CACHE, world))
    cln_p, _, _, cmeta = load_panel(os.path.join(config.CACHE, world + "clean"))
    inherited = [c for c in cmeta["cols"] if c not in CP.LEAKING]
    for c in inherited:
        assert np.array_equal(np.asarray(raw_p[:, :, rmeta["cols"].index(c)]), np.asarray(cln_p[:, :, cmeta["cols"].index(c)])), \
            f"{world}clean column {c} differs from the raw cache: it cannot inherit the raw result"
    L = line_arrays(S, CS)
    pr = poison_raw(S, R, weeks, rng); log(f"[{world}] raw poison done ({time.time() - t:.0f}s)", fh)
    pc = poison_clean(CS, Cc, weeks, rng); log(f"[{world}] clean poison done ({time.time() - t:.0f}s)", fh)
    ar = alignment_share(R, weeks, L, ALIGN_TOL)
    ac = alignment_share(Cc, weeks, L, ALIGN_TOL); log(f"[{world}] alignment done ({time.time() - t:.0f}s)", fh)
    readers = {}
    for e in reg:
        readers.setdefault((e["family"], e["column"]), e["readers"])
    rows = []
    for panel, cols in (("raw", list(R)), ("clean", list(R))):
        for c in cols:
            if panel == "clean" and c in Cc:
                ps, al, how = pc[c], ac[c], "clean_columns"
            else:
                ps, al, how = pr[c], ar[c], "generator rebuild" + (" (inherited, bytes identical)" if panel == "clean" else "")
            v, failed = column_verdict(ps, al)
            rows.append(dict(panel=panel, column=c, family=f"panel_{panel}", built_by=how, verdict=v, failed_tests=failed,
                             poison_change_share=float(np.mean(ps)), poison_any=bool(max(ps) > 0),
                             alignment_share=float(al["share"]), alignment_share_literal=float(al["share_literal"]),
                             alignment_active=al["active"], alignment_aligned=al["aligned"],
                             known_fail=(panel, c) in KNOWN_FAIL, readers=readers.get((f"panel_{panel}", c), [])))
    fam = family_checks(world, rng, fh=fh)
    for e in reg:
        f = e["family"]
        if f.startswith("panel"):
            continue
        if f in fam:
            v = fam[f]["verdict"]; why = f"falsify {fam[f]['falsify']}; poison {fam[f].get('poison', 'n/a')}"
        elif f == "fwd_season":
            v = fam["fwd_load"]["verdict"]; why = "inherits fwd_load"
        elif f in ("static", "flat"):
            v, why = "N/A", "time-invariant masters: N/A (no time dimension)"
        elif f == "row_own":
            v, why = "N/A", "the row's own order-time value"
        elif f == "UNREGISTERED":
            v, why = "UNCOVERED", e["test_kind"]
        else:
            v, why = "PASS", e["test_kind"]
        rows.append(dict(panel=None, column=e["column"], family=f, built_by=e["source_table"], verdict=v,
                         failed_tests=[] if v != "FAIL" else [why], note=why, readers=e["readers"], known_fail=False))
    rows.append(dict(panel=None, column="(self-exclusion)", family="self_exclusion", verdict="PASS", failed_tests=[],
                     note="covered by ml/tests/test_phase22_clean_panel.py test_self_exclusion and "
                          "ml/tests/test_phase21_grpstats.py", readers=[], known_fail=False))
    return dict(world=world, weeks=[str((S.W0 + pd.Timedelta(days=7 * w)).date()) for w in weeks], rows=rows,
                family=fam, seconds=time.time() - t, align_fail=ALIGN_FAIL)


def write_guard(results, stamp):
    os.makedirs(OUT_ART, exist_ok=True); os.makedirs(OUT_REP, exist_ok=True)
    for w, R in results.items():
        json.dump(dict(R, stamp=stamp), open(os.path.join(OUT_ART, f"guard_{w}.json"), "w"), indent=1, default=str)
    with open(os.path.join(OUT_REP, "guard.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["world", "panel", "family", "column", "verdict", "failed_tests", "poison_change_share", "alignment_share",
                     "alignment_share_literal", "known_fail", "finding", "readers", "note"])
        for w, R in results.items():
            for r in R["rows"]:
                fnd = r["verdict"] in ("FAIL", "UNCOVERED") and not r.get("known_fail")
                wr.writerow([w, r.get("panel") or "", r["family"], r["column"], r["verdict"], "+".join(map(str, r["failed_tests"])),
                             "" if "poison_change_share" not in r else f"{r['poison_change_share']:.4f}",
                             "" if "alignment_share" not in r else f"{r['alignment_share']:.4f}",
                             "" if "alignment_share_literal" not in r else f"{r['alignment_share_literal']:.4f}",
                             r.get("known_fail", False), "FINDING" if fnd else "", "; ".join(r["readers"]), r.get("note", "")])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", default="v8,v8w1002"); ap.add_argument("--weeks", type=int, default=12)
    a = ap.parse_args(argv)
    st = C.require_clean()
    PP.register()
    os.makedirs(OUT_ART, exist_ok=True)
    fh = open(os.path.join(OUT_ART, "guard.log"), "w")
    log(f"leak guard {st['code_version']} worlds={a.worlds} weeks={a.weeks} ALIGN_FAIL={ALIGN_FAIL}", fh)
    reg = build_registry(); write_registry(reg)
    log(f"registry: {len(reg)} entries", fh)
    results = {}
    for w in a.worlds.split(","):
        results[w] = run_world(w, a.weeks, reg, fh)
        for r in results[w]["rows"]:
            if r.get("panel"):
                log(f"  {w:8s} {r['panel']:5s} {r['column']:24s} {r['verdict']:4s} poison {r['poison_change_share']:.4f} "
                    f"align {r['alignment_share']:.4f} (literal {r['alignment_share_literal']:.4f}) {'KNOWN' if r['known_fail'] else ''}", fh)
    write_guard(results, st)
    code = exit_code(results)
    if code == 2:
        try:
            known_answer(results)
        except GuardInvalid as e:
            log(f"GUARD INVALID: {e}", fh)
    for f in findings(results):
        log(f"FINDING: {f}", fh)
    log(f"exit {code}", fh)
    fh.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
