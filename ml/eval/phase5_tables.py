"""Phase 5 — build the report's tables from phase5_scores.json, the grids and the selections.

Prints markdown to stdout and writes ml/artifacts/phase5_tables.md. Pure bookkeeping: no metric is
computed here that phase5_score.py did not already compute with its bootstrap interval, except the
seed spreads, which are max - min over the three seeds' point scores.
"""
from __future__ import annotations
import os, sys, json, glob
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
import numpy as np
from config import ARTIFACTS

S = json.load(open(os.path.join(ARTIFACTS, "phase5_scores.json")))
GRID = [r for g in sorted(glob.glob(os.path.join(ARTIFACTS, "phase5_grid_*.json"))) for r in json.load(open(g))]
SEL = {}
for f in glob.glob(os.path.join(ARTIFACTS, "phase5_selection_*.json")):
    SEL.update(json.load(open(f)))
HEAD = {"arrival_week": "hazard", "fill_rate": "cdf22", "capacity_strain": "quantile"}
out = []
P = lambda *a: out.append(" ".join(str(x) for x in a))


def lab(w, task, lr, seed=7, gate=0, wsla=0, loss=None):
    return (f"{w}|{task}|NEW_{HEAD[task]}_none_h0_s{seed}_lr{lr:g}_g{gate}_w{wsla}"
            + (f"_loss-{loss}" if loss else ""))


def ci(e, m, d=4):
    if e is None or m not in e: return "—"
    v = e[m]
    if len(v) < 3 or v[1] is None or (isinstance(v[1], float) and np.isnan(v[1])):
        return f"{v[0]:.{d}f}"
    return f"{v[0]:.{d}f} [{v[1]:.{d}f}, {v[2]:.{d}f}]"


def pt(e, m):
    return e[m][0] if e and m in e else float("nan")


def cell_row(r):
    return r


def grid_row(w, task, lr, seed=7, gate=0, wsla=0, loss=None):
    for r in GRID:
        if (r["world"], r["task"], r["lr"], r["seed"], r["gate"], r["wsla"], r.get("fill_loss")) == \
                (w, task, lr, seed, gate, wsla, loss):
            return r
    return None


HEADLINE = {"arrival_week": ("cindex", True), "fill_rate": ("crps_exact", False),
            "capacity_strain": ("pinball_mean", False)}

# ------------------------------------------------------------------ sweeps
for task, sel in SEL.items():
    m, _ = HEADLINE[task]
    P(f"\n### LR sweep — {task} (v6, h⁰, seed 7; selected on validation)\n")
    P("| lr | validation | test (headline) | best epoch | epochs | stop | s/epoch |")
    P("|---|---|---|---|---|---|---|")
    for s in sel["sweep"]:
        e = S.get(lab("v6", task, s["lr"]))
        star = "**" if s["lr"] == sel["chosen"] else ""
        P(f"| {star}{s['lr']:g}{star} | {star}{s['val']:.5f}{star} | {ci(e, m)} | {s['best_epoch']} | "
          f"{s['epochs']} | {s['stop']} | {s['sec_per_epoch']:.1f} |")
    P(f"\nchosen **{sel['chosen']:g}**; boundary after extension: {sel['boundary_after_extension']}")

# ------------------------------------------------------------------ noise bands
P("\n### Per-metric seed bands (v6, h⁰, chosen rate, seeds 7 / 17 / 27)\n")
P("| task | metric | seed 7 | seed 17 | seed 27 | spread | sd |")
P("|---|---|---|---|---|---|---|")
BAND = {}
METRICS = {"arrival_week": ["cindex", "roc_auc_late", "ece_week"],
           "fill_rate": ["crps_exact", "crps_legacy", "ece20", "ece22", "rel_one"],
           "capacity_strain": ["pinball_mean", "pinball_10", "pinball_50", "pinball_90", "coverage80"]}
for task, sel in SEL.items():
    for m in METRICS[task]:
        vals = [pt(S.get(lab("v6", task, sel["chosen"], s)), m) for s in (7, 17, 27)]
        if any(np.isnan(vals)): continue
        BAND[(task, m)] = max(vals) - min(vals)
        P(f"| {task} | {m} | " + " | ".join(f"{v:.4f}" for v in vals) +
          f" | **{BAND[(task, m)]:.4f}** | {np.std(vals, ddof=1):.4f} |")


def verdict(delta, band, better_if_positive=True):
    if band is None or np.isnan(delta): return "—"
    if abs(delta) <= band: return "inside band"
    good = (delta > 0) == better_if_positive
    return f"{'new better' if good else 'old better'} ({abs(delta)/band:.1f}× band)"


# ------------------------------------------------------------------ fill
if "fill_rate" in SEL:
    lr = SEL["fill_rate"]["chosen"]
    P("\n### Fill — old head, new head, LightGBM (h⁰, test 2025)\n")
    P("| world | model | CRPS exact ↓ | CRPS legacy ↓ | ECE 20-bin ↓ | ECE 22-cell ↓ | reliability P(f=1) ↓ | reliability P(f≥.95) ↓ | P(complete) pred / obs |")
    P("|---|---|---|---|---|---|---|---|---|")
    for w in ("v6", "v7"):
        for name, key in (("old CE-20 head", f"{w}|fill_rate|OLD_ce20_h0"),
                          ("**new point-mass head**", lab(w, "fill_rate", lr)),
                          ("LightGBM-20 (phase1_2)", f"{w}|fill_rate|BASE_lightgbm_run9"),
                          ("LightGBM-20 (refit)", f"{w}|fill_rate|BASE_lightgbm20"),
                          ("LightGBM-22 point masses", f"{w}|fill_rate|BASE_lightgbm22"),
                          ("naive (phase1_2)", f"{w}|fill_rate|BASE_naive_run9")):
            e = S.get(key)
            pc = f"{pt(e, 'p_complete_pred'):.4f} / {pt(e, 'p_complete_obs'):.4f}" if e else "—"
            P(f"| {w} | {name} | {ci(e, 'crps_exact')} | {ci(e, 'crps_legacy')} | {ci(e, 'ece20')} | "
              f"{ci(e, 'ece22')} | {ci(e, 'rel_one')} | {ci(e, 'rel_top20')} | {pc} |")
    P("\n### Fill — old vs new margins against the new head's seed band\n")
    P("| world | metric | old | new | old − new | band | verdict |")
    P("|---|---|---|---|---|---|---|")
    for w in ("v6", "v7"):
        o, n = S.get(f"{w}|fill_rate|OLD_ce20_h0"), S.get(lab(w, "fill_rate", lr))
        for m in ("crps_exact", "crps_legacy", "ece20"):
            d = pt(o, m) - pt(n, m); b = BAND.get(("fill_rate", m))
            P(f"| {w} | {m} | {pt(o, m):.4f} | {pt(n, m):.4f} | {d:+.4f} | {b if b is None else f'{b:.4f}'} | {verdict(d, b)} |")
    P("\n### Fill — loss ablation on the 22 cells (same partition, same rate)\n")
    P("| world | loss | CRPS exact | ECE 20 | ECE 22 | reliability P(f=1) | best epoch / epochs |")
    P("|---|---|---|---|---|---|---|")
    for w in ("v6", "v7"):
        for loss in (None, "ce", "rps+ce"):
            e = S.get(lab(w, "fill_rate", lr, loss=loss)); g = grid_row(w, "fill_rate", lr, loss=loss)
            ep = f"{g['best_epoch']} / {g['epochs_run']}" if g else "—"
            P(f"| {w} | {loss or 'RPS (specified)'} | {ci(e, 'crps_exact')} | {ci(e, 'ece20')} | {ci(e, 'ece22')} | {ci(e, 'rel_one')} | {ep} |")
    P("\n#### 22-cell reliability table, new head (pred / observed)\n")
    for w in ("v6", "v7"):
        e = S.get(lab(w, "fill_rate", lr))
        if e and "_table22" in e:
            P(f"{w}: " + ", ".join(f"{b}:{p:.4f}/{o:.4f}" for b, p, o in e["_table22"]))

# ------------------------------------------------------------------ arrival
if "arrival_week" in SEL:
    lr = SEL["arrival_week"]["chosen"]
    P("\n### Arrival — old MSE head vs hazard head (h⁰, test 2025)\n")
    P("| world | model | C-index ↑ | ROC-AUC late (E[T] − promise) ↑ | ROC-AUC late, P(T > promise) ↑ | week-ECE ↓ | max |err| w1–6 | reliability P(arrive ≤ 12) ↓ |")
    P("|---|---|---|---|---|---|---|---|")
    for w in ("v6", "v7"):
        for name, key in (("old MSE head (uncensored rows)", f"{w}|arrival_week|OLD_mse_h0"),
                          ("**hazard head (all rows)**", lab(w, "arrival_week", lr)),
                          ("LightGBM", f"{w}|arrival_week|BASE_lightgbm"),
                          ("naive", f"{w}|arrival_week|BASE_naive")):
            e = S.get(key)
            P(f"| {w} | {name} | {ci(e, 'cindex')} | {ci(e, 'roc_auc_late')} | {ci(e, 'roc_auc_late_ptail')} | "
              f"{ci(e, 'ece_week')} | {ci(e, 'max_abs_err_w1_6')} | {ci(e, 'rel_arrive_by_12')} |")
    P("\n| world | old C-index | new | Δ | band | verdict | old ROC-AUC | new | Δ | band | verdict |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    for w in ("v6", "v7"):
        o, n = S.get(f"{w}|arrival_week|OLD_mse_h0"), S.get(lab(w, "arrival_week", lr))
        dc = pt(n, "cindex") - pt(o, "cindex"); dr = pt(n, "roc_auc_late") - pt(o, "roc_auc_late")
        bc, br = BAND.get(("arrival_week", "cindex")), BAND.get(("arrival_week", "roc_auc_late"))
        P(f"| {w} | {pt(o,'cindex'):.4f} | {pt(n,'cindex'):.4f} | {dc:+.4f} | {bc if bc is None else f'{bc:.4f}'} | {verdict(dc, bc)} | "
          f"{pt(o,'roc_auc_late'):.4f} | {pt(n,'roc_auc_late'):.4f} | {dr:+.4f} | {br if br is None else f'{br:.4f}'} | {verdict(dr, br)} |")
    for w in ("v6", "v7"):
        o = S.get(f"{w}|arrival_week|OLD_mse_h0")
        if o:
            P(f"\n{w}: phase1_2 printed ROC-AUC {pt(o,'roc_auc_late_as_phase1_2_units_mismatch'):.4f} (units mismatch); "
              f"consistent units {pt(o,'roc_auc_late'):.4f}; promise week alone {pt(o,'roc_auc_late_promise_only'):.4f}")
        e = S.get(lab(w, "arrival_week", lr))
        if e and "_table_week" in e:
            P(f"{w} week table pred/obs: " + ", ".join(f"{a}:{p:.3f}/{b:.3f}" for a, p, b in e["_table_week"]))
            P(f"{w} checks: sum dev {pt(e,'sum_check_max_dev'):.2e}, monotone violations {int(pt(e,'monotone_violations'))}")

# ------------------------------------------------------------------ capacity
if "capacity_strain" in SEL:
    lr = SEL["capacity_strain"]["chosen"]
    P("\n### Capacity — first measurement (h⁰, test 2025)\n")
    P("| world | model | pinball P10 ↓ | pinball P50 ↓ | pinball P90 ↓ | mean pinball ↓ | 80% coverage | crossing rate | Spearman P50 | mean width |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for w in ("v6", "v7"):
        for name, key in (("naive global", f"{w}|capacity_strain|BASE_naive_global"),
                          ("naive per-supplier *(identity-keyed)*", f"{w}|capacity_strain|BASE_naive_supplier"),
                          ("naive per-channel *(identity-keyed)*", f"{w}|capacity_strain|BASE_naive_channel"),
                          ("LightGBM quantile (inductive)", f"{w}|capacity_strain|BASE_lightgbm"),
                          ("**neural quantile head h⁰**", lab(w, "capacity_strain", lr))):
            e = S.get(key)
            P(f"| {w} | {name} | {ci(e,'pinball_10')} | {ci(e,'pinball_50')} | {ci(e,'pinball_90')} | {ci(e,'pinball_mean')} | "
              f"{ci(e,'coverage80',3)} | {ci(e,'crossing_rate',4)} | {ci(e,'spearman_p50',3)} | {ci(e,'width80_mean',3)} |")

# ------------------------------------------------------------------ gate
P("\n### 5.0 — gate off vs on (h⁰, new head, chosen rate)\n")
P("| world | task | metric | base (no wsla, no gate) | + wsla, gate off | + wsla, gate on | on − off | band | verdict | mean g | share g < 0.99 |")
P("|---|---|---|---|---|---|---|---|---|---|---|")
for task, m, hib in (("arrival_week", "cindex", True), ("fill_rate", "crps_exact", False), ("fill_rate", "ece20", False)):
    if task not in SEL: continue
    lr = SEL[task]["chosen"]
    for w in ("v6", "v7"):
        a, b, c = (S.get(lab(w, task, lr, gate=g, wsla=ws)) for g, ws in ((0, 0), (0, 1), (1, 1)))
        d = pt(c, m) - pt(b, m); band = BAND.get((task, m))
        gr = grid_row(w, task, lr, gate=1, wsla=1)
        gs = gr["gate_diag"]["test"] if gr and "gate_diag" in gr else None
        P(f"| {w} | {task} | {m} | {ci(a,m)} | {ci(b,m)} | {ci(c,m)} | {d:+.4f} | {band if band is None else f'{band:.4f}'} | "
          f"{verdict(d if hib else -d, band)} | {gs['mean_g']:.4f} | {gs['frac_lt_099']:.4f} |" if gs else
          f"| {w} | {task} | {m} | {ci(a,m)} | {ci(b,m)} | {ci(c,m)} | {d:+.4f} | — | — | — | — |")

# ------------------------------------------------------------------ convergence and cost
P("\n### Every Phase 5 cell — convergence and cost\n")
P("| world | task | tag | lr | seed | gate | wsla | loss | epochs | best | stop | s/epoch | wall s | RSS GB |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
tot = 0.0; ep = 0; rss = 0.0
for r in GRID:
    tot += r["wall"]; ep += r["epochs_run"]; rss = max(rss, r["rss"])
    P(f"| {r['world']} | {r['task']} | {r['tag']} | {r['lr']:g} | {r['seed']} | {r['gate']} | {r['wsla']} | "
      f"{r.get('fill_loss') or 'rps' if r['task']=='fill_rate' else '—'} | {r['epochs_run']} | {r['best_epoch']} | "
      f"{r['stop']} | {r['sec_per_epoch']:.1f} | {r['wall']:.0f} | {r['rss']:.2f} |")
P(f"\n**{len(GRID)} cells, {ep} epochs, {tot:.0f} s = {tot/3600:.2f} h summed wall, peak RSS {rss:.2f} GB, "
  f"{sum(1 for r in GRID if r['stop'].startswith('CAP'))} at the cap.**")

txt = "\n".join(out)
open(os.path.join(ARTIFACTS, "phase5_tables.md"), "w").write(txt)
print(txt)
