"""Phase 4 re-derived on the Phase 5 heads — depth and encoder tables.

Reads phase5_scores.json (run phase5_score.py first), the Phase 5 h0 grids and the phase5_grid_p4_*
grids. Selection is on VALIDATION; the test score of every depth not selected is shown beside it.
Margins are read against the Phase 5 per-metric seed bands, which were measured at h0 on v6 -- a
borrowed band at h1 / h4, stated wherever it is used.
"""
from __future__ import annotations
import os, sys, json, glob
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
import numpy as np
from config import ARTIFACTS

S = json.load(open(os.path.join(ARTIFACTS, "phase5_scores.json")))
HEAD = {"arrival_week": "hazard", "capacity_strain": "quantile", "fill_rate": "cdf22"}
PRIMARY = {"arrival_week": ("cindex", True), "capacity_strain": ("pinball_mean", False),
           "fill_rate": ("crps_exact", False)}
SECONDARY = {"arrival_week": ["roc_auc_late", "ece_week"], "capacity_strain": ["coverage80", "spearman_p50"],
             "fill_rate": ["ece20", "rel_one"]}
out = []
P = lambda *a: out.append(" ".join(str(x) for x in a))


def rows(path):
    return json.load(open(path)) if os.path.exists(path) else []


def lab(w, task, lr, arch, depth, seed=7):
    return f"{w}|{task}|NEW_{HEAD[task]}_{arch}_h{depth}_s{seed}_lr{lr:g}_g0_w0"


def ci(e, m):
    if not e or m not in e: return "—"
    v = e[m]
    return f"{v[0]:.4f} [{v[1]:.4f}, {v[2]:.4f}]" if len(v) >= 3 and v[1] is not None else f"{v[0]:.4f}"


def pt(e, m):
    return e[m][0] if e and m in e else float("nan")


def verdict(delta, band, higher):
    if np.isnan(delta) or not np.isfinite(band) or band <= 0: return "—"
    if abs(delta) <= band: return "**no preference**"
    return ("deeper better" if (delta > 0) == higher else "shallower better") + f" ({abs(delta) / band:.1f}×)"


for task in ("capacity_strain", "arrival_week", "fill_rate"):
    selp = os.path.join(ARTIFACTS, f"phase5_selection_{task}.json")
    g4 = rows(os.path.join(ARTIFACTS, f"phase5_grid_p4_{task.split('_')[0]}.json"))
    if not g4 or not os.path.exists(selp):
        continue
    lr_p5 = json.load(open(selp))[task]["chosen"]
    lr = lr_p5
    ratesp = os.path.join(ARTIFACTS, "phase4r_rates.json")
    if os.path.exists(ratesp) and task in json.load(open(ratesp)):
        lr = json.load(open(ratesp))[task]["lr"]          # a documented override beats the Phase 5 selection
    g0 = rows(os.path.join(ARTIFACTS, f"phase5_grid_{task.split('_')[0]}.json"))
    m, higher = PRIMARY[task]
    # seed repeats exist only at the Phase 5 selected rate; when Phase 4 runs at an override rate the
    # band is BORROWED from there, and the header says so
    seeds = [pt(S.get(lab("v6", task, lr_p5, "none", 0, s)), m) for s in (7, 17, 27)]
    band = max(seeds) - min(seeds) if not any(np.isnan(seeds)) else float("nan")
    band_note = "h⁰ v6 seeds" if lr == lr_p5 else f"BORROWED from h⁰ v6 seeds at {lr_p5:g}"

    def val(w, arch, depth):
        src = g0 if depth == 0 else g4
        for r in src:
            if (r["world"], r["arch"], r["depth"], r["seed"], r["lr"], r["gate"], r["wsla"], r.get("fill_loss")) == \
                    (w, "none" if depth == 0 else arch, depth, 7, lr, 0, 0, None):
                return r
        return None

    P(f"\n### {task} — {m} {'↑' if higher else '↓'}  (rate {lr:g} {'frozen from Phase 5' if lr == lr_p5 else 'OVERRIDE, see phase4r_rates.json'}; band {band:.4f}, {band_note})\n")
    P("| world | encoder | h⁰ | h¹ | h⁴ | val-selected | test of depths not chosen | h⁰ → h¹ | h¹ → h⁴ |")
    P("|---|---|---|---|---|---|---|---|---|")
    for w in ("v6", "v7"):
        for arch, name in (("lite", "SHARE-lite"), ("mp", "HeteroMP")):
            E = {d: S.get(lab(w, task, lr, "none" if d == 0 else arch, d)) for d in (0, 1, 4)}
            V = {d: val(w, arch, d) for d in (0, 1, 4)}
            have = [d for d in (0, 1, 4) if V[d] and E[d]]
            if not have:
                continue
            best = (max if higher else min)(have, key=lambda d: V[d]["best_val"])
            others = ", ".join(f"h{d} {pt(E[d], m):.4f}" for d in have if d != best)
            flag = lambda d: " ⚠cap" if V[d] and V[d]["stop"].startswith("CAP") else ""
            d01 = pt(E[1], m) - pt(E[0], m); d14 = pt(E[4], m) - pt(E[1], m)
            P(f"| {w} | {name} | {ci(E[0], m)}{flag(0)} | {ci(E[1], m)}{flag(1)} | {ci(E[4], m)}{flag(4)} | **h{best}** | {others} | "
              f"{d01:+.4f} {verdict(d01, band, higher)} | {d14:+.4f} {verdict(d14, band, higher)} |")
    P(f"\n#### {task} — SHARE-lite vs HeteroMP at the same depth (lite − mp)\n")
    P("| world | depth | SHARE-lite | HeteroMP | lite − mp | verdict |")
    P("|---|---|---|---|---|---|")
    for w in ("v6", "v7"):
        for d in (1, 4):
            a, b = S.get(lab(w, task, lr, "lite", d)), S.get(lab(w, task, lr, "mp", d))
            delta = pt(a, m) - pt(b, m)
            v = "—" if (np.isnan(delta) or not np.isfinite(band) or band <= 0) else ("**no difference**" if abs(delta) <= band else
                                             (f"SHARE-lite better ({abs(delta)/band:.1f}×)" if (delta > 0) == higher
                                              else f"HeteroMP better ({abs(delta)/band:.1f}×)"))
            P(f"| {w} | h{d} | {ci(a, m)} | {ci(b, m)} | {delta:+.4f} | {v} |")
    P(f"\n#### {task} — secondary metrics\n")
    P("| world | config | " + " | ".join(SECONDARY[task]) + " |")
    P("|---|---|" + "---|" * len(SECONDARY[task]))
    for w in ("v6", "v7"):
        for arch, d in (("none", 0), ("lite", 1), ("mp", 1), ("lite", 4), ("mp", 4)):
            e = S.get(lab(w, task, lr, arch, d))
            if e:
                P(f"| {w} | {arch} h{d} | " + " | ".join(ci(e, x) for x in SECONDARY[task]) + " |")
    if task == "fill_rate":
        # depth must be judged on the head AS IT SHIPS: validation-recalibrated (Phase 5 Addendum A)
        rp = os.path.join(ARTIFACTS, "phase5_recal_scores.json")
        RC = json.load(open(rp)) if os.path.exists(rp) else {}
        P("\n#### fill_rate — calibration per depth, raw and validation-recalibrated (test ECE 20-bin ↓)\n")
        P("| world | config | raw ECE 20 | recal method (val-selected) | **recal ECE 20** | recal CRPS exact | recal reliability P(f=1) |")
        P("|---|---|---|---|---|---|---|")
        for w in ("v6", "v7"):
            for arch, d in (("none", 0), ("lite", 1), ("mp", 1), ("lite", 4), ("mp", 4)):
                e = RC.get(f"{w}_fill_rate_cdf22_{arch}_h{d}_s7_lr{lr:g}_g0_w0")
                if not e:
                    continue
                sel = e["selected_on_validation"]; t = e[sel]["test"]; t0 = e["none"]["test"]
                P(f"| {w} | {arch} h{d} | {t0['ece20'][0]:.4f} | {sel} | **{ci(t, 'ece20')}** | "
                  f"{ci(t, 'crps_exact')} | {ci(t, 'rel_one')} |")
    P(f"\n#### {task} — cells\n")
    P("| world | encoder | depth | params | epochs | best | stop | s/epoch | wall s | RSS GB |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for r in g4:
        P(f"| {r['world']} | {r['arch']} | h{r['depth']} | {r['params']:,} | {r['epochs_run']} | {r['best_epoch']} | "
          f"{r['stop']} | {r['sec_per_epoch']:.1f} | {r['wall']:.0f} | {r['rss']:.2f} |")

txt = "\n".join(out)
open(os.path.join(ARTIFACTS, "phase4r_tables.md"), "w").write(txt)
print(txt if txt else "no Phase 4 re-derivation cells scored yet")
