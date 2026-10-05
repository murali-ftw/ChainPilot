"""Phase 23AC T4.3 -- render the progress charts from the CSVs written by phase23ac_t4_metrics.py (deviation 224).

RENDERING ONLY. Runs under a SEPARATE interpreter that has matplotlib (the project venv does not, and installing it there
could move numpy). Imports numpy / pandas / matplotlib / json / os / argparse only -- no repo module, no model, no score.
Every number drawn is read from reports/part2/phase23ac/t4/charts/data/*.csv.

Style: large fonts, one message per chart, the Okabe-Ito colour-blind-safe palette, the base rate as a reference line,
intervals as error bars.
  1_headline.png            headline metric per use case, INCUMBENT -> BEST PUBLISHED (LEAKY) -> CLEAN
  2_precision_vs_base.png   precision at the flagged cases (top 5%) against the base rate
  3_capacity_snapshots.png  capacity precision at 5% by test snapshot, worst snapshot marked
  4_order_time.png          order-time expected-date error vs promise date and channel averages (only if T1's CSV exists)
  5_leak_story.png          the nine leaking columns, and the drop per use case when they are removed

  <python-with-matplotlib> ml/eval/phase23ac_t4_charts.py [--data reports/part2/phase23ac/t4/charts/data] [--out reports/part2/phase23ac/t4/charts]
"""
from __future__ import annotations
import os, json, argparse
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA = os.path.join(REPO, "reports", "part2", "phase23ac", "t4", "charts", "data")
OUT = os.path.join(REPO, "reports", "part2", "phase23ac", "t4", "charts")
# Okabe-Ito
OI = dict(orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442", blue="#0072B2", vermillion="#D55E00",
          purple="#CC79A7", black="#000000", grey="#999999")
COLS = ("INCUMBENT", "BEST PUBLISHED", "CLEAN")
COLOR = {"INCUMBENT": OI["orange"], "BEST PUBLISHED": OI["sky"], "CLEAN": OI["green"]}
HATCH = {"INCUMBENT": "//", "BEST PUBLISHED": "//", "CLEAN": ""}        # hatched = LEAKY (superseded, Phase 22)
TICK = {"INCUMBENT": "Incumbent\n(leaky)", "BEST PUBLISHED": "Best published\n(leaky)", "CLEAN": "Clean"}

plt.rcParams.update({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
                     "legend.fontsize": 13, "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})


def _yerr(df):
    v = df["value"].to_numpy(float)
    lo = pd.to_numeric(df["lo"], errors="coerce").to_numpy(float); hi = pd.to_numeric(df["hi"], errors="coerce").to_numpy(float)
    lo = np.where(np.isfinite(lo), v - lo, 0.0); hi = np.where(np.isfinite(hi), hi - v, 0.0)
    return np.vstack([np.clip(lo, 0, None), np.clip(hi, 0, None)])


def _bars(ax, df, valcol="value"):
    df = df.set_index("column").reindex([c for c in COLS if c in set(df["column"])]).reset_index()
    x = np.arange(len(df))
    d = df.rename(columns={valcol: "value"})
    err = _yerr(d)
    for xi, (c, v) in enumerate(zip(d["column"], d["value"])):     # one bar at a time: per-bar hatch works on every matplotlib
        ax.bar([xi], [v], color=COLOR[c], hatch=HATCH[c], edgecolor="black", linewidth=0.8, yerr=err[:, [xi]], capsize=6,
               error_kw=dict(lw=1.6))
    ax.set_xticks(x); ax.set_xticklabels([TICK[c] for c in d["column"]])
    for xi, v in zip(x, d["value"]):
        ax.annotate(f"{v:.3g}", (xi, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=12)
    return d


def chart1(data, out):
    df = pd.read_csv(os.path.join(data, "chart1_headline.csv"))
    ucs = list(dict.fromkeys(df["use_case"]))
    n = len(ucs); ncol = 4; nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 5.2 * nrow), squeeze=False)
    for ax, uc in zip(axs.ravel(), ucs):
        d = df[df["use_case"] == uc]
        dd = _bars(ax, d)
        b = pd.to_numeric(d["base_rate"], errors="coerce").dropna()
        if len(b):
            ax.axhline(float(b.iloc[0]), color=OI["black"], ls="--", lw=1.6)
            # labelled on the line itself (a legend box hid the clean column's value label)
            ax.annotate(f"base rate {float(b.iloc[0]):.2f}", (len(d) - 0.5, float(b.iloc[0])), textcoords="offset points",
                        xytext=(-4, -16), ha="right", fontsize=11, color=OI["black"],
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        top = np.nanmax(pd.to_numeric(d[["value", "hi"]].stack(), errors="coerce").to_numpy())
        ax.set_ylim(0, top * 1.18)                       # headroom for the value labels
        if uc == "FILL":
            ax.annotate("bars: absolute CRPS, which varies by snapshot;\nclean vs Phase 19 recipe, paired: +0.0013 [0.0003, 0.0025] better",
                        (0.02, 0.03), xycoords="axes fraction", fontsize=10, color=OI["black"],
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
        hib = int(d["higher_is_better"].iloc[0]) == 1
        cls = [str(c) for c in dd["phase15_class"].fillna("")]
        ax.set_title(f"{uc}\n{d['metric'].iloc[0]} ({'higher' if hib else 'lower'} = better)", fontsize=14)
        for xi, c in enumerate(cls):
            if c and c != "—" and c != "nan":
                ax.annotate(c, (xi, 0), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=11, color="white",
                            fontweight="bold")
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Every headline, three columns: hatched = LEAKY (superseded, Phase 22); green = clean", fontsize=19)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(out, "1_headline.png"); fig.savefig(p); plt.close(fig)
    return p


def chart2(data, out):
    df = pd.read_csv(os.path.join(data, "chart2_precision_vs_base.csv"))
    ucs = list(dict.fromkeys(df["use_case"]))
    fig, ax = plt.subplots(figsize=(15, 6.5))
    w = 0.26
    for j, c in enumerate(COLS):
        d = df[df["column"] == c].set_index("use_case").reindex(ucs)
        x = np.arange(len(ucs)) + (j - 1) * w
        v = d["precision_at_5pct"].to_numpy(float)
        lo = pd.to_numeric(d["lo"], errors="coerce").to_numpy(float); hi = pd.to_numeric(d["hi"], errors="coerce").to_numpy(float)
        err = np.vstack([np.where(np.isfinite(lo), v - lo, 0), np.where(np.isfinite(hi), hi - v, 0)])
        err = np.nan_to_num(np.clip(err, 0, None))
        ax.bar(x, np.nan_to_num(v), w, color=COLOR[c], hatch=HATCH[c], edgecolor="black", linewidth=0.8, yerr=err, capsize=5,
               label=TICK[c].replace("\n", " "))
    for i, uc in enumerate(ucs):
        b = pd.to_numeric(df[df["use_case"] == uc]["base_rate"], errors="coerce").dropna()
        if len(b):
            ax.hlines(float(b.iloc[0]), i - 1.6 * w, i + 1.6 * w, colors=OI["black"], linestyles="--", lw=2.2,
                      label="base rate (chance)" if i == 0 else None)
    ax.set_xticks(np.arange(len(ucs))); ax.set_xticklabels(ucs)
    ax.set_ylim(0, 1.05); ax.set_ylabel("precision in the top 5%")
    ax.set_title("At the flagged cases: how often the list is right, against the base rate (dashed)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)
    fig.tight_layout()
    p = os.path.join(out, "2_precision_vs_base.png"); fig.savefig(p); plt.close(fig)
    return p


def chart3(data, out):
    df = pd.read_csv(os.path.join(data, "chart3_capacity_snapshots.csv"))
    fig, ax = plt.subplots(figsize=(14, 6.5))
    snaps = sorted(set(df["snapshot"]))
    x = np.arange(len(snaps))
    mk = {"INCUMBENT": "o", "BEST PUBLISHED": "s", "CLEAN": "D"}
    worst = {}
    for c in COLS:
        d = df[df["column"] == c].set_index("snapshot").reindex(snaps)
        if d["precision_at_5pct"].notna().sum() == 0:
            continue
        ax.plot(x, d["precision_at_5pct"], marker=mk[c], ms=9, lw=2.4, color=COLOR[c], label=TICK[c].replace("\n", " "))
        w = d[d["is_worst"] == 1]
        for s_, r in w.iterrows():
            xi = snaps.index(s_)
            ax.scatter([xi], [r["precision_at_5pct"]], s=320, facecolors="none", edgecolors=OI["vermillion"], linewidths=2.6, zorder=5)
            worst.setdefault(s_, []).append((TICK[c].replace("\n", " "), r["precision_at_5pct"]))
    for s_, items in worst.items():              # one combined note per worst snapshot (separate notes overlapped)
        xi = snaps.index(s_); y = min(v for _, v in items)
        txt = f"worst snapshot {s_}:\n" + "\n".join(f"{n} {v:.2f}" for n, v in items)
        ax.annotate(txt, (xi, y), textcoords="offset points", xytext=(-24, -10), ha="right", va="top", fontsize=12,
                    color=OI["vermillion"])
    ax.set_xticks(x); ax.set_xticklabels(snaps, rotation=30, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("precision in the top 5% of the snapshot")
    ax.set_title("Capacity precision in the top 5%, by test snapshot (worst snapshot circled)")
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    p = os.path.join(out, "3_capacity_snapshots.png"); fig.savefig(p); plt.close(fig)
    return p


def chart4(data, out):
    f = os.path.join(data, "chart4_order_time.csv")
    if not os.path.exists(f):
        print("chart 4 SKIPPED: chart4_order_time.csv does not exist (Track T1 not done)")
        return None
    df = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(13, 6.5))
    names = df["estimator"].astype(str).tolist()
    colors = [OI["green"] if "product" in n.lower() else OI["grey"] for n in names]
    y = np.arange(len(df))
    ax.barh(y, df["a3_days"], color=colors, edgecolor="black")
    for yi, (a3, w7) in enumerate(zip(df["a3_days"], df["share_within_7d"])):
        lab = f"{a3:.1f} d" + (f"   ({w7:.0%} within ±7 d)" if pd.notna(w7) else "")
        ax.annotate(lab, (a3, yi), textcoords="offset points", xytext=(6, -5), fontsize=12)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.invert_yaxis(); ax.set_xlabel("A3: median |predicted − actual arrival| (days)")
    ax.set_title("Order-time expected date vs the promise date and channel averages (lower is better)")
    fig.tight_layout()
    p = os.path.join(out, "4_order_time.png"); fig.savefig(p); plt.close(fig)
    return p


def chart5(data, out):
    cols = pd.read_csv(os.path.join(data, "chart5_leak_columns.csv"))
    drop = pd.read_csv(os.path.join(data, "chart5_leak_drop.csv"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(19, 7), gridspec_kw=dict(width_ratios=[1, 1.15]))
    c = cols.sort_values("share_channels_changed_H_week")
    a1.barh(c["column_name"], c["share_channels_changed_H_week"], color=OI["vermillion"], edgecolor="black")
    a1.set_xlim(0, 1); a1.set_xlabel("share of channels whose row moved when\nfuture rows were poisoned")
    a1.set_title("The nine leaking columns (Phase 22 audit)")
    # drop per use case, as a signed change in the direction 'better'
    sgn = np.where(drop["higher_is_better"] == 1, 1.0, -1.0)
    v = sgn * drop["delta_clean_minus_published"].to_numpy(float)
    lo, hi = np.sort(np.vstack([sgn * drop["lo"].to_numpy(float), sgn * drop["hi"].to_numpy(float)]), axis=0)
    err = np.vstack([np.clip(v - lo, 0, None), np.clip(hi - v, 0, None)])
    y = np.arange(len(drop))
    col = [OI["vermillion"] if x < 0 else OI["blue"] for x in v]
    a2.barh(y, v, color=col, edgecolor="black", xerr=err, capsize=5)
    a2.axvline(0, color=OI["black"], lw=1.4)
    a2.set_yticks(y); a2.set_yticklabels(drop["use_case"])
    a2.invert_yaxis()
    a2.set_xlabel("clean − published, signed so that left = worse\n(neural ensemble; 95% snapshot-block interval)")
    a2.set_title("What removing the leak costs, per use case")
    for yi, (x, m) in enumerate(zip(v, drop["verdict"])):
        a2.annotate(m, (x, yi), textcoords="offset points", xytext=(6 if x >= 0 else -6, 8), ha="left" if x >= 0 else "right", fontsize=11)
    fig.tight_layout()
    p = os.path.join(out, "5_leak_story.png"); fig.savefig(p); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default=DATA); ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = {}
    for name, fn in (("chart1", chart1), ("chart2", chart2), ("chart3", chart3), ("chart4", chart4), ("chart5", chart5)):
        res[name] = fn(a.data, a.out)
        print(f"{name}: {res[name] if res[name] else 'skipped'}")
    json.dump(res, open(os.path.join(a.out, "charts_written.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
