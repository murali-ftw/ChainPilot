"""Phase 24 Stage 3 -- render rescue_published_vs_clean.png. RENDERING ONLY (Phase 23AC deviation 224's arrangement, deviation 265).

Runs under a separate interpreter with matplotlib (the project venv has none). Imports numpy / pandas / matplotlib / os /
argparse only; every number drawn is read from reports/part2/phase24/charts/data/rescue_published_vs_clean.csv, written by
ml/eval/phase24_metrics_v2.py. Style as Phase 23AC: Okabe-Ito palette, leaky bars hatched, base rate as a reference line,
intervals as error bars, the source named on the chart.

  <python-with-matplotlib> ml/eval/phase24_chart.py
"""
from __future__ import annotations
import os, argparse
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA = os.path.join(REPO, "reports", "part2", "phase24", "charts", "data", "rescue_published_vs_clean.csv")
OUT = os.path.join(REPO, "reports", "part2", "phase24", "charts", "rescue_published_vs_clean.png")
OI = dict(orange="#E69F00", sky="#56B4E9", green="#009E73", blue="#0072B2", vermillion="#D55E00", grey="#999999", black="#000000")
plt.rcParams.update({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15, "xtick.labelsize": 12, "ytick.labelsize": 13,
                     "legend.fontsize": 12, "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default=DATA); ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    df = pd.read_csv(a.data)
    fig, ax = plt.subplots(figsize=(12, 6.8))
    x = np.arange(len(df))
    col = [OI["sky"] if "imulation" in r else OI["grey"] if "history" in r else OI["green"] for r in df["arm"]]
    v = df["value"].to_numpy(float)
    err = np.vstack([v - df["lo"].to_numpy(float), df["hi"].to_numpy(float) - v])
    bars = ax.bar(x, v, color=col, edgecolor=OI["black"], linewidth=1.0,
                  hatch=["//" if bool(l) else "" for l in df["leaky"]], yerr=err, capsize=7, error_kw=dict(lw=1.6))
    base = float(df["base_rate"].iloc[0])
    ax.axhline(base, color=OI["black"], ls="--", lw=1.6)
    ax.text(len(df) - 0.5, base + 0.012, f"base rate {base:.3f} (a transfer-in that week)", ha="right", va="bottom", fontsize=12)
    for i, (b, r) in enumerate(zip(bars, df.itertuples())):
        ax.text(b.get_x() + b.get_width() / 2, r.hi + 0.015, f"{r.value:.3f}" + (" (Q)" if r.status == "QUOTED" else ""),
                ha="center", va="bottom", fontsize=13)
    ax.set_xticks(x); ax.set_xticklabels(df["arm"])
    ax.set_ylim(0, 1.0); ax.set_ylabel("precision at the Phase 17 operating point")
    ax.set_title("Predict-the-rescue: published (leaky, hatched) vs clean", loc="left")
    fig.text(0.01, 0.01, "Test 2025, v8 seed 1001. 5-seed ensembles; error bars = snapshot-block 95% (1,000 resamples). (Q) = QUOTED "
             "from reports/part2/phase23b/stage2_rescue_clean.md;\nsimulation bars RECOMPUTED, ml/artifacts/phase24/stage2_sim.json "
             "(5 fill seeds, mean of per-seed precision).", fontsize=10, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out); plt.close(fig)
    print(a.out)


if __name__ == "__main__":
    main()
