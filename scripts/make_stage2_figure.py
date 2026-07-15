#!/usr/bin/env python3
"""
Stage-2 headline figure: small-job walltime cap vs large-job starvation.

Reads results/stage2_small_walltime_sweep/sweep_results.csv (produced by
`python sweep.py --grid configs/sweeps/stage2_small_walltime_sweep.yaml`) and
draws the slide-ready tradeoff: as small jobs are allowed to run longer
(24h -> 7 days), does the large (>=1920-node) job wait time blow up? With the
draining EASY reservation, it does not.

Run from repo root:  python scripts/make_stage2_figure.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "docs", "figures")
os.makedirs(OUT, exist_ok=True)
CSV = "results/stage2_small_walltime_sweep/sweep_results.csv"


def _small_cap(bp_str):
    # breakpoints stored as string like "[[1, 168], [512, 48], [1920, 24]]"
    import ast
    bp = ast.literal_eval(bp_str)
    return bp[0][1]  # walltime at the 1-node breakpoint


def main():
    df = pd.read_csv(CSV)
    df = df[df["status"] == "OK"].copy()
    df["small_cap_h"] = df["walltime_policy.breakpoints"].map(_small_cap)
    df = df.sort_values("small_cap_h")

    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(df))  # categorical even spacing
    ax.plot(x, df["big_wait_p95_h"], "o-", lw=2.5, ms=10,
            color="#c0392b", label="Large job (≥1920 nodes) p95 wait")
    ax.plot(x, df["small_wait_p95_h"], "s--", lw=2, ms=8,
            color="#2c7fb8", label="Small job p95 wait")
    # value labels on each marker
    for xi, yb, ys in zip(x, df["big_wait_p95_h"], df["small_wait_p95_h"]):
        ax.annotate(f"{yb:.0f}h", (xi, yb), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=9, color="#c0392b")
        ax.annotate(f"{ys:.0f}h", (xi, ys), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9, color="#2c7fb8")
    ax.set_xlabel("Small-job walltime cap")
    ax.set_ylabel("p95 queued time (hours)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(c)}h" + (f"\n({c/24:.0f}d)" if c >= 48 else "")
                        for c in df["small_cap_h"]])
    ax.set_ylim(0, max(df["small_wait_p95_h"].max(), df["big_wait_p95_h"].max()) * 1.35)
    ax.grid(alpha=.3, axis="y")
    ax.legend(loc="upper left", fontsize=10)
    util_txt = "   ".join(f"{int(c)}h→{u:.0f}%" for c, u in
                          zip(df["small_cap_h"], df["avg_util_pct"]))
    ax.set_title("Small jobs can run up to 7 days at ~zero cost to large-job wait\n"
                 "single queue + max_walltime(nodes) + draining EASY reservation "
                 "(9600 production nodes)\n"
                 f"large-job wait flat ~11h · 0 large jobs starved · "
                 f"utilization: {util_txt}", fontsize=9.5)
    fig.tight_layout()
    path = os.path.join(OUT, "stage2_small_walltime_vs_bigwait.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.relpath(path)}")
    print(df[["small_cap_h", "big_wait_p95_h", "n_big_unstarted",
              "small_wait_p95_h", "avg_util_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
