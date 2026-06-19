#!/usr/bin/env python3
"""
Visualize per-program profiles: node-size, walltime, and seasonal burn curves.

Produces results/programs/:
  - program_seasonal.png   : monthly burn multipliers per program
  - program_sizes.png      : node-size CDFs per program
  - program_walltime.png   : walltime CDFs per program
  - program_summary.png    : delivered-share bar + summary table
"""
import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import program_profiles as pp

# Dark theme to match existing slides
plt.rcParams.update({
    "figure.facecolor": "#0a1628", "axes.facecolor": "#112240",
    "axes.edgecolor": "#8892b0", "axes.labelcolor": "#e6f1ff",
    "text.color": "#e6f1ff", "xtick.color": "#8892b0",
    "ytick.color": "#8892b0", "grid.color": "#1d3461",
    "grid.alpha": 0.5, "legend.facecolor": "#112240",
    "legend.edgecolor": "#8892b0", "font.size": 12,
})
COLORS = {pp.INCITE: "#64ffda", pp.ALCC: "#57cbff",
          pp.DD: "#ffb347", pp.GENESIS: "#ff6b8a"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def plot_seasonal(profiles, outdir):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(12)
    for name, prof in profiles.items():
        ax.plot(x, prof.monthly_multiplier, marker="o", linewidth=2,
                color=COLORS.get(name, "#ccc"), label=name)
    ax.axhline(1.0, color="#8892b0", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(MONTHS)
    ax.set_ylabel("Seasonal arrival multiplier (mean=1.0)")
    ax.set_title("Per-program seasonal burn curve\n"
                 "INCITE: Nov year-end burn + Jan 13th-month  |  "
                 "ALCC: May strong finish  |  Genesis: Jul ramp")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{outdir}/program_seasonal.png", dpi=130)
    plt.close(fig)


def _cdf(ax, arr, label, color, logx=False):
    a = np.sort(arr)
    y = np.arange(1, len(a) + 1) / len(a)
    ax.plot(a, y, linewidth=2, color=color, label=label)
    if logx:
        ax.set_xscale("log")


def plot_sizes(profiles, outdir):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, prof in profiles.items():
        _cdf(ax, prof.nodes_arr, name, COLORS.get(name, "#ccc"), logx=True)
    ax.set_xlabel("Nodes per job (log)"); ax.set_ylabel("CDF")
    ax.set_title("Job size distribution by program")
    ax.legend(); ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(f"{outdir}/program_sizes.png", dpi=130)
    plt.close(fig)


def plot_walltime(profiles, outdir):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, prof in profiles.items():
        _cdf(ax, prof.walltime_arr, name, COLORS.get(name, "#ccc"))
    ax.set_xlabel("Requested walltime (hours)"); ax.set_ylabel("CDF")
    ax.set_title("Walltime distribution by program "
                 "(Genesis: 7-day dominated)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{outdir}/program_walltime.png", dpi=130)
    plt.close(fig)


def plot_summary(profiles, outdir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    names = list(profiles.keys())
    shares = [profiles[n].annual_share * 100 for n in names]
    colors = [COLORS.get(n, "#ccc") for n in names]
    bars = ax1.bar(names, shares, color=colors, alpha=0.85)
    for b, s in zip(bars, shares):
        ax1.text(b.get_x() + b.get_width() / 2, s, f"{s:.0f}%",
                 ha="center", va="bottom", fontsize=12)
    ax1.set_ylabel("Annual allocation share (%)")
    ax1.set_title("Target allocation shares")
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.axis("off")
    rows = [["program", "n_jobs", "λ/day", "nodes\np50/p90",
             "walltime h\np50/p90", "share", "overburn"]]
    for n in names:
        p = profiles[n]
        rows.append([
            n,
            f"{p.n_source_jobs:,}" if not p.synthetic else "synth",
            f"{p.arrival_rate_h*24:.0f}",
            f"{int(np.percentile(p.nodes_arr,50))}/{int(np.percentile(p.nodes_arr,90))}",
            f"{np.percentile(p.walltime_arr,50):.0f}/{np.percentile(p.walltime_arr,90):.0f}",
            f"{p.annual_share:.0%}",
            f"+{p.overburn:.0%}" if p.overburn else "—",
        ])
    tbl = ax2.table(cellText=rows[1:], colLabels=rows[0], loc="center",
                    cellLoc="center",
                    colWidths=[0.16, 0.16, 0.11, 0.16, 0.16, 0.12, 0.13])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#1d3461")
        if r == 0:
            cell.set_facecolor("#1d3461"); cell.set_text_props(color="#e6f1ff")
        else:
            cell.set_facecolor("#112240"); cell.set_text_props(color="#e6f1ff")
    ax2.set_title("Program profile summary")
    fig.tight_layout()
    fig.savefig(f"{outdir}/program_summary.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-db",
                    default=os.environ.get("PBS_SIM_DB",
                                           "/Users/jchilders/pbs_monitor_aurora.db"))
    ap.add_argument("--outdir", default="results/programs")
    ap.add_argument("--machine-nodes", type=int, default=10624)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    profiles = pp.build_profiles(args.trace_db, machine_nodes=args.machine_nodes)
    pp.print_summary(profiles)
    plot_seasonal(profiles, args.outdir)
    plot_sizes(profiles, args.outdir)
    plot_walltime(profiles, args.outdir)
    plot_summary(profiles, args.outdir)
    print(f"\nPlots written to {args.outdir}/")


if __name__ == "__main__":
    main()
