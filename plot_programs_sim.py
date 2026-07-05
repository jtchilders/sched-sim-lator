#!/usr/bin/env python3
"""
plot_programs_sim.py — Committee-facing figures for sim_programs.py outputs.

Reads:
  1. Decision table CSV  (--csv)                  : per-program summary
  2. <stem>_telemetry.csv (--telemetry, inferred) : cumulative delivered node-hours over time
  3. <stem>_util.csv      (--util, inferred)       : machine utilisation samples

Produces (saved to --outdir):
  A. delivered_vs_target.png   — target vs delivered share grouped bar chart
  B. stacked_delivered_nh.png  — stacked area: cumulative NH by program over time
  C. budget_burn.png           — horizontal bars: budget burn per program
  D. wait_percentiles.png      — grouped bars: wait p50/p95/max per program

Usage:
  python3 plot_programs_sim.py --csv results/programs_sim/decision.csv \\
      [--telemetry PATH] [--util PATH] [--outdir results/programs_plots] \\
      [--title "Policy=budget, 365d"]

If --telemetry/--util are not given they are inferred as <csv_stem>_telemetry.csv
and <csv_stem>_util.csv.  If a file is missing the corresponding figure is
skipped with a warning rather than crashing.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")                # must come before pyplot import
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# House style — dark theme matching existing slides
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.facecolor": "#0a1628",
    "axes.facecolor": "#112240",
    "axes.edgecolor": "#8892b0",
    "axes.labelcolor": "#e6f1ff",
    "text.color": "#e6f1ff",
    "xtick.color": "#8892b0",
    "ytick.color": "#8892b0",
    "grid.color": "#1d3461",
    "grid.alpha": 0.5,
    "legend.facecolor": "#112240",
    "legend.edgecolor": "#8892b0",
    "font.size": 12,
})

# Per-program colour palette — consistent across ALL figures.
PROG_COLORS = {
    "INCITE":  "#64ffda",   # teal-green  — largest allocation
    "ALCC":    "#57cbff",   # sky-blue
    "DD":      "#ffb347",   # amber
    "Genesis": "#ff6b8a",   # coral-pink  — the new entrant
}
PROG_ORDER = ["INCITE", "ALCC", "DD", "Genesis"]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _color(prog: str) -> str:
    return PROG_COLORS.get(prog, "#cccccc")


def _ensure_outdir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)


def _load_csv(path: str, label: str) -> "pd.DataFrame | None":
    """Load CSV or return None with a clear warning."""
    if path is None or not os.path.exists(path):
        warnings.warn(
            f"[plot_programs_sim] {label} file not found: {path!r} — "
            "skipping dependent figure(s).",
            stacklevel=3,
        )
        return None
    return pd.read_csv(path)


def _infer_path(csv_path: str, suffix: str) -> str:
    """Derive telemetry/util path from the decision-table CSV stem."""
    stem = csv_path
    for ext in (".csv", ".CSV"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    return stem + suffix


def _bar_label(ax, rects, fmt="{:.2f}", fontsize=9, padding=3, color="#e6f1ff"):
    """Annotate bar tops with formatted values."""
    for rect in rects:
        h = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            h + padding * (ax.get_ylim()[1] - ax.get_ylim()[0]) / 200.0,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=color,
        )


# ---------------------------------------------------------------------------
# Figure A — delivered vs target share
# ---------------------------------------------------------------------------

def fig_delivered_vs_target(df: pd.DataFrame, outdir: str, title: str = "") -> str:
    progs = [p for p in PROG_ORDER if p in df["program"].values]
    n = len(progs)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    targets = [df.loc[df["program"] == p, "target_share"].values[0] for p in progs]
    delivered = [df.loc[df["program"] == p, "delivered_share"].values[0] for p in progs]

    rects_t = ax.bar(x - width / 2, targets, width,
                     color=[_color(p) for p in progs], alpha=0.45,
                     label="Target share", edgecolor="#8892b0", linewidth=0.6)
    rects_d = ax.bar(x + width / 2, delivered, width,
                     color=[_color(p) for p in progs], alpha=0.90,
                     label="Delivered share", edgecolor="#8892b0", linewidth=0.6)

    # Bar labels (percentage)
    for rect in list(rects_t) + list(rects_d):
        h = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            h + 0.003,
            f"{h:.1%}",
            ha="center", va="bottom", fontsize=8.5, color="#e6f1ff",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(progs, fontsize=12)
    ax.set_ylabel("Share of total node-hours delivered")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, max(max(targets), max(delivered)) * 1.20)
    ax.grid(True, axis="y", alpha=0.4)
    ax.legend(fontsize=10, loc="upper right")

    head = "Delivered vs Target Node-Hour Share"
    if title:
        head = f"{head}\n{title}"
    ax.set_title(head, fontsize=14, pad=12)

    # Horizontal policy note
    ax.axhline(0.0, color="#8892b0", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.text(n - 0.5, ax.get_ylim()[1] * 0.97,
            "Bars closer in height = better policy alignment",
            ha="right", va="top", fontsize=8, color="#8892b0", style="italic")

    fig.tight_layout()
    out = os.path.join(outdir, "delivered_vs_target.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")
    return out


# ---------------------------------------------------------------------------
# Figure B — stacked cumulative delivered node-hours over time
# ---------------------------------------------------------------------------

def fig_stacked_delivered_nh(tel: pd.DataFrame, outdir: str, title: str = "") -> str:
    # Expected columns: t_h, program, delivered_nh
    required = {"t_h", "program", "delivered_nh"}
    if not required.issubset(tel.columns):
        missing = required - set(tel.columns)
        warnings.warn(
            f"[plot_programs_sim] telemetry CSV missing columns {missing} — "
            "skipping stacked_delivered_nh.png"
        )
        return ""

    progs = [p for p in PROG_ORDER if p in tel["program"].unique()]
    pivoted = tel.pivot_table(index="t_h", columns="program",
                              values="delivered_nh", aggfunc="last").ffill().fillna(0.0)
    # Reindex to the desired order (subset present)
    progs = [p for p in PROG_ORDER if p in pivoted.columns]
    pivoted = pivoted[progs]

    days = pivoted.index / 24.0

    fig, ax = plt.subplots(figsize=(12, 6))

    bottom = np.zeros(len(days))
    for prog in progs:
        vals = pivoted[prog].to_numpy()
        ax.fill_between(days, bottom, bottom + vals,
                        color=_color(prog), alpha=0.75, label=prog,
                        linewidth=0)
        ax.plot(days, bottom + vals, color=_color(prog), linewidth=0.8, alpha=0.6)
        bottom = bottom + vals

    ax.set_xlabel("Simulation time (days)")
    ax.set_ylabel("Cumulative delivered node-hours")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"
    ))
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=10, loc="upper left")

    head = "Cumulative Delivered Node-Hours by Program (Stacked)"
    if title:
        head = f"{head}\n{title}"
    ax.set_title(head, fontsize=14, pad=12)

    fig.tight_layout()
    out = os.path.join(outdir, "stacked_delivered_nh.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")
    return out


# ---------------------------------------------------------------------------
# Figure C — budget burn horizontal bar chart
# ---------------------------------------------------------------------------

def fig_budget_burn(df: pd.DataFrame, outdir: str, title: str = "") -> str:
    progs = [p for p in PROG_ORDER if p in df["program"].values]
    burns = [df.loc[df["program"] == p, "budget_burn"].values[0] for p in progs]

    fig, ax = plt.subplots(figsize=(9, 5))

    y = np.arange(len(progs))
    colors = []
    for prog, burn in zip(progs, burns):
        c = _color(prog)
        if burn > 1.0:
            # Overburn: use a more saturated/warning tint of the program colour
            colors.append("#ff4444")   # clear overburn indicator
        else:
            colors.append(c)

    bars = ax.barh(y, burns, color=colors, alpha=0.85,
                   edgecolor="#8892b0", linewidth=0.6, height=0.5)

    # Value labels on bars
    for bar, burn in zip(bars, burns):
        ax.text(
            burn + 0.01,
            bar.get_y() + bar.get_height() / 2.0,
            f"{burn:.2f}×",
            va="center", ha="left", fontsize=10, color="#e6f1ff",
        )

    # Reference lines
    ax.axvline(1.0, color="#e6f1ff", linewidth=1.4, linestyle="--",
               label="Full budget (1.0×)", alpha=0.85)
    ax.axvline(1.25, color="#ffb347", linewidth=1.2, linestyle=":",
               label="INCITE overburn ceiling (1.25×)", alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(progs, fontsize=12)
    ax.set_xlabel("Budget burn (delivered ÷ annual budget)")
    ax.invert_yaxis()
    ax.set_xlim(0, max(max(burns) * 1.25, 1.35))
    ax.grid(True, axis="x", alpha=0.35)
    ax.legend(fontsize=9, loc="lower right")

    head = "Budget Burn by Program"
    if title:
        head = f"{head}\n{title}"
    ax.set_title(head, fontsize=14, pad=12)

    fig.tight_layout()
    out = os.path.join(outdir, "budget_burn.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")
    return out


# ---------------------------------------------------------------------------
# Figure D — wait percentiles grouped bar chart
# ---------------------------------------------------------------------------

def fig_wait_percentiles(df: pd.DataFrame, outdir: str, title: str = "") -> str:
    progs = [p for p in PROG_ORDER if p in df["program"].values]
    n = len(progs)
    x = np.arange(n)
    width = 0.24

    metrics = [
        ("wait_p50_h", "p50", -1.0 * width),
        ("wait_p95_h", "p95",  0.0),
        ("wait_max_h", "max",  1.0 * width),
    ]
    # Check all metric columns exist
    missing_cols = [m for m, _, _ in metrics if m not in df.columns]
    if missing_cols:
        warnings.warn(
            f"[plot_programs_sim] decision table missing columns {missing_cols} — "
            "skipping wait_percentiles.png"
        )
        return ""

    fig, ax = plt.subplots(figsize=(11, 6))

    alphas = [0.65, 0.82, 0.95]
    all_vals = []
    for (col, lbl, offset), alpha in zip(metrics, alphas):
        vals = [df.loc[df["program"] == p, col].values[0] for p in progs]
        all_vals.extend(vals)
        rects = ax.bar(x + offset, vals, width,
                       color=[_color(p) for p in progs],
                       alpha=alpha,
                       label=lbl,
                       edgecolor="#8892b0", linewidth=0.5)
        # Label bars only for p50 and p95 (max can be very tall on log scale)
        if col != "wait_max_h":
            for rect, v in zip(rects, vals):
                ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    rect.get_height() * 1.04,
                    f"{v:.1f}h",
                    ha="center", va="bottom", fontsize=7.5, color="#e6f1ff",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(progs, fontsize=12)
    ax.set_ylabel("Wait time (hours)")

    # Use log scale when max dwarfs p50 by more than 50×
    if all_vals and max(all_vals) > 50 * np.median([v for v in all_vals if v > 0]):
        ax.set_yscale("log")
        ax.set_ylabel("Wait time (hours) — log scale")

    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(fontsize=10, title="Percentile", title_fontsize=9, loc="upper right")

    head = "Queue Wait Times by Program"
    if title:
        head = f"{head}\n{title}"
    ax.set_title(head, fontsize=14, pad=12)

    fig.tight_layout()
    out = os.path.join(outdir, "wait_percentiles.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Generate committee-facing figures from sim_programs.py CSV outputs."
    )
    ap.add_argument(
        "--csv", required=True,
        help="Decision table CSV produced by sim_programs.py --csv PATH.",
    )
    ap.add_argument(
        "--telemetry", default=None,
        help="Telemetry CSV (<stem>_telemetry.csv).  Inferred from --csv stem if omitted.",
    )
    ap.add_argument(
        "--util", default=None,
        help="Util CSV (<stem>_util.csv).  Inferred from --csv stem if omitted.  "
             "(Reserved for future figures; currently unused.)",
    )
    ap.add_argument(
        "--outdir", default="results/programs_plots",
        help="Directory for output PNGs (created if absent).  Default: results/programs_plots",
    )
    ap.add_argument(
        "--title", default="",
        help="Optional subtitle appended to every figure title (e.g. policy/duration).",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    # Resolve inferred paths
    tel_path = args.telemetry or _infer_path(args.csv, "_telemetry.csv")
    util_path = args.util or _infer_path(args.csv, "_util.csv")

    _ensure_outdir(args.outdir)

    # Load mandatory decision table
    if not os.path.exists(args.csv):
        sys.exit(f"ERROR: decision table CSV not found: {args.csv!r}")
    df = pd.read_csv(args.csv)
    print(f"Loaded decision table: {args.csv}  ({len(df)} programs)")

    # Load optional files
    tel = _load_csv(tel_path, "telemetry")
    # util is reserved for future figures; load anyway for forward compat
    _load_csv(util_path, "util")

    produced = []

    print("\nFigure A — delivered vs target share")
    produced.append(fig_delivered_vs_target(df, args.outdir, args.title))

    print("Figure B — stacked delivered node-hours")
    if tel is not None:
        produced.append(fig_stacked_delivered_nh(tel, args.outdir, args.title))
    else:
        print("  SKIPPED (telemetry file not found)")

    print("Figure C — budget burn")
    produced.append(fig_budget_burn(df, args.outdir, args.title))

    print("Figure D — wait percentiles")
    produced.append(fig_wait_percentiles(df, args.outdir, args.title))

    print(f"\n{'='*60}")
    print(f"Done.  {sum(1 for p in produced if p)} PNG(s) saved to {args.outdir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
