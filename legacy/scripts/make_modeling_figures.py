#!/usr/bin/env python3
"""
Generate the modeling figures embedded in MODELING.md.

Reproducible: reads the same trace the generator uses (via load_trace) and the
same Genesis synthesis, writes PNGs to docs/figures/. Run from the repo root:

    python scripts/make_modeling_figures.py

All figures are small and committed (see .gitignore exception) so MODELING.md
renders on GitHub.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SimConfig, GenesisConfig
from generator import load_trace, fit_burn_curve
from genesis import build_genesis_arrays, genesis_burn

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "docs", "figures")
os.makedirs(OUT, exist_ok=True)

PROGS = ["INCITE", "ALCC", "DD"]
COL = {"INCITE": "#1f77b4", "ALCC": "#ff7f0e", "DD": "#2ca02c",
       "Genesis": "#d62728"}
TIERS = ["capacity", "small", "medium", "large"]
CFG_PATH = "configs/validate_baseline.yaml"


def _cdf(ax, data, label, color, logx=False):
    x = np.sort(np.asarray(data, float))
    x = x[np.isfinite(x)]
    if logx:
        x = np.clip(x, 1e-2, None)
    y = np.linspace(0, 1, len(x))
    ax.plot(x, y, label=label, color=color, lw=2)


def fig_distributions(df):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for p in PROGS:
        s = df[df["program"] == p]
        _cdf(ax[0], s["nodes"], p, COL[p], logx=True)
        _cdf(ax[1], s["walltime_h"], p, COL[p])
        _cdf(ax[2], s["runtime_h"], p, COL[p], logx=True)
    ax[0].set_xscale("log"); ax[0].set_xlabel("nodes (log)")
    ax[0].set_title("Node-count CDF"); ax[0].legend()
    ax[1].set_xlabel("requested walltime (h)"); ax[1].set_title("Walltime CDF")
    ax[1].set_xlim(0, 80)
    ax[2].set_xscale("log"); ax[2].set_xlabel("actual runtime (h, log)")
    ax[2].set_title("Runtime CDF")
    for a in ax:
        a.set_ylabel("CDF"); a.grid(alpha=.3)
    fig.suptitle("Empirical per-program distributions the generator samples "
                 "(Aurora trace: 357 days, 414K finished jobs)", fontsize=12)
    fig.tight_layout()
    _save(fig, "program_distributions.png")


def fig_size_tier_mix(df):
    df = df.copy()
    df["nh"] = df["nodes"] * df["runtime_h"]
    by_count = pd.crosstab(df["program"], df["size_tier"], normalize="index") * 100
    by_nh = df.pivot_table(index="program", columns="size_tier", values="nh",
                           aggfunc="sum")
    by_nh = by_nh.div(by_nh.sum(axis=1), axis=0) * 100
    by_count = by_count.reindex(index=PROGS, columns=TIERS)
    by_nh = by_nh.reindex(index=PROGS, columns=TIERS)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    tier_col = ["#7fb3d5", "#f7cb4f", "#82c99a", "#c0392b"]
    for a, data, title in [(ax[0], by_count, "By job COUNT"),
                           (ax[1], by_nh, "By NODE-HOURS delivered")]:
        bottom = np.zeros(len(PROGS))
        for ti, tier in enumerate(TIERS):
            vals = data[tier].to_numpy()
            a.bar(PROGS, vals, bottom=bottom, label=tier, color=tier_col[ti])
            bottom += vals
        a.set_ylabel("%"); a.set_title(title); a.set_ylim(0, 100)
        a.legend(title="size tier", loc="upper right", fontsize=8)
    fig.suptitle("Size-tier composition per program — the crux of the "
                 "capacity-protection question:\n~85–92% of JOBS are ≤128 nodes, "
                 "but they use only ~11–15% of NODE-HOURS", fontsize=11)
    fig.tight_layout()
    _save(fig, "size_tier_mix.png")


def fig_burn_curves(df):
    cfg = SimConfig.from_yaml(CFG_PATH)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    off = np.arange(12)
    for p in cfg.programs:
        bc = fit_burn_curve(df, p)
        ax.plot(off, bc, marker="o", label=f"{p.name} (yr starts M{p.alloc_year_start_month})",
                color=COL.get(p.name, "gray"), lw=2)
    ax.axhline(1.0, color="k", ls="--", alpha=.4, label="mean (1.0)")
    ax.set_xlabel("months since program's allocation-year start (offset 0..11)")
    ax.set_ylabel("arrival-rate multiplier")
    ax.set_title("Seasonal burn curves keyed on allocation-year offset\n"
                 "(INCITE fast start + year-end burn; ALCC slow post-July pickup; "
                 "DD flat) — note: noisy, 357-day trace")
    ax.set_xticks(off); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "burn_curves.png")


def fig_genesis(cfg_nodes=10624):
    scenarios = ["ai_default", "incite_like", "bursty_campaign"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    rng = np.random.default_rng(7)
    for sc in scenarios:
        gc = GenesisConfig(enabled=True, scenario=sc, share=0.15)
        n, w, r = build_genesis_arrays(gc, rng)
        _cdf(ax[0], n, sc, None, logx=True)
        _cdf(ax[1], w, sc, None)
    # node CDF
    ax[0].set_xscale("log"); ax[0].set_xlabel("nodes (log)")
    ax[0].set_title("Genesis node-count CDF by scenario"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("walltime (h)"); ax[1].set_title("Genesis walltime CDF by scenario")
    ax[1].legend(fontsize=8)
    # ramp
    for sc in scenarios:
        gc = GenesisConfig(enabled=True, scenario=sc, share=0.15)
        ax[2].plot(range(1, 13), genesis_burn(gc), marker="o", label=sc)
    ax[2].set_xlabel("calendar month"); ax[2].set_ylabel("ramp multiplier")
    ax[2].set_title("Genesis ramp (Jul→Oct, or Jul→Aug for bursty)")
    ax[2].set_xticks(range(1, 13)); ax[2].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=.3)
    ax[0].set_ylabel("CDF"); ax[1].set_ylabel("CDF")
    fig.suptitle("Genesis Mission — synthesized (no historical data): three "
                 "labeled scenarios, all knobs configurable", fontsize=12)
    fig.tight_layout()
    _save(fig, "genesis_scenarios.png")


def fig_projects(cfg_path="configs/validate_projects.yaml"):
    """Project layer: per-project under-use + deadline submission spikes.
    Runs a short project-enabled sim to visualize the emergent behavior."""
    import dataclasses as dc
    from generator import JobGenerator
    from scheduler import Scheduler
    cfg = SimConfig.from_yaml(cfg_path)
    df = load_trace(cfg)
    g = JobGenerator(cfg, df)
    rng = np.random.default_rng(42)
    jobs = g.generate(rng)

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))

    # (a) per-project delivered-node-hours share within each program (award tail)
    import pandas as pd
    for prog, samplers in g.projects_by_prog.items():
        shares = np.sort([ps.delivered_nh for ps in samplers])[::-1]
        if shares.sum() == 0:
            continue
        shares = shares / shares.sum()
        ax[0].plot(np.arange(1, len(shares) + 1), np.cumsum(shares),
                   marker=".", label=f"{prog} ({len(shares)} proj)",
                   color=COL.get(prog, "gray"))
    ax[0].set_xlabel("project rank (largest first)")
    ax[0].set_ylabel("cumulative share of program node-hours")
    ax[0].set_title("Award concentration: few projects dominate each program")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

    # (b) deadline effect as in-window vs baseline submission rate (bar chart).
    # A detrended time series hides the spike (a rolling baseline absorbs a
    # multi-week window), so we show the measured ratio directly: mean daily
    # submissions INSIDE each deadline window vs the non-window baseline.
    days = np.array([int(j.submit_time_h // 24) for j in jobs])
    counts = np.bincount(days, minlength=int(cfg.run.duration_days)).astype(float)
    sim_start = cfg.run.start_month
    win_days = np.zeros(len(counts), bool)
    dl_info = []
    for dl in cfg.deadlines.deadlines:
        d = ((dl.month - sim_start) % 12) * 30 + (dl.day - 1)
        lo, hi = max(0, d - dl.lead_days), min(len(counts) - 1, d)
        dl_info.append((dl.name, lo, hi))
        win_days[lo:hi + 1] = True
    baseline = counts[~win_days].mean() if (~win_days).any() else counts.mean()
    names = [n for (n, _, _) in dl_info]
    ratios = [counts[lo:hi + 1].mean() / baseline for (_, lo, hi) in dl_info]
    bars = ax[1].bar(names, ratios, color="#c0392b", alpha=.8)
    ax[1].axhline(1.0, color="k", ls=":", label="non-deadline baseline")
    for b, r in zip(bars, ratios):
        ax[1].text(b.get_x() + b.get_width() / 2, r + 0.02, f"{r:.2f}x",
                   ha="center", fontsize=9)
    ax[1].set_ylabel("in-window submissions / baseline")
    ax[1].set_title("Conference-deadline effect (mean submission rate in lead window)")
    ax[1].set_ylim(0, max(ratios) * 1.25); ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3, axis="y")

    fig.suptitle("Project layer: competitive-award concentration + deadline-driven "
                 "submission bursts (project-enabled config)", fontsize=12)
    fig.tight_layout()
    _save(fig, "project_layer.png")


def _save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.relpath(path)}")


def main():
    print(f"Loading trace via {CFG_PATH} ...")
    cfg = SimConfig.from_yaml(CFG_PATH)
    df = load_trace(cfg)
    print(f"  {len(df):,} rows")
    fig_distributions(df)
    fig_size_tier_mix(df)
    fig_burn_curves(df)
    fig_genesis(cfg.machine.total_nodes)
    fig_projects()
    print("Done.")


if __name__ == "__main__":
    main()
