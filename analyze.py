#!/usr/bin/env python3
"""
Deep-dive analysis of a SINGLE config: produces the six requested metrics as
CSV data + PNG plots. Use on a config of interest (e.g. a scan winner), where
you want the rich per-run detail the scan's summary rows don't carry.

Metrics produced:
  1. Utilization over time  — daily utilization plot across the run.
  2. Integrated utilization — single number (avg over the year, vs production nodes).
  3. Per-QUEUE job queued-time percentiles (p50/p90/p95/p99/max).
  4. Program allocation: total usage / total ALLOCATED to that program (burn %).
  5. Program allocation: total usage / total ANNUAL SYSTEM node-hours (share).
  6. Histogram of PROJECTS grouped by program, binned by % of allocation used.

Usage:
  python analyze.py --config configs/menu_scan_base.yaml \
      --cache data/profiles_cache.npz --out results/analysis_run
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

from config import SimConfig
from generator import JobGenerator, load_trace
from scheduler import Scheduler
from compile_profiles import load_cache_df
import metrics as M


def _load(cfg, cache):
    return load_cache_df(cache, cfg) if cache else load_trace(cfg)


def analyze(cfg: SimConfig, df, seed: int):
    duration_h = cfg.run.duration_days * 24.0
    prod = cfg.machine.prod()
    rng = np.random.default_rng(seed)
    gen = JobGenerator(cfg, df)
    jobs = gen.generate(rng)
    sched = Scheduler(cfg)
    if cfg.projects.enabled:
        sched.attach_projects(gen.projects_by_prog)
    sched.run(jobs)

    out = {}

    # 1 + 2. Utilization over time (daily) + integrated.
    util = pd.DataFrame(sched.util_samples, columns=["t_h", "busy"])
    util = util[util["t_h"] <= duration_h].copy()
    util["day"] = (util["t_h"] // 24).astype(int)
    util["util_frac"] = util["busy"] / prod
    daily = util.groupby("day")["util_frac"].mean().reset_index()
    out["utilization_daily"] = daily
    out["integrated_utilization"] = float(util["util_frac"].mean())

    # 3. Per-queue (size-tier) queued-time percentiles.
    started = [j for j in jobs if j.start_time_h is not None]
    rows = []
    for tier in [t.name for t in cfg.size_tiers]:
        w = np.array([j.start_time_h - j.submit_time_h
                      for j in started if j.size_tier == tier])
        if len(w) == 0:
            continue
        rows.append({
            "queue": tier, "n": len(w),
            "wait_p50_h": float(np.percentile(w, 50)),
            "wait_p90_h": float(np.percentile(w, 90)),
            "wait_p95_h": float(np.percentile(w, 95)),
            "wait_p99_h": float(np.percentile(w, 99)),
            "wait_max_h": float(w.max()),
        })
    out["queue_wait_percentiles"] = pd.DataFrame(rows)

    # 4 + 5. Program allocation usage / allocated, and / annual system node-hours.
    # NOTE: allocated_nh is the FULL-YEAR budget (share * annual system NH). Over a
    # partial-year window a program can only reach ~window/year of that, so we also
    # report a window-pro-rated burn (usage / (allocated * window_frac)) which is
    # window-length-independent and comparable across runs.
    annual_system_nh = prod * 24 * 365.0
    window_frac = min(1.0, duration_h / (365 * 24.0))
    rows = []
    for prog in sched.shares:
        used = sched.delivered_nh.get(prog, 0.0)
        budget = sched.budget_nh.get(prog, float("nan"))  # allocated (share*year)
        prorated = budget * window_frac if budget and budget > 0 else float("nan")
        rows.append({
            "program": prog,
            "delivered_nh": used,
            "allocated_nh": budget,
            "target_share": sched.shares.get(prog, 0.0),
            "usage_over_allocated": (used / budget) if budget and budget > 0 else float("nan"),
            "usage_over_allocated_prorated": (used / prorated) if prorated and prorated > 0 else float("nan"),
            "usage_over_annual_system": used / annual_system_nh,
            "delivered_share_of_used": used / (sum(sched.delivered_nh.values()) or 1.0),
        })
    out["program_allocation"] = pd.DataFrame(rows)
    out["_window_frac"] = window_frac

    # 6. Per-project utilization histogram (delivered / award), grouped by program.
    proj_rows = []
    if cfg.projects.enabled and sched.project_award_nh:
        # map project -> program from generated jobs
        proj_prog = {}
        for j in jobs:
            if j.project and j.project not in proj_prog:
                proj_prog[j.project] = j.program
        for proj, award in sched.project_award_nh.items():
            if award and award > 0:
                proj_rows.append({
                    "project": proj,
                    "program": proj_prog.get(proj, "?"),
                    "pct_alloc_used": 100.0 * sched.project_delivered_nh.get(proj, 0.0) / award,
                })
    out["project_utilization"] = pd.DataFrame(proj_rows)

    out["_summary"] = M.summary_stats(jobs, sched, cfg)
    return out


def write_outputs(out: dict, cfg: SimConfig, outdir: pathlib.Path):
    outdir.mkdir(parents=True, exist_ok=True)
    # CSVs
    out["utilization_daily"].to_csv(outdir / "utilization_daily.csv", index=False)
    out["queue_wait_percentiles"].to_csv(outdir / "queue_wait_percentiles.csv", index=False)
    out["program_allocation"].to_csv(outdir / "program_allocation.csv", index=False)
    out["project_utilization"].to_csv(outdir / "project_utilization.csv", index=False)
    (outdir / "summary.json").write_text(json.dumps(
        {"integrated_utilization": out["integrated_utilization"],
         **out["_summary"]}, indent=2, default=str))

    # Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable; wrote CSVs only.")
        return

    prod = cfg.machine.prod()

    # (1) daily utilization
    d = out["utilization_daily"]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(d["day"], d["util_frac"] * 100, lw=1.2, color="#1f77b4")
    ax.axhline(out["integrated_utilization"] * 100, ls="--", color="k", alpha=.6,
               label=f"integrated {out['integrated_utilization']*100:.1f}%")
    ax.set_xlabel("day"); ax.set_ylabel("utilization (% of production nodes)")
    ax.set_ylim(0, 105); ax.grid(alpha=.3); ax.legend()
    ax.set_title(f"Daily utilization ({prod} production nodes)")
    fig.tight_layout(); fig.savefig(outdir / "utilization_daily.png", dpi=110); plt.close(fig)

    # (3) per-queue wait percentiles
    q = out["queue_wait_percentiles"]
    if not q.empty:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        x = np.arange(len(q)); w = 0.18
        for i, pc in enumerate(["wait_p50_h", "wait_p90_h", "wait_p95_h", "wait_p99_h"]):
            ax.bar(x + (i-1.5)*w, q[pc], w, label=pc.replace("wait_", "").replace("_h", ""))
        ax.set_xticks(x); ax.set_xticklabels(q["queue"])
        ax.set_ylabel("queued time (h)"); ax.legend(title="percentile")
        ax.set_title("Per-queue queued-time percentiles"); ax.grid(alpha=.3, axis="y")
        fig.tight_layout(); fig.savefig(outdir / "queue_wait_percentiles.png", dpi=110); plt.close(fig)

    # (4+5) program allocation
    p = out["program_allocation"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].bar(p["program"], p["usage_over_allocated"] * 100, color="#2ca02c")
    ax[0].axhline(100, ls="--", color="k", alpha=.5, label="100% of allocation")
    ax[0].set_ylabel("% of program allocation used"); ax[0].legend()
    ax[0].set_title("Usage / allocated (budget burn)")
    ax[1].bar(p["program"], p["usage_over_annual_system"] * 100, color="#ff7f0e")
    for i, ts in enumerate(p["target_share"]):
        ax[1].plot([i-0.4, i+0.4], [ts*100, ts*100], color="k", lw=2)
    ax[1].set_ylabel("% of annual system node-hours")
    ax[1].set_title("Usage / annual system (black = target share)")
    for a in ax: a.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(outdir / "program_allocation.png", dpi=110); plt.close(fig)

    # (6) project utilization histogram by program (STACKED so all programs
    # are visible — overlapping alpha bars hid the smaller ALCC series).
    pu = out["project_utilization"]
    if not pu.empty:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        bins = np.arange(0, 210, 10)
        progs = sorted(pu["program"].unique())
        data = [pu[pu["program"] == pr]["pct_alloc_used"].clip(0, 200).to_numpy()
                for pr in progs]
        labels = [f"{pr} (n={len(pu[pu['program']==pr])})" for pr in progs]
        ax.hist(data, bins=bins, stacked=True, label=labels)
        ax.axvline(100, ls="--", color="k", alpha=.6, label="100% of award")
        ax.set_xlabel("% of allocation utilized"); ax.set_ylabel("# projects")
        ax.set_title("Projects by program, binned by % of allocation used")
        ax.legend(); ax.grid(alpha=.3, axis="y")
        fig.tight_layout(); fig.savefig(outdir / "project_utilization.png", dpi=110); plt.close(fig)

    print(f"Wrote CSVs + plots -> {outdir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="results/analysis")
    args = ap.parse_args()
    cfg = SimConfig.from_yaml(args.config)
    seed = args.seed if args.seed is not None else cfg.run.base_seed
    df = _load(cfg, args.cache)
    out = analyze(cfg, df, seed)
    print(f"integrated utilization: {out['integrated_utilization']*100:.1f}%")
    print("\nper-queue wait percentiles:")
    print(out["queue_wait_percentiles"].to_string(index=False))
    print("\nprogram allocation:")
    print(out["program_allocation"].to_string(index=False))
    if not out["project_utilization"].empty:
        pu = out["project_utilization"]
        print(f"\nproject utilization: {len(pu)} projects, "
              f"{(pu['pct_alloc_used']<100).mean()*100:.0f}% under 100% of award")
    write_outputs(out, cfg, pathlib.Path(args.out))


if __name__ == "__main__":
    main()
