#!/usr/bin/env python3
"""
Parameter sweep runner.

Takes a base YAML config + an override grid, runs the cartesian product, and
writes one tidy results table (results/<sweep>/sweep_results.csv) with the
three headline objectives per run:
  - system_util_pct          (system utilization fraction * 100)
  - alloc_util               (delivered node-h / offered node-h for STARTED work)
  - wait_p50_h / wait_p95_h  (job queued-time)
plus per-program delivered/target share and budget burn.

Grid is given as a small YAML/JSON, e.g.:

  base: configs/validate_baseline.yaml
  name: score_sweep
  grid:
    scheduler.score_expr:
      - "base + aging_rate*wait"
      - "base + aging_rate*wait - 0.002*nodes"
      - "base + aging_rate*wait + 30*(target_share-delivered_share)"
    capacity_protection.pool_nodes: [512, 1024, 2048]

Each grid key is a dotted path into the SimConfig dict; values are the list to
sweep. Runs = product of all lists. Oversubscribed configs are caught by the
pre-flight guard and recorded as status=SATURATED (not fatal to the sweep).

Usage:
  python sweep.py --grid configs/sweeps/score_sweep.yaml
  python sweep.py --grid configs/sweeps/score_sweep.yaml --seeds 3   # override n_seeds
"""
from __future__ import annotations

import argparse
import copy
import itertools
import pathlib
import time

import numpy as np
import pandas as pd
import yaml

from config import SimConfig
from generator import JobGenerator, load_trace
from scheduler import Scheduler, SaturationError
import metrics as M


def _set_dotted(d: dict, dotted: str, value):
    """Set d['a']['b']=value for dotted='a.b', creating dicts as needed."""
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def _alloc_util(jobs, duration_h) -> float:
    """Delivered node-h / offered node-h, over jobs that were STARTED.
    Answers 'of the work that ran, how efficiently did allocation convert to
    delivered compute' — 1.0 means every started job ran its full requested
    walltime within the window."""
    started = [j for j in jobs if j.start_time_h is not None]
    delivered = sum(j.nodes * M._win_rt(j, duration_h) for j in started)
    offered = sum(j.nodes * j.walltime_h for j in started) or 1.0
    return delivered / offered


def run_grid(grid_path: str, seeds_override: int | None):
    with open(grid_path) as f:
        spec = yaml.safe_load(f)
    base_path = spec["base"]
    name = spec.get("name", pathlib.Path(grid_path).stem)
    grid = spec["grid"]

    base_cfg = SimConfig.from_yaml(base_path)
    base_dict = base_cfg.to_dict()
    df = load_trace(base_cfg)   # load ONCE, reuse across all runs
    print(f"Loaded trace ({len(df):,} rows). Sweep '{name}' over {base_path}.")

    keys = list(grid.keys())
    value_lists = [grid[k] for k in keys]
    combos = list(itertools.product(*value_lists))
    print(f"{len(combos)} configuration(s) in the grid.")

    outdir = pathlib.Path("results") / name
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, combo in enumerate(combos):
        d = copy.deepcopy(base_dict)
        label = {}
        for k, v in zip(keys, combo):
            _set_dotted(d, k, v)
            label[k] = v
        if seeds_override:
            _set_dotted(d, "run.n_seeds", seeds_override)
        cfg = SimConfig.from_dict(d)
        chash = cfg.config_hash()
        duration_h = cfg.run.duration_days * 24.0
        seeds = [cfg.run.base_seed + s for s in range(cfg.run.n_seeds)]

        t0 = time.time()
        try:
            per_seed = []
            for seed in seeds:
                rng = np.random.default_rng(seed)
                gen = JobGenerator(cfg, df)
                jobs = gen.generate(rng)
                sched = Scheduler(cfg)
                if cfg.projects.enabled:
                    sched.attach_projects(gen.projects_by_prog)
                sched.run(jobs)
                s = M.summary_stats(jobs, sched, cfg)
                dt = M.decision_table(jobs, sched, cfg)
                s["alloc_util"] = _alloc_util(jobs, duration_h)
                s["_dt"] = dt
                per_seed.append(s)
            # aggregate over seeds
            rec = {"config_hash": chash, **label, "status": "OK",
                   "runtime_s": round(time.time() - t0, 1)}
            for m in ("avg_util_pct", "alloc_util", "wait_p50_h", "wait_p95_h",
                      "wait_max_h", "n_started", "n_jobs", "throughput_jobs_per_day",
                      "big_wait_p50_h", "big_wait_p95_h", "big_wait_max_h",
                      "n_big_jobs", "n_big_unstarted",
                      "small_wait_p50_h", "small_wait_p95_h"):
                if m in per_seed[0]:
                    rec[m] = float(np.mean([ps[m] for ps in per_seed]))
            # per-program delivered share (mean across seeds)
            dt_all = pd.concat([ps["_dt"] for ps in per_seed], ignore_index=True)
            for prog, g in dt_all.groupby("program"):
                rec[f"{prog}_delivered"] = round(g["delivered_share"].mean(), 4)
                rec[f"{prog}_wait_p95"] = round(g["wait_p95_h"].mean(), 2)
            rows.append(rec)
            print(f"  [{i+1}/{len(combos)}] util={rec['avg_util_pct']:.1f}% "
                  f"alloc_util={rec['alloc_util']:.3f} "
                  f"wait_p95={rec['wait_p95_h']:.2f}h  {label}")
        except SaturationError as e:
            rows.append({"config_hash": chash, **label, "status": "SATURATED",
                         "note": str(e)[:120], "runtime_s": round(time.time()-t0, 1)})
            print(f"  [{i+1}/{len(combos)}] SATURATED  {label}")

    res = pd.DataFrame(rows)
    res_path = outdir / "sweep_results.csv"
    res.to_csv(res_path, index=False)
    print(f"\nSweep results -> {res_path}")
    # print a compact leaderboard by the three objectives
    ok = res[res["status"] == "OK"].copy()
    if not ok.empty:
        print("\n=== Top configs by system utilization ===")
        print(ok.nlargest(5, "avg_util_pct")[
            [*keys, "avg_util_pct", "alloc_util", "wait_p95_h"]].to_string(index=False))
        print("\n=== Lowest p95 queued-time ===")
        print(ok.nsmallest(5, "wait_p95_h")[
            [*keys, "wait_p95_h", "avg_util_pct", "alloc_util"]].to_string(index=False))
        # Starvation view: large-job wait vs small-job wait (Stage-2 headline).
        if "big_wait_p95_h" in ok.columns:
            print("\n=== Large-job (capability) vs small-job wait — starvation check ===")
            cols = [*keys, "big_wait_p95_h", "n_big_unstarted",
                    "small_wait_p95_h", "avg_util_pct"]
            print(ok[cols].to_string(index=False))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, help="Path to sweep grid YAML.")
    ap.add_argument("--seeds", type=int, default=None,
                    help="Override run.n_seeds for every run in the sweep.")
    args = ap.parse_args()
    run_grid(args.grid, args.seeds)


if __name__ == "__main__":
    main()
