#!/usr/bin/env python3
"""
Single entrypoint:  python run.py --config path/to/config.yaml

Runs one (multi-seed-aware) simulation fully described by a YAML config, writes
the decision table + tidy time-series + a run manifest tagging outputs with the
config hash for provenance.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import pandas as pd

from config import SimConfig
from generator import JobGenerator, load_trace
from scheduler import Scheduler, SaturationError
import metrics as M


def run_one(cfg: SimConfig, df, seed: int):
    rng = np.random.default_rng(seed)
    gen = JobGenerator(cfg, df)
    jobs = gen.generate(rng)
    sched = Scheduler(cfg)
    if cfg.projects.enabled:
        sched.attach_projects(gen.projects_by_prog)
    sched.run(jobs)
    return jobs, sched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to SimConfig YAML.")
    ap.add_argument("--print-only", action="store_true",
                    help="Print resolved config + hash and exit (no run).")
    args = ap.parse_args()

    cfg = SimConfig.from_yaml(args.config)
    chash = cfg.config_hash()
    print(f"Config: {args.config}  hash={chash}")
    if args.print_only:
        print(cfg.to_yaml())
        return

    outdir = pathlib.Path(cfg.output.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(str(outdir / "resolved_config.yaml"))

    print(f"Loading trace {cfg.trace_db} ...")
    t0 = time.time()
    df = load_trace(cfg)
    print(f"  {len(df):,} trace rows loaded in {time.time()-t0:.1f}s")

    seeds = [cfg.run.base_seed + i for i in range(cfg.run.n_seeds)]
    dtables, summaries = [], []
    ts_last = None
    for si, seed in enumerate(seeds):
        t0 = time.time()
        try:
            jobs, sched = run_one(cfg, df, seed)
        except SaturationError as e:
            print(f"\n*** SATURATION (seed {seed}):\n    {e}\n")
            print("Run aborted by the oversubscription guard. Adjust the config "
                  "and retry.")
            raise SystemExit(2)
        dt = M.decision_table(jobs, sched, cfg)
        dt["seed"] = seed
        dtables.append(dt)
        s = M.summary_stats(jobs, sched, cfg); s["seed"] = seed
        summaries.append(s)
        ts_last = M.timeseries(sched, cfg)
        print(f"  seed {seed}: {len(jobs):,} jobs, "
              f"util={s['avg_util_pct']}%, started={s['n_started']:,}/"
              f"{s['n_jobs']:,}, wait_p95={s['wait_p95_h']}h "
              f"({time.time()-t0:.1f}s)")

    all_dt = pd.concat(dtables, ignore_index=True)
    sdf = pd.DataFrame(summaries)

    # aggregate decision table across seeds (mean + std)
    numeric = [c for c in all_dt.columns if c not in ("program", "seed")]
    agg = all_dt.groupby("program")[numeric].agg(["mean", "std"])
    print("\n=== Decision table (mean over {} seed(s)) ===".format(len(seeds)))
    show = all_dt.groupby("program")[numeric].mean().round(4)
    print(show.to_string())
    print("\n=== Run summary (mean over seeds) ===")
    print(sdf.drop(columns=["seed"]).mean(numeric_only=True).round(2).to_string())

    if cfg.output.write_decision_table:
        all_dt.to_csv(outdir / "decision_table.csv", index=False)
        agg.to_csv(outdir / "decision_table_agg.csv")
    if cfg.output.write_timeseries and ts_last is not None:
        ts_last.to_csv(outdir / "timeseries.csv", index=False)
    manifest = {
        "config_hash": chash, "config_path": args.config,
        "seeds": seeds, "summary": summaries,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nOutputs -> {outdir}/ (config_hash={chash})")


if __name__ == "__main__":
    main()
