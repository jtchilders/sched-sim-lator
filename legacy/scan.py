#!/usr/bin/env python3
"""
Parameter-scan engine: experiment spec -> manifest -> parallel run -> results.

Built for Crux `parton` (128 cores, dedicated node): embarrassingly parallel
scan over score-function / queue / behavior parameters to find good queue+PBS
settings. Each (config x seed) task is one independent sim run; a process pool
fills the node. Results append to one parquet keyed by config_hash (idempotent /
resumable — re-running skips completed cells).

Workers load a compiled PROFILE CACHE (.npz, ~3 MB), not the 9 GB trace DB, so
startup is instant and the DB never has to reach the cluster.

Experiment spec (YAML), e.g.:

    name: score_x_walltime_scan
    base: configs/stage2_base.yaml
    cache: data/profiles_cache.npz            # or pass --cache
    method: grid                              # grid | lhs | list
    seeds: 3                                  # runs per parameter point
    n_lhs: 200                                # samples if method=lhs
    params:
      scheduler.score_expr:
        - "base + aging_rate*wait"
        - "base + aging_rate*wait + 30*(target_share-delivered_share)"
      walltime_policy.breakpoints:
        - [[1,168],[512,48],[1920,24]]
        - [[1,24],[512,24],[1920,24]]
      generator.load_multiplier: {min: 0.4, max: 0.9}   # lhs range form

Usage:
    python scan.py --spec experiments/foo.yaml --workers 128 \
        --out /lus/eagle/projects/datascience/parton-ai/schedsim/experiments/foo
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yaml

from config import SimConfig
from generator import JobGenerator
from scheduler import Scheduler, SaturationError
from compile_profiles import load_cache_df
import metrics as M

# One cache load per worker process, reused across that worker's tasks.
_WORKER_DF = None
_WORKER_CACHE = None


def _worker_init(cache_path: str):
    global _WORKER_DF, _WORKER_CACHE
    _WORKER_CACHE = cache_path
    # df built lazily per config (size_tier depends on cfg); cache raw load once
    _WORKER_DF = None


def _set_dotted(d: dict, dotted: str, value):
    """Set a value by dotted path into the config dict.

    Supports two special cases beyond plain nesting:
      - size_tiers.<tier_name>.<field>  -> sets that field on the tier whose
        name matches (e.g. size_tiers.large.base_priority). Lets a scan sweep a
        single tier's base_priority/aging_rate/walltime_cap_h without pasting the
        whole size_tiers list.
      - programs.<program_name>.<field> -> same, for programs by name.
    """
    keys = dotted.split(".")
    # tier/program by-name addressing
    if keys[0] in ("size_tiers", "programs") and len(keys) == 3:
        lst = d.get(keys[0], [])
        for item in lst:
            if item.get("name") == keys[1]:
                item[keys[2]] = value
                return
        raise KeyError(f"{keys[0]} has no entry named {keys[1]!r}")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def _alloc_util(jobs, duration_h) -> float:
    started = [j for j in jobs if j.start_time_h is not None]
    delivered = sum(j.nodes * M._win_rt(j, duration_h) for j in started)
    offered = sum(j.nodes * j.walltime_h for j in started) or 1.0
    return delivered / offered


def run_task(task: dict) -> dict:
    """Execute one (config, seed). Runs in a worker process. Loads the cache
    once per process. Returns a flat result row."""
    global _WORKER_DF
    cfg = SimConfig.from_dict(task["config_dict"])
    seed = task["seed"]
    chash = task["config_hash"]
    duration_h = cfg.run.duration_days * 24.0
    t0 = time.time()
    try:
        df = load_cache_df(_WORKER_CACHE, cfg)  # cfg-specific (size_tier)
        rng = np.random.default_rng(seed)
        gen = JobGenerator(cfg, df)
        jobs = gen.generate(rng)
        sched = Scheduler(cfg)
        if cfg.projects.enabled:
            sched.attach_projects(gen.projects_by_prog)
        sched.run(jobs)
        s = M.summary_stats(jobs, sched, cfg)
        s["alloc_util"] = _alloc_util(jobs, duration_h)
        dt = M.decision_table(jobs, sched, cfg)
        row = {"config_hash": chash, "seed": seed, "status": "OK",
               "runtime_s": round(time.time() - t0, 1), **task["label"]}
        for k, v in s.items():
            row[k] = v
        for _, r in dt.iterrows():
            p = r["program"]
            row[f"{p}_delivered"] = r["delivered_share"]      # share of delivered
            row[f"{p}_wait_p95"] = r["wait_p95_h"]
            if "budget_burn" in r:
                row[f"{p}_burn"] = r["budget_burn"]           # delivered/allocated
        # Absolute per-program delivered node-hours (the DELIVERY quantity, not a
        # share) — what the sensitivity study is ultimately about.
        for prog, nh in sched.delivered_nh.items():
            row[f"{prog}_delivered_nh"] = float(nh)
        # Per-QUEUE and per-PROGRAM breakdowns for cross-analysis (wait, throughput,
        # delivered node-h per queue/program) + per-PROJECT long rows.
        bd, proj_df = M.breakdowns(jobs, sched, cfg)
        row.update(bd)
        if not proj_df.empty:
            proj_df = proj_df.copy()
            proj_df.insert(0, "seed", seed)
            proj_df.insert(0, "config_hash", chash)
            for kk, vv in task["label"].items():
                proj_df[kk] = (json.dumps(vv) if isinstance(vv, (list, dict)) else vv)
            row["_project_rows"] = proj_df.to_dict("records")
        return row
    except SaturationError as e:
        return {"config_hash": chash, "seed": seed, "status": "SATURATED",
                "note": str(e)[:120], "runtime_s": round(time.time() - t0, 1),
                **task["label"]}
    except Exception as e:  # never let one task sink the pool
        return {"config_hash": chash, "seed": seed, "status": "ERROR",
                "note": f"{type(e).__name__}: {str(e)[:120]}",
                "runtime_s": round(time.time() - t0, 1), **task["label"]}


# --- manifest generation ---------------------------------------------------

def _expand_grid(params: dict):
    keys = list(params.keys())
    vals = [params[k] if isinstance(params[k], list) else [params[k]] for k in keys]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))


def _expand_lhs(params: dict, n: int, rng: np.random.Generator):
    """Latin-hypercube over range-form params ({min,max}); list-form params are
    sampled uniformly per draw. Good for coarse screening of a big space."""
    keys = list(params.keys())
    # separate continuous ranges from discrete lists
    cont = {k: v for k, v in params.items() if isinstance(v, dict) and "min" in v}
    disc = {k: v for k, v in params.items() if not (isinstance(v, dict) and "min" in v)}
    # LHS for continuous
    cont_keys = list(cont.keys())
    if cont_keys:
        d = len(cont_keys)
        cube = np.zeros((n, d))
        for j in range(d):
            perm = rng.permutation(n)
            cube[:, j] = (perm + rng.random(n)) / n
    for i in range(n):
        point = {}
        for j, k in enumerate(cont_keys):
            lo, hi = cont[k]["min"], cont[k]["max"]
            point[k] = float(lo + cube[i, j] * (hi - lo))
        for k, v in disc.items():
            opts = v if isinstance(v, list) else [v]
            point[k] = opts[rng.integers(len(opts))]
        yield point


def build_manifest(spec: dict) -> tuple[list, str, int]:
    base_cfg = SimConfig.from_yaml(spec["base"])
    base_dict = base_cfg.to_dict()
    method = spec.get("method", "grid")
    seeds = int(spec.get("seeds", 1))
    params = spec.get("params", {})

    if method == "grid":
        points = list(_expand_grid(params))
    elif method == "lhs":
        rng = np.random.default_rng(spec.get("lhs_seed", 12345))
        points = list(_expand_lhs(params, int(spec.get("n_lhs", 100)), rng))
    elif method == "list":
        points = params.get("points", [])
    else:
        raise ValueError(f"unknown method {method!r} (grid|lhs|list)")

    tasks = []
    base_seed = int(base_cfg.run.base_seed)
    for pt in points:
        d = copy.deepcopy(base_dict)
        for k, v in pt.items():
            _set_dotted(d, k, v)
        cfg = SimConfig.from_dict(d)   # validates
        chash = cfg.config_hash()
        for s in range(seeds):
            tasks.append({
                "config_dict": cfg.to_dict(),
                "config_hash": chash,
                "seed": base_seed + s,
                "label": {k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                          for k, v in pt.items()},
            })
    return tasks, method, len(points)


# --- driver ----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="Experiment spec YAML.")
    ap.add_argument("--cache", default=None,
                    help="Profile cache .npz (overrides spec.cache).")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=None, help="Output dir (overrides spec).")
    args = ap.parse_args()

    with open(args.spec) as f:
        spec = yaml.safe_load(f)
    cache = args.cache or spec.get("cache")
    if not cache or not os.path.exists(cache):
        raise SystemExit(f"profile cache not found: {cache!r} "
                         f"(compile with compile_profiles.py, or pass --cache)")
    outdir = pathlib.Path(args.out or spec.get("out")
                          or f"results/{spec.get('name', 'scan')}")
    outdir.mkdir(parents=True, exist_ok=True)

    tasks, method, n_points = build_manifest(spec)
    print(f"Scan '{spec.get('name')}' : method={method} points={n_points} "
          f"seeds={spec.get('seeds', 1)} -> {len(tasks)} tasks, "
          f"{args.workers} workers")

    # Idempotent resume: skip (config_hash, seed) already in results.
    res_path = outdir / "results.parquet"
    done = set()
    if res_path.exists():
        prev = pd.read_parquet(res_path)
        done = set(zip(prev["config_hash"], prev["seed"]))
        print(f"  resume: {len(done)} tasks already done, skipping them")
    todo = [t for t in tasks if (t["config_hash"], t["seed"]) not in done]
    print(f"  running {len(todo)} tasks")
    if not todo:
        print("  nothing to do."); return

    # write manifest for provenance
    pd.DataFrame([{"config_hash": t["config_hash"], "seed": t["seed"], **t["label"]}
                  for t in tasks]).to_parquet(outdir / "manifest.parquet", index=False)

    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_worker_init,
                             initargs=(cache,)) as ex:
        futs = [ex.submit(run_task, t) for t in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % max(1, len(todo) // 20) == 0 or i == len(todo):
                el = time.time() - t0
                rate = i / el if el > 0 else 0
                eta = (len(todo) - i) / rate if rate > 0 else 0
                print(f"  [{i}/{len(todo)}] {rate:.1f} tasks/s  ETA {eta/60:.1f} min",
                      flush=True)

    # Separate per-project long rows into a companion file so the main results
    # table stays flat/wide-but-bounded.
    proj_records = []
    for r in rows:
        pr = r.pop("_project_rows", None)
        if pr:
            proj_records.extend(pr)

    new = pd.DataFrame(rows)
    if res_path.exists():
        new = pd.concat([pd.read_parquet(res_path), new], ignore_index=True)
    new.to_parquet(res_path, index=False)
    ok = (new["status"] == "OK").sum()
    print(f"\nDone. {ok}/{len(new)} OK -> {res_path} ({time.time()-t0:.0f}s)")

    if proj_records:
        proj_path = outdir / "results_projects.parquet"
        pdf = pd.DataFrame(proj_records)
        if proj_path.exists():
            prev = pd.read_parquet(proj_path)
            # drop already-present (config_hash, seed, project) to stay idempotent
            key = ["config_hash", "seed", "project"]
            merged = pd.concat([prev, pdf], ignore_index=True)
            merged = merged.drop_duplicates(subset=key, keep="last")
            pdf = merged
        pdf.to_parquet(proj_path, index=False)
        print(f"Per-project rows -> {proj_path} ({len(pdf)} rows)")


if __name__ == "__main__":
    main()
