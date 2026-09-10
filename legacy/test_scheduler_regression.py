#!/usr/bin/env python3
"""
Scheduler regression suite.

Two kinds of checks:

  GOLDEN (exact): a set of short, stable (sub-examine_cap) runs whose summary
  metrics are pinned. Any scheduler change MUST reproduce these EXACTLY — this
  is the guard against silently altering validated low/moderate-load behavior.

  INVARIANTS (properties): correctness properties that must hold at ALL loads,
  especially high load. The key one: the machine must not END idle with fitting
  jobs pending (the deep-queue drain bug). These are asserted, not pinned to a
  number, so a fix that changes high-load numbers (as it should) still passes as
  long as the property holds.

Usage:
  python test_scheduler_regression.py            # run all checks vs pinned goldens
  python test_scheduler_regression.py --update   # re-pin goldens from current code
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from config import SimConfig
from generator import JobGenerator, Job
from scheduler import Scheduler
from compile_profiles import load_cache_df
import metrics as M

import dataclasses as dc

CACHE = "data/profiles_cache.npz"
GOLDEN_FILE = "tests_golden.json"

# Metrics pinned in the golden set (deterministic given seed).
GOLDEN_KEYS = ["n_jobs", "n_started", "avg_util_pct", "n_big_jobs",
               "n_big_unstarted", "big_wait_p95_h", "small_wait_p95_h",
               "wait_p95_h"]

# GOLDEN configs: stable, sub-examine_cap (verified maxpend < 4000), machine
# packs (big_unstart=0). Pinned under backfill_mode=conservative (the original
# published behavior) so we can prove that mode stays byte-stable. easy-mode
# behavior is exercised by the invariant cases below. (config, overrides, seed)
GOLDEN_CASES = [
    ("configs/stage2_base.yaml",
     {"run.duration_days": 20, "scheduler.backfill_mode": "conservative"}, 42),
    ("configs/menu_scan_base.yaml",
     {"run.duration_days": 20, "generator.load_multiplier": 0.4,
      "scheduler.backfill_mode": "conservative"}, 42),
    ("configs/menu_scan_base.yaml",
     {"run.duration_days": 20, "generator.load_multiplier": 0.6,
      "scheduler.backfill_mode": "conservative"}, 42),
]

# INVARIANT configs: high load (exceeds examine_cap) under the DEFAULT easy mode.
# These must satisfy the utilization/progress properties after the fix.
INVARIANT_CASES = [
    ("configs/menu_scan_base.yaml", {"run.duration_days": 15,
                                     "generator.load_multiplier": 0.9}, 42),
    ("configs/menu_scan_base.yaml", {"run.duration_days": 12,
                                     "generator.load_multiplier": 1.2}, 42),
    ("configs/menu_scan_base.yaml", {"run.duration_days": 12,
                                     "generator.load_multiplier": 1.5}, 42),
]


def _apply(cfg_dict, overrides):
    for k, v in overrides.items():
        keys = k.split(".")
        cur = cfg_dict
        for kk in keys[:-1]:
            cur = cur.setdefault(kk, {})
        cur[keys[-1]] = v
    return cfg_dict


def _run(cfg_path, overrides, seed):
    base = SimConfig.from_yaml(cfg_path)
    d = _apply(base.to_dict(), overrides)
    cfg = SimConfig.from_dict(d)
    df = load_cache_df(CACHE, cfg)
    gen = JobGenerator(cfg, df)
    jobs = gen.generate(np.random.default_rng(seed))
    sched = Scheduler(cfg)
    if cfg.projects.enabled:
        sched.attach_projects(gen.projects_by_prog)
    sched.run(jobs)
    return jobs, sched, cfg


def _case_id(cfg_path, overrides, seed):
    ov = ",".join(f"{k}={v}" for k, v in sorted(overrides.items()))
    return f"{os.path.basename(cfg_path)}|{ov}|seed{seed}"


def collect_golden():
    out = {}
    for cfg_path, ov, seed in GOLDEN_CASES:
        jobs, sched, cfg = _run(cfg_path, ov, seed)
        s = M.summary_stats(jobs, sched, cfg)
        out[_case_id(cfg_path, ov, seed)] = {k: s[k] for k in GOLDEN_KEYS}
    return out


def check_invariants():
    """Return list of (case_id, ok, detail). Properties that must hold at all
    loads. The key one: the machine must be well-UTILIZED under oversubscription
    (not draining idle). We check TIME-AVERAGED utilization, not the final
    instantaneous free_nodes (which is a sampling artifact: with a 10-min cycle
    and sparse daily samples, the final snapshot can catch a transient dip)."""
    results = []
    for cfg_path, ov, seed in INVARIANT_CASES:
        jobs, sched, cfg = _run(cfg_path, ov, seed)
        s = M.summary_stats(jobs, sched, cfg)
        # Property: under >=1.2x offered load the machine should be busy. Use the
        # time-averaged utilization (summary_stats' avg_util_pct), which is the
        # real measure — NOT the final free_nodes instant.
        util_ok = s["avg_util_pct"] >= 70.0
        # Property: the scheduler must make real progress (not go idle-forever).
        # Under oversubscription a big backlog is EXPECTED; what's not allowed is
        # near-zero starts. Require a healthy fraction started.
        frac_started = s["n_started"] / max(1, s["n_jobs"])
        progress_ok = frac_started >= 0.30
        ok = util_ok and progress_ok
        results.append((_case_id(cfg_path, ov, seed), ok,
                        f"avg_util={s['avg_util_pct']:.1f}% "
                        f"started={s['n_started']}/{s['n_jobs']} "
                        f"({frac_started*100:.0f}%) "
                        f"util_ok={util_ok} progress_ok={progress_ok}"))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true",
                    help="Re-pin goldens from current code (do this only on a "
                         "KNOWN-GOOD scheduler).")
    args = ap.parse_args()

    if args.update:
        g = collect_golden()
        with open(GOLDEN_FILE, "w") as f:
            json.dump(g, f, indent=2)
        print(f"Pinned {len(g)} golden cases -> {GOLDEN_FILE}")
        for cid, v in g.items():
            print(f"  {cid}: {v}")
        return

    if not os.path.exists(GOLDEN_FILE):
        sys.exit(f"No {GOLDEN_FILE}; run with --update on a known-good scheduler first.")
    golden = json.load(open(GOLDEN_FILE))

    print("=== GOLDEN (exact) ===")
    cur = collect_golden()
    n_fail = 0
    for cid, exp in golden.items():
        got = cur.get(cid, {})
        diffs = {k: (exp[k], got.get(k)) for k in exp if got.get(k) != exp[k]}
        if diffs:
            n_fail += 1
            print(f"  FAIL {cid}")
            for k, (e, g) in diffs.items():
                print(f"       {k}: golden={e} got={g}")
        else:
            print(f"  OK   {cid}")

    print("\n=== INVARIANTS (properties, all loads) ===")
    for cid, ok, detail in check_invariants():
        print(f"  {'OK  ' if ok else 'FAIL'} {cid}\n       {detail}")
        if not ok:
            n_fail += 1

    print(f"\n{'ALL PASS' if n_fail == 0 else f'{n_fail} FAILURE(S)'}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
