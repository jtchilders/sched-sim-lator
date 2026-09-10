#!/usr/bin/env python3
"""
Adaptive-seeding helper: from a screening scan (few seeds/config), pick the most
promising configs and emit a refinement experiment spec that re-runs ONLY those
with many more seeds — the "hone in where it matters" step of the staged
Monte-Carlo strategy.

Because scan.py is idempotent by (config_hash, seed), the refinement spec can
point at the SAME output dir and it will only run the additional seeds.

Usage:
  python refine.py --results results/<scan>/results.parquet \
      --base configs/menu_scan_base.yaml \
      --objective avg_util_pct --top 8 --seeds 40 \
      --out experiments/<scan>_refine.yaml
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--base", required=True, help="base config the scan used")
    ap.add_argument("--objective", default="avg_util_pct")
    ap.add_argument("--top", type=int, default=8, help="how many top configs to refine")
    ap.add_argument("--seeds", type=int, default=40, help="seeds per refined config")
    ap.add_argument("--maximize", action="store_true", default=True)
    ap.add_argument("--minimize", dest="maximize", action="store_false",
                    help="pick LOWEST objective (e.g. for wait times)")
    ap.add_argument("--cache", default="data/profiles_cache.npz")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.results)
    ok = df[df["status"] == "OK"].copy()
    obj = args.objective
    if obj not in ok.columns:
        raise SystemExit(f"objective {obj!r} not in results columns")

    # mean objective per config, pick top-K
    meta = {"config_hash", "seed", "status", "runtime_s", "note"}
    param_cols = [c for c in ok.columns
                  if c not in meta and not any(c.startswith(p) or c.endswith(s)
                     for p in ("n_", "wait_", "big_", "small_", "avg_")
                     for s in ("_delivered", "_delivered_nh", "_wait_p95", "_burn"))
                  and c not in ("alloc_util", "node_hours_delivered",
                                "throughput_jobs_per_day", "avg_util_pct",
                                "wait_p50_h", "wait_p95_h", "wait_max_h")]
    means = ok.groupby("config_hash")[obj].mean().sort_values(ascending=not args.maximize)
    top_hashes = means.head(args.top).index.tolist()

    # recover each top config's swept-param values (the label columns)
    points = []
    for h in top_hashes:
        row = ok[ok.config_hash == h].iloc[0]
        pt = {}
        for pc in param_cols:
            v = row[pc]
            # de-serialize list/dict labels stored as JSON strings by scan.py
            if isinstance(v, str) and v[:1] in "[{":
                try:
                    v = json.loads(v)
                except Exception:
                    pass
            pt[pc] = v
        points.append(pt)

    spec = {
        "name": args.name or "refine",
        "base": args.base,
        "cache": args.cache,
        "method": "list",
        "seeds": args.seeds,
        "params": {"points": points},
    }
    with open(args.out, "w") as f:
        yaml.safe_dump(spec, f, sort_keys=False)
    print(f"Refinement spec -> {args.out}")
    print(f"  {len(points)} top configs by {obj} "
          f"({'max' if args.maximize else 'min'}), {args.seeds} seeds each")
    for h, pt in zip(top_hashes, points):
        print(f"  {h[:8]}  {obj}={means[h]:.3f}  {pt}")


if __name__ == "__main__":
    main()
