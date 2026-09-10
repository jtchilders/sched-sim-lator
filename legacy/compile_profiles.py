#!/usr/bin/env python3
"""
Compile the 9 GB Aurora PBS trace DB into a compact profile cache (~2 MB .npz).

Run this ONCE on the machine that has the trace DB (e.g. the Mac mini), then scp
the resulting .npz to Crux. Simulation workers load the cache instead of the DB —
the DB never has to travel to the cluster, and each worker starts in ~0.01 s
instead of paying a multi-second SQLite read.

The cache stores exactly the columns the generator's samplers need
(program, nodes, walltime_h, rt_ratio, cal_month, project), categorical-encoded
for compactness. The samplers (ConditionalSampler / ProjectSampler / burn curves)
are rebuilt cheaply from these arrays in-memory at run time — so the cache is
data, not pickled objects (portable, version-robust).

Usage:
  python compile_profiles.py --config configs/validate_baseline.yaml \
      --out data/profiles_cache.npz
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from config import SimConfig
from generator import load_trace

CACHE_VERSION = 1
NEEDED = ["program", "nodes", "walltime_h", "rt_ratio", "cal_month", "project"]


def compile_cache(cfg: SimConfig, out_path: str) -> dict:
    t0 = time.time()
    df = load_trace(cfg)
    load_s = time.time() - t0
    # span of the trace (hours) — needed by the generator to normalize annual
    # arrival rates. Stored so cache-fit rates match DB-fit rates exactly.
    span_h = (df["submit_time"].max() - df["submit_time"].min()).total_seconds() / 3600.0

    prog_cat, prog_codes = np.unique(df["program"].to_numpy(), return_inverse=True)
    proj_cat, proj_codes = np.unique(df["project"].astype(str).to_numpy(),
                                     return_inverse=True)
    # Cast category arrays to fixed-width unicode so npz can load without pickle
    # (allow_pickle=False on the worker side — safer + faster).
    prog_cat = prog_cat.astype("U")
    proj_cat = proj_cat.astype("U")
    arrs = dict(
        cache_version=np.array([CACHE_VERSION], dtype=np.int32),
        min_runtime_s=np.array([cfg.generator.min_runtime_s], dtype=np.int64),
        span_h=np.array([span_h], dtype=np.float64),
        prog_cat=prog_cat,
        prog_codes=prog_codes.astype(np.int16),
        proj_cat=proj_cat,
        proj_codes=proj_codes.astype(np.int32),
        nodes=df["nodes"].to_numpy(np.int32),
        walltime_h=df["walltime_h"].to_numpy(np.float64),
        rt_ratio=df["rt_ratio"].to_numpy(np.float64),
        runtime_h=df["runtime_h"].to_numpy(np.float64),
        cal_month=df["cal_month"].to_numpy(np.int8),
    )
    import os
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    np.savez_compressed(out_path, **arrs)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Compiled {len(df):,} trace rows -> {out_path} "
          f"({size_mb:.1f} MB) in {load_s:.1f}s")
    print(f"  programs: {prog_cat.tolist()}  projects: {len(proj_cat)}")
    return arrs


def load_cache_df(path: str, cfg: "SimConfig" = None):
    """Rebuild the sampler-input DataFrame from a compiled cache (fast, ~0.01s).
    Returns a DataFrame with the same columns load_trace() produces for the
    generator (program, nodes, walltime_h, rt_ratio, cal_month, project,
    runtime_h, size_tier). `cfg` is required to assign size_tier from the config's
    size tiers (matching load_trace)."""
    import pandas as pd
    z = np.load(path, allow_pickle=False)
    ver = int(z["cache_version"][0]) if "cache_version" in z else 0
    if ver != CACHE_VERSION:
        raise ValueError(f"profile cache version {ver} != expected {CACHE_VERSION}; "
                         f"recompile with compile_profiles.py")
    prog = z["prog_cat"][z["prog_codes"]]
    proj = z["proj_cat"][z["proj_codes"]]
    df = pd.DataFrame({
        "program": prog,
        "project": proj,
        "nodes": z["nodes"].astype(np.int64),
        "walltime_h": z["walltime_h"].astype(np.float64),
        "rt_ratio": z["rt_ratio"].astype(np.float64),
        "cal_month": z["cal_month"].astype(np.int64),
    })
    # runtime_h: use the stored original (runtime_s/3600) when present, so burn
    # curves and project node-hours match the DB fit exactly. Fall back to
    # reconstruction from rt_ratio only for older caches without it.
    if "runtime_h" in z.files:
        df["runtime_h"] = z["runtime_h"].astype(np.float64)
    else:
        df["runtime_h"] = (df["rt_ratio"] * df["walltime_h"]).clip(lower=0.0)
    # size_tier: assign from config tiers (same searchsorted as load_trace).
    if cfg is not None:
        tiers = sorted(cfg.size_tiers, key=lambda t: t.min_nodes)
        edges = np.array([t.max_nodes for t in tiers])
        names = np.array([t.name for t in tiers])
        idx = np.searchsorted(edges, df["nodes"].to_numpy(), side="left")
        idx = np.clip(idx, 0, len(tiers) - 1)
        df["size_tier"] = names[idx]
    # span_h (trace duration in hours) drives annual arrival-rate normalization.
    # Attach it so the generator uses the exact DB span without needing the
    # submit_time column (which the cache omits).
    df.attrs["span_h"] = float(z["span_h"][0]) if "span_h" in z else None
    return df


def cache_span_h(path: str) -> float:
    """Read just the trace span (hours) from a cache without loading rows."""
    z = np.load(path, allow_pickle=False)
    return float(z["span_h"][0])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/validate_baseline.yaml",
                    help="Config providing trace_db path + generator settings.")
    ap.add_argument("--out", default="data/profiles_cache.npz")
    args = ap.parse_args()
    cfg = SimConfig.from_yaml(args.config)
    compile_cache(cfg, args.out)
