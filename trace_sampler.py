"""
Sample jobs from real Aurora pbs_monitor traces.

Three sampling modes:
  - EmpiricalSampler: draw (nodes, walltime_h, runtime_h) from per-queue
    empirical distributions. Arrival times come from the simulator's
    Poisson model. Good for stress tests at arbitrary loads.

  - FittedSampler: fit per-bucket distributions from production queues only
    (tiny/small/medium/large/capacity), apply a two-component walltime model
    to account for the inverted queue policy (small node counts now get longer
    walltime caps), then sample from those fits. This is the recommended mode
    for policy exploration.

  - ReplaySampler: replay jobs in real submit order with original
    inter-arrival times. Good for validating scheduler against history.

Queue mapping from DB → simulation buckets
------------------------------------------
The DB contains Aurora's historical production queues. We map them as follows:

  DB queue     -> sim bucket   notes
  ------------ -------------- ----------------------------------------
  tiny         -> tiny         Old queue (removed ~Feb 2026). Node dist
                               used as-is; walltime rescaled (6h->168h).
  capacity     -> tiny         Current long-walltime small-node queue.
                               Used for cap-packer walltime shape.
  small        -> small        12h cap -> 72h cap
  medium       -> medium       18h cap -> 48h cap
  large        -> large        24h cap -> 24h (no change)

Walltime rescaling (two-component model)
-----------------------------------------
For each bucket, we split jobs into:
  - Cap-packers  : jobs with walltime >= CAP_PACKER_THRESHOLD * old_cap
  - Interior jobs: the rest

Interior jobs keep their original walltime as-is. They needed that amount of
time under the old policy and the new policy doesn't change their actual
computational needs.

Cap-packers are reassigned: their new walltime is sampled from the `capacity`
queue's empirical walltime distribution, scaled to the new cap. This models
users who were artificially inflated to the old cap and will now take
advantage of the new longer cap -- the capacity queue is our best empirical
signal for "what do users ask for when given 168h at small node counts."

For `tiny` (old tiny queue): cap-packers shift from 6h to sampling from
the capacity distribution directly (already 168h, no rescaling needed).
For `small`: cap-packers shift from 12h to capacity-dist * (72/168).
For `medium`: cap-packers shift from 18h to capacity-dist * (48/168).
For `large`: cap is unchanged (24h), so no two-component split needed.
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Simulation bucket boundaries: (lo_nodes, hi_nodes) inclusive.
# Must match sim.QUEUES.
DEFAULT_BUCKETS = {
    "capacity": (1,    128),
    "small":    (129,  512),
    "medium":   (513,  2048),
    "large":    (2049, 10_624),
}

# Old walltime caps (hours) that existed when DB jobs were submitted.
# "capacity" maps to the old Aurora `tiny` queue (6h cap).
OLD_CAPS_H = {
    "capacity": 6.0,
    "small":    12.0,
    "medium":   18.0,
    "large":    24.0,
}

# New walltime caps (hours) under the proposed policy.
NEW_CAPS_H = {
    "capacity": 168.0,
    "small":    72.0,
    "medium":   48.0,
    "large":    24.0,
}

# Capacity queue resource partition (nodes reserved, carved from total).
CAPACITY_NODES_RESERVED = 128
CAPACITY_NODE_MAX       = 16
CAPACITY_WALLTIME_CAP_H = 168.0

# A job is a "cap-packer" if its requested walltime >= this fraction of old cap.
CAP_PACKER_THRESHOLD = 0.90

# Production DB queues that map to each simulation bucket.
# capacity <- old Aurora `tiny` queue (removed Feb 2026, replaced by `capacity`)
PROD_QUEUE_MAP = {
    "capacity": ["tiny"],   # old tiny is the source for node dist + interior walltimes
    "small":    ["small"],
    "medium":   ["medium"],
    "large":    ["large"],
}

# ---------------------------------------------------------------------------
# Walltime parsing
# ---------------------------------------------------------------------------

def _parse_walltime(w) -> float:
    """PBS HH:MM:SS string -> hours. Returns NaN on garbage."""
    if not isinstance(w, str) or not w:
        return float("nan")
    parts = w.split(":")
    try:
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h, m, s = "0", parts[0], parts[1]
        else:
            return float("nan")
        return int(h) + int(m) / 60.0 + int(s) / 3600.0
    except ValueError:
        return float("nan")


def _bucket_of(nodes: int, buckets: dict) -> Optional[str]:
    for name, (lo, hi) in buckets.items():
        if lo <= nodes <= hi:
            return name
    return None


# ---------------------------------------------------------------------------
# DB loader
# ---------------------------------------------------------------------------

def _load_raw(db_path: str, min_runtime_s: int = 30) -> pd.DataFrame:
    """Pull all finished jobs from sqlite. Returns raw DataFrame."""
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT submit_time, start_time, end_time,
                   queue AS orig_queue, nodes, walltime,
                   actual_runtime_seconds AS runtime_s
            FROM jobs
            WHERE state = 'FINISHED'
              AND nodes IS NOT NULL
              AND nodes > 0
              AND actual_runtime_seconds IS NOT NULL
              AND actual_runtime_seconds >= ?
              AND submit_time IS NOT NULL
            """,
            con, params=[min_runtime_s],
        )
    finally:
        con.close()

    df["submit_time"] = pd.to_datetime(df["submit_time"], errors="coerce")
    df = df.dropna(subset=["submit_time"])
    df["walltime_h"] = df["walltime"].apply(_parse_walltime)
    df["runtime_h"] = df["runtime_s"] / 3600.0

    # Fill missing/bad walltimes with runtime * 1.5
    bad = df["walltime_h"].isna() | (df["walltime_h"] < df["runtime_h"])
    df.loc[bad, "walltime_h"] = (df.loc[bad, "runtime_h"] * 1.5).clip(lower=0.1)

    t0 = df["submit_time"].min()
    df["submit_h"] = (df["submit_time"] - t0).dt.total_seconds() / 3600.0
    return df.sort_values("submit_h").reset_index(drop=True)


def load_trace(
    db_path: str,
    buckets: dict = None,
    min_runtime_s: int = 30,
    max_nodes: Optional[int] = None,
    cache_path: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Legacy loader used by EmpiricalSampler and replay mode.
    Maps ALL finished jobs to buckets by node count.
    Returns DataFrame with: submit_h, runtime_h, walltime_h, nodes,
    orig_queue, bucket.
    """
    if cache_path is None:
        cache_path = db_path + ".trace_cache.pkl"
    if use_cache and os.path.exists(cache_path):
        try:
            return pd.read_pickle(cache_path)
        except Exception:
            pass

    if buckets is None:
        buckets = DEFAULT_BUCKETS
    if max_nodes is None:
        max_nodes = max(hi for _, hi in buckets.values())

    df = _load_raw(db_path, min_runtime_s)
    df = df[df["nodes"] <= max_nodes]
    df["bucket"] = df["nodes"].apply(lambda n: _bucket_of(int(n), buckets))
    df = df.dropna(subset=["bucket"])

    out = df[["submit_h", "runtime_h", "walltime_h", "nodes",
              "orig_queue", "bucket"]].copy()
    if use_cache:
        try:
            out.to_pickle(cache_path)
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# FittedSampler — the main new sampler
# ---------------------------------------------------------------------------

@dataclass
class _BucketFit:
    """Per-bucket fitted distributions ready for sampling."""
    name: str
    new_cap_h: float

    # Node distribution: empirical array (log-space sampling)
    nodes_arr: np.ndarray          # int array of node counts

    # Walltime: two-component model
    interior_wt_arr: np.ndarray    # walltime_h for non-cap-packers
    cap_packer_wt_arr: np.ndarray  # walltime_h for cap-packers (already rescaled)
    cap_packer_frac: float         # fraction of jobs that are cap-packers

    # Runtime: empirical ratio (runtime / walltime) for realistic actual runtimes
    rt_ratio_arr: np.ndarray       # runtime_h / walltime_h ratios

    # Arrival rate (jobs/hour) from empirical inter-arrival times
    arrival_rate_h: float


class FittedSampler:
    """
    Fit per-bucket distributions from production queues only (tiny, small,
    medium, large), applying the two-component walltime model to account for
    the inverted policy (small nodes now get longer walltime caps).

    Sources:
      tiny   <- old 'tiny' DB queue (nodes 1-128 only) for arrival rate +
                node distribution + interior walltimes.
                cap-packer walltimes drawn from 'capacity' queue dist
                (already on 168h cap, no rescale needed).
      small  <- DB 'small' queue; cap-packers rescaled via capacity dist
                * (72/168).
      medium <- DB 'medium' queue; cap-packers rescaled via capacity dist
                * (48/168).
      large  <- DB 'large' queue; no walltime change (cap stays 24h).
      capacity <- separate pool; not in main 4-bucket scheduler.
    """

    def __init__(self, db_path: str, rng: np.random.Generator,
                 buckets: dict = None, min_runtime_s: int = 30,
                 cache_path: Optional[str] = None, use_cache: bool = True):
        self._rng = rng
        self._buckets: dict[str, _BucketFit] = {}

        fitted_cache = (cache_path or db_path) + ".fitted_cache.pkl"
        if use_cache and os.path.exists(fitted_cache):
            try:
                import pickle
                with open(fitted_cache, "rb") as f:
                    self._buckets = pickle.load(f)
                print(f"[FittedSampler] Loaded from cache: {fitted_cache}")
                return
            except Exception:
                pass

        print("[FittedSampler] Building fits from DB (one-time, ~30s)...")
        df = _load_raw(db_path, min_runtime_s)
        self._fit(df, buckets or DEFAULT_BUCKETS)

        if use_cache:
            try:
                import pickle
                with open(fitted_cache, "wb") as f:
                    pickle.dump(self._buckets, f)
                print(f"[FittedSampler] Cache written: {fitted_cache}")
            except Exception:
                pass

    # ------------------------------------------------------------------

    def _fit(self, df: pd.DataFrame, buckets: dict):
        # Pull capacity queue walltime array — used as cap-packer shape donor
        cap_q = df[df["orig_queue"] == "capacity"].copy()
        cap_q = cap_q[(cap_q["walltime_h"] > 0) & cap_q["walltime_h"].notna()]
        cap_wt_arr = cap_q["walltime_h"].to_numpy(np.float64)
        if len(cap_wt_arr) == 0:
            raise ValueError("No 'capacity' queue jobs found in DB — "
                             "needed for cap-packer walltime shape.")
        print(f"  capacity queue: {len(cap_wt_arr):,} jobs for cap-packer shape")

        for bucket_name, (lo, hi) in buckets.items():
            fit = self._fit_bucket(bucket_name, lo, hi, df, cap_wt_arr)
            self._buckets[bucket_name] = fit
            print(f"  {bucket_name:10s}: {len(fit.nodes_arr):,} jobs | "
                  f"cap-packers={fit.cap_packer_frac:.1%} | "
                  f"λ={fit.arrival_rate_h:.4f}/h ({fit.arrival_rate_h*24:.1f}/day)")

    def _fit_bucket(self, name: str, lo: int, hi: int,
                    df: pd.DataFrame, cap_wt_arr: np.ndarray) -> _BucketFit:
        old_cap = OLD_CAPS_H[name]
        new_cap = NEW_CAPS_H[name]

        # Select source rows
        src_queues = PROD_QUEUE_MAP[name]
        sub = df[df["orig_queue"].isin(src_queues)].copy()

        # For tiny: restrict to node range (old tiny went up to 512; we want 1-128 only)
        sub = sub[(sub["nodes"] >= lo) & (sub["nodes"] <= hi)].copy()
        sub = sub[sub["walltime_h"].notna() & (sub["walltime_h"] > 0)].copy()

        if len(sub) == 0:
            raise ValueError(f"No jobs found for bucket '{name}' "
                             f"(queues={src_queues}, nodes {lo}-{hi})")

        # --- Arrival rate from empirical inter-arrival times ---
        # Use the full submit_h span of the source data, not the whole DB.
        span_h = sub["submit_h"].max() - sub["submit_h"].min()
        arrival_rate_h = len(sub) / span_h if span_h > 0 else 1.0 / 24.0

        # --- Node distribution (empirical, integer array) ---
        nodes_arr = sub["nodes"].to_numpy(np.int64)

        # --- Two-component walltime model ---
        threshold = CAP_PACKER_THRESHOLD * old_cap
        is_cap_packer = sub["walltime_h"] >= threshold
        cap_packer_frac = is_cap_packer.mean()

        interior = sub[~is_cap_packer]["walltime_h"].to_numpy(np.float64)
        # Interior jobs: keep walltime as-is — they needed that time,
        # the new policy doesn't change their actual computation.
        interior_wt_arr = np.clip(interior, 0.0, new_cap)

        # Cap-packer walltimes: sample from capacity distribution,
        # scaled to the new cap (capacity dist is already on 168h scale).
        if name == "large":
            # No cap change — treat all as interior (no rescaling needed)
            cap_packer_wt_arr = sub[is_cap_packer]["walltime_h"].to_numpy(np.float64)
            cap_packer_wt_arr = np.clip(cap_packer_wt_arr, 0.0, new_cap)
        else:
            scale = new_cap / CAPACITY_WALLTIME_CAP_H  # e.g. 72/168 for small
            rescaled = cap_wt_arr * scale
            cap_packer_wt_arr = np.clip(rescaled, 0.0, new_cap)

        # --- Runtime ratio (runtime / walltime_requested) ---
        # Clamp to (0, 1] so sampled runtimes are always <= requested walltime.
        ratio = (sub["runtime_h"] / sub["walltime_h"]).clip(0.001, 1.0)
        rt_ratio_arr = ratio.to_numpy(np.float64)

        return _BucketFit(
            name=name,
            new_cap_h=new_cap,
            nodes_arr=nodes_arr,
            interior_wt_arr=interior_wt_arr,
            cap_packer_wt_arr=cap_packer_wt_arr,
            cap_packer_frac=float(cap_packer_frac),
            rt_ratio_arr=rt_ratio_arr,
            arrival_rate_h=float(arrival_rate_h),
        )

    # ------------------------------------------------------------------

    def sample(self, bucket: str) -> tuple[int, float, float]:
        """
        Returns (nodes, walltime_h, runtime_h) for one job from `bucket`.
        walltime reflects the new-policy cap; runtime <= walltime.
        """
        fit = self._buckets.get(bucket)
        if fit is None:
            raise ValueError(f"Unknown bucket: {bucket!r}")
        rng = self._rng

        # Sample nodes from empirical distribution
        nodes = int(rng.choice(fit.nodes_arr))

        # Two-component walltime
        if (len(fit.cap_packer_wt_arr) > 0 and
                fit.cap_packer_frac > 0 and
                rng.random() < fit.cap_packer_frac):
            wt = float(rng.choice(fit.cap_packer_wt_arr))
        else:
            if len(fit.interior_wt_arr) == 0:
                wt = fit.new_cap_h * 0.5
            else:
                wt = float(rng.choice(fit.interior_wt_arr))

        wt = float(np.clip(wt, 0.083, fit.new_cap_h))  # 5 min floor

        # Runtime from empirical ratio
        ratio = float(rng.choice(fit.rt_ratio_arr))
        runtime = float(np.clip(ratio * wt, 0.05, wt))

        return nodes, wt, runtime

    def arrival_rate(self, bucket: str) -> float:
        """Empirical arrival rate (jobs/hour) for this bucket."""
        return self._buckets[bucket].arrival_rate_h

    def cap_packer_frac(self, bucket: str) -> float:
        return self._buckets[bucket].cap_packer_frac

    def print_fit_summary(self):
        print("\n=== FittedSampler — per-bucket fit summary ===")
        header = f"{'bucket':10s}  {'n_jobs':>8s}  {'cap_pack%':>9s}  "
        header += f"{'λ/h':>7s}  {'λ/day':>7s}  {'new_cap':>8s}"
        print(header)
        for name, fit in self._buckets.items():
            print(f"  {name:10s}  {len(fit.nodes_arr):>8,d}  "
                  f"{fit.cap_packer_frac:>8.1%}  "
                  f"{fit.arrival_rate_h:>7.4f}  "
                  f"{fit.arrival_rate_h*24:>7.1f}  "
                  f"{fit.new_cap_h:>7.0f}h")

        print("\n  Walltime shape — interior jobs (hours):")
        hdr2 = f"  {'bucket':10s}  {'n':>7s}  {'p50':>6s}  {'p90':>6s}  {'p99':>6s}  {'max':>6s}"
        print(hdr2)
        for name, fit in self._buckets.items():
            arr = fit.interior_wt_arr
            if len(arr) == 0:
                print(f"  {name:10s}  {'(empty)':>7s}")
                continue
            print(f"  {name:10s}  {len(arr):>7,d}  "
                  f"{np.percentile(arr,50):>6.1f}  "
                  f"{np.percentile(arr,90):>6.1f}  "
                  f"{np.percentile(arr,99):>6.1f}  "
                  f"{arr.max():>6.1f}")

        print("\n  Walltime shape — cap-packer component (hours):")
        hdr3 = f"  {'bucket':10s}  {'n':>7s}  {'p50':>6s}  {'p90':>6s}  {'p99':>6s}  {'max':>6s}"
        print(hdr3)
        for name, fit in self._buckets.items():
            arr = fit.cap_packer_wt_arr
            if len(arr) == 0:
                print(f"  {name:10s}  (no cap-packers)")
                continue
            print(f"  {name:10s}  {len(arr):>7,d}  "
                  f"{np.percentile(arr,50):>6.1f}  "
                  f"{np.percentile(arr,90):>6.1f}  "
                  f"{np.percentile(arr,99):>6.1f}  "
                  f"{arr.max():>6.1f}")


# ---------------------------------------------------------------------------
# EmpiricalSampler (legacy — maps all jobs by node count, no rescaling)
# ---------------------------------------------------------------------------

class EmpiricalSampler:
    """
    Per-queue empirical distribution. Pre-extracts each bucket's
    (nodes, walltime_h, runtime_h) arrays into numpy for O(1) sampling.
    No walltime rescaling — use FittedSampler for policy exploration.
    """

    def __init__(self, trace: pd.DataFrame):
        self._pools: dict[str, dict[str, np.ndarray]] = {}
        for bucket, sub in trace.groupby("bucket"):
            self._pools[bucket] = {
                "nodes": sub["nodes"].to_numpy(np.int64),
                "walltime_h": sub["walltime_h"].to_numpy(np.float64),
                "runtime_h": sub["runtime_h"].to_numpy(np.float64),
            }

    def sample(self, bucket: str, rng: np.random.Generator) -> tuple[int, float, float]:
        pool = self._pools.get(bucket)
        if pool is None or len(pool["nodes"]) == 0:
            raise ValueError(f"no trace samples for bucket {bucket!r}")
        i = int(rng.integers(0, len(pool["nodes"])))
        return (int(pool["nodes"][i]),
                float(pool["walltime_h"][i]),
                float(pool["runtime_h"][i]))


# ---------------------------------------------------------------------------
# Replay helper
# ---------------------------------------------------------------------------

def replay_jobs(trace: pd.DataFrame, start_h: float = 0.0,
                duration_h: Optional[float] = None) -> pd.DataFrame:
    """
    Return a view of `trace` shifted so its first arrival is at `start_h`.
    Optionally clip to a duration.
    """
    if trace.empty:
        return trace
    out = trace.copy()
    offset = start_h - out["submit_h"].iloc[0]
    out["submit_h"] = out["submit_h"] + offset
    if duration_h is not None:
        out = out[out["submit_h"] <= start_h + duration_h]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------------------

def summarize(trace: pd.DataFrame) -> None:
    """Print per-bucket summary stats for the loaded trace."""
    print(f"\nTrace summary: {len(trace):,} jobs, "
          f"{trace['submit_h'].max()/24:.1f} days of arrivals")
    print(f"Original queues represented: {trace['orig_queue'].nunique()}")
    g = trace.groupby("bucket").agg(
        n=("nodes", "size"),
        nodes_mean=("nodes", "mean"),
        nodes_p50=("nodes", "median"),
        nodes_p95=("nodes", lambda s: s.quantile(0.95)),
        rt_mean_h=("runtime_h", "mean"),
        rt_p50_h=("runtime_h", "median"),
        rt_p95_h=("runtime_h", lambda s: s.quantile(0.95)),
        wall_mean_h=("walltime_h", "mean"),
    ).round(2)
    order = ["capacity", "small", "medium", "large"]
    g = g.reindex([b for b in order if b in g.index])
    print(g.to_string())

    span_h = trace["submit_h"].max() - trace["submit_h"].min()
    print(f"\nArrival rates over {span_h/24:.1f} days:")
    rates = trace.groupby("bucket").size() / span_h
    for b in order:
        if b in rates.index:
            print(f"  {b:10s} {rates[b]:.3f} jobs/h  ({rates[b]*24:.1f} jobs/day)")
