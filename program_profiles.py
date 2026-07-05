"""
Per-program allocation profiles for the PBS queue simulator.

A ProgramProfile bundles everything the simulator needs to generate and account
for one allocation program's workload:

  - size/walltime/runtime distributions (empirical, sampled with replacement)
  - arrival rate (jobs/hour)
  - allocation calendar (program-year start/end + optional extension)
  - seasonal burn curve (monthly multipliers on node-hour demand)
  - annual budget (node-hours) + overburn tolerance

Real programs (INCITE / ALCC / DD) are FITTED from the pbs_monitor DB via the
`allocation_type` column. Genesis has no history, so it is SYNTHESIZED from
explicit, labeled assumptions (see GenesisScenario).

Decisions baked in (Taylor, 2026-06-19):
  - Allocation split is a *yearly average* target, not instantaneous. Programs
    ramp slowly and finish strong; we model that with a seasonal burn curve.
  - Fair-share is NOT enforced. We model per-program budgets with overburn
    tolerance (INCITE may run +25% over allocation). The scheduler damps a
    program's priority as it approaches budget; it is hard-blocked only at the
    overburn ceiling.
  - Default share INCITE/ALCC/DD/Genesis = 50/25/10/15.
  - Genesis = AI-centric: 1-node, ~7-day jobs; starts mid-July 2026, ramps by
    Sep/Oct 2026.
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Program identifiers
# ---------------------------------------------------------------------------

INCITE = "INCITE"
ALCC = "ALCC"
DD = "DD"
GENESIS = "Genesis"

ALL_PROGRAMS = [INCITE, ALCC, DD, GENESIS]

# DB allocation_type -> program. Discretionary -> DD; UNKNOWN dropped by default.
DB_ALLOC_MAP = {
    "INCITE": INCITE,
    "ALCC": ALCC,
    "Discretionary": DD,
}

# Default annual allocation shares (yearly-average target node-hour fractions).
# Genesis carved from the others; configurable on the CLI.
DEFAULT_SHARES = {INCITE: 0.50, ALCC: 0.25, DD: 0.10, GENESIS: 0.15}

# Overburn tolerance: a program may deliver up to (1 + overburn) * budget before
# being hard-capped. INCITE gets +25% headroom; others none by default (DD is a
# soft floor, handled separately — it may exceed when capacity is idle).
DEFAULT_OVERBURN = {INCITE: 0.25, ALCC: 0.0, DD: 0.0, GENESIS: 0.0}

# Allocation calendars: (start_month, end_month) 1-indexed, inclusive.
# INCITE Jan-Dec (+13th-month extension exploits slow-start, handled via burn
# curve, not a separate window). ALCC Jul-Jun. DD continuous. Genesis: ramp
# starts mid-July 2026 (handled in GenesisScenario).
CALENDARS = {
    INCITE: (1, 12),
    ALCC: (7, 6),     # wraps year boundary
    DD: (1, 12),      # continuous
    GENESIS: (7, 6),  # mission-year placeholder; ramp handled separately
}


# ---------------------------------------------------------------------------
# Walltime parsing (shared with trace_sampler conventions)
# ---------------------------------------------------------------------------

def _parse_walltime_h(w) -> float:
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


# ---------------------------------------------------------------------------
# ProgramProfile
# ---------------------------------------------------------------------------

@dataclass
class ProgramProfile:
    """Everything needed to generate + account one program's workload."""
    name: str

    # Empirical sampling arrays (parallel is NOT required; sampled independently)
    nodes_arr: np.ndarray            # int node counts
    walltime_arr: np.ndarray         # requested walltime (h)
    rt_ratio_arr: np.ndarray         # runtime/walltime ratios in (0, 1]

    # Arrival
    arrival_rate_h: float            # mean jobs/hour (annual average)

    # Calendar + seasonality
    calendar: tuple                  # (start_month, end_month) inclusive, 1-idx
    monthly_multiplier: np.ndarray   # length-12, mean ~1.0, indexed by month-1
                                     # multiplies *arrival rate* to reproduce burn

    # Accounting
    annual_share: float              # yearly-average target fraction of machine
    overburn: float = 0.0            # fractional headroom above budget

    # Provenance
    synthetic: bool = False          # True for Genesis (assumption-driven)
    n_source_jobs: int = 0

    def sample_job(self, rng: np.random.Generator,
                   walltime_cap_h: float = 168.0) -> tuple[int, float, float]:
        """Return (nodes, walltime_h, runtime_h) for one job."""
        nodes = int(rng.choice(self.nodes_arr))
        wt = float(rng.choice(self.walltime_arr))
        wt = float(np.clip(wt, 0.083, walltime_cap_h))  # 5-min floor
        ratio = float(rng.choice(self.rt_ratio_arr))
        runtime = float(np.clip(ratio * wt, 0.05, wt))
        return nodes, wt, runtime

    def rate_at_month(self, month_1idx: int) -> float:
        """Arrival rate (jobs/h) modulated by the seasonal burn curve."""
        return self.arrival_rate_h * float(self.monthly_multiplier[month_1idx - 1])

    def summary_row(self) -> dict:
        return {
            "program": self.name,
            "n_jobs": self.n_source_jobs,
            "synthetic": self.synthetic,
            "lambda_per_day": round(self.arrival_rate_h * 24, 1),
            "nodes_p50": int(np.percentile(self.nodes_arr, 50)),
            "nodes_p90": int(np.percentile(self.nodes_arr, 90)),
            "nodes_p99": int(np.percentile(self.nodes_arr, 99)),
            "wt_p50_h": round(float(np.percentile(self.walltime_arr, 50)), 1),
            "wt_p90_h": round(float(np.percentile(self.walltime_arr, 90)), 1),
            "share": self.annual_share,
            "overburn": self.overburn,
        }


# ---------------------------------------------------------------------------
# DB loader + fitter for real programs
# ---------------------------------------------------------------------------

def _load_program_jobs(db_path: str, min_runtime_s: int = 30) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT submit_time, nodes, walltime,
                   actual_runtime_seconds AS runtime_s,
                   allocation_type
            FROM jobs
            WHERE state = 'FINISHED'
              AND nodes IS NOT NULL AND nodes > 0
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
    df["walltime_h"] = df["walltime"].apply(_parse_walltime_h)
    df["runtime_h"] = df["runtime_s"] / 3600.0
    # Fill bad walltimes with runtime * 1.5
    bad = df["walltime_h"].isna() | (df["walltime_h"] < df["runtime_h"])
    df.loc[bad, "walltime_h"] = (df.loc[bad, "runtime_h"] * 1.5).clip(lower=0.1)
    df["program"] = df["allocation_type"].map(DB_ALLOC_MAP)
    df = df.dropna(subset=["program"])
    df["month"] = df["submit_time"].dt.month
    return df


def _fit_monthly_multiplier(sub: pd.DataFrame) -> np.ndarray:
    """
    Fit a length-12 seasonal multiplier from a program's node-hour-by-month.

    We normalize node-hours per calendar month to mean 1.0 across the 12 months
    present. Months with no data get multiplier 1.0 (neutral). This captures
    the slow-start / strong-finish burn shape Taylor described.
    """
    sub = sub.copy()
    sub["nh"] = sub["nodes"] * sub["runtime_h"]
    by_month = sub.groupby("month")["nh"].sum()
    mult = np.ones(12)
    if by_month.sum() > 0:
        for m in range(1, 13):
            if m in by_month.index:
                mult[m - 1] = by_month[m]
        # Normalize present months to mean 1.0
        present = [m - 1 for m in by_month.index]
        vals = mult[present]
        if vals.mean() > 0:
            mult[present] = vals / vals.mean()
        # Any absent month stays at 1.0 (neutral)
    return mult


def fit_real_profiles(db_path: str, min_runtime_s: int = 30,
                      shares: dict = None,
                      overburn: dict = None) -> dict[str, ProgramProfile]:
    """Fit INCITE / ALCC / DD profiles from the DB."""
    shares = shares or DEFAULT_SHARES
    overburn = overburn or DEFAULT_OVERBURN
    df = _load_program_jobs(db_path, min_runtime_s)

    span_h = (df["submit_time"].max() - df["submit_time"].min()).total_seconds() / 3600.0
    profiles: dict[str, ProgramProfile] = {}

    for prog in (INCITE, ALCC, DD):
        sub = df[df["program"] == prog]
        if len(sub) == 0:
            raise ValueError(f"No jobs for program {prog!r} in DB")
        ratio = (sub["runtime_h"] / sub["walltime_h"]).clip(0.001, 1.0)
        profiles[prog] = ProgramProfile(
            name=prog,
            nodes_arr=sub["nodes"].to_numpy(np.int64),
            walltime_arr=sub["walltime_h"].to_numpy(np.float64),
            rt_ratio_arr=ratio.to_numpy(np.float64),
            arrival_rate_h=len(sub) / span_h if span_h > 0 else 1.0,
            calendar=CALENDARS[prog],
            monthly_multiplier=_fit_monthly_multiplier(sub),
            annual_share=shares.get(prog, 0.0),
            overburn=overburn.get(prog, 0.0),
            synthetic=False,
            n_source_jobs=len(sub),
        )
    return profiles


# ---------------------------------------------------------------------------
# Genesis synthesis (assumption-driven)
# ---------------------------------------------------------------------------

@dataclass
class GenesisScenario:
    """
    Explicit, labeled assumptions for the Genesis Mission program.

    Default reflects Taylor's description: AI-centric, 1-node / 7-day dominated,
    starts mid-July 2026, ramps to full by Sep/Oct 2026.
    """
    name: str = "genesis_ai_default"
    share: float = 0.15

    # Node-size mix: list of (nodes, weight). Default heavily 1-node.
    node_mix: tuple = ((1, 0.70), (2, 0.12), (4, 0.08), (8, 0.05),
                       (16, 0.03), (64, 0.015), (256, 0.005))
    # Walltime mix (hours): list of (walltime_h, weight). Default ~7-day heavy.
    walltime_mix: tuple = ((168.0, 0.55), (96.0, 0.20), (48.0, 0.12),
                           (24.0, 0.08), (6.0, 0.05))
    # Runtime/walltime ratio: AI jobs often run close to the wall.
    rt_ratio_mean: float = 0.85
    rt_ratio_spread: float = 0.12

    # Ramp: month index (1-12) when Genesis is at each fraction of full rate.
    # Slow mid-July start, ~full by Oct. Months outside the ramp before start
    # are 0; after full, 1.0.
    ramp_start_month: int = 7    # July
    ramp_full_month: int = 10    # October

    # Target full-load arrival rate (jobs/day) once ramped. Derived from share
    # if None (see synthesize()).
    full_lambda_per_day: Optional[float] = None

    def _arrays(self, n: int = 20000):
        nodes_vals, nodes_w = zip(*self.node_mix)
        wt_vals, wt_w = zip(*self.walltime_mix)
        rng = np.random.default_rng(20260719)
        nodes_w = np.array(nodes_w) / sum(nodes_w)
        wt_w = np.array(wt_w) / sum(wt_w)
        nodes_arr = rng.choice(nodes_vals, size=n, p=nodes_w).astype(np.int64)
        wt_arr = rng.choice(wt_vals, size=n, p=wt_w).astype(np.float64)
        rt_ratio = np.clip(
            rng.normal(self.rt_ratio_mean, self.rt_ratio_spread, size=n),
            0.05, 1.0)
        return nodes_arr, wt_arr, rt_ratio

    def _monthly_multiplier(self) -> np.ndarray:
        """Ramp curve: 0 before start, linear to 1.0 by full month, 1.0 after."""
        mult = np.zeros(12)
        for m in range(1, 13):
            if m < self.ramp_start_month:
                mult[m - 1] = 0.0
            elif m >= self.ramp_full_month:
                mult[m - 1] = 1.0
            else:
                frac = (m - self.ramp_start_month) / \
                       max(1, self.ramp_full_month - self.ramp_start_month)
                mult[m - 1] = frac
        # Normalize present (nonzero) months to mean 1.0 so arrival_rate_h is the
        # average over active months (keeps budget math consistent).
        active = mult[mult > 0]
        if active.mean() > 0:
            mult[mult > 0] = active / active.mean()
        return mult

    def synthesize(self, machine_nodes: int = 10624,
                   ref_profiles: dict = None) -> ProgramProfile:
        """
        Build a ProgramProfile for Genesis. If full_lambda_per_day is None,
        derive arrival rate so annual node-hours ≈ share × machine annual NH.
        """
        nodes_arr, wt_arr, rt_ratio = self._arrays()

        if self.full_lambda_per_day is not None:
            rate_h = self.full_lambda_per_day / 24.0
        else:
            # mean node-hours per job from the synthesized mix
            mean_nh_per_job = float(np.mean(nodes_arr * wt_arr * self.rt_ratio_mean))
            annual_nh_target = self.share * machine_nodes * 24 * 365
            # active-month fraction of the year (ramp): integrate multiplier/12
            mult = self._monthly_multiplier()
            active_frac = float(np.count_nonzero(mult)) / 12.0
            # jobs/year so that delivered NH ~ target (over active months)
            jobs_per_year = annual_nh_target / max(mean_nh_per_job, 1e-9)
            # spread over the active fraction of the year
            rate_h = jobs_per_year / (365 * 24 * max(active_frac, 1e-9))

        return ProgramProfile(
            name=GENESIS,
            nodes_arr=nodes_arr,
            walltime_arr=wt_arr,
            rt_ratio_arr=rt_ratio,
            arrival_rate_h=rate_h,
            calendar=CALENDARS[GENESIS],
            monthly_multiplier=self._monthly_multiplier(),
            annual_share=self.share,
            overburn=0.0,
            synthetic=True,
            n_source_jobs=0,
        )


# ---------------------------------------------------------------------------
# Convenience: named Genesis scenario factory
# ---------------------------------------------------------------------------

def genesis_scenario(name: str, share: float) -> GenesisScenario:
    """
    Return a labeled GenesisScenario for the given scenario name and share.

    Scenarios
    ---------
    ai_default      : AI-centric, 1-node / 7-day dominated, ramp Jul->Oct.
                      Reflects Taylor's description of Genesis Mission Phase 1.
    incite_like     : Capability jobs — large node counts (256/512/1024/2048),
                      6-24h walltimes, rt_ratio close to 1.0. Ramp Jul->Oct.
    bursty_campaign : Mid-to-large nodes with a rapid seasonal surge.
                      Ramp compresses to Jul->Aug (fast rise) to approximate
                      bursty demand; burstiness is captured via the steep ramp
                      multiplier shape (no AR(1) field exists on GenesisScenario).
    """
    name = name.lower()
    if name == "ai_default":
        return GenesisScenario(
            name="genesis_ai_default",
            share=share,
            node_mix=((1, 0.70), (2, 0.12), (4, 0.08), (8, 0.05),
                      (16, 0.03), (64, 0.015), (256, 0.005)),
            walltime_mix=((168.0, 0.55), (96.0, 0.20), (48.0, 0.12),
                          (24.0, 0.08), (6.0, 0.05)),
            rt_ratio_mean=0.85,
            rt_ratio_spread=0.12,
            ramp_start_month=7,
            ramp_full_month=10,
        )
    elif name == "incite_like":
        # Capability jobs: large nodes, shorter walltimes, high rt_ratio.
        return GenesisScenario(
            name="genesis_incite_like",
            share=share,
            node_mix=((256, 0.30), (512, 0.30), (1024, 0.25), (2048, 0.15)),
            walltime_mix=((6.0, 0.20), (12.0, 0.35), (18.0, 0.25), (24.0, 0.20)),
            rt_ratio_mean=0.90,
            rt_ratio_spread=0.08,
            ramp_start_month=7,
            ramp_full_month=10,
        )
    elif name == "bursty_campaign":
        # Mid-to-large nodes with a rapid Jul->Aug surge (compressed ramp
        # approximates bursty campaign behaviour; GenesisScenario has no
        # explicit AR(1)/burst amplitude field so we encode burstiness via
        # the steep seasonal multiplier shape).
        return GenesisScenario(
            name="genesis_bursty_campaign",
            share=share,
            node_mix=((64, 0.20), (128, 0.25), (256, 0.30), (512, 0.20),
                      (1024, 0.05)),
            walltime_mix=((6.0, 0.30), (12.0, 0.30), (24.0, 0.25),
                          (48.0, 0.10), (96.0, 0.05)),
            rt_ratio_mean=0.80,
            rt_ratio_spread=0.15,
            ramp_start_month=7,
            ramp_full_month=8,   # fast surge: fully ramped by August
        )
    else:
        raise ValueError(
            f"Unknown genesis scenario {name!r}. "
            f"Choose from: ai_default, incite_like, bursty_campaign")


# ---------------------------------------------------------------------------
# Convenience: build the full program set
# ---------------------------------------------------------------------------

def build_profiles(db_path: str, shares: dict = None,
                   overburn: dict = None,
                   genesis: Optional[GenesisScenario] = None,
                   machine_nodes: int = 10624,
                   include_genesis: bool = True) -> dict[str, ProgramProfile]:
    """Fit real programs + (optionally) synthesize Genesis. Returns dict."""
    shares = shares or DEFAULT_SHARES
    profiles = fit_real_profiles(db_path, shares=shares, overburn=overburn)
    if include_genesis:
        gs = genesis or GenesisScenario(share=shares.get(GENESIS, 0.15))
        profiles[GENESIS] = gs.synthesize(machine_nodes=machine_nodes,
                                          ref_profiles=profiles)
    return profiles


def print_summary(profiles: dict[str, ProgramProfile]) -> None:
    rows = [p.summary_row() for p in profiles.values()]
    df = pd.DataFrame(rows)
    print("\n=== Program profiles ===")
    print(df.to_string(index=False))
    print("\n=== Seasonal multipliers (monthly, mean~1.0 over active months) ===")
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    hdr = "  program   " + "".join(f"{m:>6s}" for m in months)
    print(hdr)
    for p in profiles.values():
        vals = "".join(f"{x:>6.2f}" for x in p.monthly_multiplier)
        print(f"  {p.name:9s} {vals}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Fit + report program profiles.")
    ap.add_argument("--trace-db",
                    default=os.environ.get("PBS_SIM_DB",
                                           "/Users/jchilders/pbs_monitor_aurora.db"))
    ap.add_argument("--machine-nodes", type=int, default=10624)
    ap.add_argument("--no-genesis", action="store_true")
    args = ap.parse_args()

    profiles = build_profiles(args.trace_db,
                              machine_nodes=args.machine_nodes,
                              include_genesis=not args.no_genesis)
    print_summary(profiles)
