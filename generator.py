"""
Joint conditional Monte Carlo job generator.

The v1 generator sampled marginals independently: nodes ~ P(nodes), walltime ~
P(walltime), each drawn separately. That destroys the correlation between a
job's size and its duration. This generator instead bootstraps WHOLE trace rows
from cells conditioned on (program, alloc-month-offset, size_tier), so a sampled
job keeps its real (nodes, walltime, runtime) TOGETHER.

Arrivals are a non-homogeneous Poisson process per program, with the rate
modulated by a per-program seasonal burn curve keyed on months-since-alloc-year-
start. This makes "ALCC ramps slowly after its July start" and "INCITE starts
fast in January" intrinsic program properties rather than calendar artifacts.

Genesis has no history, so it is synthesized from explicit GenesisConfig
assumptions (kept compatible with the existing scenario names).

Everything here is driven by SimConfig; no hard-coded policy values.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config import SimConfig, ProgramConfig


# DB allocation_type -> program name used in config
DB_ALLOC_MAP = {"INCITE": "INCITE", "ALCC": "ALCC", "Discretionary": "DD"}


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


def _parse_walltime_vec(s: "pd.Series") -> "pd.Series":
    """Vectorized HH:MM:SS -> hours. ~100x faster than row-wise apply."""
    parts = s.astype(str).str.split(":", expand=True)
    ncol = parts.shape[1]
    if ncol >= 3:
        h = pd.to_numeric(parts[0], errors="coerce")
        m = pd.to_numeric(parts[1], errors="coerce")
        sec = pd.to_numeric(parts[2], errors="coerce")
    elif ncol == 2:
        h = 0.0
        m = pd.to_numeric(parts[0], errors="coerce")
        sec = pd.to_numeric(parts[1], errors="coerce")
    else:
        return pd.Series(np.nan, index=s.index)
    return h + m / 60.0 + sec / 3600.0


@dataclass
class Job:
    """A generated job. Program-tagged; size_tier assigned from node count."""
    job_id: int
    program: str
    size_tier: str
    nodes: int
    walltime_h: float
    actual_runtime_h: float
    submit_time_h: float
    base_priority: float
    aging_rate: float
    start_time_h: Optional[float] = None
    end_time_h: Optional[float] = None

    def wait_h(self, now: float) -> float:
        return max(0.0, now - self.submit_time_h)


# ---------------------------------------------------------------------------
# DB load + fit
# ---------------------------------------------------------------------------

def load_trace(cfg: SimConfig) -> pd.DataFrame:
    con = sqlite3.connect(cfg.trace_db)
    try:
        df = pd.read_sql_query(
            """
            SELECT submit_time, nodes, walltime,
                   actual_runtime_seconds AS runtime_s,
                   queue_time_seconds AS qtime_s,
                   allocation_type
            FROM jobs
            WHERE state='FINISHED' AND nodes IS NOT NULL AND nodes>0
              AND actual_runtime_seconds IS NOT NULL
              AND actual_runtime_seconds >= ?
              AND submit_time IS NOT NULL
            """,
            con, params=[cfg.generator.min_runtime_s],
        )
    finally:
        con.close()

    df["submit_time"] = pd.to_datetime(df["submit_time"], errors="coerce")
    df = df.dropna(subset=["submit_time"])
    df["program"] = df["allocation_type"].map(DB_ALLOC_MAP)
    df = df.dropna(subset=["program"])
    df["walltime_h"] = _parse_walltime_vec(df["walltime"])
    df["runtime_h"] = df["runtime_s"] / 3600.0
    bad = df["walltime_h"].isna() | (df["walltime_h"] < df["runtime_h"])
    df.loc[bad, "walltime_h"] = (df.loc[bad, "runtime_h"] * 1.5).clip(lower=0.1)
    df["rt_ratio"] = (df["runtime_h"] / df["walltime_h"]).clip(0.001, 1.0)
    df["cal_month"] = df["submit_time"].dt.month

    # size_tier per row from config (vectorized via searchsorted on tier edges)
    tiers = sorted(cfg.size_tiers, key=lambda t: t.min_nodes)
    edges = np.array([t.max_nodes for t in tiers])
    names = np.array([t.name for t in tiers])
    idx = np.searchsorted(edges, df["nodes"].to_numpy(), side="left")
    idx = np.clip(idx, 0, len(tiers) - 1)
    df["size_tier"] = names[idx]
    return df


def _alloc_month_offset(cal_month: int, start_month: int) -> int:
    """Months since the program's allocation-year start (0..11)."""
    return (cal_month - start_month) % 12


class ConditionalSampler:
    """Bootstrap whole (nodes, walltime, runtime) rows from conditioned cells."""

    def __init__(self, df: pd.DataFrame, prog_cfg: ProgramConfig,
                 min_cell_rows: int):
        self.prog = prog_cfg.name
        self.start_month = prog_cfg.alloc_year_start_month
        self.min_cell_rows = min_cell_rows
        sub = df[df["program"] == self.prog].copy()
        sub["alloc_off"] = sub["cal_month"].apply(
            lambda m: _alloc_month_offset(m, self.start_month))
        # Cells keyed by (alloc_off, size_tier). Store index arrays for fast draw.
        self._cells: dict = {}
        for (off, tier), g in sub.groupby(["alloc_off", "size_tier"]):
            self._cells[(off, tier)] = g[["nodes", "walltime_h", "rt_ratio"]].to_numpy()
        # Parent (program-wide) pool for fallback
        self._parent = sub[["nodes", "walltime_h", "rt_ratio"]].to_numpy()
        # size_tier pool (program + tier) for intermediate fallback
        self._tier_pool: dict = {}
        for tier, g in sub.groupby("size_tier"):
            self._tier_pool[tier] = g[["nodes", "walltime_h", "rt_ratio"]].to_numpy()
        self.n_source = len(sub)

    def sample_row(self, alloc_off: int, rng: np.random.Generator) -> tuple:
        """Draw one joint (nodes, walltime_h, runtime_h). Picks a size_tier by
        the empirical tier mix in this alloc-offset, then a row within cell."""
        # Choose a cell for this alloc offset weighted by cell size; fall back
        # progressively if the offset is sparse.
        candidates = [(off, tier) for (off, tier) in self._cells if off == alloc_off]
        pool = None
        if candidates:
            weights = np.array([len(self._cells[c]) for c in candidates], float)
            weights /= weights.sum()
            ci = rng.choice(len(candidates), p=weights)
            cell = self._cells[candidates[ci]]
            if len(cell) >= self.min_cell_rows:
                pool = cell
        if pool is None:
            pool = self._parent  # program-wide fallback
        idx = rng.integers(len(pool))
        nodes, wt, ratio = pool[idx]
        runtime = float(np.clip(ratio * wt, 0.05, wt))
        return int(nodes), float(wt), runtime


def fit_burn_curve(df: pd.DataFrame, prog_cfg: ProgramConfig) -> np.ndarray:
    """Length-12 multiplier on ARRIVAL RATE keyed on alloc-month-offset.
    Normalized to mean 1.0 over active months so the base arrival rate stays the
    annual average."""
    sub = df[df["program"] == prog_cfg.name].copy()
    sub["alloc_off"] = sub["cal_month"].apply(
        lambda m: _alloc_month_offset(m, prog_cfg.alloc_year_start_month))
    sub["nh"] = sub["nodes"] * sub["runtime_h"]
    by_off = sub.groupby("alloc_off")["nh"].sum()
    mult = np.ones(12)
    if by_off.sum() > 0:
        for off in range(12):
            if off in by_off.index:
                mult[off] = by_off[off]
        present = [o for o in by_off.index]
        vals = mult[present]
        if vals.mean() > 0:
            mult[present] = vals / vals.mean()
    return mult


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class JobGenerator:
    def __init__(self, cfg: SimConfig, df: pd.DataFrame):
        self.cfg = cfg
        self.df = df
        span_h = (df["submit_time"].max() - df["submit_time"].min()).total_seconds() / 3600.0
        self.span_h = span_h
        self._tier_cfg = {t.name: t for t in cfg.size_tiers}

        self.samplers: dict = {}
        self.rates_h: dict = {}       # annual-average arrival rate jobs/h
        self.burn: dict = {}          # alloc-offset multiplier
        self.start_month: dict = {}
        for p in cfg.programs:
            self.samplers[p.name] = ConditionalSampler(
                df, p, cfg.generator.min_cell_rows)
            n = int((df["program"] == p.name).sum())
            self.rates_h[p.name] = n / span_h if span_h > 0 else 1.0
            self.burn[p.name] = fit_burn_curve(df, p)
            self.start_month[p.name] = p.alloc_year_start_month

    def _cal_month_at(self, t_h: float, sim_start_month: int) -> int:
        day = int(t_h // 24)
        month_offset = day // 30
        return ((sim_start_month - 1 + month_offset) % 12) + 1

    def generate(self, rng: np.random.Generator) -> list[Job]:
        cfg = self.cfg
        duration_h = cfg.run.duration_days * 24.0
        sim_start = cfg.run.start_month
        jobs: list[Job] = []

        for p in cfg.programs:
            base_rate = self.rates_h[p.name]
            burn = self.burn[p.name]
            smonth = self.start_month[p.name]
            t = 0.0
            while t < duration_h:
                cal_month = self._cal_month_at(t, sim_start)
                alloc_off = _alloc_month_offset(cal_month, smonth)
                rate = base_rate * float(burn[alloc_off])
                if rate <= 0:
                    t += 24.0
                    continue
                t += rng.exponential(1.0 / rate)
                if t >= duration_h:
                    break
                nodes, wt, runtime = self.samplers[p.name].sample_row(alloc_off, rng)
                tier = self.cfg.tier_for_nodes(nodes)
                wt = min(wt, tier.walltime_cap_h)
                runtime = min(runtime, wt)
                jobs.append(Job(
                    job_id=0, program=p.name, size_tier=tier.name,
                    nodes=nodes, walltime_h=wt, actual_runtime_h=runtime,
                    submit_time_h=t,
                    base_priority=tier.base_priority, aging_rate=tier.aging_rate))

        # Genesis (synthetic)
        if cfg.genesis.enabled:
            jobs += self._generate_genesis(rng)

        jobs.sort(key=lambda j: j.submit_time_h)
        for i, j in enumerate(jobs):
            j.job_id = i
        return jobs

    def _generate_genesis(self, rng: np.random.Generator) -> list[Job]:
        from genesis import build_genesis_arrays, genesis_burn, genesis_rate_h
        cfg = self.cfg
        gc = cfg.genesis
        duration_h = cfg.run.duration_days * 24.0
        sim_start = cfg.run.start_month
        nodes_arr, wt_arr, rt_arr = build_genesis_arrays(gc, rng)
        burn = genesis_burn(gc)   # calendar-month keyed ramp
        rate_h = genesis_rate_h(gc, cfg.machine.total_nodes,
                                nodes_arr, wt_arr, rt_arr)
        jobs = []
        t = 0.0
        while t < duration_h:
            cal_month = self._cal_month_at(t, sim_start)
            rate = rate_h * float(burn[cal_month - 1])
            if rate <= 0:
                t += 24.0
                continue
            t += rng.exponential(1.0 / rate)
            if t >= duration_h:
                break
            i = rng.integers(len(nodes_arr))
            nodes = int(nodes_arr[i]); wt = float(wt_arr[i])
            tier = cfg.tier_for_nodes(nodes)
            wt = min(wt, tier.walltime_cap_h)
            runtime = float(np.clip(rt_arr[i] * wt, 0.05, wt))
            jobs.append(Job(
                job_id=0, program="Genesis", size_tier=tier.name,
                nodes=nodes, walltime_h=wt, actual_runtime_h=runtime,
                submit_time_h=t,
                base_priority=tier.base_priority, aging_rate=tier.aging_rate))
        return jobs
