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


@dataclass(eq=False)
class Job:
    """A generated job. Program-tagged; size_tier assigned from node count.

    eq=False -> identity equality/hash, so `job in list` and list.remove(job)
    use `is` (O(1) compare) instead of field-by-field dataclass __eq__, which
    was a scheduler hot-path cost. Jobs are unique objects; identity is correct.
    """
    job_id: int
    program: str
    size_tier: str
    nodes: int
    walltime_h: float
    actual_runtime_h: float
    submit_time_h: float
    base_priority: float
    aging_rate: float
    project: str = ""              # project id (project layer); "" if disabled
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
                   allocation_type, project
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


def fit_burn_curve(df: pd.DataFrame, prog_cfg: ProgramConfig,
                   min_coverage_frac: float = 0.40,
                   min_mult_floor: float = 0.40) -> np.ndarray:
    """Length-12 multiplier on ARRIVAL RATE keyed on alloc-month-offset.
    Normalized to mean 1.0 over active months so the base arrival rate stays the
    annual average.

    LOW-DATA MONTH SMOOTHING: the trace under-covers some months. **June** is a
    true coverage artifact (trace starts mid-June / ends early-June: ~3 weeks
    split across two partial years, e.g. 3 INCITE jobs in 2025-06). **July** is a
    real but low summer lull. Both otherwise fit tiny multipliers that produce a
    spurious ~50-day mid-year utilization collapse. Any month whose job COUNT is
    below `min_coverage_frac` x the median month's count has its multiplier
    replaced by the mean of its (circular) neighbors, then the curve is
    renormalized.

    NOTE on the 0.40 default: this smooths BOTH June (coverage artifact) AND July
    (a real summer lull) — a deliberate modeling CHOICE to keep the simulated
    year flat rather than reproduce the trace's summer dip. It discards a real
    (if noisy) signal; set min_coverage_frac lower (~0.20) to keep July's lull,
    or 0 to disable smoothing entirely. Config-adjustable per study.
    """
    sub = df[df["program"] == prog_cfg.name].copy()
    sub["alloc_off"] = sub["cal_month"].apply(
        lambda m: _alloc_month_offset(m, prog_cfg.alloc_year_start_month))
    sub["nh"] = sub["nodes"] * sub["runtime_h"]
    by_off_nh = sub.groupby("alloc_off")["nh"].sum()
    by_off_n = sub.groupby("alloc_off").size()
    mult = np.ones(12)
    if by_off_nh.sum() > 0:
        for off in range(12):
            if off in by_off_nh.index:
                mult[off] = by_off_nh[off]
        present = [o for o in by_off_nh.index]
        vals = mult[present]
        if vals.mean() > 0:
            mult[present] = vals / vals.mean()

        # Low-data / low-demand month smoothing: interpolate a month from its
        # neighbors if EITHER it is under-covered (job count < min_coverage_frac
        # x median) OR its fitted multiplier is below min_mult_floor (a month so
        # low it would idle the machine — whether from thin coverage or a real
        # lull). Both criteria are config knobs; set to 0 to disable.
        if len(by_off_n) > 2 and (min_coverage_frac > 0 or min_mult_floor > 0):
            counts = np.zeros(12)
            for off in range(12):
                counts[off] = by_off_n.get(off, 0)
            present_counts = counts[counts > 0]
            med = np.median(present_counts) if len(present_counts) else 0.0
            cthresh = med * min_coverage_frac
            low = [off for off in range(12)
                   if (min_coverage_frac > 0 and counts[off] < cthresh)
                   or (min_mult_floor > 0 and off in present and mult[off] < min_mult_floor)]
            for off in low:
                # neighbor mean over the nearest non-low months (circular)
                lo = mult[(off - 1) % 12]
                hi = mult[(off + 1) % 12]
                mult[off] = (lo + hi) / 2.0
            if mult[present].mean() > 0:
                mult[present] = mult[present] / mult[present].mean()
    return mult


# ---------------------------------------------------------------------------
# Project layer (optional): per-project pools + awards + arrival rates
# ---------------------------------------------------------------------------

class ProjectSampler:
    """Holds one project's real job rows for joint bootstrap + its award/rate."""

    def __init__(self, project: str, program: str, rows: np.ndarray,
                 arrival_rate_h: float, delivered_nh: float,
                 award_nh: float):
        self.project = project
        self.program = program
        self._rows = rows                     # (nodes, walltime_h, rt_ratio)
        self.arrival_rate_h = arrival_rate_h  # annual-avg jobs/h for this project
        self.delivered_nh = delivered_nh      # historical delivered node-h
        self.award_nh = award_nh              # notional award (over-allocated)

    def sample_row(self, rng: np.random.Generator) -> tuple:
        idx = rng.integers(len(self._rows))
        nodes, wt, ratio = self._rows[idx]
        runtime = float(np.clip(ratio * wt, 0.05, wt))
        return int(nodes), float(wt), runtime


def build_project_samplers(df: pd.DataFrame, cfg: SimConfig) -> dict:
    """Partition each program's jobs into per-project ProjectSamplers.

    Award model: a project's notional award = its historical delivered node-h
    scaled by the program's over_allocation factor. Because jobs are
    bootstrapped from the project's REAL delivered mix, simulated delivery lands
    near the historical (under-used) level — the target-vs-delivered gap and the
    over-allocation both fall out of the data rather than being imposed.

    Small projects (< min_project_jobs) are folded into a program-wide
    '<PROG>_misc' pseudo-project so we don't carry thousands of 1-job projects.
    """
    span_h = df.attrs.get("span_h") if hasattr(df, "attrs") else None
    if not span_h:
        span_h = (df["submit_time"].max() - df["submit_time"].min()).total_seconds() / 3600.0
    over = cfg.projects.over_allocation
    min_jobs = cfg.projects.min_project_jobs
    load = cfg.generator.load_multiplier
    prog_names = {p.name for p in cfg.programs}

    out: dict[str, list[ProjectSampler]] = {p: [] for p in prog_names}
    df = df.copy()
    df["nh"] = df["nodes"] * df["runtime_h"]
    for prog in prog_names:
        sub = df[df["program"] == prog]
        if len(sub) == 0:
            continue
        counts = sub.groupby("project").size()
        big = set(counts[counts >= min_jobs].index)
        # assign a synthetic project id for small ones
        proj_col = sub["project"].where(sub["project"].isin(big), f"{prog}_misc")
        for proj, g in sub.assign(_p=proj_col).groupby("_p"):
            rows = g[["nodes", "walltime_h", "rt_ratio"]].to_numpy()
            rate_h = (len(g) / span_h if span_h > 0 else 1.0) * load
            delivered = float(g["nh"].sum())
            award = delivered * over
            out[prog].append(ProjectSampler(
                project=str(proj), program=prog, rows=rows,
                arrival_rate_h=rate_h, delivered_nh=delivered, award_nh=award))
    return out


def build_deadline_windows(cfg: SimConfig, projects_by_prog: dict,
                           rng: np.random.Generator) -> list:
    """Return [(start_h, end_h, rate_multiplier, set_of_affected_projects)].

    Deadlines are calendar-month based; we map them into sim-time using
    run.start_month and 30-day months (consistent with _cal_month_at).
    A configurable fraction of projects 'chase' each deadline.
    """
    if not cfg.deadlines.enabled:
        return []
    all_projects = [ps.project for lst in projects_by_prog.values() for ps in lst]
    windows = []
    sim_start = cfg.run.start_month
    duration_h = cfg.run.duration_days * 24.0
    for dl in cfg.deadlines.deadlines:
        # month offset from sim start (0..) -> approx day -> hour
        month_off = (dl.month - sim_start) % 12
        deadline_day = month_off * 30 + (dl.day - 1)
        end_h = deadline_day * 24.0
        start_h = end_h - dl.lead_days * 24.0
        # also add the next-year occurrence if the run is long enough
        for shift in range(0, int(cfg.run.duration_days // 365) + 1):
            s = start_h + shift * 365 * 24.0
            e = end_h + shift * 365 * 24.0
            if e < 0 or s > duration_h:
                continue
            k = max(1, int(round(dl.affected_fraction * len(all_projects))))
            affected = set(rng.choice(all_projects, size=min(k, len(all_projects)),
                                      replace=False)) if all_projects else set()
            windows.append((max(0.0, s), e, dl.rate_multiplier, affected))
    return windows


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class JobGenerator:
    def __init__(self, cfg: SimConfig, df: pd.DataFrame):
        self.cfg = cfg
        self.df = df
        span_h = df.attrs.get("span_h") if hasattr(df, "attrs") else None
        if not span_h:
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
            self.rates_h[p.name] = (n / span_h if span_h > 0 else 1.0) * cfg.generator.load_multiplier
            self.burn[p.name] = fit_burn_curve(df, p)
            self.start_month[p.name] = p.alloc_year_start_month

        # Optional project layer
        self.projects_by_prog: dict = {}
        if cfg.projects.enabled:
            self.projects_by_prog = build_project_samplers(df, cfg)

        # Stage-2: walltime policy + behavioral size-choice
        self._wp = cfg.walltime_policy
        self._bh = cfg.behavior
        self._adapt_by_prog = {p: cfg.behavior.adapt_fraction for p in
                               [pc.name for pc in cfg.programs] + ["Genesis"]}
        for prog, frac in cfg.behavior.program_adapt:
            self._adapt_by_prog[prog] = frac

    def _finalize(self, program, nodes, wt, runtime, rng):
        """Apply the (optional) behavioral size-choice + walltime policy, then
        return (nodes, size_tier, walltime_h, runtime_h) ready for a Job.

        Order:
          1. Determine the desired walltime = the originally-sampled wt.
          2. If behavior enabled and this job adapts and desires a long run that
             the policy would cap: resize UP to the smallest node count whose
             policy cap >= desired wt (bounded by max_resize_nodes).
          3. Apply the walltime policy (or tier fallback cap) at the final size.
          4. Clamp runtime <= final walltime (scale runtime proportionally so a
             capped job doesn't keep an impossible runtime).
        """
        desired_wt = wt
        # (2) behavioral resize to chase walltime
        if (self._bh.enabled and self._wp.enabled
                and desired_wt >= self._bh.min_desired_walltime_h
                and rng.random() < self._adapt_by_prog.get(program, self._bh.adapt_fraction)):
            tier0 = self.cfg.tier_for_nodes(nodes)
            cap_now = self._wp.cap_for(nodes, tier0.walltime_cap_h)
            if cap_now < desired_wt:
                new_nodes = self._smallest_nodes_for_walltime(desired_wt, nodes)
                if new_nodes is not None and new_nodes != nodes:
                    # resize: scale runtime fraction with the new (larger) wt cap,
                    # keep the runtime/walltime ratio the user originally had
                    ratio = runtime / wt if wt > 0 else 1.0
                    nodes = new_nodes
                    wt = desired_wt
                    runtime = ratio * wt
        # (3) apply walltime cap at final size
        tier = self.cfg.tier_for_nodes(nodes)
        cap = self._wp.cap_for(nodes, tier.walltime_cap_h)
        if wt > cap:
            ratio = runtime / wt if wt > 0 else 1.0
            wt = cap
            runtime = ratio * wt
        runtime = min(runtime, wt)
        return nodes, tier, wt, runtime

    def _smallest_nodes_for_walltime(self, desired_wt, cur_nodes):
        """Smallest node count (>= cur_nodes, <= max_resize_nodes) whose walltime
        policy cap >= desired_wt. None if the policy never allows desired_wt in
        that range (job can't get its wish; stays put)."""
        if not self._wp.enabled or not self._wp.breakpoints:
            return None
        cap_ceiling = self._bh.max_resize_nodes
        # breakpoints sorted; find the first min_nodes whose cap >= desired_wt
        best = None
        for min_n, cap_wt in sorted(self._wp.breakpoints, key=lambda b: b[0]):
            if cap_wt >= desired_wt and min_n <= cap_ceiling:
                best = int(min_n) if best is None else min(best, int(min_n))
        # must be at least cur_nodes (resize UP only) and within ceiling
        if best is None:
            return None
        return max(best, cur_nodes) if max(best, cur_nodes) <= cap_ceiling else None

    def _cal_month_at(self, t_h: float, sim_start_month: int) -> int:
        day = int(t_h // 24)
        month_offset = day // 30
        return ((sim_start_month - 1 + month_offset) % 12) + 1

    def generate(self, rng: np.random.Generator) -> list[Job]:
        cfg = self.cfg
        duration_h = cfg.run.duration_days * 24.0
        sim_start = cfg.run.start_month
        jobs: list[Job] = []

        if cfg.projects.enabled:
            jobs += self._generate_by_project(rng)
        else:
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
                    nodes, tier, wt, runtime = self._finalize(p.name, nodes, wt, runtime, rng)
                    jobs.append(Job(
                        job_id=0, program=p.name, size_tier=tier.name,
                        nodes=nodes, walltime_h=wt, actual_runtime_h=runtime,
                        submit_time_h=t,
                        base_priority=tier.base_priority, aging_rate=tier.aging_rate))

        # Genesis (synthetic) — program-level regardless of project layer
        if cfg.genesis.enabled:
            jobs += self._generate_genesis(rng)

        jobs.sort(key=lambda j: j.submit_time_h)
        for i, j in enumerate(jobs):
            j.job_id = i
        return jobs

    def _generate_by_project(self, rng: np.random.Generator) -> list[Job]:
        """Per-project arrivals. Each project uses its own arrival rate, its
        program's seasonal burn curve, and any conference-deadline spike that
        applies to it in the current time window."""
        cfg = self.cfg
        duration_h = cfg.run.duration_days * 24.0
        sim_start = cfg.run.start_month
        burn_by_prog = {p.name: self.burn[p.name] for p in cfg.programs}
        smonth_by_prog = {p.name: p.alloc_year_start_month for p in cfg.programs}
        windows = build_deadline_windows(cfg, self.projects_by_prog, rng)
        jobs: list[Job] = []

        def deadline_mult(project: str, t: float) -> float:
            m = 1.0
            for (s, e, mult, affected) in windows:
                if s <= t <= e and project in affected:
                    m *= mult
            return m

        for prog, samplers in self.projects_by_prog.items():
            burn = burn_by_prog[prog]
            smonth = smonth_by_prog[prog]
            for ps in samplers:
                if ps.arrival_rate_h <= 0:
                    continue
                t = 0.0
                while t < duration_h:
                    cal_month = self._cal_month_at(t, sim_start)
                    alloc_off = _alloc_month_offset(cal_month, smonth)
                    rate = ps.arrival_rate_h * float(burn[alloc_off]) \
                        * deadline_mult(ps.project, t)
                    if rate <= 0:
                        t += 24.0
                        continue
                    t += rng.exponential(1.0 / rate)
                    if t >= duration_h:
                        break
                    nodes, wt, runtime = ps.sample_row(rng)
                    nodes, tier, wt, runtime = self._finalize(prog, nodes, wt, runtime, rng)
                    jobs.append(Job(
                        job_id=0, program=prog, size_tier=tier.name,
                        nodes=nodes, walltime_h=wt, actual_runtime_h=runtime,
                        submit_time_h=t, project=ps.project,
                        base_priority=tier.base_priority, aging_rate=tier.aging_rate))
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
            runtime = float(np.clip(rt_arr[i] * wt, 0.05, wt))
            nodes, tier, wt, runtime = self._finalize("Genesis", nodes, wt, runtime, rng)
            jobs.append(Job(
                job_id=0, program="Genesis", size_tier=tier.name,
                nodes=nodes, walltime_h=wt, actual_runtime_h=runtime,
                submit_time_h=t,
                base_priority=tier.base_priority, aging_rate=tier.aging_rate))
        return jobs
