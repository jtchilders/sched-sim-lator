#!/usr/bin/env python3
"""
Program-aware PBS queue simulation: INCITE / ALCC / DD / Genesis.

Builds on sim.py's discrete-event Scheduler but adds the allocation-program
dimension that the v1 baseline lacks. Answers the INCITE/ALCC-vs-Genesis
balancing question.

Model of the policy lever (per Taylor, 2026-06-19)
--------------------------------------------------
Fair-share is NOT enforced. ALCF allocates a yearly-average node-hour budget per
program and *aims* for full utilization; many projects under-use. So instead of
a fair-share steering term we model:

  * Per-program annual budget  B_p = share_p * machine_node_hours_per_year.
  * A program may overburn up to (1 + overburn_p) * B_p before being hard-capped
    (INCITE overburn = 0.25; others 0 by default; DD is a soft floor so it is
    never hard-capped when capacity is idle).
  * Priority DAMPS smoothly as a program's delivered node-hours approach its
    (pro-rated) budget, so under-using programs keep easy access and heavy
    burners yield — but nothing is blocked until the overburn ceiling.

Policies (--policy)
  blind     : program-agnostic (reproduces v1 behavior; validation baseline)
  budget    : budget-damped priority + overburn hard-cap (the realistic model)

Usage
  python sim_programs.py --duration-days 365 --policy budget --plot
  python sim_programs.py --duration-days 365 --policy budget \
      --shares 0.50,0.25,0.10,0.15 --genesis-from incite
"""
from __future__ import annotations

import argparse
import os
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import sim          # v1 engine (Job, Scheduler, QUEUES, QueueConfig)
import program_profiles as pp


# ----------------------------------------------------------------------
# Program-aware job: extend sim.Job with a program tag + node-hour cost
# ----------------------------------------------------------------------

def _bucket_for_nodes(nodes: int) -> str:
    for qc in sim.QUEUES:
        if qc.min_nodes <= nodes <= qc.max_nodes:
            return qc.name
    return sim.QUEUES[-1].name


# ----------------------------------------------------------------------
# Program-aware generator
# ----------------------------------------------------------------------

class ProgramJobGenerator:
    """
    Generate jobs across programs using fitted ProgramProfiles, modulated by
    each program's seasonal burn curve over the simulated calendar.

    The simulation clock starts at `start_month` (1-12) so the seasonal curves
    align with a real calendar (default: Jan = month 1).
    """

    def __init__(self, profiles: dict[str, pp.ProgramProfile],
                 rng: np.random.Generator, start_month: int = 1):
        self.profiles = profiles
        self.rng = rng
        self.start_month = start_month

    def _month_at(self, t_h: float) -> int:
        """Calendar month (1-12) at simulation time t_h, wrapping each year."""
        day = int(t_h // 24)
        month_offset = day // 30          # approx 30-day months — fine for burn
        return ((self.start_month - 1 + month_offset) % 12) + 1

    def generate(self, duration_h: float) -> list[sim.Job]:
        jobs: list[sim.Job] = []
        # Bucket scoring params come from the size bucket the job lands in.
        bucket_cfg = {qc.name: qc for qc in sim.QUEUES}

        for prog, prof in self.profiles.items():
            # Non-homogeneous Poisson via month-wise thinning on the seasonal
            # multiplier. Walk day-by-day; rate = arrival_rate_h * mult(month).
            t = 0.0
            while t < duration_h:
                month = self._month_at(t)
                rate = prof.rate_at_month(month)
                if rate <= 0:
                    t += 24.0  # skip an inactive day (e.g. Genesis pre-ramp)
                    continue
                dt = self.rng.exponential(1.0 / rate)
                t += dt
                if t >= duration_h:
                    break
                nodes, wt, runtime = prof.sample_job(self.rng)
                bucket = _bucket_for_nodes(nodes)
                qc = bucket_cfg[bucket]
                wt = min(wt, qc.walltime_cap_h)
                runtime = min(runtime, wt)
                j = sim.Job(
                    job_id=0, queue=bucket, nodes=nodes,
                    walltime_h=wt, actual_runtime_h=runtime,
                    submit_time_h=t,
                    base_priority=qc.base_priority, aging_rate=qc.aging_rate,
                )
                j.program = prog                       # attach program tag
                jobs.append(j)

        jobs.sort(key=lambda j: j.submit_time_h)
        for i, j in enumerate(jobs):
            j.job_id = i
        return jobs


# ----------------------------------------------------------------------
# Program-aware scheduler: budget-damped priority + overburn hard-cap
# ----------------------------------------------------------------------

class ProgramScheduler(sim.Scheduler):
    """
    Extends the v1 Scheduler with per-program budget accounting.

    delivered_nh[p] accrues as jobs run. The pro-rated budget at time t is
    B_p * (t / year_h) — i.e. the share a program "should" have consumed by now
    if burning evenly. We compute a damping factor on each job's score based on
    how far over its pro-rated budget the program already is, and we hard-block a
    job only if starting it would exceed the overburn ceiling AND the program is
    not DD (DD is a soft floor: allowed when nodes are otherwise idle).
    """

    def __init__(self, *args, profiles: dict[str, pp.ProgramProfile] = None,
                 policy: str = "budget", machine_nodes: int = 10624,
                 year_h: float = 365 * 24.0,
                 damp_strength: float = 8.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.profiles = profiles or {}
        self.policy = policy
        self.year_h = year_h
        self.damp_strength = damp_strength
        # Annual node-hour budgets
        machine_year_nh = machine_nodes * year_h
        self.budget_nh = {p: prof.annual_share * machine_year_nh
                          for p, prof in self.profiles.items()}
        self.overburn = {p: prof.overburn for p, prof in self.profiles.items()}
        self.delivered_nh: dict[str, float] = {p: 0.0 for p in self.profiles}
        # Telemetry: (t, {program: delivered_nh})
        self.program_nh_samples: list = []
        # Precomputed ceiling constants (immutable for the run). DD is omitted
        # because it is a soft floor (never hard-capped) -> .get() returns None.
        self._budget_policy = (self.policy == "budget")
        self._ceiling_nh: dict[str, float] = {
            p: self.budget_nh[p] * (1.0 + self.overburn.get(p, 0.0))
            for p in self.budget_nh if p != pp.DD
        }
        # Perf caches
        self._min_pending_cache: Optional[int] = None

    # -- perf helpers ---------------------------------------------------

    def _min_pending_nodes(self) -> int:
        """Smallest node request among pending jobs (cached). Used as an exact
        early-exit guard.

        Maintenance is exact and cheap: the cache is recomputed lazily whenever
        it is None. Arrivals (which only *append* to pending) update it
        incrementally via _note_arrival; removals (job starts) invalidate it so
        it is recomputed on next use.
        """
        if self._min_pending_cache is None:
            self._min_pending_cache = min((j.nodes for j in self.pending),
                                          default=1 << 30)
        return self._min_pending_cache

    def _note_arrival(self, job):
        """An arrival only lowers the min, so update incrementally (no rescan)."""
        if self._min_pending_cache is None:
            return  # will be recomputed lazily anyway
        if job.nodes < self._min_pending_cache:
            self._min_pending_cache = job.nodes

    def _invalidate_pending_cache(self):
        self._min_pending_cache = None

    # -- budget helpers -------------------------------------------------

    def _prorated_budget(self, prog: str) -> float:
        frac = min(1.0, self.now_h / self.year_h) if self.year_h > 0 else 1.0
        return self.budget_nh.get(prog, float("inf")) * frac

    def _damp_factor(self, prog: str) -> float:
        """
        Multiplicative factor in (0, 1] applied to a job's score. 1.0 when under
        budget; decays toward 0 as delivered/prorated_budget exceeds 1.
        Under-using programs are never penalized (factor stays 1.0).
        """
        if self.policy != "budget" or prog not in self.budget_nh:
            return 1.0
        pb = self._prorated_budget(prog)
        if pb <= 0:
            return 1.0
        ratio = self.delivered_nh.get(prog, 0.0) / pb
        if ratio <= 1.0:
            return 1.0
        # Smoothly damp once over pro-rated budget.
        return math.exp(-self.damp_strength * (ratio - 1.0))

    def _over_ceiling(self, job) -> bool:
        """True if starting `job` would breach the program's overburn ceiling."""
        if not self._budget_policy:
            return False
        prog = getattr(job, "program", None)
        ceiling = self._ceiling_nh.get(prog)
        if ceiling is None:  # unknown program or DD (soft floor) -> never capped
            return False
        job_nh = job.nodes * job.walltime_h
        return (self.delivered_nh.get(prog, 0.0) + job_nh) > ceiling

    # -- overrides ------------------------------------------------------

    def _can_start(self, job) -> bool:
        # Inlined hot path (called ~14M times). Avoids super() + helper-call +
        # max() overhead that dominated the profile. Semantics identical to
        # sim.Scheduler._can_start + the overburn ceiling check.
        if job.nodes > self.free_nodes:
            return False
        if job.queue == sim.CAPACITY_QUEUE:
            pool_free = self.capacity_pool - self._capacity_nodes_in_use
            if job.nodes > pool_free:
                return False
        if self._over_ceiling(job):
            return False
        return True

    def _program_score(self, job, now: float,
                       damp_cache: dict = None) -> float:
        """Base score times program budget damping."""
        prog = getattr(job, "program", None)
        if damp_cache is not None:
            damp = damp_cache.get(prog)
            if damp is None:
                damp = self._damp_factor(prog)
                damp_cache[prog] = damp
        else:
            damp = self._damp_factor(prog)
        return job.score(now) * damp

    def _try_schedule(self):
        """
        Same EASY-backfill logic as v1, but the priority ordering uses the
        budget-damped program score instead of the raw score.

        Perf (semantics-preserving fast paths):
          1. Min-nodes guard. A job can only start now if free_nodes >=
             job.nodes (true for both the greedy pass and backfill). We track
             the smallest node request in `pending`; if free_nodes is below it,
             NOTHING can start, so we skip the O(n log n) sort + O(n) walk
             entirely. This is exact, not heuristic. Finish events free only a
             small slice of nodes, so most of them hit this guard and return
             immediately instead of re-sorting a deep pending list.
          2. Damp cache. The budget damp factor depends only on (program, now),
             not the individual job, so compute it once per program per pass.
        """
        if not self.pending:
            return
        # (1) Exact min-nodes guard: if even the smallest pending job can't fit,
        # no greedy start and no backfill start is possible this pass.
        if self.free_nodes < self._min_pending_nodes():
            return
        now = self.now_h
        damp_cache: dict = {}
        # Precompute the sort key once per job (avoids recomputation inside the
        # comparator, which Python would otherwise call O(n log n) times).
        keyed = [(-self._program_score(j, now, damp_cache), j.submit_time_h, i)
                 for i, j in enumerate(self.pending)]
        keyed.sort()
        order = [k[2] for k in keyed]
        reservation_time: Optional[float] = None
        reserved_idx: Optional[int] = None
        removed: set = set()

        for idx in order:
            j = self.pending[idx]
            if self._can_start(j):
                self._start_at_index(j, idx, removed)
            elif reserved_idx is None and not self._over_ceiling(j):
                # Don't let an over-ceiling job hold a reservation.
                reserved_idx = idx
                reservation_time = self._estimate_reservation(j)

        if reservation_time is not None and self.enable_backfill:
            for idx in order:
                if idx in removed or idx == reserved_idx:
                    continue
                j = self.pending[idx]
                if not self._can_start(j):
                    continue
                if now + j.walltime_h <= reservation_time + 1e-9:
                    self._start_at_index(j, idx, removed)

        if removed:
            self.pending = [j for i, j in enumerate(self.pending)
                            if i not in removed]
            self._invalidate_pending_cache()  # membership shrank

    def _start_at_index(self, job, idx, removed):
        super()._start_at_index(job, idx, removed)
        # Accrue delivered node-hours (use actual runtime — real consumption).
        prog = getattr(job, "program", None)
        if prog in self.delivered_nh:
            self.delivered_nh[prog] += job.nodes * job.actual_runtime_h

    def run(self, jobs, duration_h, sample_dt_h: float = 0.25):
        """Event loop mirroring sim.Scheduler.run, with min-pending-cache
        maintenance so the _try_schedule fast-path stays exact.

        Identical event semantics to v1: arrivals append to pending and trigger
        a schedule attempt only when the job could plausibly fit; finishes free
        nodes and always attempt a schedule; samples record telemetry.
        """
        for j in jobs:
            self._push_event(j.submit_time_h, "arrive", j)
        t = 0.0
        sample_job = sim.Job(-1, "_sample", 0, 0, 0, 0)
        while t <= duration_h:
            self._push_event(t, "sample", sample_job)
            t += sample_dt_h

        # Event coalescing: process ALL events sharing the same timestamp, apply
        # their state changes, then run _try_schedule at most once for that
        # timestamp. This is exact: greedy scheduling at a fixed `now` is
        # deterministic, so the union of jobs started by one combined pass over
        # the post-batch state equals the union started by per-event passes
        # (intermediate passes at the same `now` only start a subset). Same for
        # samples — but samples must reflect post-scheduling state, so we run
        # them after the schedule pass. This collapses bursts of same-time
        # finishes (very common) from N full O(n log n) passes down to 1.
        EPS = 1e-9
        while self.events:
            ev = sim.heapq.heappop(self.events)
            t_now = ev.time_h
            # Gather this timestamp's batch.
            batch = [ev]
            while self.events and abs(self.events[0].time_h - t_now) <= EPS:
                batch.append(sim.heapq.heappop(self.events))

            if t_now > duration_h:
                # Past horizon: only drain finishes for clean accounting.
                for e in batch:
                    if e.kind == "finish":
                        self.now_h = duration_h
                        self._finish(e.job)
                self._invalidate_pending_cache()
                continue

            self.now_h = t_now
            sched_needed = False
            sample_events = []
            for e in batch:
                if e.kind == "arrive":
                    self.pending.append(e.job)
                    self._note_arrival(e.job)
                    if e.job.nodes <= self.free_nodes:
                        sched_needed = True
                elif e.kind == "finish":
                    self._finish(e.job)
                    sched_needed = True
                elif e.kind == "sample":
                    sample_events.append(e)

            if sched_needed:
                self._try_schedule()

            # Samples reflect post-scheduling state at this timestamp.
            for _e in sample_events:
                busy = self.total_nodes - self.free_nodes
                self.utilization_samples.append((self.now_h, busy))
                depth = {q.name: 0 for q in sim.QUEUES}
                for j in self.pending:
                    depth[j.queue] = depth.get(j.queue, 0) + 1
                self.queue_depth_samples.append((self.now_h, depth))
                self.capacity_pool_samples.append(
                    (self.now_h, self._capacity_nodes_in_use))
                self.program_nh_samples.append(
                    (self.now_h, dict(self.delivered_nh)))


# ----------------------------------------------------------------------
# Reporting: the decision table
# ----------------------------------------------------------------------

def summarize_programs(jobs, sched: ProgramScheduler, duration_h: float,
                       machine_nodes: int) -> pd.DataFrame:
    rows = []
    started = [j for j in jobs if j.start_time_h is not None]
    total_delivered = sum(
        j.nodes * max(0.0, min(duration_h, (j.end_time_h or duration_h))
                      - max(0.0, j.start_time_h))
        for j in started) or 1.0

    progs = list(sched.profiles.keys())
    for prog in progs:
        pj_all = [j for j in jobs if getattr(j, "program", None) == prog]
        pj = [j for j in pj_all if j.start_time_h is not None]
        unstarted = len(pj_all) - len(pj)
        if pj:
            waits = np.array([j.start_time_h - j.submit_time_h for j in pj])
            nh = sum(j.nodes * max(0.0,
                     min(duration_h, (j.end_time_h or duration_h))
                     - max(0.0, j.start_time_h)) for j in pj)
        else:
            waits = np.array([0.0]); nh = 0.0
        share_delivered = nh / total_delivered
        budget = sched.budget_nh.get(prog, float("nan"))
        burn = (sched.delivered_nh.get(prog, 0.0) / budget
                if budget and budget > 0 else float("nan"))
        rows.append({
            "program": prog,
            "n_jobs": len(pj_all),
            "unstarted": unstarted,
            "target_share": round(sched.profiles[prog].annual_share, 3),
            "delivered_share": round(share_delivered, 3),
            "budget_burn": round(burn, 3),
            "wait_p50_h": round(float(np.percentile(waits, 50)), 2),
            "wait_p95_h": round(float(np.percentile(waits, 95)), 2),
            "wait_max_h": round(float(waits.max()), 2),
        })
    df = pd.DataFrame(rows)

    util = pd.DataFrame(sched.utilization_samples, columns=["t", "busy"])
    util = util[util["t"] <= duration_h]
    avg_util = (util["busy"] / machine_nodes).mean() * 100

    print(f"\n=== Program decision table  (policy={sched.policy}, "
          f"{duration_h/24:.0f}d, util={avg_util:.1f}%) ===")
    print(df.to_string(index=False))
    print(f"\n  target vs delivered share — the balancing-act result.")
    print(f"  budget_burn = delivered_nh / annual_budget "
          f"(INCITE overburn ceiling = {1+sched.overburn.get(pp.INCITE,0):.2f}x).")
    return df


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def _parse_shares(s: str) -> dict:
    vals = [float(x) for x in s.split(",")]
    if len(vals) != 4:
        raise argparse.ArgumentTypeError("shares need 4 values I,A,D,G")
    return dict(zip([pp.INCITE, pp.ALCC, pp.DD, pp.GENESIS], vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-db",
                    default=os.environ.get("PBS_SIM_DB",
                                           "/Users/jchilders/pbs_monitor_aurora.db"))
    ap.add_argument("--total-nodes", type=int, default=10624)
    ap.add_argument("--duration-days", type=float, default=365.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--policy", choices=["blind", "budget"], default="budget")
    ap.add_argument("--shares", type=_parse_shares, default="0.50,0.25,0.10,0.15",
                    help="I,A,D,G annual shares (default 0.50,0.25,0.10,0.15)")
    ap.add_argument("--no-genesis", action="store_true")
    ap.add_argument("--start-month", type=int, default=1,
                    help="Calendar month the sim clock starts on (1=Jan).")
    ap.add_argument("--capacity-pool", type=int, default=sim.CAPACITY_POOL_NODES)
    ap.add_argument("--damp-strength", type=float, default=8.0)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--outdir", default="results/programs_sim")
    args = ap.parse_args()

    shares = args.shares if isinstance(args.shares, dict) else _parse_shares(args.shares)
    duration_h = args.duration_days * 24.0
    rng = np.random.default_rng(args.seed)

    print(f"Machine: {args.total_nodes} nodes | policy={args.policy} | "
          f"shares I/A/D/G = {[shares[p] for p in pp.ALL_PROGRAMS]}")
    genesis = pp.GenesisScenario(share=shares[pp.GENESIS])
    profiles = pp.build_profiles(
        args.trace_db, shares=shares, genesis=genesis,
        machine_nodes=args.total_nodes,
        include_genesis=not args.no_genesis)
    pp.print_summary(profiles)

    gen = ProgramJobGenerator(profiles, rng, start_month=args.start_month)
    jobs = gen.generate(duration_h)
    print(f"\nGenerated {len(jobs):,} jobs over {args.duration_days:.0f} days "
          f"({len(jobs)/args.duration_days:.0f}/day)")

    sched = ProgramScheduler(
        total_nodes=args.total_nodes, enable_backfill=True,
        capacity_pool=args.capacity_pool,
        profiles=profiles, policy=args.policy,
        machine_nodes=args.total_nodes, year_h=365 * 24.0,
        damp_strength=args.damp_strength)
    sched.run(jobs, duration_h=duration_h)

    df = summarize_programs(jobs, sched, duration_h, args.total_nodes)

    if args.csv:
        d = os.path.dirname(os.path.abspath(args.csv))
        if d:
            os.makedirs(d, exist_ok=True)
        df.to_csv(args.csv, index=False)
        print(f"\nDecision table CSV → {args.csv}")


if __name__ == "__main__":
    main()
