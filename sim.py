#!/usr/bin/env python3
"""
PBS-style job queue Monte Carlo simulator.

Single-machine model with 4 queues (capacity / small / medium / large),
priority scoring, aging, EASY backfill, and a pool-cap constraint for
the capacity queue. Pure-Python discrete event simulation (no simpy dep).

Queue design
------------
  capacity  1-128 nodes, 168h cap, 512-node running pool cap
            FIFO + backfill; low flat priority + mild aging.
            AI training / long small-node jobs.
  small     129-512 nodes, 72h cap
  medium    513-2048 nodes, 48h cap
  large     2049-(10624 - on_demand) nodes, 24h cap (MTBF-limited)

Resource partitions (carved from 10,624 total)
-----------------------------------------------
  capacity pool  : CAPACITY_POOL_NODES (default 512) — shared running cap
                   across all running capacity jobs; NOT a physical partition.
                   These nodes are still visible to main scheduler when idle.
  on-demand      : ON_DEMAND_NODES (default 0, sweep via --on-demand-nodes)
                   Dedicated nodes for reservation/preemptable use; removed
                   from large-queue node ceiling entirely.

Usage:
  python sim.py --source fitted --duration-days 30 --plot
  python sim.py --source fitted --duration-days 30 --capacity-pool 256 --on-demand-nodes 512
"""
from __future__ import annotations

import argparse
import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import trace_sampler

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

@dataclass
class QueueConfig:
    name: str
    min_nodes: int
    max_nodes: int
    walltime_cap_h: float       # hard upper limit (PBS-style)
    base_priority: float        # higher = scheduled first all else equal
    aging_rate: float           # priority added per hour waiting
    # Poisson arrival rate (jobs per hour)
    arrival_rate_per_h: float
    # Log-normal parameters for sampling job size & walltime
    # We sample in log-space within the queue's [min,max] bounds.
    size_logmean_frac: float = 0.3   # mean ~ this fraction of the bucket range (log scale)
    walltime_mean_frac: float = 0.5  # mean walltime as fraction of the cap

# Arrival rates below are placeholders for synthetic mode only.
# --source fitted always overrides them with empirical rates from the DB.
#
# Scoring philosophy
# ------------------
# capacity : low flat base_priority; mild aging (FIFO + backfill does the work).
#            We do NOT give capacity higher priority than main queues — it
#            competes on FIFO+aging only. The pool cap is the protective lever.
# small    : moderate priority + aging. Should clear faster than capacity
#            long-wallers but not crowd out medium/large.
# medium   : higher priority; larger jobs deserve faster scheduling.
# large    : highest priority + steepest aging; MTBF makes every hour count.
QUEUES = [
    QueueConfig(
        name="capacity", min_nodes=1,    max_nodes=128,
        walltime_cap_h=168.0,
        base_priority=5.0,   # intentionally low — FIFO+backfill is the policy
        aging_rate=0.5,      # mild: protect long-waiting jobs without crowding main queues
        arrival_rate_per_h=25.4,  # overridden by fitted sampler
        size_logmean_frac=0.25, walltime_mean_frac=0.3,
    ),
    QueueConfig(
        name="small",    min_nodes=129,  max_nodes=512,
        walltime_cap_h=72.0,
        base_priority=20.0, aging_rate=2.0,
        arrival_rate_per_h=0.8,
        size_logmean_frac=0.3, walltime_mean_frac=0.4,
    ),
    QueueConfig(
        name="medium",   min_nodes=513,  max_nodes=2048,
        walltime_cap_h=48.0,
        base_priority=40.0, aging_rate=5.0,
        arrival_rate_per_h=0.25,
        size_logmean_frac=0.35, walltime_mean_frac=0.5,
    ),
    QueueConfig(
        name="large",    min_nodes=2049, max_nodes=10_624,
        walltime_cap_h=24.0,
        base_priority=80.0, aging_rate=10.0,
        arrival_rate_per_h=0.08,
        size_logmean_frac=0.3, walltime_mean_frac=0.6,
    ),
]

DEFAULT_TOTAL_NODES    = 10_624  # Aurora nominal node count
CAPACITY_POOL_NODES    = 512    # max nodes in use by capacity queue simultaneously
ON_DEMAND_NODES        = 0      # default on-demand partition size (sweep via CLI)

# Queue names
CAPACITY_QUEUE = "capacity"
MAIN_QUEUES    = ["small", "medium", "large"]

# ----------------------------------------------------------------------
# Job model
# ----------------------------------------------------------------------

@dataclass
class Job:
    job_id: int
    queue: str
    nodes: int
    walltime_h: float          # requested walltime
    actual_runtime_h: float    # what it actually runs for (<= walltime)
    submit_time_h: float
    start_time_h: Optional[float] = None
    end_time_h: Optional[float] = None
    base_priority: float = 0.0
    aging_rate: float = 0.0

    def score(self, now_h: float) -> float:
        wait = max(0.0, now_h - self.submit_time_h)
        return self.base_priority + self.aging_rate * wait

    @property
    def wait_time_h(self) -> float:
        if self.start_time_h is None:
            return float("nan")
        return self.start_time_h - self.submit_time_h

# ----------------------------------------------------------------------
# Job generator (synthetic Poisson)
# ----------------------------------------------------------------------

class JobGenerator:
    def __init__(self, queues: list[QueueConfig], rng: np.random.Generator,
                 sampler=None, bursty: bool = False,
                 bursty_rho: float = 0.90, bursty_cv: float = 0.74):
        """
        sampler=None          -> pure synthetic (lognormal sizes & walltimes)
        EmpiricalSampler set  -> Poisson arrivals + empirical (nodes,wt,rt) draws
        FittedSampler set     -> Poisson arrivals + fitted two-component draws

        bursty=True  -> AR(1) modulated Poisson arrivals.
          The daily arrival rate λ(t) follows an AR(1) process in log-space:
            log(λ_today) = ρ × log(λ_yesterday) + (1-ρ) × log(λ_mean) + σ × N(0,1)
          where ρ = bursty_rho (autocorrelation, default 0.90 from Aurora data)
          and σ is calibrated so the marginal CV of daily counts matches bursty_cv
          (default 0.74 from Aurora data).
        """
        self.queues = queues
        self.rng = rng
        self.sampler = sampler
        self.bursty = bursty
        self.bursty_rho = bursty_rho
        self.bursty_cv = bursty_cv
        self._next_id = 0

    def _generate_daily_multipliers(self, n_days: int) -> np.ndarray:
        """
        Generate daily rate multipliers via AR(1) in log-space.

        We calibrate σ so that the *finite-window* CV of the multipliers
        matches bursty_cv. The stationary formula σ² = Var(log) × (1-ρ²)
        underestimates variance in short windows because the AR(1) with
        high ρ hasn't explored its full range.

        Calibration (Monte Carlo, 10k trials):
          ρ=0.90, target CV=0.74, n=30 days  →  σ = 0.4791
          ρ=0.90, target CV=0.74, n=365 days →  σ ≈ 0.29 (stationary formula)

        For arbitrary (n_days, ρ, cv) we use the stationary formula as a
        starting point and apply a finite-window correction via a quick
        Monte Carlo calibration.
        """
        rho = self.bursty_rho
        cv = self.bursty_cv
        sigma = self._calibrate_sigma(n_days, rho, cv)

        # AR(1) process in log-space (centered at 0 so mean multiplier = exp(var/2))
        log_m = np.zeros(n_days)
        log_m[0] = self.rng.normal(0, np.sqrt(var_log_m))  # draw from stationary dist
        for i in range(1, n_days):
            log_m[i] = rho * log_m[i-1] + sigma * self.rng.normal()

        # Convert to multipliers, normalize so mean ≈ 1.0
        multipliers = np.exp(log_m - var_log_m / 2)  # bias correction
        multipliers *= n_days / multipliers.sum()  # ensure total count is preserved
        return multipliers

    @staticmethod
    def _calibrate_sigma(n_days: int, rho: float, target_cv: float,
                         n_trials: int = 5000) -> float:
        """Binary search for AR(1) sigma that yields target_cv in n_days."""
        cal_rng = np.random.default_rng(9999)  # fixed seed for reproducibility
        lo, hi = 0.01, 5.0
        for _ in range(40):
            sigma = (lo + hi) / 2
            var_stat = sigma**2 / (1 - rho**2)
            cvs = []
            for _ in range(n_trials):
                log_m = np.zeros(n_days)
                log_m[0] = cal_rng.normal(0, np.sqrt(var_stat))
                for i in range(1, n_days):
                    log_m[i] = rho * log_m[i-1] + sigma * cal_rng.normal()
                m = np.exp(log_m)
                m *= n_days / m.sum()
                cvs.append(np.std(m) / np.mean(m))
            if np.median(cvs) < target_cv:
                lo = sigma
            else:
                hi = sigma
        return (lo + hi) / 2

    def generate(self, duration_h: float) -> list[Job]:
        """Pre-generate all jobs that will arrive during the run window."""
        jobs: list[Job] = []

        if self.bursty:
            n_days = int(np.ceil(duration_h / 24.0))
            multipliers = self._generate_daily_multipliers(n_days)

            for qc in self.queues:
                base_rate = qc.arrival_rate_per_h  # hourly rate
                t = 0.0
                for day in range(n_days):
                    day_start = day * 24.0
                    day_end = min((day + 1) * 24.0, duration_h)
                    day_rate = base_rate * multipliers[day]
                    if day_rate <= 0:
                        continue
                    # Poisson arrivals within this day at modulated rate
                    t = day_start
                    while True:
                        dt = self.rng.exponential(1.0 / day_rate)
                        t += dt
                        if t >= day_end:
                            break
                        jobs.append(self._make_job(qc, submit_time_h=t))
        else:
            for qc in self.queues:
                # Flat Poisson arrivals: exponential inter-arrival times
                t = 0.0
                while True:
                    dt = self.rng.exponential(1.0 / qc.arrival_rate_per_h)
                    t += dt
                    if t > duration_h:
                        break
                    jobs.append(self._make_job(qc, submit_time_h=t))

        jobs.sort(key=lambda j: j.submit_time_h)
        for i, j in enumerate(jobs):
            j.job_id = i
        return jobs

    def from_replay(self, trace_df: pd.DataFrame) -> list[Job]:
        """Build job list directly from a real trace DataFrame."""
        by_q = {q.name: q for q in self.queues}
        jobs: list[Job] = []
        for i, row in trace_df.iterrows():
            qc = by_q[row["bucket"]]
            # Cap walltime/runtime at queue's documented cap to keep
            # scheduler reservation math sane (some trace walltimes are huge).
            walltime = min(float(row["walltime_h"]), qc.walltime_cap_h)
            runtime = min(float(row["runtime_h"]), walltime)
            jobs.append(Job(
                job_id=i,
                queue=qc.name,
                nodes=int(row["nodes"]),
                walltime_h=walltime,
                actual_runtime_h=runtime,
                submit_time_h=float(row["submit_h"]),
                base_priority=qc.base_priority,
                aging_rate=qc.aging_rate,
            ))
        return jobs

    def _make_job(self, qc: QueueConfig, submit_time_h: float) -> Job:
        if self.sampler is not None:
            # FittedSampler uses different signature (no rng arg — owns its rng)
            if isinstance(self.sampler, trace_sampler.FittedSampler):
                nodes, walltime, runtime = self.sampler.sample(qc.name)
            else:
                nodes, walltime, runtime = self.sampler.sample(qc.name, self.rng)
            # Cap to queue's documented walltime
            walltime = min(walltime, qc.walltime_cap_h)
            runtime = min(runtime, walltime)
            nodes = max(qc.min_nodes, min(qc.max_nodes, nodes))
            return Job(
                job_id=self._next_id, queue=qc.name, nodes=nodes,
                walltime_h=walltime, actual_runtime_h=runtime,
                submit_time_h=submit_time_h,
                base_priority=qc.base_priority, aging_rate=qc.aging_rate,
            )

        # --- Pure synthetic fallback ---
        lo = math.log(qc.min_nodes)
        hi = math.log(qc.max_nodes)
        u = self.rng.beta(1.5, 3.0)
        nodes = int(round(math.exp(lo + (hi - lo) * u)))
        nodes = max(qc.min_nodes, min(qc.max_nodes, nodes))

        mean_req = qc.walltime_cap_h * qc.walltime_mean_frac
        sigma = 0.8
        mu = math.log(max(0.5, mean_req)) - 0.5 * sigma**2
        requested = float(np.clip(self.rng.lognormal(mu, sigma), 0.5, qc.walltime_cap_h))

        actual = requested * float(self.rng.beta(3.0, 2.0))
        actual = max(0.05, min(actual, requested))

        return Job(
            job_id=self._next_id,
            queue=qc.name, nodes=nodes,
            walltime_h=requested, actual_runtime_h=actual,
            submit_time_h=submit_time_h,
            base_priority=qc.base_priority, aging_rate=qc.aging_rate,
        )

# ----------------------------------------------------------------------
# Scheduler
# ----------------------------------------------------------------------

@dataclass(order=True)
class _Event:
    time_h: float
    seq: int                                  # tiebreaker
    kind: str = field(compare=False)          # "arrive" | "finish"
    job: Job = field(compare=False)

class Scheduler:
    def __init__(self, total_nodes: int, enable_backfill: bool = True,
                 capacity_pool: int = CAPACITY_POOL_NODES,
                 on_demand_nodes: int = 0):
        """
        total_nodes     : total physical nodes on the machine.
        capacity_pool   : max nodes that capacity-queue jobs may occupy
                          simultaneously (running jobs only). These nodes are
                          still in the shared pool — the constraint is a
                          running-job cap, not a physical partition.
        on_demand_nodes : nodes removed from the large-queue ceiling for the
                          on-demand/preemptable partition. Reduces effective
                          large-queue max_nodes.
        """
        self.total_nodes    = total_nodes
        self.capacity_pool  = capacity_pool
        self.on_demand_nodes = on_demand_nodes
        self.enable_backfill = enable_backfill
        self.free_nodes     = total_nodes
        self.now_h          = 0.0
        self.pending: list[Job] = []           # waiting jobs
        self.running: list[Job] = []           # currently executing
        self.events: list[_Event] = []
        self._seq = 0
        # Running node-count for capacity queue (pool cap enforcement)
        self._capacity_nodes_in_use: int = 0
        # Telemetry
        self.utilization_samples: list[tuple[float, int]] = []   # (t, busy_nodes)
        self.queue_depth_samples: list[tuple[float, dict[str, int]]] = []
        self.capacity_pool_samples: list[tuple[float, int]] = []  # (t, nodes_in_use)
        # Starvation tracking: max observed wait per queue
        self.max_wait_h: dict[str, float] = {q.name: 0.0 for q in QUEUES}

    def _push_event(self, time_h: float, kind: str, job: Job):
        self._seq += 1
        heapq.heappush(self.events, _Event(time_h, self._seq, kind, job))

    def run(self, jobs: list[Job], duration_h: float, sample_dt_h: float = 0.25):
        # Seed arrival events
        for j in jobs:
            self._push_event(j.submit_time_h, "arrive", j)
        # Seed periodic sampling events
        t = 0.0
        sample_job = Job(-1, "_sample", 0, 0, 0, 0)
        while t <= duration_h:
            self._push_event(t, "sample", sample_job)
            t += sample_dt_h

        while self.events:
            ev = heapq.heappop(self.events)
            # Skip arrivals/samples past the horizon. Drain finish events
            # for jobs already started, so utilization accounting is clean.
            if ev.time_h > duration_h:
                if ev.kind == "finish":
                    # Cap finish at horizon for accounting; job effectively ran
                    # only up to the window.
                    self.now_h = duration_h
                    self._finish(ev.job)
                continue
            self.now_h = ev.time_h

            if ev.kind == "arrive":
                self.pending.append(ev.job)
                # Fast-path: only re-schedule if the new arrival could
                # plausibly start now (fits in free nodes). Otherwise the
                # existing schedule is unchanged — no need to re-sort
                # potentially-thousands of pending jobs.
                if ev.job.nodes <= self.free_nodes:
                    self._try_schedule()
            elif ev.kind == "finish":
                self._finish(ev.job)
                self._try_schedule()
            elif ev.kind == "sample":
                busy = self.total_nodes - self.free_nodes
                self.utilization_samples.append((self.now_h, busy))
                depth = {q.name: 0 for q in QUEUES}
                for j in self.pending:
                    depth[j.queue] = depth.get(j.queue, 0) + 1
                self.queue_depth_samples.append((self.now_h, depth))
                self.capacity_pool_samples.append((self.now_h, self._capacity_nodes_in_use))

    # ------------------------------------------------------------------

    def _capacity_pool_free(self) -> int:
        """Remaining capacity-pool headroom (nodes)."""
        return max(0, self.capacity_pool - self._capacity_nodes_in_use)

    def _can_start(self, job: Job) -> bool:
        """True if `job` can start right now given node and pool constraints."""
        if job.nodes > self.free_nodes:
            return False
        if job.queue == CAPACITY_QUEUE:
            if job.nodes > self._capacity_pool_free():
                return False
        return True

    def _try_schedule(self):
        """
        Priority scheduling with EASY backfill, plus capacity pool-cap enforcement.

        Scoring notes
        -------------
        Main queues (small/medium/large) use higher base_priority so they are
        always preferred over capacity jobs when competing for the same nodes.
        Within the capacity queue, FIFO order is preserved via submit-time
        tiebreaking; the mild aging_rate just prevents exact-time-tie starvation.
        Backfill then lets short capacity jobs fill gaps without violating the
        reservation of the head-of-queue job.
        """
        if not self.pending:
            return

        now = self.now_h
        # Sort by score descending; use submit_time as tiebreaker for FIFO
        order = sorted(
            range(len(self.pending)),
            key=lambda i: (-self.pending[i].score(now),
                           self.pending[i].submit_time_h),
        )

        reservation_time: Optional[float] = None
        reserved_idx: Optional[int] = None
        removed: set[int] = set()

        # First pass: greedy start in priority order
        for idx in order:
            j = self.pending[idx]
            if self._can_start(j):
                self._start_at_index(j, idx, removed)
            elif reserved_idx is None:
                # First job we couldn't start becomes the reservation holder
                reserved_idx = idx
                reservation_time = self._estimate_reservation(j)

        if reservation_time is not None and self.enable_backfill:
            # Backfill: any lower-priority job that fits AND finishes before
            # the reservation time may run now.
            for idx in order:
                if idx in removed or idx == reserved_idx:
                    continue
                j = self.pending[idx]
                if not self._can_start(j):
                    continue
                projected_end = now + j.walltime_h
                if projected_end <= reservation_time + 1e-9:
                    self._start_at_index(j, idx, removed)

        # Compact pending list (remove started jobs)
        if removed:
            self.pending = [j for i, j in enumerate(self.pending)
                            if i not in removed]

    def _start_at_index(self, job: Job, idx: int, removed: set[int]):
        """Start a job that's at known position `idx` in self.pending."""
        if not self._can_start(job):
            return
        assert self.pending[idx] is job
        removed.add(idx)
        self.running.append(job)
        self.free_nodes -= job.nodes
        if job.queue == CAPACITY_QUEUE:
            self._capacity_nodes_in_use += job.nodes
        assert self.free_nodes >= 0
        job.start_time_h = self.now_h
        job.end_time_h = self.now_h + job.actual_runtime_h
        self._start_job(job)
        self._push_event(job.end_time_h, "finish", job)

    def _estimate_reservation(self, job: Job) -> float:
        """
        Earliest time `job` can start, accounting for both the global free-node
        count and (for capacity jobs) the pool cap.
        Uses requested walltime as a conservative proxy (scheduler can't see actual).
        """
        ends = sorted(
            [(r.start_time_h + r.walltime_h, r.nodes, r.queue)
             for r in self.running]
        )

        free_now = self.free_nodes
        pool_used = self._capacity_nodes_in_use

        if job.queue == CAPACITY_QUEUE:
            # Need both a free node slot AND pool headroom
            for end_t, nodes, q in ends:
                node_ok  = free_now  >= job.nodes
                pool_ok  = (self.capacity_pool - pool_used) >= job.nodes
                if node_ok and pool_ok:
                    return end_t  # could have started just before this release
                free_now += nodes
                if q == CAPACITY_QUEUE:
                    pool_used -= nodes
            return self.now_h + job.walltime_h
        else:
            if free_now >= job.nodes:
                return self.now_h
            for end_t, nodes, _ in ends:
                free_now += nodes
                if free_now >= job.nodes:
                    return end_t
            return self.now_h + job.walltime_h

    def _start(self, job: Job):
        if job.nodes > self.free_nodes:
            return
        assert job in self.pending, f"job {job.job_id} not in pending"
        self.pending.remove(job)
        self.running.append(job)
        self.free_nodes -= job.nodes
        assert self.free_nodes >= 0, (
            f"overcommit: free={self.free_nodes} after starting job {job.job_id} "
            f"nodes={job.nodes}")
        job.start_time_h = self.now_h
        job.end_time_h = self.now_h + job.actual_runtime_h
        self._push_event(job.end_time_h, "finish", job)

    def _start_job(self, job: Job):
        """Record start time and update starvation tracker."""
        wait = job.start_time_h - job.submit_time_h
        if job.queue in self.max_wait_h:
            self.max_wait_h[job.queue] = max(self.max_wait_h[job.queue], wait)

    def _finish(self, job: Job):
        if job in self.running:
            self.running.remove(job)
            self.free_nodes += job.nodes
            if job.queue == CAPACITY_QUEUE:
                self._capacity_nodes_in_use -= job.nodes
                self._capacity_nodes_in_use = max(0, self._capacity_nodes_in_use)

# ----------------------------------------------------------------------
# Analysis / plotting
# ----------------------------------------------------------------------

def summarize(jobs: list[Job], total_nodes: int, duration_h: float,
              sched: Scheduler, capacity_nodes: int = 0) -> pd.DataFrame:
    rows = []
    for j in jobs:
        if j.start_time_h is None:
            continue
        # Bound runtime accounting to the simulation window.
        eff_start = max(0.0, j.start_time_h)
        eff_end = min(duration_h, j.end_time_h if j.end_time_h is not None else duration_h)
        eff_rt = max(0.0, eff_end - eff_start)
        rows.append({
            "job_id": j.job_id,
            "queue": j.queue,
            "nodes": j.nodes,
            "walltime_h": j.walltime_h,
            "runtime_h": j.actual_runtime_h,
            "runtime_in_window_h": eff_rt,
            "submit_h": j.submit_time_h,
            "start_h": j.start_time_h,
            "end_h": j.end_time_h,
            "wait_h": j.wait_time_h,
            "started_in_window": j.start_time_h <= duration_h,
            "completed_in_window": (j.end_time_h is not None
                                    and j.end_time_h <= duration_h),
        })
    df = pd.DataFrame(rows)

    machine_nodes = total_nodes + capacity_nodes
    print(f"\n=== Simulation summary ===")
    print(f"Total nodes:     {machine_nodes} ({total_nodes} main + {capacity_nodes} capacity partition)")
    print(f"Duration:        {duration_h:.1f} h ({duration_h/24:.2f} d)")
    print(f"Jobs submitted:    {len(jobs)}")
    print(f"Jobs that started: {len(df)}")
    print(f"Jobs unstarted:    {len(jobs) - len(df)} (still queued at horizon)")
    if not df.empty:
        n_completed = int(df["completed_in_window"].sum())
        print(f"Jobs completed:    {n_completed} (finished within the {duration_h:.0f}h window)")

    if df.empty:
        return df

    # Wait time stats: only jobs that actually started during the window.
    started = df[df["started_in_window"]]
    print("\n--- Per-queue wait time (hours, only jobs started in window) ---")
    by_q = started.groupby("queue")["wait_h"].agg(
        ["count", "mean", "median",
         lambda s: s.quantile(0.95), "max"]
    )
    by_q.columns = ["count", "mean", "median", "p95", "max"]
    print(by_q.round(2).to_string())

    # Starvation report: max wait vs walltime cap
    print("\n--- Starvation check (max wait vs walltime cap) ---")
    from trace_sampler import NEW_CAPS_H
    starvation_found = False
    for qname, max_w in sched.max_wait_h.items():
        cap = NEW_CAPS_H.get(qname, float("nan"))
        ratio = max_w / cap if cap > 0 else float("nan")
        flag = " ⚠️  STARVATION" if ratio > 1.0 else ""
        print(f"  {qname:8s}  max_wait={max_w:7.1f}h  cap={cap:5.0f}h  "
              f"ratio={ratio:5.2f}x{flag}")
        if ratio > 1.0:
            starvation_found = True
    if not starvation_found:
        print("  No starvation detected (max wait < walltime cap for all queues)")

    # Utilization (sampled within window only)
    util = pd.DataFrame(sched.utilization_samples, columns=["t", "busy"])
    util = util[util["t"] <= duration_h]
    util["frac"] = util["busy"] / total_nodes
    avg_util = util["frac"].mean()
    print(f"\nAverage utilization:  {avg_util*100:.1f}%")

    # Per-queue node-hours breakdown
    print("\n--- Node-hours delivered per queue ---")
    for qname in ["capacity", "small", "medium", "large"]:
        sub = df[df["queue"] == qname]
        nh = (sub["nodes"] * sub["runtime_in_window_h"]).sum()
        print(f"  {qname:10s}  {nh:>12,.0f} node-hours")

    # Throughput: node-hours delivered WITHIN the window.
    node_hours = (df["nodes"] * df["runtime_in_window_h"]).sum()
    capacity_nh = total_nodes * duration_h
    print(f"\nNode-hours delivered: {node_hours:,.0f} / {capacity_nh:,.0f} capacity "
          f"({100*node_hours/capacity_nh:.1f}%)")
    print(f"Throughput:           {n_completed/duration_h*24:.1f} completed jobs/day")
    print(f"                      {node_hours/duration_h*24:,.0f} node-hours/day")
    print(f"Theoretical max:      {total_nodes*24:,.0f} node-hours/day")

    # Capacity pool peak usage
    cap_running = df[df["queue"] == "capacity"]
    if not cap_running.empty:
        print(f"\nCapacity pool: peak concurrent nodes not directly tracked in df "
              f"(see scheduler telemetry); pool cap = {sched.capacity_pool}")

    return df


def plot(df: pd.DataFrame, sched: Scheduler, total_nodes: int, outdir: str):
    import matplotlib.pyplot as plt
    import os
    os.makedirs(outdir, exist_ok=True)

    # 1) Wait time CDF per queue
    fig, ax = plt.subplots(figsize=(8, 5))
    for q in ["capacity", "small", "medium", "large"]:
        sub = df[df["queue"] == q]["wait_h"].sort_values().to_numpy()
        if len(sub) == 0:
            continue
        cdf = np.arange(1, len(sub) + 1) / len(sub)
        ax.plot(sub, cdf, label=f"{q} (n={len(sub)})")
    ax.set_xlabel("Wait time (hours)")
    ax.set_ylabel("CDF")
    ax.set_title("Wait time CDF by queue")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{outdir}/wait_cdf.png", dpi=120)
    plt.close(fig)

    # 2) Utilization timeline
    util = pd.DataFrame(sched.utilization_samples, columns=["t", "busy"])
    util["frac"] = util["busy"] / total_nodes
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(util["t"] / 24.0, util["frac"] * 100)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Utilization (%)")
    ax.set_title(f"Node utilization (avg = {util['frac'].mean()*100:.1f}%)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{outdir}/utilization.png", dpi=120)
    plt.close(fig)

    # 3) Queue depth over time
    depth_rows = []
    for t, depth in sched.queue_depth_samples:
        for q, n in depth.items():
            depth_rows.append({"t": t, "queue": q, "depth": n})
    ddf = pd.DataFrame(depth_rows)
    fig, ax = plt.subplots(figsize=(10, 4))
    for q in ["capacity", "small", "medium", "large"]:
        sub = ddf[ddf["queue"] == q]
        ax.plot(sub["t"] / 24.0, sub["depth"], label=q)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Pending jobs")
    ax.set_title("Queue depth over time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{outdir}/queue_depth.png", dpi=120)
    plt.close(fig)

    # 4) Capacity pool utilization over time
    if sched.capacity_pool_samples:
        pool_df = pd.DataFrame(sched.capacity_pool_samples, columns=["t", "in_use"])
        pool_df = pool_df[pool_df["t"] <= pool_df["t"].max()]
        pool_frac = pool_df["in_use"] / sched.capacity_pool
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(pool_df["t"] / 24.0, pool_frac * 100, color="steelblue")
        ax.axhline(100, color="red", linestyle="--", linewidth=0.8, label=f"Pool cap ({sched.capacity_pool} nodes)")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Capacity pool in use (%)")
        ax.set_title(f"Capacity pool utilization (avg = {pool_frac.mean()*100:.1f}%, "
                     f"peak = {pool_frac.max()*100:.1f}%)")
        ax.set_ylim(0, 110)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{outdir}/capacity_pool_utilization.png", dpi=120)
        plt.close(fig)

    print(f"\nPlots written to {outdir}/")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-nodes", type=int, default=DEFAULT_TOTAL_NODES)
    ap.add_argument("--duration-days", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-backfill", action="store_true",
                    help="Disable EASY backfill for comparison runs.")
    ap.add_argument("--source", choices=["synthetic", "empirical", "fitted", "replay"],
                    default="synthetic",
                    help="Job source. synthetic=Poisson+lognormal; "
                         "empirical=Poisson arrivals + real (nodes,wt,rt) draws (no rescaling); "
                         "fitted=Poisson arrivals + two-component walltime model "
                         "(recommended for policy exploration); "
                         "replay=real submit times verbatim.")
    ap.add_argument("--trace-db", default="/Users/jchilders/pbs_monitor_aurora.db",
                    help="Path to pbs_monitor sqlite (for empirical/replay).")
    ap.add_argument("--use-real-arrival-rates", action="store_true",
                    help="For --source empirical: override QUEUES arrival rates "
                         "with empirical per-bucket Poisson rates from trace. "
                         "Always on for --source fitted.")
    ap.add_argument("--capacity-pool", type=int, default=CAPACITY_POOL_NODES,
                    help=f"Max nodes capacity queue may use simultaneously "
                         f"(running jobs only). Default {CAPACITY_POOL_NODES}.")
    ap.add_argument("--on-demand-nodes", type=int, default=ON_DEMAND_NODES,
                    help="Nodes reserved for on-demand/preemptable partition; "
                         "subtracted from large-queue node ceiling. Default 0.")
    ap.add_argument("--replay-start-day", type=float, default=0.0,
                    help="For --source replay: offset into trace (days) to begin.")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--csv", default=None, help="Write completed-jobs CSV to this path.")
    ap.add_argument("--arrival-rate-scale", type=float, default=1.0,
                    help="Scale fitted arrival rates by this factor (capacity queue only). "
                         "1.0 = empirical, 0.5 = half, 0.1 = 10%%. Used for sensitivity sweeps.")
    ap.add_argument("--bursty", action="store_true",
                    help="Use AR(1) modulated Poisson arrivals instead of flat Poisson. "
                         "Calibrated from Aurora empirical data: rho=0.90, CV=0.74.")
    ap.add_argument("--bursty-rho", type=float, default=0.90,
                    help="AR(1) autocorrelation parameter (default 0.90 from Aurora data).")
    ap.add_argument("--bursty-cv", type=float, default=0.74,
                    help="Target CV for daily rate multipliers (default 0.74 from Aurora data).")
    args = ap.parse_args()

    duration_h = args.duration_days * 24.0
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    on_demand = args.on_demand_nodes
    cap_pool  = args.capacity_pool
    # Adjust large queue ceiling for on-demand partition
    for qc in QUEUES:
        if qc.name == "large" and on_demand > 0:
            qc.max_nodes = max(qc.min_nodes, args.total_nodes - on_demand)
    print(f"Machine: {args.total_nodes} total nodes")
    print(f"  Capacity pool cap : {cap_pool} nodes (running cap, not physical partition)")
    print(f"  On-demand partition: {on_demand} nodes (removed from large ceiling)")
    print(f"  Large queue ceiling: {next(q.max_nodes for q in QUEUES if q.name=='large')} nodes")

    if args.source == "synthetic":
        gen = JobGenerator(QUEUES, rng)
        jobs = gen.generate(duration_h)
    elif args.source == "fitted":
        fitted = trace_sampler.FittedSampler(
            db_path=args.trace_db, rng=rng,
            use_cache=not getattr(args, 'no_cache', False),
        )
        fitted.print_fit_summary()
        # Override QUEUES arrival rates with empirical values
        for qc in QUEUES:
            rate = fitted.arrival_rate(qc.name)
            if qc.name == CAPACITY_QUEUE and args.arrival_rate_scale != 1.0:
                rate *= args.arrival_rate_scale
                print(f"  [sweep] capacity arrival_rate scaled {args.arrival_rate_scale:.0%}: "
                      f"{rate:.4f}/h ({rate*24:.1f}/day)")
            qc.arrival_rate_per_h = rate
        bursty_label = ""
        if args.bursty:
            bursty_label = f" [BURSTY: \u03c1={args.bursty_rho}, CV={args.bursty_cv}]"
        print(f"\nGenerating {args.duration_days:.1f} days of fitted jobs...{bursty_label}")
        gen = JobGenerator(QUEUES, rng, sampler=fitted,
                           bursty=args.bursty,
                           bursty_rho=args.bursty_rho,
                           bursty_cv=args.bursty_cv)
        jobs = gen.generate(duration_h)
    elif args.source == "empirical":
        trace = trace_sampler.load_trace(args.trace_db)
        print(f"Loaded trace: {len(trace):,} jobs, "
              f"{trace['submit_h'].max()/24:.1f} days")
        if args.use_real_arrival_rates:
            span_h = trace["submit_h"].max() - trace["submit_h"].min()
            for qc in QUEUES:
                n = (trace["bucket"] == qc.name).sum()
                if n > 0:
                    qc.arrival_rate_per_h = n / span_h
                    print(f"  override {qc.name} arrival_rate = "
                          f"{qc.arrival_rate_per_h:.3f}/h")
        sampler = trace_sampler.EmpiricalSampler(trace=trace)
        gen = JobGenerator(QUEUES, rng, sampler=sampler)
        jobs = gen.generate(duration_h)
    elif args.source == "replay":
        trace = trace_sampler.load_trace(args.trace_db)
        start_h_in_trace = args.replay_start_day * 24.0
        # Slice the trace window first (in trace coords), then re-anchor to 0.
        window = trace[(trace["submit_h"] >= start_h_in_trace) &
                       (trace["submit_h"] <  start_h_in_trace + duration_h)].copy()
        window["submit_h"] = window["submit_h"] - start_h_in_trace
        window = window.reset_index(drop=True)
        print(f"Replay: {len(window):,} real jobs from trace day "
              f"{args.replay_start_day:.1f} → day {args.replay_start_day + args.duration_days:.1f}")
        gen = JobGenerator(QUEUES, rng)
        jobs = gen.from_replay(window)
    else:
        raise ValueError(args.source)

    print(f"\nGenerated {len(jobs)} jobs over {args.duration_days:.1f} days "
          f"({len(jobs)/args.duration_days:.1f} jobs/day)")

    sched = Scheduler(total_nodes=args.total_nodes,
                      enable_backfill=not args.no_backfill,
                      capacity_pool=cap_pool,
                      on_demand_nodes=on_demand)
    sched.run(jobs, duration_h=duration_h)
    df = summarize(jobs, args.total_nodes, duration_h, sched,
                   capacity_nodes=on_demand)

    if args.csv:
        import os
        csv_dir = os.path.dirname(os.path.abspath(args.csv))
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        df.to_csv(args.csv, index=False)
        print(f"\nCompleted-jobs CSV → {args.csv}")
    if args.plot and not df.empty:
        plot(df, sched, args.total_nodes, args.outdir)


if __name__ == "__main__":
    main()
