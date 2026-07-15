"""
Discrete-event scheduler with:
  - configurable score expression (AST-safe, from YAML)
  - pluggable capacity-protection strategies (the 512-cap is now ONE option)
  - per-program budget accounting + budget_damp variable exposed to the score
  - EASY backfill
  - Delta-t as MEASUREMENT granularity only (never gates scheduling)
  - a saturation guard so an oversubscribed run reports instead of hanging

Everything is driven by SimConfig. No hard-coded policy constants.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import SimConfig
from generator import Job
from score_expr import compile_expr


# Variables available to a score expression, per (job, sim-state).
SCORE_VARS = {
    "base",            # size-tier base priority
    "wait",            # hours waited so far
    "nodes",           # job node count
    "walltime",        # requested walltime (h)
    "aging_rate",      # size-tier aging rate
    "budget_ratio",    # delivered / pro-rated budget for the program
    "budget_damp",     # smooth damping factor in (0,1]
    "project_damp",    # per-project damping in (0,1] (1.0 if project layer off)
    "delivered_share", # program's delivered share so far
    "target_share",    # program's target share
    "now",             # sim time (h)
    "queue_depth",     # current pending count
    "free_nodes",      # currently free nodes
}


@dataclass(order=True)
class _Event:
    time_h: float
    seq: int
    kind: str = field(compare=False)   # arrive | finish | sample
    job: Optional[Job] = field(compare=False, default=None)


class SaturationError(RuntimeError):
    """Raised when pending grows past the configured guard threshold."""


class Scheduler:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.total_nodes = cfg.machine.total_nodes
        self.free_nodes = self.total_nodes
        self.enable_backfill = cfg.scheduler.enable_backfill
        self.now_h = 0.0
        self.pending: list[Job] = []
        self.running: list[Job] = []
        self.events: list[_Event] = []
        self._seq = 0

        # --- capacity protection strategy ---
        cp = cfg.capacity_protection
        self.cp_strategy = cp.strategy
        self.cp_pool_nodes = cp.pool_nodes
        self.cp_protected_tier = cp.protected_tier
        self.cp_partition_nodes = cp.partition_nodes
        self.cp_big_min = cp.big_job_min_nodes
        self._protected_nodes_in_use = 0
        # dedicated_partition reserves nodes for big jobs => small jobs see fewer
        self._small_job_ceiling = self.total_nodes - (
            cp.partition_nodes if cp.strategy == "dedicated_partition" else 0)

        # --- programs / budgets ---
        from genesis import reallocate_shares
        self.shares = reallocate_shares(cfg.programs, cfg.genesis)
        self.overburn = {p.name: p.overburn for p in cfg.programs}
        self.soft_floor = {p.name: p.soft_floor for p in cfg.programs}
        if cfg.genesis.enabled:
            self.overburn["Genesis"] = 0.0
            self.soft_floor["Genesis"] = False
        year_h = 365 * 24.0
        machine_year_nh = self.total_nodes * year_h
        self.year_h = year_h
        self.budget_nh = {p: self.shares.get(p, 0.0) * machine_year_nh
                          for p in self.shares}
        self.delivered_nh = {p: 0.0 for p in self.shares}
        self._ceiling_nh = {
            p: self.budget_nh[p] * (1.0 + self.overburn.get(p, 0.0))
            for p in self.budget_nh if not self.soft_floor.get(p, False)}

        # --- score ---
        self.policy = cfg.scheduler.policy
        self.damp_strength = cfg.scheduler.budget_damp_strength
        self._score_fn = compile_expr(cfg.scheduler.score_expr, SCORE_VARS)

        # --- optional project layer: per-project awards + delivered accounting ---
        self.projects_enabled = cfg.projects.enabled
        self.project_award_nh: dict = {}
        self.project_delivered_nh: dict = {}
        self.project_damp_strength = cfg.projects.project_damp_strength
        # populated by attach_projects() before run (needs the generator's awards)

        # --- telemetry ---
        self.util_samples: list[tuple[float, int]] = []
        self.program_nh_samples: list[tuple[float, dict]] = []
        self.queue_depth_samples: list[tuple[float, dict]] = []
        self._sat_limit = cfg.output.saturation_abort_pending
        self._min_pending_cache: Optional[int] = None

    # -- events ---------------------------------------------------------

    def _push(self, time_h: float, kind: str, job: Optional[Job]):
        self._seq += 1
        heapq.heappush(self.events, _Event(time_h, self._seq, kind, job))

    # -- budget helpers -------------------------------------------------

    def _prorated_budget(self, prog: str) -> float:
        frac = min(1.0, self.now_h / self.year_h) if self.year_h > 0 else 1.0
        return self.budget_nh.get(prog, float("inf")) * frac

    def _budget_ratio(self, prog: str) -> float:
        pb = self._prorated_budget(prog)
        if pb <= 0:
            return 0.0
        return self.delivered_nh.get(prog, 0.0) / pb

    def _damp_factor(self, prog: str) -> float:
        if self.policy != "budget":
            return 1.0
        r = self._budget_ratio(prog)
        if r <= 1.0:
            return 1.0
        return math.exp(-self.damp_strength * (r - 1.0))

    def _over_ceiling(self, job: Job) -> bool:
        if self.policy != "budget":
            return False
        ceiling = self._ceiling_nh.get(job.program)
        if ceiling is None:
            return False
        return (self.delivered_nh.get(job.program, 0.0)
                + job.nodes * job.walltime_h) > ceiling

    # -- capacity protection --------------------------------------------

    def _is_protected(self, job: Job) -> bool:
        return job.size_tier == self.cp_protected_tier

    def _can_start(self, job: Job) -> bool:
        strat = self.cp_strategy
        if strat == "dedicated_partition":
            # small (non-big) jobs limited to total - partition; big jobs full.
            if job.nodes < self.cp_big_min:
                busy_small = self.total_nodes - self.free_nodes
                if (busy_small + job.nodes) > self._small_job_ceiling:
                    return False
            if job.nodes > self.free_nodes:
                return False
        else:
            if job.nodes > self.free_nodes:
                return False
            if strat == "running_pool_cap" and self._is_protected(job):
                if (self._protected_nodes_in_use + job.nodes) > self.cp_pool_nodes:
                    return False
        if self._over_ceiling(job):
            return False
        return True

    # -- scoring --------------------------------------------------------

    def _score(self, job: Job, prog_cache: dict, proj_cache: dict) -> float:
        """Score one job. Per-program values (budget_ratio, budget_damp,
        delivered_share, target_share) are computed ONCE per program per pass
        and cached; only per-job fields vary. The score expression is compiled
        bytecode, so this is a single fast eval."""
        prog = job.program
        pc = prog_cache.get(prog)
        if pc is None:
            damp = self._damp_factor(prog)
            ratio = self._budget_ratio(prog)
            pc = {
                "budget_ratio": ratio,
                "budget_damp": damp,
                "delivered_share": self._delivered_share(prog),
                "target_share": self.shares.get(prog, 0.0),
                "now": self.now_h,
                "queue_depth": len(self.pending),
                "free_nodes": self.free_nodes,
            }
            prog_cache[prog] = pc
        ns = {
            "base": job.base_priority,
            "wait": self.now_h - job.submit_time_h if self.now_h > job.submit_time_h else 0.0,
            "nodes": job.nodes,
            "walltime": job.walltime_h,
            "aging_rate": job.aging_rate,
            "project_damp": self._project_damp(job, proj_cache),
            **pc,
        }
        return float(self._score_fn(ns))

    def _project_damp(self, job: Job, proj_cache: dict) -> float:
        """Per-project damping in (0,1]: 1.0 while under the project's pro-rated
        award, decaying as delivered exceeds it. 1.0 when the project layer is
        off or the project has no award. Cached per project per pass."""
        if not self.projects_enabled or self.project_damp_strength <= 0:
            return 1.0
        proj = job.project
        cached = proj_cache.get(proj)
        if cached is not None:
            return cached
        award = self.project_award_nh.get(proj)
        if not award or award <= 0:
            proj_cache[proj] = 1.0
            return 1.0
        frac = min(1.0, self.now_h / self.year_h) if self.year_h > 0 else 1.0
        prorated = award * frac
        ratio = self.project_delivered_nh.get(proj, 0.0) / prorated if prorated > 0 else 0.0
        d = 1.0 if ratio <= 1.0 else math.exp(-self.project_damp_strength * (ratio - 1.0))
        proj_cache[proj] = d
        return d

    def attach_projects(self, projects_by_prog: dict) -> None:
        """Register per-project awards (from the generator's ProjectSamplers)
        so the scheduler can do per-project budget accounting + damping."""
        for lst in projects_by_prog.values():
            for ps in lst:
                self.project_award_nh[ps.project] = ps.award_nh
                self.project_delivered_nh[ps.project] = 0.0

    def _delivered_share(self, prog: str) -> float:
        tot = sum(self.delivered_nh.values()) or 1.0
        return self.delivered_nh.get(prog, 0.0) / tot

    # -- min-pending guard ----------------------------------------------

    def _min_pending_nodes(self) -> int:
        if self._min_pending_cache is None:
            self._min_pending_cache = min((j.nodes for j in self.pending),
                                          default=1 << 30)
        return self._min_pending_cache

    # -- scheduling -----------------------------------------------------

    def _try_schedule(self):
        if not self.pending:
            return
        if self.free_nodes < self._min_pending_nodes():
            return
        prog_cache: dict = {}
        proj_cache: dict = {}
        keyed = [(-self._score(j, prog_cache, proj_cache), j.submit_time_h, i)
                 for i, j in enumerate(self.pending)]
        keyed.sort()
        order = [k[2] for k in keyed]

        reservation_time: Optional[float] = None
        reserved_idx: Optional[int] = None
        removed: set = set()

        for idx in order:
            j = self.pending[idx]
            if self._can_start(j):
                self._start(j, idx, removed)
            elif reserved_idx is None and not self._over_ceiling(j):
                reserved_idx = idx
                reservation_time = self._estimate_reservation(j)

        if reservation_time is not None and self.enable_backfill:
            for idx in order:
                if idx in removed or idx == reserved_idx:
                    continue
                j = self.pending[idx]
                if not self._can_start(j):
                    continue
                if self.now_h + j.walltime_h <= reservation_time + 1e-9:
                    self._start(j, idx, removed)

        if removed:
            self.pending = [j for i, j in enumerate(self.pending)
                            if i not in removed]
            self._min_pending_cache = None

    def _start(self, job: Job, idx: int, removed: set):
        removed.add(idx)
        self.running.append(job)
        self.free_nodes -= job.nodes
        if self._is_protected(job):
            self._protected_nodes_in_use += job.nodes
        job.start_time_h = self.now_h
        job.end_time_h = self.now_h + job.actual_runtime_h
        if job.program in self.delivered_nh:
            self.delivered_nh[job.program] += job.nodes * job.actual_runtime_h
        if self.projects_enabled and job.project in self.project_delivered_nh:
            self.project_delivered_nh[job.project] += job.nodes * job.actual_runtime_h
        self._push(job.end_time_h, "finish", job)

    def _finish(self, job: Job):
        if job in self.running:
            self.running.remove(job)
            self.free_nodes += job.nodes
            if self._is_protected(job):
                self._protected_nodes_in_use = max(
                    0, self._protected_nodes_in_use - job.nodes)

    def _estimate_reservation(self, job: Job) -> float:
        ends = sorted((r.start_time_h + r.walltime_h, r.nodes)
                      for r in self.running)
        free = self.free_nodes
        if free >= job.nodes:
            return self.now_h
        for end_t, nodes in ends:
            free += nodes
            if free >= job.nodes:
                return end_t
        return self.now_h + job.walltime_h

    # -- run loop -------------------------------------------------------

    def _preflight_check(self, jobs: list[Job], duration_h: float):
        """Cheap up-front oversubscription detection.

        For a running_pool_cap strategy, the protected tier can deliver at most
        pool_nodes * duration_h node-hours. If the offered protected-tier load
        exceeds that (with a small tolerance), the backlog is guaranteed to grow
        without bound — so we abort immediately with an actionable message
        instead of discovering it via a quadratic slowdown mid-run.

        Also checks total offered load vs whole-machine capacity.
        """
        if duration_h <= 0:
            return
        cp = self.cfg.capacity_protection
        # total offered
        total_offered = sum(j.nodes * j.actual_runtime_h for j in jobs)
        machine_cap = self.total_nodes * duration_h
        if total_offered > machine_cap * 1.05:
            # not necessarily fatal (jobs can spill past horizon) but warn
            print(f"  [preflight] NOTE offered load {total_offered/machine_cap:.2f}x "
                  f"whole-machine capacity over the window; expect a growing queue.")
        if cp.strategy == "running_pool_cap":
            prot = [j for j in jobs if j.size_tier == cp.protected_tier]
            offered = sum(j.nodes * j.actual_runtime_h for j in prot)
            ceiling = cp.pool_nodes * duration_h
            if ceiling > 0 and offered > ceiling * 1.02:
                raise SaturationError(
                    f"OVERSUBSCRIBED: protected tier {cp.protected_tier!r} is "
                    f"offered {offered:,.0f} node-h but the running_pool_cap of "
                    f"{cp.pool_nodes} nodes can deliver at most {ceiling:,.0f} "
                    f"node-h over {duration_h/24:.0f}d "
                    f"({offered/ceiling:.1f}x oversubscribed). The backlog would "
                    f"grow without bound. Raise capacity_protection.pool_nodes, "
                    f"widen the protected tier, or switch strategy "
                    f"(e.g. dedicated_partition / none).")
        elif cp.strategy == "dedicated_partition":
            # small jobs limited to total - partition
            small = [j for j in jobs if j.nodes < cp.big_job_min_nodes]
            offered = sum(j.nodes * j.actual_runtime_h for j in small)
            ceiling = (self.total_nodes - cp.partition_nodes) * duration_h
            if ceiling > 0 and offered > ceiling * 1.02:
                raise SaturationError(
                    f"OVERSUBSCRIBED: small jobs offered {offered:,.0f} node-h but "
                    f"only {self.total_nodes - cp.partition_nodes} nodes remain "
                    f"after a {cp.partition_nodes}-node big-job partition "
                    f"({offered/ceiling:.1f}x). Shrink the partition or change "
                    f"strategy.")

    def run(self, jobs: list[Job]):
        cfg = self.cfg
        duration_h = cfg.run.duration_days * 24.0
        sample_dt = cfg.run.sample_dt_h

        # -- pre-flight oversubscription check (cheap, before the event loop) --
        # If the protected tier is offered more node-hours than its protection
        # strategy can deliver, the backlog will grow without bound. Detect it
        # up front instead of grinding through a quadratic slowdown to find out.
        self._preflight_check(jobs, duration_h)

        for j in jobs:
            self._push(j.submit_time_h, "arrive", j)
        t = 0.0
        while t <= duration_h:
            self._push(t, "sample", None)
            t += sample_dt

        EPS = 1e-9
        while self.events:
            ev = heapq.heappop(self.events)
            t_now = ev.time_h
            batch = [ev]
            while self.events and abs(self.events[0].time_h - t_now) <= EPS:
                batch.append(heapq.heappop(self.events))

            if t_now > duration_h:
                for e in batch:
                    if e.kind == "finish":
                        self.now_h = duration_h
                        self._finish(e.job)
                self._min_pending_cache = None
                continue

            self.now_h = t_now
            sched_needed = False
            samples = []
            for e in batch:
                if e.kind == "arrive":
                    self.pending.append(e.job)
                    if self._min_pending_cache is not None and \
                            e.job.nodes < self._min_pending_cache:
                        self._min_pending_cache = e.job.nodes
                    if e.job.nodes <= self.free_nodes:
                        sched_needed = True
                elif e.kind == "finish":
                    self._finish(e.job)
                    sched_needed = True
                elif e.kind == "sample":
                    samples.append(e)

            if sched_needed:
                self._try_schedule()
                if len(self.pending) > self._sat_limit:
                    raise SaturationError(
                        f"pending={len(self.pending)} exceeded guard "
                        f"{self._sat_limit} at t={self.now_h:.1f}h "
                        f"({self.now_h/24:.1f}d). System is oversubscribed under "
                        f"this config (offered load > capacity). Raise "
                        f"capacity, lower arrival, or change capacity_protection.")

            for _ in samples:
                busy = self.total_nodes - self.free_nodes
                self.util_samples.append((self.now_h, busy))
                self.program_nh_samples.append((self.now_h, dict(self.delivered_nh)))
                depth = {}
                for j in self.pending:
                    depth[j.size_tier] = depth.get(j.size_tier, 0) + 1
                self.queue_depth_samples.append((self.now_h, depth))
