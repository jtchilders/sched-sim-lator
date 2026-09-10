"""Event-driven PBS-style scheduler.

Model
  * Scheduling PASSES run when something changed (a job arrived or finished) —
    PBS Pro starts a cycle on those events — and at least every `cycle_h`
    (scheduler_iteration). A pass lasts ~1 s per job examined (capped); the
    next pass waits for it to end, and jobs start during the pass in sort
    order, so bursts of events are batched and low-priority jobs pay more
    cycle latency, as on the real system.
  * ROUTING / QUEUE ADMISSION: a queue's `max_queued_per_project` /
    `max_queued_per_user` caps how many of an entity's jobs may sit in the
    execution queue (queued + running). Excess jobs wait in the routing queue in
    submit order, are not scheduled, and do not accrue eligible time — exactly
    PBS's max_queued semantics. They are admitted as slots free up.
  * Each pass: score eligible pending jobs (vectorised), sort descending, walk
    in order. A job starts if its footprint (nodes x requested walltime) fits in
    the free-node PROFILE — availability minus running jobs' walltime footprints
    minus already-placed reservations. Otherwise, if fewer than
    `backfill_depth` reservations exist, it gets a reservation at the earliest
    time its footprint fits (PBS backfill_depth; 1 = EASY). Later jobs may
    still start if they fit around the reservations (they cannot delay them,
    because reserved footprints are subtracted from the profile).
  * Availability comes from `Machine` (usable-node series minus reservation
    windows), so the profile drains ahead of a full-machine maintenance window.
  * PARTITIONS: queues can be pinned to dedicated node pools (Aurora's debug
    queue: 64 nodes); those nodes are subtracted from the general pool.
  * RUN limits (`max_run_per_user`, `max_run_per_project`, `max_nodes_total`)
    are enforced at start; a limit-blocked job is skipped and holds no
    reservation.
  * Running jobs finish at start + runtime_h (actual) but occupy the profile
    until start + walltime_h (requested), as the real scheduler believes.

Output: per-job start_h (NaN if never started by t_end), end_h, eligible_h
(time the job became eligible; = time it entered the execution queue).

Debugging: set SCHEDSIM_TRACE_JOB=<job_id> to print, at every pass, why that
job did or did not start (rank, free nodes, profile minimum, limits).
"""
from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .jobs import JobArrays, validate
from .machine import Machine
from .menu import Menu
from .priority import ExprPriority


@dataclass(frozen=True)
class SchedulerSpec:
    cycle_h: float = 600.0 / 3600.0       # periodic pass (PBS scheduler_iteration)
    trigger_on_events: bool = True        # also pass on arrival / finish
    # Cycle timing (calibrated on replay, see docs/VALIDATION.md). A pass (PBS
    # cycle) lasts max(min_pass_gap_h, pass_time_per_job_h * jobs examined) and
    # the next pass cannot start before it ends. Jobs start DURING the pass in
    # sort order, so a job's recorded start = pass start + dispatch latency +
    # its rank fraction x pass duration (low-score jobs pay more cycle latency).
    min_pass_gap_h: float = 1.0 / 60.0    # 1 min floor between passes
    pass_time_per_job_h: float = 1.0 / 3600.0   # ~1 s of cycle time per examined job
    max_pass_time_h: float = 5.0 / 60.0   # cap on a cycle duration
    dispatch_latency_h: float = 0.04      # PBS->MOM dispatch + node prologue before a
                                          # job is recorded as started (~2.5 min on Aurora)
    backfill_depth: int = 1
    examine_cap: int = 5000               # max eligible jobs walked per pass
    enforce_limits: bool = True
    enforce_queued_limits: bool = False   # routing-queue holdback (max_queued). Off for
                                          # replay: recorded submissions already reflect
                                          # users pacing under the cap. On for synthetic
                                          # workloads.
    runtime_source: str = "actual"        # actual | walltime (jobs run to walltime)
    # Who may start out of priority order once a reservation is held?
    #   backfill_all      any fitting job (stock PBS EASY/conservative backfill)
    #   backfill_flagged  only jobs whose enable_backfill flag is 1 (ALCF hypothesis:
    #                     prod queues carry enable_backfill=0, backfill-* queues 1)
    #   strict            nobody (strict ordering)
    #   family_flagged    strictness only among jobs of `strict_queues`: a job in a
    #                     strict queue may not start while a higher-priority strict-
    #                     queue job holds a reservation (unless flagged); other
    #                     queues always backfill
    #   queue_flagged     same, but only a blocked job of the SAME queue blocks
    #   strict_groups     PBS strict_ordering per scheduling group: within a group
    #                     (a tuple of queues in `strict_groups`), the first
    #                     `backfill_depth` blocked jobs get reservations and the next
    #                     blocked job stops the pass for the rest of that group.
    #                     Queues outside every group backfill freely.
    ordering: str = "backfill_all"
    strict_groups: tuple = (("small", "medium", "large", "backfill-small",
                             "backfill-medium", "backfill-large"),)
    # Dedicated node partitions: (name, queues, nodes). Jobs of those queues run
    # only on that partition; everyone else gets availability minus the
    # partitions. Aurora's debug queue draws from 64 dedicated (shared) nodes.
    partitions: tuple = (("debug", ("debug",), 64),)
    strict_queues: tuple = ("small", "medium", "large", "backfill-small",
                            "backfill-medium", "backfill-large")


@dataclass
class SimResult:
    jobs: pd.DataFrame          # input table + start_h, end_h, eligible_h, sim_wait_h, started
    t_end_h: float
    n_passes: int
    machine: Machine
    log: list = field(default_factory=list)


class Engine:
    def __init__(self, machine: Machine, menu: Menu, spec: SchedulerSpec,
                 priority: ExprPriority | None = None):
        self.machine = machine
        self.menu = menu
        self.spec = spec
        self.priority = priority or ExprPriority(total_nodes=machine.total_nodes)

    # -----------------------------------------------------------------------
    def run(self, table: pd.DataFrame, t_end_h: float) -> SimResult:
        table = validate(table)
        J = JobArrays.from_table(table)
        spec = self.spec
        mach = self.machine
        lim = self.menu.limit_arrays(J.queue_names)
        nq = len(J.queue_names)
        self._strict_qids = {i for i, n in enumerate(J.queue_names) if n in set(spec.strict_queues)}
        # partition 0 = general pool; k>=1 = dedicated pools
        self._part_of = np.zeros(nq, np.int64)
        self._part_nodes = [0.0]
        for pi, (pname, pqueues, pnodes) in enumerate(spec.partitions, start=1):
            self._part_nodes.append(float(pnodes))
            for i, n in enumerate(J.queue_names):
                if n in pqueues:
                    self._part_of[i] = pi
        self._group_of = {}
        for gi, grp in enumerate(spec.strict_groups):
            for i, n in enumerate(J.queue_names):
                if n in grp:
                    self._group_of[i] = gi

        runtime = J.runtime_h if spec.runtime_source == "actual" else J.walltime_h
        runtime = np.minimum(runtime, J.walltime_h)   # PBS kills at walltime

        start_h = np.full(J.n, np.nan)
        end_h = np.full(J.n, np.nan)
        eligible_h = np.full(J.n, np.nan)
        self._starts = start_h                        # shared with _pass
        self._eligible = eligible_h
        import os
        tj = os.environ.get("SCHEDSIM_TRACE_JOB")
        self._trace_idx = int(np.where(table["job_id"].astype(str).to_numpy() == tj)[0][0]) if tj and (table["job_id"].astype(str) == tj).any() else -1

        # --- state -----------------------------------------------------------
        running: set[int] = set()
        finish_heap: list[tuple[float, int]] = []
        q_nodes = np.zeros(nq, float)
        proj_run: dict = {}; user_run: dict = {}
        proj_q: dict = {}; user_q: dict = {}          # in execution queue (pending+running)
        routing: dict[int, deque] = {q: deque() for q in range(nq)}   # held back, per queue

        def _inc(d, k):
            d[k] = d.get(k, 0) + 1

        def _dec(d, k):
            d[k] -= 1

        def _admissible(j) -> bool:
            q = J.queue_id[j]
            if proj_q.get((q, J.project_id[j]), 0) >= lim["max_queued_per_project"][q]:
                return False
            if user_q.get((q, J.user_id[j]), 0) >= lim["max_queued_per_user"][q]:
                return False
            return True

        def _admit(j, now):
            q = J.queue_id[j]
            _inc(proj_q, (q, J.project_id[j])); _inc(user_q, (q, J.user_id[j]))
            eligible_h[j] = now
            pending.append(j)

        def _arrive(j, now):
            if spec.enforce_queued_limits and not _admissible(j):
                routing[J.queue_id[j]].append(j)
            else:
                _admit(j, now)

        lat = spec.dispatch_latency_h

        def _occupy(j: int, t0: float, pre: bool = False, offset: float = 0.0):
            # nodes are held from the decision time t0; the recorded start (what
            # PBS stamps as stime) lags by dispatch latency + within-cycle offset
            start_h[j] = t0 if pre else t0 + lat + offset
            end_h[j] = start_h[j] + runtime[j]
            running.add(j)
            heapq.heappush(finish_heap, (end_h[j], j))
            q = J.queue_id[j]
            q_nodes[q] += J.nodes[j]
            _inc(proj_run, (q, J.project_id[j])); _inc(user_run, (q, J.user_id[j]))

        def _release(j: int, now: float):
            running.discard(j)
            q = J.queue_id[j]
            q_nodes[q] -= J.nodes[j]
            _dec(proj_run, (q, J.project_id[j])); _dec(user_run, (q, J.user_id[j]))
            _dec(proj_q, (q, J.project_id[j])); _dec(user_q, (q, J.user_id[j]))
            # a slot opened: admit the oldest held-back job(s) of this queue that fit
            rq = routing[q]
            if rq:
                keep = deque()
                while rq:
                    k = rq.popleft()
                    if _admissible(k):
                        _admit(k, now)
                    else:
                        keep.append(k)
                routing[q] = keep

        pending: list[int] = []

        # jobs already running at t=0 (replay warm state)
        pre = np.where(~np.isnan(J.initial_start_h))[0]
        for j in pre:
            j = int(j)
            eligible_h[j] = J.submit_h[j]
            q = J.queue_id[j]
            _inc(proj_q, (q, J.project_id[j])); _inc(user_q, (q, J.user_id[j]))
            _occupy(j, float(J.initial_start_h[j]), pre=True)
        arr_idx = np.setdiff1d(np.arange(J.n), pre)
        arr_idx = arr_idx[np.argsort(J.submit_h[arr_idx], kind="stable")]
        arr_ptr = 0

        cycle = spec.cycle_h
        gap = spec.min_pass_gap_h
        n_passes = 0
        now = 0.0
        next_periodic = 0.0
        last_pass = -math.inf
        next_pass_ok = 0.0
        dirty = True

        while now <= t_end_h + 1e-9:
            # --- apply all events at `now` --------------------------------------
            while arr_ptr < len(arr_idx) and J.submit_h[arr_idx[arr_ptr]] <= now + 1e-9:
                _arrive(int(arr_idx[arr_ptr]), now); arr_ptr += 1; dirty = True
            while finish_heap and finish_heap[0][0] <= now + 1e-9:
                _, j = heapq.heappop(finish_heap)
                if j in running:
                    _release(j, now); dirty = True
            periodic_due = now >= next_periodic - 1e-9
            if periodic_due:
                next_periodic = (math.floor(now / cycle + 1e-9) + 1) * cycle

            # --- scheduling pass ------------------------------------------------
            want = pending and (periodic_due or (spec.trigger_on_events and dirty))
            if want and now >= next_pass_ok - 1e-9:
                started, dur = self._pass(now, J, pending, running, q_nodes, proj_run,
                                          user_run, lim, mach, _occupy)
                n_passes += 1
                last_pass = now
                next_pass_ok = now + max(gap, dur)
                dirty = False
                if started:
                    s = set(started)
                    pending = [j for j in pending if j not in s]

            # --- next time ------------------------------------------------------
            cands = [next_periodic if pending else math.inf]
            if arr_ptr < len(arr_idx):
                cands.append(float(J.submit_h[arr_idx[arr_ptr]]))
            if finish_heap:
                cands.append(finish_heap[0][0])
            if pending and dirty and spec.trigger_on_events:
                cands.append(next_pass_ok)           # rate-limited retry
            nxt = min(cands)
            if nxt is math.inf or nxt > t_end_h + 1e-9:
                break
            now = max(nxt, now + 1e-9)

        df = table.copy()
        df["start_h"] = start_h
        df["end_h"] = end_h
        df["eligible_h"] = eligible_h
        df["started"] = ~np.isnan(start_h)
        df["sim_wait_h"] = start_h - J.submit_h
        df.loc[~np.isnan(J.initial_start_h), "sim_wait_h"] = np.nan   # pre-placed
        return SimResult(jobs=df, t_end_h=t_end_h, n_passes=n_passes, machine=mach)

    # -----------------------------------------------------------------------
    def _pass(self, now, J, pending, running, q_nodes, proj_run, user_run,
              lim, mach, occupy) -> list[int]:
        spec = self.spec
        pend = np.fromiter(pending, dtype=np.int64, count=len(pending))
        scores = self.priority.score(now, J, pend, self._eligible[pend])
        # highest score first; ties by earlier submit, then index
        order = np.lexsort((pend, J.submit_h[pend], -scores))
        if len(order) > spec.examine_cap:
            order = order[: spec.examine_cap]
        pass_dur = min(spec.max_pass_time_h, spec.pass_time_per_job_h * len(order))
        n_order = max(1, len(order))

        # ---- free-node profiles over [now, horizon], one per partition ----------
        part_of = self._part_of
        npart = len(self._part_nodes)
        max_wt_pend = float(J.walltime_h[pend].max())
        if running:
            run = np.fromiter(running, dtype=np.int64, count=len(running))
            wt_end_all = np.maximum(self._starts[run] + J.walltime_h[run], now + 1e-9)
        else:
            run = np.empty(0, np.int64); wt_end_all = np.empty(0)
        horizon = now + max_wt_pend + (float(wt_end_all.max() - now) if len(wt_end_all) else 0.0) + 1e-6
        T = np.unique(np.concatenate([[now], wt_end_all[wt_end_all < horizon],
                                      mach.breakpoints(now, horizon), [horizon]]))
        avail = mach.available_at(T)
        dedicated = sum(self._part_nodes[1:])
        Fs = []
        run_part = part_of[J.queue_id[run]] if len(run) else np.empty(0, np.int64)
        for pi in range(npart):
            sel = run_part == pi
            wt_end = wt_end_all[sel]; run_nodes = J.nodes[run[sel]].astype(float)
            o = np.argsort(wt_end, kind="stable")
            wt_end_s = wt_end[o]; cum_nodes = np.cumsum(run_nodes[o])
            if len(cum_nodes):
                released = np.searchsorted(wt_end_s, T, side="right")
                footprint = float(run_nodes.sum()) - np.concatenate([[0.0], cum_nodes])[released]
            else:
                footprint = np.zeros(len(T))
            cap = np.clip(avail - dedicated, 0, None) if pi == 0 else np.minimum(avail, self._part_nodes[pi])
            Fs.append(np.floor(cap - footprint + 1e-9))
        Ms = [np.minimum.accumulate(F) for F in Fs]
        free_now = Fs[0][0]

        n_res = 0
        started: list[int] = []
        min_nodes_pending = float(J.nodes[pend].min())
        ordering = spec.ordering
        can_bf = J.scoring["enable_backfill"]
        strict_q = self._strict_qids
        res_queues: set = set()          # queue ids holding a reservation
        strict_res = False               # any strict-queue job holds a reservation
        group_of = self._group_of
        grp_blocked: dict = {}           # group -> blocked jobs seen so far
        grp_stopped: set = set()

        trace_j = self._trace_idx
        for rank, oi in enumerate(order):
            j = int(pend[oi]); nn = float(J.nodes[j]); wt = float(J.walltime_h[j])
            q = J.queue_id[j]
            if j == trace_j:
                pi_ = part_of[q]; i1_ = max(int(np.searchsorted(T, now + wt - 1e-9, side="left")), 1)
                print(f"[trace] t={now:.3f} job {j} q={J.queue_names[q]} nodes={nn} wt={wt} part={pi_} "
                      f"rank={int(np.where(order == oi)[0][0])}/{len(order)} F0={Fs[pi_][0]} Mwt={Ms[pi_][i1_-1]} "
                      f"n_res={n_res} q_nodes={q_nodes[q]} user_run={user_run.get((q, J.user_id[j]), 0)} "
                      f"proj_run={proj_run.get((q, J.project_id[j]), 0)} T[:4]={np.round(T[:4]-now,3)} avail0={avail[0]}")
            if spec.enforce_limits:
                if q_nodes[q] + nn > lim["max_nodes_total"][q]:
                    continue
                if proj_run.get((q, J.project_id[j]), 0) >= lim["max_run_per_project"][q]:
                    continue
                if user_run.get((q, J.user_id[j]), 0) >= lim["max_run_per_user"][q]:
                    continue
            if ordering == "strict_groups" and group_of.get(q, -1) in grp_stopped:
                continue
            if n_res > 0 and ordering not in ("backfill_all", "strict_groups") and can_bf[j] < 0.5:
                if ordering == "strict":
                    continue
                if ordering == "backfill_flagged":
                    continue
                if ordering == "family_flagged" and strict_res and q in strict_q:
                    continue
                if ordering == "queue_flagged" and q in res_queues:
                    continue
            pi = part_of[q]; F = Fs[pi]; M = Ms[pi]
            i1 = max(int(np.searchsorted(T, now + wt - 1e-9, side="left")), 1)
            if M[i1 - 1] >= nn:
                occupy(j, now, offset=pass_dur * rank / n_order)
                started.append(j)
                F[:i1] -= nn
                Ms[pi] = np.minimum.accumulate(F)
                if pi == 0:
                    free_now -= nn
                continue
            if ordering == "strict_groups":
                g = group_of.get(q, -1)
                grp_blocked[g] = grp_blocked.get(g, 0) + 1
                if grp_blocked[g] > spec.backfill_depth:
                    if g >= 0:
                        grp_stopped.add(g)
                    continue
                grant = True
            else:
                grant = n_res < spec.backfill_depth or (
                    ordering == "queue_flagged" and q not in res_queues)
            if grant:
                k0 = _earliest_fit(T, F, nn, wt)
                if k0 is not None:
                    k1 = int(np.searchsorted(T, T[k0] + wt - 1e-9, side="left"))
                    F[k0:max(k1, k0 + 1)] -= nn
                    Ms[pi] = np.minimum.accumulate(F)
                n_res += 1
                res_queues.add(q)
                if q in strict_q:
                    strict_res = True
                continue
            if free_now < min_nodes_pending and ordering not in ("queue_flagged", "strict_groups"):
                break
        return started, pass_dur


def _earliest_fit(T: np.ndarray, F: np.ndarray, nodes: float, wt: float):
    """Smallest k such that min(F[k : k1]) >= nodes where [T[k], T[k]+wt) spans
    segments k..k1-1. None if it never fits within the profile horizon."""
    n = len(T)
    ends = np.searchsorted(T, T + wt - 1e-9, side="left")
    for k in range(n - 1):
        k1 = min(max(int(ends[k]), k + 1), n)
        if F[k:k1].min() >= nodes:
            return k
    return None
