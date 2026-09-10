"""
Metrics + reporting. Delta-t sampled telemetry -> tidy long-format time series
and a per-program decision table. Utilization / throughput / queue-time are the
three headline metrics the study cares about.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import SimConfig
from scheduler import Scheduler
from generator import Job


def decision_table(jobs: list[Job], sched: Scheduler, cfg: SimConfig) -> pd.DataFrame:
    duration_h = cfg.run.duration_days * 24.0
    started = [j for j in jobs if j.start_time_h is not None]
    total_delivered = sum(j.nodes * _win_rt(j, duration_h) for j in started) or 1.0

    rows = []
    progs = list(sched.shares.keys())
    for prog in progs:
        allj = [j for j in jobs if j.program == prog]
        sj = [j for j in allj if j.start_time_h is not None]
        unstarted = len(allj) - len(sj)
        if sj:
            waits = np.array([j.start_time_h - j.submit_time_h for j in sj])
            nh = sum(j.nodes * _win_rt(j, duration_h) for j in sj)
        else:
            waits = np.array([0.0]); nh = 0.0
        budget = sched.budget_nh.get(prog, float("nan"))
        burn = (sched.delivered_nh.get(prog, 0.0) / budget
                if budget and budget > 0 else float("nan"))
        rows.append({
            "program": prog,
            "n_jobs": len(allj),
            "unstarted": unstarted,
            "target_share": round(sched.shares.get(prog, 0.0), 4),
            "delivered_share": round(nh / total_delivered, 4),
            "budget_burn": round(burn, 3),
            "wait_p50_h": round(float(np.percentile(waits, 50)), 2),
            "wait_p95_h": round(float(np.percentile(waits, 95)), 2),
            "wait_max_h": round(float(waits.max()), 2),
        })
    return pd.DataFrame(rows)


def _win_rt(j: Job, duration_h: float) -> float:
    end = min(duration_h, j.end_time_h if j.end_time_h is not None else duration_h)
    return max(0.0, end - max(0.0, j.start_time_h))


def breakdowns(jobs, sched, cfg):
    """Per-QUEUE and per-PROGRAM breakdowns for cross-analysis, plus a per-PROJECT
    long-format DataFrame. Returns (flat_dict, projects_df).

    flat_dict keys (added to a scan row):
      q_<tier>_wait_p50/p95/max_h, q_<tier>_started, q_<tier>_submitted,
      q_<tier>_delivered_nh, q_<tier>_throughput_jpd
      prog_<P>_wait_p50/p95_h, prog_<P>_started, prog_<P>_delivered_nh
    So you can see how a knob trades wait/throughput/delivery ACROSS queues and
    programs, not just in the pooled aggregate.
    """
    duration_h = cfg.run.duration_days * 24.0
    days = max(1e-9, cfg.run.duration_days)
    started = [j for j in jobs if j.start_time_h is not None]
    flat = {}

    def _stats(prefix, group_all, group_started):
        n_sub = len(group_all)
        n_st = len(group_started)
        w = np.array([j.start_time_h - j.submit_time_h for j in group_started]) \
            if group_started else np.array([0.0])
        nh = sum(j.nodes * _win_rt(j, duration_h) for j in group_started)
        completed = sum(1 for j in group_started
                        if j.end_time_h is not None and j.end_time_h <= duration_h)
        flat[f"{prefix}_submitted"] = n_sub
        flat[f"{prefix}_started"] = n_st
        flat[f"{prefix}_unstarted"] = n_sub - n_st
        flat[f"{prefix}_delivered_nh"] = float(nh)
        flat[f"{prefix}_throughput_jpd"] = round(completed / days, 2)
        flat[f"{prefix}_wait_p50_h"] = round(float(np.percentile(w, 50)), 2)
        flat[f"{prefix}_wait_p95_h"] = round(float(np.percentile(w, 95)), 2)
        flat[f"{prefix}_wait_max_h"] = round(float(w.max()), 2)

    # per queue (size tier)
    for t in cfg.size_tiers:
        allq = [j for j in jobs if j.size_tier == t.name]
        stq = [j for j in started if j.size_tier == t.name]
        _stats(f"q_{t.name}", allq, stq)

    # per program
    progs = list(sched.shares.keys()) if hasattr(sched, "shares") else \
        sorted({j.program for j in jobs})
    for p in progs:
        allp = [j for j in jobs if j.program == p]
        stp = [j for j in started if j.program == p]
        _stats(f"prog_{p}", allp, stp)

    # per project (long format) — only meaningful with the project layer
    proj_rows = []
    if cfg.projects.enabled and getattr(sched, "project_award_nh", None):
        proj_prog = {}
        for j in jobs:
            if j.project and j.project not in proj_prog:
                proj_prog[j.project] = j.program
        deliv = getattr(sched, "project_delivered_nh", {})
        award = sched.project_award_nh
        for proj, aw in award.items():
            d = deliv.get(proj, 0.0)
            proj_rows.append({
                "project": proj, "program": proj_prog.get(proj, "?"),
                "delivered_nh": float(d), "award_nh": float(aw),
                "pct_alloc_used": (100.0 * d / aw) if aw > 0 else float("nan"),
            })
    import pandas as pd
    return flat, pd.DataFrame(proj_rows)


def timeseries(sched: Scheduler, cfg: SimConfig) -> pd.DataFrame:
    """Tidy long-format: (t_h, metric, program|_all, value).
    Utilization is measured against PRODUCTION nodes (the accountability
    denominator), consistent with summary_stats."""
    rows = []
    prod = cfg.machine.prod()
    for t, busy in sched.util_samples:
        rows.append({"t_h": t, "metric": "utilization", "program": "_all",
                     "value": busy / prod})
        rows.append({"t_h": t, "metric": "busy_nodes", "program": "_all",
                     "value": busy})
    for t, prog_nh in sched.program_nh_samples:
        for prog, nh in prog_nh.items():
            rows.append({"t_h": t, "metric": "delivered_nh", "program": prog,
                         "value": nh})
    for t, depth in sched.queue_depth_samples:
        total_depth = sum(depth.values())
        rows.append({"t_h": t, "metric": "queue_depth", "program": "_all",
                     "value": total_depth})
        for tier, n in depth.items():
            rows.append({"t_h": t, "metric": "queue_depth_tier", "program": tier,
                         "value": n})
    return pd.DataFrame(rows)


def summary_stats(jobs: list[Job], sched: Scheduler, cfg: SimConfig) -> dict:
    duration_h = cfg.run.duration_days * 24.0
    started = [j for j in jobs if j.start_time_h is not None]
    util = pd.DataFrame(sched.util_samples, columns=["t", "busy"])
    util = util[util["t"] <= duration_h]
    prod = cfg.machine.prod()
    avg_util = float((util["busy"] / prod).mean()) if len(util) else 0.0
    completed = [j for j in started
                 if j.end_time_h is not None and j.end_time_h <= duration_h]
    nh = sum(j.nodes * _win_rt(j, duration_h) for j in started)
    all_waits = np.array([j.start_time_h - j.submit_time_h for j in started]) \
        if started else np.array([0.0])

    # Large-job (capability) vs small-job wait — the starvation metric.
    # "Large" = >= the capacity-job threshold (20% of production nodes).
    big_thresh = int(round(0.20 * prod))
    big = [j for j in started if j.nodes >= big_thresh]
    small = [j for j in started if j.nodes < big_thresh]
    big_all = [j for j in jobs if j.nodes >= big_thresh]
    big_waits = (np.array([j.start_time_h - j.submit_time_h for j in big])
                 if big else np.array([0.0]))
    small_waits = (np.array([j.start_time_h - j.submit_time_h for j in small])
                   if small else np.array([0.0]))

    return {
        "n_jobs": len(jobs),
        "n_started": len(started),
        "n_unstarted": len(jobs) - len(started),
        "n_completed": len(completed),
        "avg_util_pct": round(avg_util * 100, 2),   # vs PRODUCTION nodes
        "node_hours_delivered": round(nh),
        "throughput_jobs_per_day": round(len(completed) / cfg.run.duration_days, 1),
        "wait_p50_h": round(float(np.percentile(all_waits, 50)), 2),
        "wait_p95_h": round(float(np.percentile(all_waits, 95)), 2),
        "wait_max_h": round(float(all_waits.max()), 2),
        # starvation metric: capability jobs (>= 20% of production nodes)
        "big_thresh_nodes": big_thresh,
        "n_big_jobs": len(big_all),
        "n_big_unstarted": len(big_all) - len(big),
        "big_wait_p50_h": round(float(np.percentile(big_waits, 50)), 2),
        "big_wait_p95_h": round(float(np.percentile(big_waits, 95)), 2),
        "big_wait_max_h": round(float(big_waits.max()), 2),
        "small_wait_p50_h": round(float(np.percentile(small_waits, 50)), 2),
        "small_wait_p95_h": round(float(np.percentile(small_waits, 95)), 2),
    }
