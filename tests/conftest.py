import numpy as np
import pandas as pd
import pytest

from schedsim.jobs import SCORING_DEFAULTS
from schedsim.machine import Machine
from schedsim.menu import Menu, Queue
from schedsim.engine import Engine, SchedulerSpec
from schedsim.priority import ExprPriority


def make_jobs(rows):
    """rows: list of dicts with nodes, walltime_h, runtime_h, submit_h (+optional
    queue/project/user/enable_backfill). Fills scoring defaults."""
    df = pd.DataFrame(rows)
    for k, v in SCORING_DEFAULTS.items():
        if k not in df:
            df[k] = v
    for c, d in [("project", "p"), ("user", "u"), ("program", "INCITE"), ("queue", "q")]:
        df[c] = df[c].fillna(d) if c in df else d
    df["job_id"] = [f"j{i}" for i in range(len(df))]
    return df


# instantaneous engine: no cycle latency, so start times are exact
FAST = dict(cycle_h=0.5, min_pass_gap_h=0.0, pass_time_per_job_h=0.0,
            dispatch_latency_h=0.0, ordering="backfill_all", partitions=())


@pytest.fixture
def machine100():
    return Machine(total_nodes=100, reportable_nodes=100, schedulable_nodes=100)


@pytest.fixture
def menu_q():
    return Menu([Queue("q", 1, 100, 200)])


def run(machine, menu, jobs, t_end=200.0, priority="eligible_s", **spec):
    s = dict(FAST); s.update(spec)
    eng = Engine(machine, menu, SchedulerSpec(**s), ExprPriority(priority, machine.total_nodes))
    return eng.run(jobs, t_end).jobs
