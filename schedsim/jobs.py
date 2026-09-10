"""The JobTable contract.

A JobTable is a pandas DataFrame with a fixed set of columns; it is the only
thing the scheduler consumes, whether the rows came from the real trace
(replay) or from a workload model (Monte Carlo). Times are hours relative to
the simulation origin (t = 0); submit_h may be negative for jobs that were
already queued when the window opens.

Required columns
  job_id        str    unique
  queue         str    execution queue the job sits in
  project, user, program   str   (program = INCITE/ALCC/DD/... or 'other')
  nodes         int
  walltime_h    float  requested walltime (drives backfill footprint)
  runtime_h     float  actual run time (drives when nodes free)
  submit_h      float
  base_score, score_boost, project_priority, enable_wfp, enable_fifo,
  enable_backfill, wfp_factor, fifo_factor, backfill_factor, backfill_max,
  queue_priority          float  scoring parameters PBS attached to the job

Optional columns
  initial_start_h  float  for jobs already RUNNING at t=0: their real start (<0).
                           NaN otherwise. These are pre-placed, not scheduled.
Any other columns are carried through untouched (e.g. observed wait for replay).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SCORING_COLUMNS = [
    "base_score", "score_boost", "project_priority", "enable_wfp", "enable_fifo",
    "enable_backfill", "wfp_factor", "fifo_factor", "backfill_factor",
    "backfill_max", "queue_priority",
]
REQUIRED = ["job_id", "queue", "project", "user", "program", "nodes",
            "walltime_h", "runtime_h", "submit_h"] + SCORING_COLUMNS

SCORING_DEFAULTS = dict(base_score=51.0, score_boost=0.0, project_priority=25.0,
                        enable_wfp=1.0, enable_fifo=0.0, enable_backfill=0.0,
                        wfp_factor=100_000.0, fifo_factor=1800.0,
                        backfill_factor=84_600.0, backfill_max=50.0,
                        queue_priority=0.0)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"JobTable missing columns: {missing}")
    if df["job_id"].duplicated().any():
        raise ValueError("JobTable job_id must be unique")
    for c in ("queue", "project", "user", "program"):
        if df[c].isna().any():
            raise ValueError(f"JobTable column {c!r} has null values")
    bad = (df["nodes"] <= 0) | (df["walltime_h"] <= 0) | (df["runtime_h"] < 0)
    if bad.any():
        raise ValueError(f"JobTable has {int(bad.sum())} rows with non-positive "
                         f"nodes/walltime or negative runtime")
    if "initial_start_h" not in df.columns:
        df = df.assign(initial_start_h=np.nan)
    for c in SCORING_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(SCORING_DEFAULTS[c]).astype(float)
    return df.reset_index(drop=True)


@dataclass
class JobArrays:
    """Struct-of-arrays view used inside the engine (all aligned to row order)."""
    n: int
    nodes: np.ndarray
    walltime_h: np.ndarray
    runtime_h: np.ndarray
    submit_h: np.ndarray
    initial_start_h: np.ndarray
    queue_id: np.ndarray
    project_id: np.ndarray
    user_id: np.ndarray
    queue_names: list
    scoring: dict     # name -> float array

    @classmethod
    def from_table(cls, df: pd.DataFrame) -> "JobArrays":
        df = validate(df)
        q_codes, q_names = pd.factorize(df["queue"].astype(str), sort=True)
        p_codes, _ = pd.factorize(df["project"].astype(str), sort=True)
        u_codes, _ = pd.factorize(df["user"].astype(str), sort=True)
        return cls(
            n=len(df),
            nodes=df["nodes"].to_numpy(np.int64),
            walltime_h=df["walltime_h"].to_numpy(float),
            runtime_h=df["runtime_h"].to_numpy(float),
            submit_h=df["submit_h"].to_numpy(float),
            initial_start_h=df["initial_start_h"].to_numpy(float),
            queue_id=q_codes.astype(np.int64),
            project_id=p_codes.astype(np.int64),
            user_id=u_codes.astype(np.int64),
            queue_names=list(q_names),
            scoring={c: df[c].to_numpy(float) for c in SCORING_COLUMNS},
        )
