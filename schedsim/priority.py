"""Job-scoring (priority) models. Higher score runs first.

ALCF runs a custom PBS sort formula whose exact form is not published. The
per-job parameters it uses ARE recorded in every job's Resource_List
(base_score, score_boost, project_priority, enable_wfp/wfp_factor,
enable_fifo/fifo_factor, enable_backfill/backfill_factor/backfill_max), and the
public description says: larger jobs gain priority faster, shorter jobs gain
priority faster, INCITE/ALCC outrank discretionary, negative-balance projects
are demoted. That is the Cobalt "WFP" lineage: (wait / walltime)^3 * size.

ALCF_EXACT is the server's job_sort_formula (qstat -Bf); ALCF_FITTED is the form
recovered from recorded scores before the server formula was available. Variables available to an
expression, all per-job numpy arrays at scoring time:

  eligible_h, eligible_s   accrued eligible time (since the job entered its
                           execution queue), hours / seconds
  walltime_h, nodes, total_nodes, queue_priority
  base_score, score_boost, project_priority,
  enable_wfp, wfp_factor, enable_fifo, fifo_factor,
  enable_backfill, backfill_factor, backfill_max
"""
from __future__ import annotations

import numpy as np

from .jobs import JobArrays, SCORING_COLUMNS
from .safe_expr import compile_expr

VARIABLES = set(SCORING_COLUMNS) | {"eligible_h", "eligible_s", "walltime_h",
                                    "nodes", "total_nodes"}

# The exact server formula (qstat -Bf, Sept 2026):
#   job_sort_formula = base_score + score_boost
#     + enable_wfp * wfp_factor * (eligible_time**2 / min(max(walltime,21600),43200)**3
#                                  * project_priority * nodect / total_cpus)
#     + enable_backfill * min(backfill_max, eligible_time / backfill_factor)
#     + enable_fifo * eligible_time / fifo_factor
# with eligible_time and walltime in SECONDS, walltime clamped to [6 h, 12 h].
# Quadratic in eligible time, linear in node count, project priority as a
# multiplier; jobs requesting <= 6 h accrue 8x faster than 12 h+ jobs.
ALCF_EXACT = (
    "base_score + score_boost"
    " + enable_wfp * wfp_factor * (eligible_s ** 2 / min(max(walltime_h * 3600.0, 21600.0), 43200.0) ** 3"
    "   * project_priority * nodes / total_nodes)"
    " + enable_backfill * min(backfill_max, eligible_s / backfill_factor)"
    " + enable_fifo * eligible_s / fifo_factor"
)
# Fitted independently on 312 recorded job_history scores before the server
# formula was known (docs/PRIORITY.md): identical to ALCF_EXACT for walltime <= 6 h
# (1e5 / 21600**3 = 0.99e-8 ~ (1/1e4)**2) and missing only the 6-12 h clamp.
ALCF_FITTED = (
    "base_score + score_boost"
    " + enable_wfp * project_priority * nodes / total_nodes * (eligible_s / 1e4) ** 2"
    " + enable_fifo * eligible_h"
    " + enable_backfill * min(backfill_max, eligible_s / backfill_factor)"
)
ALCF_WFP = ALCF_EXACT

CANDIDATES = {
    "alcf_exact": ALCF_EXACT,
    "alcf_fitted": ALCF_FITTED,
    "fifo": "eligible_s",
    "size_then_fifo": "nodes * 1e6 + eligible_s",
}


class ExprPriority:
    def __init__(self, expr: str = ALCF_EXACT, total_nodes: int = 10_624):
        self.expr = expr
        self.total_nodes = float(total_nodes)
        self._fn = compile_expr(expr, VARIABLES)

    def score(self, now_h: float, J: JobArrays, idx: np.ndarray,
              eligible_since_h: np.ndarray | None = None) -> np.ndarray:
        since = J.submit_h[idx] if eligible_since_h is None else eligible_since_h
        elig_h = np.maximum(0.0, now_h - since)
        ns = {k: v[idx] for k, v in J.scoring.items()}
        ns.update(eligible_h=elig_h, eligible_s=elig_h * 3600.0,
                  walltime_h=J.walltime_h[idx], nodes=J.nodes[idx].astype(float),
                  total_nodes=self.total_nodes)
        s = np.asarray(self._fn(ns), float)
        if s.shape != elig_h.shape:
            s = np.broadcast_to(s, elig_h.shape).astype(float)
        return s
