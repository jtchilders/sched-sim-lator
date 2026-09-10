"""Job-scoring (priority) models. Higher score runs first.

ALCF runs a custom PBS sort formula whose exact form is not published. The
per-job parameters it uses ARE recorded in every job's Resource_List
(base_score, score_boost, project_priority, enable_wfp/wfp_factor,
enable_fifo/fifo_factor, enable_backfill/backfill_factor/backfill_max), and the
public description says: larger jobs gain priority faster, shorter jobs gain
priority faster, INCITE/ALCC outrank discretionary, negative-balance projects
are demoted. That is the Cobalt "WFP" lineage: (wait / walltime)^3 * size.

ALCF_FITTED below was fitted to the scores PBS actually recorded (job_history.score);
replay validation ranks alternative formulas by how well they reproduce observed waits. Variables available to an
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

# Fitted against 312 recorded (job, score) snapshots from job_history (Feb-Jun
# 2026; R^2 = 0.97 in log space, residual factor ~1.7). See docs/PRIORITY.md.
#   * project_priority MULTIPLIES the WFP term (INCITE/ALCC 25, ALCC-ish 20, DD 2)
#   * the WFP term is QUADRATIC in eligible time and linear in node count;
#     (eligible_s / 1e4)^2 * nodes / total_nodes * project_priority reproduces
#     the recorded constant (1.2e-5 h^-2 per node per priority unit) within 2%
#   * fifo queues (debug) add eligible HOURS; backfill queues add
#     min(backfill_max, eligible_s / backfill_factor)
ALCF_FITTED = (
    "base_score + score_boost"
    " + enable_wfp * project_priority * nodes / total_nodes * (eligible_s / 1e4) ** 2"
    " + enable_fifo * eligible_h"
    " + enable_backfill * min(backfill_max, eligible_s / backfill_factor)"
)
ALCF_WFP = ALCF_FITTED

CANDIDATES = {
    "alcf_fitted": ALCF_FITTED,
    # earlier reconstruction (cubic, additive project priority) kept for comparison
    "alcf_cubic": (
        "base_score + score_boost + project_priority"
        " + enable_wfp * wfp_factor * (eligible_h / walltime_h) ** 3 * nodes / total_nodes"
        " + enable_fifo * eligible_s / fifo_factor"
        " + enable_backfill * min(backfill_max, eligible_s / backfill_factor)"),
    "fifo": "eligible_s",
    "size_then_fifo": "nodes * 1e6 + eligible_s",
}


class ExprPriority:
    def __init__(self, expr: str = ALCF_FITTED, total_nodes: int = 10_624):
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
