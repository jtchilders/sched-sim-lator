"""Queue menu: node/walltime bounds, per-queue run limits and scoring defaults.

A `Menu` can be parsed from the PBS `queues` table (replay: the menu that was
actually in force) or from YAML (a hypothetical menu to evaluate). Limits are
what PBS enforces at run time: per-user / per-project running-job caps and an
aggregate node cap across the queue (Aurora's `capacity` queue).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, asdict, field, fields

import numpy as np
import pandas as pd

RESERVATION_QUEUE_RE = re.compile(r"^[RMS]\d+$")

# ALCF scoring-parameter defaults observed in the Aurora trace.
DEFAULT_SCORING = dict(base_score=51.0, enable_wfp=1.0, enable_fifo=0.0,
                       enable_backfill=0.0, wfp_factor=100_000.0,
                       fifo_factor=1800.0, backfill_factor=84_600.0,
                       backfill_max=50.0)


@dataclass(frozen=True)
class Queue:
    name: str
    min_nodes: int = 1
    max_nodes: int = 10_624
    max_walltime_h: float = 24.0
    queue_priority: float = 0.0
    max_run_per_user: float = math.inf
    max_run_per_project: float = math.inf
    max_queued_per_user: float = math.inf     # in queue (queued+running); excess held in routing
    max_queued_per_project: float = math.inf
    max_nodes_total: float = math.inf     # aggregate running nodes across the queue
    base_score: float = DEFAULT_SCORING["base_score"]
    enable_wfp: float = DEFAULT_SCORING["enable_wfp"]
    enable_fifo: float = DEFAULT_SCORING["enable_fifo"]
    enable_backfill: float = DEFAULT_SCORING["enable_backfill"]
    wfp_factor: float = DEFAULT_SCORING["wfp_factor"]
    fifo_factor: float = DEFAULT_SCORING["fifo_factor"]
    backfill_factor: float = DEFAULT_SCORING["backfill_factor"]
    backfill_max: float = DEFAULT_SCORING["backfill_max"]
    routable: bool = True   # eligible as a routing destination for synthetic jobs

    def accepts(self, nodes: int, walltime_h: float) -> bool:
        return (self.min_nodes <= nodes <= self.max_nodes
                and walltime_h <= self.max_walltime_h + 1e-9)


def _nan_to(v, default):
    try:
        return default if v is None or (isinstance(v, float) and math.isnan(v)) else v
    except TypeError:
        return default


class Menu:
    def __init__(self, queues: list[Queue]):
        self.queues: dict[str, Queue] = {q.name: q for q in queues}
        self.names = list(self.queues)
        self._id = {n: i for i, n in enumerate(self.names)}

    # -- construction -------------------------------------------------------
    @classmethod
    def from_queue_table(cls, df: pd.DataFrame, names: list[str] | None = None) -> "Menu":
        """Build from schedsim.trace.extract's queues.parquet. Execution queues
        only; reservation queues (R/M/S-numbers) are dropped."""
        qs = []
        for _, r in df.iterrows():
            n = str(r["name"])
            if r.get("queue_type") != "Execution" or RESERVATION_QUEUE_RE.match(n):
                continue
            if names is not None and n not in names:
                continue
            qs.append(Queue(
                name=n,
                min_nodes=int(_nan_to(r["min_nodes"], 1)),
                max_nodes=int(_nan_to(r["max_nodes"], 10_624)),
                max_walltime_h=float(_nan_to(r["max_walltime_h"], 24.0)),
                queue_priority=float(_nan_to(r["queue_priority"], 0.0)),
                max_run_per_user=float(_nan_to(r["max_run_per_user"], math.inf)),
                max_run_per_project=float(_nan_to(r["max_run_per_project"], math.inf)),
                max_queued_per_user=float(_nan_to(r["max_queued_per_user"], math.inf)),
                max_queued_per_project=float(_nan_to(r["max_queued_per_project"], math.inf)),
                max_nodes_total=float(_nan_to(r["max_nodes_total"], math.inf)),
                base_score=float(_nan_to(r["base_score"], DEFAULT_SCORING["base_score"])),
                enable_wfp=float(_nan_to(r["enable_wfp"], DEFAULT_SCORING["enable_wfp"])),
                enable_fifo=float(_nan_to(r["enable_fifo"], DEFAULT_SCORING["enable_fifo"])),
                enable_backfill=float(_nan_to(r["enable_backfill"], DEFAULT_SCORING["enable_backfill"])),
                routable=bool(r.get("from_route_only", False)),
            ))
        return cls(qs)

    @classmethod
    def from_dict(cls, d: dict) -> "Menu":
        valid = {f.name for f in fields(Queue)}
        qs = []
        for item in d["queues"]:
            unknown = set(item) - valid
            if unknown:
                raise ValueError(f"queue {item.get('name')!r}: unknown field(s) {sorted(unknown)}")
            item = dict(item)
            for k in ("max_run_per_user", "max_run_per_project", "max_queued_per_user",
                      "max_queued_per_project", "max_nodes_total"):
                if item.get(k) in (None, "inf"):
                    item[k] = math.inf
            qs.append(Queue(**item))
        return cls(qs)

    def to_dict(self) -> dict:
        out = []
        for q in self.queues.values():
            d = asdict(q)
            for k, v in d.items():
                if isinstance(v, float) and math.isinf(v):
                    d[k] = None
            out.append(d)
        return {"queues": out}

    # -- lookups ------------------------------------------------------------
    def __contains__(self, name: str) -> bool:
        return name in self.queues

    def __getitem__(self, name: str) -> Queue:
        return self.queues[name]

    def queue_id(self, name: str) -> int:
        return self._id[name]

    def route(self, nodes: int, walltime_h: float) -> str | None:
        """Destination for a synthetic job: the routable queue that accepts it
        (smallest node range wins if several do). None if nothing accepts."""
        cands = [q for q in self.queues.values() if q.routable and q.accepts(nodes, walltime_h)]
        if not cands:
            return None
        return min(cands, key=lambda q: (q.max_nodes - q.min_nodes, q.name)).name

    def limit_arrays(self, queue_names: list[str]) -> dict[str, np.ndarray]:
        """Per-queue-id limit arrays aligned to `queue_names` (unknown queues
        get no limits and default scoring)."""
        def arr(attr, default):
            return np.array([getattr(self.queues[n], attr) if n in self.queues else default
                             for n in queue_names], float)
        return {
            "max_run_per_user": arr("max_run_per_user", math.inf),
            "max_run_per_project": arr("max_run_per_project", math.inf),
            "max_queued_per_user": arr("max_queued_per_user", math.inf),
            "max_queued_per_project": arr("max_queued_per_project", math.inf),
            "max_nodes_total": arr("max_nodes_total", math.inf),
        }
