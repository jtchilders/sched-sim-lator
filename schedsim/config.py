"""Typed, strict configuration for replay runs.

Unknown keys anywhere raise, so a typo cannot silently fall back to a default.
Each section hashes separately: a result is tagged with the hash of the parts
that can change its numbers (window, jobs filter, machine, scheduler, priority),
never with output paths.
"""
from __future__ import annotations

import dataclasses as dc
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import typing

import yaml

from .engine import SchedulerSpec
from .priority import ALCF_WFP


def _strict(cls, d: dict | None, where: str):
    d = dict(d or {})
    names = {f.name for f in dc.fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(names)}")
    hints = typing.get_type_hints(cls)
    kw = {}
    for f in dc.fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        t = hints.get(f.name)
        if isinstance(t, type) and dc.is_dataclass(t):
            v = _strict(t, v, f"{where}.{f.name}")
        kw[f.name] = v
    return cls(**kw)


@dataclass(frozen=True)
class WindowSpec:
    start: str = "2026-03-01"
    end: str = "2026-04-30"
    warmup_days: float = 3.0     # jobs submitted before t0+warmup are excluded from comparison
    cooldown_h: float = 48.0     # ... and within cooldown of the end (right-censoring)


@dataclass(frozen=True)
class JobsSpec:
    exclude_queue_regex: str = r"^[RMS]\d+$"   # reservation queues: nodes handled as windows
    exclude_queues: tuple = ()
    max_run_count: int = 1        # requeued jobs have unreliable start times
    menu_queues: tuple = ("small", "medium", "large", "capacity", "tiny",
                          "backfill-small", "backfill-medium", "backfill-large",
                          "backfill-tiny", "debug", "debug-scaling")
    walltime_grace_h: float = 0.1  # PBS lets a job run slightly past walltime
    arrival: str = "etime"         # etime (after holds/dependencies) | qtime (raw submit)


@dataclass(frozen=True)
class MachineSpec:
    total_nodes: int = 10_624
    reportable_nodes: int = 9_600
    schedulable_nodes: Any = "auto"   # int, or "auto" = quantile of observed concurrency
    auto_quantile: float = 0.995
    auto_margin_nodes: int = 0
    reservations: bool = True
    use_node_snapshots: bool = True   # usable-node series from node_availability.parquet
    snapshot_max_gap_h: float = 3.0   # trust a snapshot only this far; fill gaps with typical
    reservation_states: tuple = ("COMPLETED", "RUNNING", "RUNNING_SHORT",
                                 "CONFIRMED", "CONFIRMED_SHORT", "DEGRADED")


@dataclass(frozen=True)
class PrioritySpec:
    expr: str = "alcf_fitted"        # a CANDIDATES name or a literal expression
    candidates: tuple = ()        # extra names/expressions to run and rank


@dataclass(frozen=True)
class OutputSpec:
    dir: str = "results/replay/default"
    plots: bool = True
    write_jobs: bool = True


@dataclass(frozen=True)
class ReplayConfig:
    name: str = "replay"
    trace_dir: str = "data/trace"
    window: WindowSpec = field(default_factory=WindowSpec)
    jobs: JobsSpec = field(default_factory=JobsSpec)
    machine: MachineSpec = field(default_factory=MachineSpec)
    scheduler: SchedulerSpec = field(default_factory=SchedulerSpec)
    priority: PrioritySpec = field(default_factory=PrioritySpec)
    output: OutputSpec = field(default_factory=OutputSpec)

    @staticmethod
    def from_dict(d: dict) -> "ReplayConfig":
        d = dict(d or {})
        def _tup(v):
            return tuple(_tup(x) for x in v) if isinstance(v, list) else v
        for k in ("jobs", "machine", "priority", "window", "scheduler"):
            if k in d and d[k] is not None:
                d[k] = {kk: _tup(v) for kk, v in d[k].items()}
        cfg = _strict(ReplayConfig, d, "replay")
        if cfg.scheduler.backfill_depth < 0 or cfg.scheduler.cycle_h <= 0:
            raise ValueError("scheduler.backfill_depth must be >= 0 and cycle_h > 0")
        modes = ("backfill_all", "backfill_flagged", "strict", "family_flagged", "queue_flagged", "strict_groups")
        if cfg.scheduler.ordering not in modes:
            raise ValueError(f"scheduler.ordering must be one of {modes}, got {cfg.scheduler.ordering!r}")
        return cfg

    @staticmethod
    def from_yaml(path: str) -> "ReplayConfig":
        with open(path) as f:
            return ReplayConfig.from_dict(yaml.safe_load(f) or {})

    def to_dict(self) -> dict:
        return _asdict(self)

    def science_hash(self) -> str:
        """Hash of everything that changes results (not name/output)."""
        d = self.to_dict()
        for k in ("name", "output"):
            d.pop(k, None)
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _asdict(obj):
    if dc.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _asdict(getattr(obj, f.name)) for f in dc.fields(obj)}
    if isinstance(obj, (tuple, list)):
        return [_asdict(v) for v in obj]
    return obj
