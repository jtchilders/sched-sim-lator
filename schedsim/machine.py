"""Node availability over time.

available(t) = min( up(t), schedulable_nodes - reserved(t) )

  up(t)         usable-node series (job-exclusive + free) from node snapshots
                when available; otherwise the constant `schedulable_nodes`.
  reserved(t)   nodes inside PBS reservations / maintenance windows. Kept as
                explicit windows so the backfill profile can see them AHEAD of
                time and drain, as the real scheduler does.

`reportable_nodes` is the DOE accounting denominator for utilisation (Aurora:
9,600); it never affects scheduling.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class DownWindow:
    start_h: float
    end_h: float
    nodes: int
    label: str = ""


@dataclass
class Machine:
    total_nodes: int = 10_624
    reportable_nodes: int = 9_600
    schedulable_nodes: int = 10_624
    windows: list = field(default_factory=list)
    # optional step series: up_nodes holds from series_t[k] until series_t[k+1]
    series_t: np.ndarray | None = None
    series_up: np.ndarray | None = None

    def __post_init__(self):
        self.windows = sorted(self.windows, key=lambda w: w.start_h)
        self._ws = np.array([w.start_h for w in self.windows], float)
        self._we = np.array([w.end_h for w in self.windows], float)
        self._wn = np.array([w.nodes for w in self.windows], float)
        if self.series_t is not None:
            self.series_t = np.asarray(self.series_t, float)
            self.series_up = np.asarray(self.series_up, float)
            o = np.argsort(self.series_t, kind="stable")
            self.series_t, self.series_up = self.series_t[o], self.series_up[o]

    def up_at(self, t: np.ndarray) -> np.ndarray:
        if self.series_t is None or len(self.series_t) == 0:
            return np.full(t.shape, float(self.schedulable_nodes))
        i = np.searchsorted(self.series_t, t, side="right") - 1
        i = np.clip(i, 0, len(self.series_t) - 1)
        return np.minimum(self.series_up[i], self.schedulable_nodes)

    def reserved_at(self, t: np.ndarray) -> np.ndarray:
        if len(self.windows) == 0:
            return np.zeros(t.shape)
        active = (self._ws[None, :] <= t[:, None]) & (t[:, None] < self._we[None, :])
        return active @ self._wn

    def available_at(self, t) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, float))
        avail = np.minimum(self.up_at(t), self.schedulable_nodes - self.reserved_at(t))
        return np.clip(avail, 0, None)

    def breakpoints(self, t_from: float, t_to: float) -> np.ndarray:
        """Times strictly inside (t_from, t_to) where availability may change."""
        parts = []
        if len(self.windows):
            parts += [self._ws, self._we]
        if self.series_t is not None and len(self.series_t):
            parts.append(self.series_t)
        if not parts:
            return np.array([])
        e = np.concatenate(parts)
        return np.unique(e[(e > t_from) & (e < t_to)])

    def mean_available(self, t0: float, t1: float, dt: float = 0.25) -> float:
        t = np.arange(t0, t1, dt)
        return float(self.available_at(t).mean()) if len(t) else float(self.schedulable_nodes)
