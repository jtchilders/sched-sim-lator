"""
Genesis Mission synthesis — assumption-driven, no historical data.

Kept deliberately small and config-driven. Three named scenarios preserve the
prior semantics (ai_default / incite_like / bursty_campaign); everything is
parameterized through GenesisConfig so scenarios can be tuned from YAML.
"""
from __future__ import annotations

import numpy as np

from config import GenesisConfig


# Scenario definitions: node_mix and walltime_mix as (value, weight) tuples.
_SCENARIOS = {
    "ai_default": dict(
        node_mix=((1, 0.70), (2, 0.12), (4, 0.08), (8, 0.05),
                  (16, 0.03), (64, 0.015), (256, 0.005)),
        walltime_mix=((168.0, 0.55), (96.0, 0.20), (48.0, 0.12),
                      (24.0, 0.08), (6.0, 0.05)),
        rt_ratio_mean=0.85, rt_ratio_spread=0.12,
    ),
    "incite_like": dict(
        node_mix=((256, 0.30), (512, 0.30), (1024, 0.25), (2048, 0.15)),
        walltime_mix=((6.0, 0.20), (12.0, 0.35), (18.0, 0.25), (24.0, 0.20)),
        rt_ratio_mean=0.90, rt_ratio_spread=0.08,
    ),
    "bursty_campaign": dict(
        node_mix=((64, 0.20), (128, 0.25), (256, 0.30), (512, 0.20), (1024, 0.05)),
        walltime_mix=((6.0, 0.30), (12.0, 0.30), (24.0, 0.25),
                      (48.0, 0.10), (96.0, 0.05)),
        rt_ratio_mean=0.80, rt_ratio_spread=0.15,
    ),
}


def _scenario(gc: GenesisConfig) -> dict:
    if gc.scenario not in _SCENARIOS:
        raise ValueError(f"unknown genesis scenario {gc.scenario!r}; "
                         f"choose from {sorted(_SCENARIOS)}")
    return _SCENARIOS[gc.scenario]


def build_genesis_arrays(gc: GenesisConfig, rng: np.random.Generator,
                         n: int = 20000) -> tuple:
    s = _scenario(gc)
    nvals, nw = zip(*s["node_mix"])
    wvals, ww = zip(*s["walltime_mix"])
    nw = np.array(nw) / sum(nw)
    ww = np.array(ww) / sum(ww)
    nodes = rng.choice(nvals, size=n, p=nw).astype(np.int64)
    wt = rng.choice(wvals, size=n, p=ww).astype(np.float64)
    rt = np.clip(rng.normal(s["rt_ratio_mean"], s["rt_ratio_spread"], size=n),
                 0.05, 1.0)
    return nodes, wt, rt


def genesis_burn(gc: GenesisConfig) -> np.ndarray:
    """Calendar-month ramp: 0 before start, linear to full, 1.0 after,
    normalized to mean 1.0 over active months."""
    mult = np.zeros(12)
    start, full = gc.ramp_start_month, gc.ramp_full_month
    for m in range(1, 13):
        if m < start:
            mult[m - 1] = 0.0
        elif m >= full:
            mult[m - 1] = 1.0
        else:
            mult[m - 1] = (m - start) / max(1, full - start)
    active = mult[mult > 0]
    if active.size and active.mean() > 0:
        mult[mult > 0] = active / active.mean()
    return mult


def genesis_rate_h(gc: GenesisConfig, machine_nodes: int,
                   nodes_arr, wt_arr, rt_arr) -> float:
    """Arrival rate (jobs/h) so annual delivered node-hours ~ share * capacity,
    spread over the active (ramped) fraction of the year."""
    mean_nh = float(np.mean(nodes_arr * wt_arr * rt_arr))
    annual_target = gc.share * machine_nodes * 24 * 365
    burn = genesis_burn(gc)
    active_frac = float(np.count_nonzero(burn)) / 12.0
    jobs_per_year = annual_target / max(mean_nh, 1e-9)
    return jobs_per_year / (365 * 24 * max(active_frac, 1e-9))


def reallocate_shares(programs, genesis: GenesisConfig) -> dict:
    """Apply genesis_from policy; return {program_name: share} summing ~1.0."""
    base = {p.name: p.target_share for p in programs}
    g = genesis.share if genesis.enabled else 0.0
    src = genesis.genesis_from
    out = dict(base)
    if not genesis.enabled:
        return out
    if src in ("proportional", "all"):
        tot = sum(base.values())
        if tot > 0:
            scale = (1.0 - g) / tot
            for k in out:
                out[k] = base[k] * scale
    elif src == "incite":
        out["INCITE"] = max(0.0, base.get("INCITE", 0.0) - g)
    elif src == "dd":
        out["DD"] = max(0.0, base.get("DD", 0.0) - g)
    out["Genesis"] = g
    return out
