"""Metrics computed from a job table with start_h / end_h.

Everything here is exact over intervals (no Δt sampling). Utilisation is
reported against `reportable_nodes` (DOE denominator) and against
`schedulable_nodes` (what the scheduler could actually fill).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# busy-node timeline
# ---------------------------------------------------------------------------

def busy_timeline(start: np.ndarray, end: np.ndarray, nodes: np.ndarray,
                  t0: float, t1: float, dt_h: float = 1.0) -> pd.DataFrame:
    """Mean busy nodes per bin of width dt_h over [t0, t1), exact integral of the
    piecewise-constant busy(t) built from job intervals."""
    ok = ~np.isnan(start) & ~np.isnan(end)
    s = np.clip(start[ok], t0, t1); e = np.clip(end[ok], t0, t1); n = nodes[ok].astype(float)
    keep = e > s
    s, e, n = s[keep], e[keep], n[keep]
    ev_t = np.concatenate([s, e]); ev_d = np.concatenate([n, -n])
    o = np.argsort(ev_t, kind="stable"); ev_t = ev_t[o]; ev_d = ev_d[o]
    busy_after = np.cumsum(ev_d)
    # cumulative node-hours C(t) is piecewise linear; C at each event time:
    seg_len = np.diff(np.concatenate([[t0], ev_t]))
    busy_before = np.concatenate([[0.0], busy_after[:-1]])
    C_ev = np.cumsum(seg_len * busy_before)              # C(ev_t[i]) just before event i

    def C(t):
        t = np.asarray(t, float)
        i = np.searchsorted(ev_t, t, side="right")        # events strictly before/at t
        base = np.where(i > 0, C_ev[np.maximum(i - 1, 0)], 0.0)
        last_t = np.where(i > 0, ev_t[np.maximum(i - 1, 0)], t0)
        rate = np.where(i > 0, busy_after[np.maximum(i - 1, 0)], 0.0)
        return base + rate * (t - last_t)

    edges = np.arange(t0, t1 + 1e-9, dt_h)
    if edges[-1] < t1 - 1e-9:
        edges = np.append(edges, t1)
    nh = np.diff(C(edges))
    width = np.diff(edges)
    return pd.DataFrame({"t_h": edges[:-1], "busy_nodes": nh / width, "node_hours": nh})


def utilization(start, end, nodes, t0, t1, denom_nodes: float) -> float:
    tl = busy_timeline(start, end, nodes, t0, t1, dt_h=(t1 - t0))
    return float(tl["node_hours"].sum() / ((t1 - t0) * denom_nodes))


# ---------------------------------------------------------------------------
# wait statistics
# ---------------------------------------------------------------------------

QS = (0.5, 0.75, 0.9, 0.95, 0.99)


def wait_summary(wait_h: np.ndarray, censored_lower_h: np.ndarray | None = None) -> dict:
    """Summary of waits. `censored_lower_h` gives, for jobs that never started,
    the lower bound on their wait (horizon - submit). Two views:
      started_*     : over started jobs only (optimistic under saturation)
      lb_*          : censored jobs included at their lower bound (a lower bound
                      on the true percentiles; honest under saturation)
    """
    w = np.asarray(wait_h, float); w = w[~np.isnan(w)]
    out = {"n_started": int(len(w)), "n_censored": 0}
    if len(w):
        out.update(started_mean_h=float(w.mean()),
                   **{f"started_p{int(q*100)}_h": float(np.quantile(w, q)) for q in QS})
    else:
        out.update(started_mean_h=np.nan, **{f"started_p{int(q*100)}_h": np.nan for q in QS})
    if censored_lower_h is not None:
        c = np.asarray(censored_lower_h, float); c = c[~np.isnan(c)]
        out["n_censored"] = int(len(c))
        all_w = np.concatenate([w, c])
        if len(all_w):
            out.update(**{f"lb_p{int(q*100)}_h": float(np.quantile(all_w, q)) for q in QS})
    return out


def bounded_slowdown(wait_h, runtime_h, tau_h=10.0 / 60.0) -> np.ndarray:
    return np.maximum(1.0, (wait_h + runtime_h) / np.maximum(runtime_h, tau_h))


# ---------------------------------------------------------------------------
# distribution comparison (replay validation)
# ---------------------------------------------------------------------------

def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(np.asarray(a, float)); b = np.sort(np.asarray(b, float))
    if len(a) == 0 or len(b) == 0:
        return np.nan
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side="right") / len(a)
    fb = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(fa - fb)))


def compare_distributions(obs: np.ndarray, sim: np.ndarray) -> dict:
    """Paired-population comparison of observed vs simulated waits (hours)."""
    obs = np.asarray(obs, float); sim = np.asarray(sim, float)
    m = ~np.isnan(obs) & ~np.isnan(sim)
    obs, sim = obs[m], sim[m]
    out = {"n": int(len(obs))}
    if len(obs) == 0:
        return out
    for q in (0.5, 0.9, 0.95):
        o, s = float(np.quantile(obs, q)), float(np.quantile(sim, q))
        out[f"obs_p{int(q*100)}_h"] = o; out[f"sim_p{int(q*100)}_h"] = s
    out["obs_mean_h"] = float(obs.mean()); out["sim_mean_h"] = float(sim.mean())
    out["ks"] = ks_statistic(obs, sim)
    eps = 1.0 / 60.0  # one minute floor so ratios of tiny waits stay finite
    out["log2_ratio_p50"] = float(np.log2((out["sim_p50_h"] + eps) / (out["obs_p50_h"] + eps)))
    out["log2_ratio_mean"] = float(np.log2((out["sim_mean_h"] + eps) / (out["obs_mean_h"] + eps)))
    # per-job agreement (same jobs, so this is meaningful): Spearman rank corr
    if len(obs) > 2:
        ro = pd.Series(obs).rank().to_numpy(); rs = pd.Series(sim).rank().to_numpy()
        out["spearman"] = float(np.corrcoef(ro, rs)[0, 1])
    return out
