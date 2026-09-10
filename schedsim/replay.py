"""Replay validation: push the REAL trace through the simulated scheduler and
compare simulated waits / utilisation with what Aurora actually did.

This calibrates the scheduler independently of any workload model. The
generator is not involved: jobs, sizes, walltimes, runtimes and submit times
are the recorded ones. What the simulator decides is only WHEN each job starts.

Population for a window [t0, t1):
  running_at_t0  start < t0 < end       pre-placed on the machine (not scheduled)
  queued_at_t0   submit < t0 <= start   pending at t0 with their true submit time
  arrivals       t0 <= submit < t1
Comparison population: arrivals with t0 + warmup <= submit <= t1 - cooldown.

Out-of-model factors (reported, not modelled): user holds, dependencies,
requeues (excluded via run_count), non-exclusive debug nodes, admin actions.
"""
from __future__ import annotations

import json
import pathlib
import re
import time

import numpy as np
import pandas as pd

from . import metrics as M
from .config import ReplayConfig
from .engine import Engine, SimResult
from .jobs import SCORING_COLUMNS, SCORING_DEFAULTS
from .machine import DownWindow, Machine
from .menu import Menu
from .priority import CANDIDATES, ExprPriority

PROGRAM_MAP = {"INCITE": "INCITE", "ALCC": "ALCC", "Discretionary": "DD"}


# ---------------------------------------------------------------------------
# population
# ---------------------------------------------------------------------------

def load_trace(trace_dir: str):
    d = pathlib.Path(trace_dir)
    na = d / "node_availability.parquet"
    return (pd.read_parquet(d / "jobs.parquet"),
            pd.read_parquet(d / "reservations.parquet"),
            pd.read_parquet(d / "queues.parquet"),
            pd.read_parquet(na) if na.exists() else None)


def build_population(jobs: pd.DataFrame, cfg: ReplayConfig) -> tuple[pd.DataFrame, dict]:
    """Filter the trace to schedulable, trustworthy rows and attach hours
    relative to the window start. Returns (table, notes)."""
    t0 = pd.Timestamp(cfg.window.start); t1 = pd.Timestamp(cfg.window.end)
    js = cfg.jobs
    df = jobs
    notes = {"n_trace_rows": int(len(df))}
    base = df[df.start_time.notna() & df.end_time.notna() & (df.nodes > 0)]
    # overlap with the window at all
    base = base[(base.submit_time < t1) & (base.end_time > t0)]
    notes["n_window_overlap"] = int(len(base))
    rx = re.compile(js.exclude_queue_regex) if js.exclude_queue_regex else None
    is_res = base.queue.astype(str).map(lambda q: bool(rx.match(q))) if rx else pd.Series(False, index=base.index)
    excl = base.queue.isin(list(js.exclude_queues))
    requeued = base.run_count.fillna(1) > js.max_run_count
    nh = (base.nodes * base.runtime_h)
    notes["dropped_reservation_queue_jobs"] = int(is_res.sum())
    notes["dropped_reservation_queue_nh_frac"] = float(nh[is_res].sum() / max(nh.sum(), 1e-9))
    notes["dropped_requeued_jobs"] = int((requeued & ~is_res).sum())
    notes["dropped_requeued_nh_frac"] = float(nh[requeued & ~is_res].sum() / max(nh.sum(), 1e-9))
    keep = base[~is_res & ~excl & ~requeued].copy()

    h = lambda ts: (ts - t0).dt.total_seconds() / 3600.0
    keep["qtime_h"] = h(keep.submit_time)
    # arrival = PBS etime (eligible after holds / dependencies) when known; the
    # observed wait is still measured from qtime, and so is the simulated one.
    if cfg.jobs.arrival == "etime" and "eligible_time_ts" in keep.columns:
        et = h(keep.eligible_time_ts)
        keep["submit_h"] = np.where(et.notna() & (et > keep.qtime_h), et, keep.qtime_h)
    else:
        keep["submit_h"] = keep.qtime_h
    keep["obs_start_h"] = h(keep.start_time)
    keep["obs_end_h"] = h(keep.end_time)
    keep["obs_wait_h"] = keep.obs_start_h - keep.qtime_h
    notes_dep = int(keep.get("has_depend", pd.Series(False, index=keep.index)).sum())
    keep["runtime_h"] = np.minimum(keep.runtime_h.clip(lower=0.0),
                                   keep.walltime_h + js.walltime_grace_h)
    keep = keep[keep.walltime_h > 0]

    running0 = (keep.obs_start_h < 0) & (keep.obs_end_h > 0)
    queued0 = (keep.submit_h < 0) & (keep.obs_start_h >= 0)
    arrivals = keep.submit_h >= 0
    keep = keep[running0 | queued0 | arrivals].copy()
    keep["initial_start_h"] = np.where(running0[keep.index], keep.obs_start_h, np.nan)
    keep["role"] = np.select([running0[keep.index], queued0[keep.index]],
                             ["running_at_t0", "queued_at_t0"], "arrival")
    notes.update(n_running_at_t0=int(running0.sum()), n_queued_at_t0=int(queued0.sum()),
                 n_arrivals=int(arrivals.sum()), n_with_depend=notes_dep,
                 n_etime_after_qtime=int((keep.submit_h > keep.qtime_h + 1 / 60).sum()))

    tbl = pd.DataFrame({
        "job_id": keep.job_id.astype(str).to_numpy(),
        "queue": keep.queue.astype(str).to_numpy(),
        "project": keep.project.astype(str).fillna("?").to_numpy(),
        "user": keep.owner.astype(str).fillna("?").to_numpy(),
        "program": keep.allocation_type.map(PROGRAM_MAP).fillna("other").astype(str).to_numpy(),
        "nodes": keep.nodes.astype(int).to_numpy(),
        "walltime_h": keep.walltime_h.to_numpy(float),
        "runtime_h": keep.runtime_h.to_numpy(float),
        "submit_h": keep.submit_h.to_numpy(float),
        "qtime_h": keep.qtime_h.to_numpy(float),
        "initial_start_h": keep.initial_start_h.to_numpy(float),
        "role": keep.role.to_numpy(),
        "obs_start_h": keep.obs_start_h.to_numpy(float),
        "obs_wait_h": keep.obs_wait_h.to_numpy(float),
        "obs_eligible_h": keep.eligible_h.to_numpy(float),
    })
    for c in SCORING_COLUMNS:
        if c in keep.columns:
            tbl[c] = pd.to_numeric(keep[c], errors="coerce").fillna(SCORING_DEFAULTS[c]).to_numpy(float)
        else:
            tbl[c] = SCORING_DEFAULTS[c]
    return tbl.reset_index(drop=True), notes


def build_machine(tbl: pd.DataFrame, reservations: pd.DataFrame, cfg: ReplayConfig,
                  t1_h: float, node_avail: pd.DataFrame | None = None) -> tuple[Machine, dict]:
    ms = cfg.machine
    t0 = pd.Timestamp(cfg.window.start)
    windows = []
    # usable-node series from node snapshots, if they cover the window. Built on
    # a regular grid: a grid point takes the nearest snapshot only if it is within
    # `snapshot_max_gap_h` (the monitor has multi-day gaps, and holding a mid-
    # maintenance value across a gap would zero the machine for days); otherwise
    # it takes the window's typical usable count. Finally the series is floored
    # at the observed concurrency (jobs + reservation nodes): the machine cannot
    # have had fewer usable nodes than were demonstrably in use.
    series_t = series_up = None
    grid = np.arange(0.0, t1_h + 0.25, 0.25)
    if ms.use_node_snapshots and node_avail is not None and len(node_avail):
        na = node_avail[(node_avail.timestamp >= t0 - pd.Timedelta(days=2))
                        & (node_avail.timestamp <= t0 + pd.Timedelta(hours=t1_h + 200))]
        if len(na) and (na.timestamp.min() - t0) <= pd.Timedelta(hours=ms.snapshot_max_gap_h):
            st = ((na.timestamp - t0).dt.total_seconds() / 3600.0).to_numpy()
            su = na.up_nodes.to_numpy(float)
            i = np.clip(np.searchsorted(st, grid), 1, len(st) - 1)
            near = np.where(np.abs(st[i] - grid) < np.abs(st[i - 1] - grid), i, i - 1)
            gap = np.abs(st[near] - grid)
            typical = float(np.quantile(su[(st >= 0) & (st <= t1_h)], 0.75)) if ((st >= 0) & (st <= t1_h)).any() else float(np.median(su))
            series_t = grid
            series_up = np.where(gap <= ms.snapshot_max_gap_h, su[near], typical)
    if ms.reservations:
        r = reservations[reservations.state.isin(list(ms.reservation_states))
                         & reservations.start_time.notna() & reservations.end_time.notna()
                         & (reservations.nodes > 0)]
        for _, row in r.iterrows():
            st_ = (row.start_time - t0).total_seconds() / 3600.0
            en_ = (row.end_time - t0).total_seconds() / 3600.0
            if en_ <= -1e-9 or st_ >= t1_h + 200:
                continue
            windows.append(DownWindow(float(st_), float(en_), int(row.nodes),
                                      f"{row.reservation_name}:{row.reservation_id}"))
    # observed concurrency (jobs + reservation nodes) -> schedulable estimate
    tl = M.busy_timeline(tbl.obs_start_h.to_numpy(), (tbl.obs_start_h + tbl.runtime_h).to_numpy(),
                         tbl.nodes.to_numpy(), 0.0, t1_h, 0.25)
    resv = np.zeros(len(tl))
    for w in windows:
        resv += np.where((tl.t_h >= w.start_h) & (tl.t_h < w.end_h), w.nodes, 0)
    conc = tl.busy_nodes.to_numpy() + resv
    q = float(np.quantile(conc, ms.auto_quantile))
    if series_t is not None:
        # floor at what was observed in use (same 15-min grid as `tl`)
        conc_grid = np.interp(series_t, tl.t_h.to_numpy(), conc, left=conc[0], right=conc[-1])
        series_up = np.maximum(series_up, np.ceil(conc_grid))
    if ms.schedulable_nodes == "auto":
        sched = ms.total_nodes if series_t is not None else int(min(ms.total_nodes, round(q) + ms.auto_margin_nodes))
    else:
        sched = int(ms.schedulable_nodes)
    machine = Machine(ms.total_nodes, ms.reportable_nodes, sched, windows,
                      series_t=series_t, series_up=series_up)
    notes = {"schedulable_nodes": sched, "availability_source": "node_snapshots" if series_t is not None else "constant",
             "mean_available_nodes": machine.mean_available(0.0, t1_h),
             "observed_concurrency_quantile": q,
             "observed_concurrency_max": float(conc.max()),
             "observed_busy_mean_jobs_only": float(tl.busy_nodes.mean()),
             "n_reservation_windows": len(windows),
             "reservation_node_hours": float(sum(max(0.0, min(w.end_h, t1_h) - max(w.start_h, 0.0)) * w.nodes
                                                 for w in windows))}
    return machine, notes


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------

def compare(res: SimResult, cfg: ReplayConfig, t1_h: float) -> tuple[pd.DataFrame, dict]:
    df = res.jobs
    lo = cfg.window.warmup_days * 24.0
    hi = t1_h - cfg.window.cooldown_h
    pop = df[(df.role == "arrival") & (df.submit_h >= lo) & (df.submit_h <= hi)].copy()
    # censored simulated waits: lower bound = horizon - submit
    pop["sim_wait_h"] = pop.start_h - pop.qtime_h
    pop["sim_wait_lb_h"] = np.where(pop.started, pop.sim_wait_h, t1_h - pop.qtime_h)
    pop["sim_censored"] = ~pop.started

    rows = []
    groups = [("ALL", pop)]
    menu_q = set(cfg.jobs.menu_queues)
    for qn in cfg.jobs.menu_queues:
        g = pop[pop.queue == qn]
        if len(g):
            groups.append((qn, g))
    other = pop[~pop.queue.isin(menu_q)]
    if len(other):
        groups.append(("other_queues", other))
    for name, g in groups:
        c = M.compare_distributions(g.obs_wait_h.to_numpy(), g.sim_wait_lb_h.to_numpy())
        c.update(group=name, n_jobs=int(len(g)), n_sim_censored=int(g.sim_censored.sum()),
                 node_hours=float((g.nodes * g.runtime_h).sum()),
                 obs_mean_eligible_h=float(g.obs_eligible_h.mean()))
        rows.append(c)
    table = pd.DataFrame(rows).set_index("group")

    # utilisation: observed vs simulated over the whole window, vs reportable nodes
    m = res.machine
    nodes = df.nodes.to_numpy()
    obs_u = M.utilization(df.obs_start_h.to_numpy(), (df.obs_start_h + df.runtime_h).to_numpy(), nodes, lo, t1_h, m.reportable_nodes)
    sim_u = M.utilization(df.start_h.to_numpy(), df.end_h.to_numpy(), nodes, lo, t1_h, m.reportable_nodes)
    summary = {
        "n_compared": int(len(pop)),
        "n_sim_censored": int(pop.sim_censored.sum()),
        "obs_util_reportable": obs_u, "sim_util_reportable": sim_u,
        "obs_node_hours": float(M.utilization(df.obs_start_h.to_numpy(), (df.obs_start_h + df.runtime_h).to_numpy(), nodes, lo, t1_h, 1.0) * (t1_h - lo)),
        "sim_node_hours": float(M.utilization(df.start_h.to_numpy(), df.end_h.to_numpy(), nodes, lo, t1_h, 1.0) * (t1_h - lo)),
        "ks_all": float(table.loc["ALL", "ks"]),
        "log2_ratio_p50_all": float(table.loc["ALL", "log2_ratio_p50"]),
        # fit score: mean over menu queues (n >= 30) of KS + |log2(sim mean / obs mean)|
        "fit_score": float(np.nanmean([abs(r.log2_ratio_mean) + r.ks for n, r in table.iterrows()
                                       if n in menu_q and r.n >= 30])) if len(table) else np.nan,
    }
    return table, summary


# ---------------------------------------------------------------------------
# plots (static PNG; observed = blue, simulated = orange)
# ---------------------------------------------------------------------------

_OBS, _SIM, _SURF, _INK, _INK2 = "#2a78d6", "#eb6834", "#fcfcfb", "#0b0b0b", "#52514e"


def _style(ax):
    ax.set_facecolor(_SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c3")
    ax.tick_params(colors=_INK2, labelsize=8)
    ax.grid(alpha=0.25, color="#c9c8c3", linewidth=0.6)
    ax.title.set_color(_INK)


def plot_wait_cdfs(res: SimResult, cfg: ReplayConfig, t1_h: float, out: pathlib.Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = res.jobs
    lo = cfg.window.warmup_days * 24.0; hi = t1_h - cfg.window.cooldown_h
    pop = df[(df.role == "arrival") & (df.submit_h >= lo) & (df.submit_h <= hi)]
    qs = [q for q in cfg.jobs.menu_queues if (pop.queue == q).sum() >= 30]
    if not qs:
        return
    ncol = min(3, len(qs)); nrow = int(np.ceil(len(qs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
    fig.patch.set_facecolor(_SURF)
    for ax, q in zip(axes.flat, qs):
        g = pop[pop.queue == q]
        o = np.sort(g.obs_wait_h.to_numpy()); s = np.sort(np.where(g.started, g.start_h - g.qtime_h, t1_h - g.qtime_h))
        y = np.arange(1, len(o) + 1) / len(o)
        ax.step(np.maximum(o, 1 / 60), y, color=_OBS, lw=2, label="observed")
        ax.step(np.maximum(s, 1 / 60), y, color=_SIM, lw=2, label="simulated")
        ax.set_xscale("log"); ax.set_ylim(0, 1)
        ax.set_title(f"{q}  (n={len(g):,})", fontsize=10, loc="left")
        ax.set_xlabel("wait (h, log)", fontsize=8, color=_INK2); ax.set_ylabel("CDF", fontsize=8, color=_INK2)
        _style(ax)
    for ax in axes.flat[len(qs):]:
        ax.axis("off")
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Queue wait: observed vs simulated (arrivals in comparison window)", fontsize=11, color=_INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=130, facecolor=_SURF); plt.close(fig)


def plot_utilization(res: SimResult, cfg: ReplayConfig, t1_h: float, out: pathlib.Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = res.jobs; m = res.machine; nodes = df.nodes.to_numpy()
    obs = M.busy_timeline(df.obs_start_h.to_numpy(), (df.obs_start_h + df.runtime_h).to_numpy(), nodes, 0, t1_h, 6.0)
    sim = M.busy_timeline(df.start_h.to_numpy(), df.end_h.to_numpy(), nodes, 0, t1_h, 6.0)
    fig, ax = plt.subplots(figsize=(11, 3.6)); fig.patch.set_facecolor(_SURF)
    days = obs.t_h / 24.0
    ax.plot(days, obs.busy_nodes, color=_OBS, lw=2, label="observed")
    ax.plot(days, sim.busy_nodes, color=_SIM, lw=2, label="simulated")
    ax.axhline(m.schedulable_nodes, color="#c9c8c3", lw=1, ls="--")
    ax.text(days.iloc[-1], m.schedulable_nodes, f" schedulable {m.schedulable_nodes:,}", fontsize=8, color=_INK2, va="center")
    for w in m.windows:
        if w.end_h > 0 and w.start_h < t1_h and w.nodes >= 0.5 * m.schedulable_nodes:
            ax.axvspan(max(w.start_h, 0) / 24, min(w.end_h, t1_h) / 24, color="#e9e8e4", zorder=0)
    ax.set_xlabel(f"days since {cfg.window.start}", fontsize=9, color=_INK2)
    ax.set_ylabel("busy nodes (6 h mean)", fontsize=9, color=_INK2)
    ax.set_title("Busy nodes: observed vs simulated (shaded = full-machine reservations)", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower left"); _style(ax)
    fig.tight_layout(); fig.savefig(out, dpi=130, facecolor=_SURF); plt.close(fig)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def _resolve_expr(name_or_expr: str) -> tuple[str, str]:
    if name_or_expr in CANDIDATES:
        return name_or_expr, CANDIDATES[name_or_expr]
    return "custom", name_or_expr


def run_replay(cfg: ReplayConfig, verbose: bool = True) -> dict:
    out = pathlib.Path(cfg.output.dir); out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    jobs, reservations, queues, node_avail = load_trace(cfg.trace_dir)
    t1_h = (pd.Timestamp(cfg.window.end) - pd.Timestamp(cfg.window.start)).total_seconds() / 3600.0
    tbl, notes = build_population(jobs, cfg)
    machine, mnotes = build_machine(tbl, reservations, cfg, t1_h, node_avail)
    menu = Menu.from_queue_table(queues)
    tbl["queue_priority"] = [menu[q].queue_priority if q in menu else 0.0 for q in tbl.queue]
    if verbose:
        print(f"[{cfg.name}] window {cfg.window.start} -> {cfg.window.end} ({t1_h/24:.0f} d), "
              f"{len(tbl):,} jobs (running@t0 {notes['n_running_at_t0']}, queued@t0 "
              f"{notes['n_queued_at_t0']}, arrivals {notes['n_arrivals']:,}); "
              f"availability={mnotes['availability_source']} mean {mnotes['mean_available_nodes']:.0f} nodes "
              f"(obs conc q{cfg.machine.auto_quantile}={mnotes['observed_concurrency_quantile']:.0f}), "
              f"{mnotes['n_reservation_windows']} reservation windows")

    exprs = [_resolve_expr(cfg.priority.expr)] + [_resolve_expr(c) for c in cfg.priority.candidates
                                                   if _resolve_expr(c) != _resolve_expr(cfg.priority.expr)]
    results = {}
    for i, (name, expr) in enumerate(exprs):
        label = name if name != "custom" else f"custom{i}"
        t0 = time.time()
        eng = Engine(machine, menu, cfg.scheduler, ExprPriority(expr, machine.total_nodes))
        res = eng.run(tbl, t1_h)
        table, summary = compare(res, cfg, t1_h)
        summary.update(priority=label, expr=expr, runtime_s=round(time.time() - t0, 1),
                       n_passes=res.n_passes, mean_available_nodes=mnotes["mean_available_nodes"])
        results[label] = {"table": table, "summary": summary, "res": res}
        if verbose:
            print(f"  [{label}] {summary['runtime_s']}s  util obs {summary['obs_util_reportable']*100:.1f}% "
                  f"sim {summary['sim_util_reportable']*100:.1f}%  KS(all)={summary['ks_all']:.3f}  "
                  f"log2(p50 sim/obs)={summary['log2_ratio_p50_all']:+.2f}  fit={summary['fit_score']:.3f}")
        table.to_csv(out / f"compare_{label}.csv")
        if cfg.output.write_jobs:
            res.jobs.drop(columns=SCORING_COLUMNS).to_parquet(out / f"jobs_{label}.parquet", index=False)
        if cfg.output.plots:
            plot_wait_cdfs(res, cfg, t1_h, out / f"wait_cdf_{label}.png")
            plot_utilization(res, cfg, t1_h, out / f"utilization_{label}.png")

    ranking = pd.DataFrame([r["summary"] for r in results.values()]).sort_values("fit_score")
    ranking.drop(columns=["expr"]).to_csv(out / "ranking.csv", index=False)
    manifest = {"config": cfg.to_dict(), "science_hash": cfg.science_hash(),
                "population": notes, "machine": mnotes,
                "results": [ {k: v for k, v in r["summary"].items()} for r in results.values()],
                "elapsed_s": round(time.time() - t_start, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    _write_report(out, cfg, notes, mnotes, results, ranking, t1_h)
    if verbose:
        print(f"  -> {out}/  (REPORT.md, ranking.csv, compare_*.csv, plots)")
    return manifest


def _write_report(out, cfg, notes, mnotes, results, ranking, t1_h):
    L = [f"# Replay validation: {cfg.name}", "",
         f"Window **{cfg.window.start} -> {cfg.window.end}** ({t1_h/24:.0f} days); comparison excludes the first "
         f"{cfg.window.warmup_days:g} days (warm-up) and the last {cfg.window.cooldown_h:g} h (right-censoring).",
         f"Science hash `{cfg.science_hash()}`.", "",
         "## Population", "",
         f"- {notes['n_window_overlap']:,} trace jobs overlap the window; replayed {notes['n_running_at_t0']} running at t0, "
         f"{notes['n_queued_at_t0']} queued at t0, {notes['n_arrivals']:,} arrivals.",
         f"- Dropped {notes['dropped_reservation_queue_jobs']:,} reservation-queue jobs "
         f"({notes['dropped_reservation_queue_nh_frac']*100:.1f}% of node-hours); their nodes are removed via "
         f"{mnotes['n_reservation_windows']} reservation windows ({mnotes['reservation_node_hours']:,.0f} node-h).",
         f"- Arrival time = PBS etime (after holds/dependencies): {notes['n_etime_after_qtime']:,} jobs became eligible "
         f"more than a minute after submission ({notes['n_with_depend']:,} carry a `depend` attribute). Waits are still measured from submission.",
         f"- Dropped {notes['dropped_requeued_jobs']:,} requeued jobs (run_count > {cfg.jobs.max_run_count}; "
         f"{notes['dropped_requeued_nh_frac']*100:.1f}% of node-hours) because their recorded start is the first attempt.",
         f"- Availability: **{mnotes['availability_source']}**, mean {mnotes['mean_available_nodes']:,.0f} usable nodes over the window "
         f"(cap {mnotes['schedulable_nodes']:,}; observed peak concurrency {mnotes['observed_concurrency_max']:,.0f}). "
         f"Utilisation is reported against {cfg.machine.reportable_nodes:,} reportable nodes.",
         "", "## Priority formulas ranked by fit (lower is better)", "",
         "fit = mean over menu queues with n >= 30 of KS distance + |log2(sim mean wait / obs mean wait)|.", ""]
    cols = ["priority", "fit_score", "ks_all", "log2_ratio_p50_all", "obs_util_reportable", "sim_util_reportable", "n_sim_censored", "runtime_s"]
    L.append(ranking[cols].to_markdown(index=False, floatfmt=".3f"))
    for label, r in results.items():
        t = r["table"].copy()
        L += ["", f"## {label}: `{r['summary']['expr']}`", "",
              f"![wait cdf](wait_cdf_{label}.png)", "", f"![utilization](utilization_{label}.png)", ""]
        show = t[["n", "n_sim_censored", "obs_p50_h", "sim_p50_h", "obs_p90_h", "sim_p90_h", "obs_mean_h", "sim_mean_h", "ks", "spearman"]]
        L.append(show.to_markdown(floatfmt=".2f"))
    L += ["", "## Reading this", "",
          "- `spearman` is the rank correlation between observed and simulated wait for the SAME jobs; it tests whether the "
          "formula orders jobs the way PBS did, independent of level.",
          "- `log2_ratio_p50` > 0 means the simulator makes jobs wait longer than reality; < 0 shorter.",
          "- Out-of-model factors: user holds and dependencies (observed eligible time < wait), admin holds, non-exclusive debug nodes, "
          "jobs deleted before running (absent from the trace), score_boost changes during the window.", ""]
    (out / "REPORT.md").write_text("\n".join(L))
