#!/usr/bin/env python3
"""
Generate stacked utilization-by-queue plots for 25% and 100% λ.
Each queue's node usage is stacked so the total equals overall utilization.
"""
import os
import sys
import copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trace_sampler
from sim import QUEUES, Scheduler, JobGenerator, CAPACITY_POOL_NODES

DB_PATH = "/Users/jchilders/pbs_monitor_aurora.db"
OUTDIR = "results/distributions"
SEED = 42
DURATION_DAYS = 30
TOTAL_NODES = 10_624
SAMPLE_DT_H = 0.25  # 15 min

plt.rcParams.update({
    'figure.facecolor': '#0a1628',
    'axes.facecolor': '#112240',
    'axes.edgecolor': '#8892b0',
    'axes.labelcolor': '#e6f1ff',
    'text.color': '#e6f1ff',
    'xtick.color': '#8892b0',
    'ytick.color': '#8892b0',
    'grid.color': '#1d3461',
    'grid.alpha': 0.5,
    'legend.facecolor': '#112240',
    'legend.edgecolor': '#8892b0',
    'font.size': 13,
})

QUEUE_COLORS = {
    'capacity': '#64ffda',
    'small':    '#57cbff',
    'medium':   '#c4b5fd',
    'large':    '#ff6b8a',
}
QUEUE_ORDER = ['large', 'medium', 'small', 'capacity']  # bottom to top


def run_sim(arrival_scale):
    """Run simulation, return (jobs, duration_h)."""
    rng = np.random.default_rng(SEED)
    import random
    random.seed(SEED)

    duration_h = DURATION_DAYS * 24.0
    fitted = trace_sampler.FittedSampler(db_path=DB_PATH, rng=rng, use_cache=True)

    queues = copy.deepcopy(QUEUES)
    for qc in queues:
        rate = fitted.arrival_rate(qc.name)
        if qc.name == "capacity" and arrival_scale != 1.0:
            rate *= arrival_scale
        qc.arrival_rate_per_h = rate

    gen = JobGenerator(queues, rng, sampler=fitted)
    jobs = gen.generate(duration_h)

    sched = Scheduler(total_nodes=TOTAL_NODES, enable_backfill=True,
                      capacity_pool=512, on_demand_nodes=0)
    sched.run(jobs, duration_h=duration_h, sample_dt_h=SAMPLE_DT_H)

    return jobs, duration_h


def compute_per_queue_utilization(jobs, duration_h):
    """Compute per-queue node counts at each sample point."""
    times = np.arange(0, duration_h, SAMPLE_DT_H)
    # Pre-filter to started jobs
    started = [j for j in jobs if j.start_time_h is not None]

    per_queue = {q: np.zeros(len(times)) for q in QUEUE_ORDER}

    for j in started:
        end = j.end_time_h if j.end_time_h is not None else duration_h
        mask = (times >= j.start_time_h) & (times < end)
        per_queue[j.queue][mask] += j.nodes

    return times, per_queue


def smooth(arr, window=16):
    """Rolling mean with min_periods=1."""
    return pd.Series(arr).rolling(window, center=True, min_periods=1).mean().to_numpy()


def plot_stacked(outdir):
    """Generate stacked utilization plots."""
    print("Running simulation at 25% λ...")
    jobs_25, dur_h = run_sim(0.25)
    print("Running simulation at 100% λ...")
    jobs_100, dur_h = run_sim(1.0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, sharey=True)
    fig.suptitle('Node Utilization by Queue Over Time', fontsize=18, y=0.98)

    for ax, jobs, label in [
        (ax1, jobs_25, '25% of empirical λ  (153 capacity jobs/day)'),
        (ax2, jobs_100, '100% of empirical λ  (610 capacity jobs/day)'),
    ]:
        times, per_queue = compute_per_queue_utilization(jobs, dur_h)
        days = times / 24.0

        # Smooth each queue
        smoothed = {q: smooth(per_queue[q] / TOTAL_NODES * 100) for q in QUEUE_ORDER}

        # Stacked area
        bottoms = np.zeros(len(times))
        for q in QUEUE_ORDER:
            vals = smoothed[q]
            ax.fill_between(days, bottoms, bottoms + vals,
                            color=QUEUE_COLORS[q], alpha=0.7, linewidth=0,
                            label=f'{q.capitalize()}')
            ax.plot(days, bottoms + vals, color=QUEUE_COLORS[q],
                    linewidth=0.5, alpha=0.8)
            bottoms = bottoms + vals

        # Total mean line
        total = sum(smoothed[q] for q in QUEUE_ORDER)
        avg = total.mean()
        ax.axhline(avg, color='#ffb347', linestyle='--', linewidth=1.5,
                   alpha=0.8, label=f'Mean: {avg:.1f}%')

        ax.set_title(label, fontsize=14)
        ax.set_ylabel('Utilization (%)')
        ax.set_ylim(0, 105)
        ax.legend(fontsize=11, loc='upper right', ncol=5)
        ax.grid(True)

    ax2.set_xlabel('Time (days)')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f'{outdir}/utilization_stacked.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {outdir}/utilization_stacked.png')


if __name__ == '__main__':
    os.makedirs(OUTDIR, exist_ok=True)
    plot_stacked(OUTDIR)
    print("\nDone!")
