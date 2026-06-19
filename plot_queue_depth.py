#!/usr/bin/env python3
"""
Generate queue depth in system-hours (pending node-hours) over time.

For each sample point, compute:
  queue_depth_nh(t) = sum over pending jobs of (nodes_j × hours_waiting_j)

This captures both the number and size of waiting jobs.
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
SAMPLE_DT_H = 0.25

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
QUEUE_ORDER = ['capacity', 'small', 'medium', 'large']


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


def compute_queue_depth_nh(jobs, duration_h):
    """
    At each sample time t, for each queue compute:
      sum of (nodes_j * (t - submit_time_j)) for all jobs j that are
      submitted before t and not yet started (or never started) at t.
    """
    times = np.arange(0, duration_h, SAMPLE_DT_H)
    per_queue = {q: np.zeros(len(times)) for q in QUEUE_ORDER}

    for j in jobs:
        submit = j.submit_time_h
        # Job is pending from submit until start (or end of sim if never started)
        if j.start_time_h is not None:
            pending_end = j.start_time_h
        else:
            pending_end = duration_h

        if pending_end <= submit:
            continue  # started instantly

        for i, t in enumerate(times):
            if t >= submit and t < pending_end:
                wait_so_far = t - submit
                per_queue[j.queue][i] += j.nodes * wait_so_far

    return times, per_queue


def smooth(arr, window=16):
    return pd.Series(arr).rolling(window, center=True, min_periods=1).mean().to_numpy()


def plot_queue_depth(outdir):
    """Generate queue depth plots."""
    print("Running simulation at 25% λ...")
    jobs_25, dur_h = run_sim(0.25)
    print("Running simulation at 100% λ...")
    jobs_100, dur_h = run_sim(1.0)

    print("Computing queue depth (system-hours)...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Queue Depth in System-Hours (Pending Node-Hours) Over Time',
                 fontsize=18, y=0.98)

    all_main_peaks = []
    for col, (jobs, scale_label, scale_pct) in enumerate([
        (jobs_25, '25% λ', '25%'),
        (jobs_100, '100% λ', '100%'),
    ]):
        print(f"  Computing for {scale_pct}...")
        times, per_queue = compute_queue_depth_nh(jobs, dur_h)
        days = times / 24.0

        # Top row: capacity queue (most interesting variation)
        ax_cap = axes[0, col]
        vals = smooth(per_queue['capacity'])
        ax_cap.fill_between(days, 0, vals, color=QUEUE_COLORS['capacity'],
                            alpha=0.5, linewidth=0)
        ax_cap.plot(days, vals, color=QUEUE_COLORS['capacity'], linewidth=1.2)
        peak = vals.max()
        ax_cap.set_title(f'Capacity Queue — {scale_label}', fontsize=14,
                         color=QUEUE_COLORS['capacity'])
        ax_cap.set_ylabel('System-hours\n(nodes × hours waiting)')
        ax_cap.grid(True)
        ax_cap.text(0.98, 0.95, f'Peak: {peak:,.0f}',
                    transform=ax_cap.transAxes, ha='right', va='top',
                    fontsize=12, color=QUEUE_COLORS['capacity'],
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#112240',
                              edgecolor=QUEUE_COLORS['capacity'], alpha=0.8))

        # Bottom row: other queues (overlaid, not stacked)
        ax_other = axes[1, col]
        max_main_vals = []
        for q in ['large', 'medium', 'small']:
            vals_q = smooth(per_queue[q])
            ax_other.plot(days, vals_q, color=QUEUE_COLORS[q],
                          linewidth=1.5, label=q.capitalize())
            ax_other.fill_between(days, 0, vals_q, color=QUEUE_COLORS[q],
                                  alpha=0.2, linewidth=0)
            max_main_vals.append(vals_q.max())
        ax_other.set_title(f'Main Queues — {scale_label}', fontsize=14)
        ax_other.set_xlabel('Time (days)')
        ax_other.set_ylabel('System-hours')
        ax_other.legend(fontsize=10, loc='upper right')
        ax_other.grid(True)
        all_main_peaks.append(max(max_main_vals))

    # Lock bottom row to shared y-axis
    shared_max = max(all_main_peaks) * 1.15
    axes[1, 0].set_ylim(0, shared_max)
    axes[1, 1].set_ylim(0, shared_max)

    # Also lock top row to shared y-axis
    top_max = max(axes[0, 0].get_ylim()[1], axes[0, 1].get_ylim()[1])
    axes[0, 0].set_ylim(0, top_max)
    axes[0, 1].set_ylim(0, top_max)

    # Format large numbers on y-axes
    from matplotlib.ticker import FuncFormatter
    def fmt_k(x, _):
        if x >= 1e6:
            return f'{x/1e6:.0f}M'
        elif x >= 1e3:
            return f'{x/1e3:.0f}K'
        return f'{x:.0f}'
    for row in axes:
        for ax in row:
            ax.yaxis.set_major_formatter(FuncFormatter(fmt_k))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f'{outdir}/queue_depth_systemhours.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {outdir}/queue_depth_systemhours.png')


if __name__ == '__main__':
    os.makedirs(OUTDIR, exist_ok=True)
    plot_queue_depth(OUTDIR)
    print("\nDone!")
