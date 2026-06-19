#!/usr/bin/env python3
"""
Generate improved utilization timeline plots for the slides.

- Side-by-side comparison of 25% vs 100% λ
- Rolling average overlay to show trend
- Horizontal mean line
- Matching dark theme
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trace_sampler
from sim import QUEUES, Scheduler, JobGenerator, summarize, CAPACITY_POOL_NODES

DB_PATH = "/Users/jchilders/pbs_monitor_aurora.db"
OUTDIR = "results/distributions"
SEED = 42
DURATION_DAYS = 30
TOTAL_NODES = 10_624

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

ACCENT = '#64ffda'
ACCENT2 = '#57cbff'
PINK = '#ff6b8a'
WARN = '#ffb347'


def run_sim(arrival_scale, duration_days=DURATION_DAYS, seed=SEED):
    """Run a simulation at the given arrival rate scale and return the scheduler."""
    rng = np.random.default_rng(seed)
    import random
    random.seed(seed)

    duration_h = duration_days * 24.0

    fitted = trace_sampler.FittedSampler(db_path=DB_PATH, rng=rng, use_cache=True)

    # Copy queue configs to avoid mutation across runs
    import copy
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
    sched.run(jobs, duration_h=duration_h, sample_dt_h=0.25)

    return sched, jobs, duration_h


def plot_utilization_comparison(outdir):
    """Side-by-side utilization timelines for 25% and 100% λ."""
    print("Running simulation at 25% λ...")
    sched_25, jobs_25, dur_h = run_sim(0.25)
    print("Running simulation at 100% λ...")
    sched_100, jobs_100, dur_h = run_sim(1.0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True, sharey=True)
    fig.suptitle('Node Utilization Over Time: Capacity Queue Arrival Rate Sensitivity',
                 fontsize=18, y=0.98)

    for ax, sched, scale_label, color in [
        (ax1, sched_25, '25% of empirical λ  (153 jobs/day)', ACCENT2),
        (ax2, sched_100, '100% of empirical λ  (610 jobs/day)', PINK),
    ]:
        util = pd.DataFrame(sched.utilization_samples, columns=['t', 'busy'])
        util = util[util['t'] <= dur_h]
        util['frac'] = util['busy'] / TOTAL_NODES * 100
        avg = util['frac'].mean()

        days = util['t'] / 24.0

        # Raw trace (very light)
        ax.plot(days, util['frac'], color=color, alpha=0.12, linewidth=0.4)

        # Rolling average (4h window = 16 samples at 0.25h interval)
        window = 16
        rolling = util['frac'].rolling(window, center=True, min_periods=1).mean()
        ax.plot(days, rolling, color=color, linewidth=1.8,
                label=f'4h rolling avg')

        # Mean line
        ax.axhline(avg, color=WARN, linestyle='--', linewidth=1.5, alpha=0.8,
                   label=f'Mean: {avg:.1f}%')

        ax.set_title(scale_label, fontsize=14, color=color)
        ax.set_ylabel('Utilization (%)')
        ax.set_ylim(-2, 105)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True)

    ax2.set_xlabel('Time (days)')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f'{outdir}/utilization_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {outdir}/utilization_comparison.png')

    # Also generate per-queue node-hours breakdown bar chart
    plot_nodehours_breakdown(sched_25, jobs_25, sched_100, jobs_100, dur_h, outdir)


def plot_nodehours_breakdown(sched_25, jobs_25, sched_100, jobs_100, dur_h, outdir):
    """Per-queue node-hours breakdown comparing 25% vs 100% λ."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Node-Hours Delivered by Queue', fontsize=18)

    queues = ['capacity', 'small', 'medium', 'large']
    colors_25 = [ACCENT2] * 4
    colors_100 = [PINK] * 4

    def get_nodehours(jobs, dur_h):
        nh = {}
        for q in queues:
            total = 0
            for j in jobs:
                if j.queue == q and j.start_time_h is not None:
                    eff_start = max(0.0, j.start_time_h)
                    eff_end = min(dur_h, j.end_time_h if j.end_time_h else dur_h)
                    total += j.nodes * max(0.0, eff_end - eff_start)
                nh[q] = total
        return nh

    nh_25 = get_nodehours(jobs_25, dur_h)
    nh_100 = get_nodehours(jobs_100, dur_h)

    x = np.arange(len(queues))
    w = 0.35
    bars1 = ax.bar(x - w/2, [nh_25.get(q, 0) for q in queues], w,
                   color=ACCENT2, alpha=0.8, label='25% λ')
    bars2 = ax.bar(x + w/2, [nh_100.get(q, 0) for q in queues], w,
                   color=PINK, alpha=0.8, label='100% λ')

    ax.set_xticks(x)
    ax.set_xticklabels([q.capitalize() for q in queues])
    ax.set_ylabel('Node-hours (millions)')
    ax.set_title(f'30-day simulation window ({TOTAL_NODES:,} nodes)')
    ax.legend(fontsize=11)
    ax.grid(True, axis='y')

    # Auto-scale y to data
    all_vals = [nh_25.get(q, 0) for q in queues] + [nh_100.get(q, 0) for q in queues]
    max_val = max(v for v in all_vals if v > 0) * 1.2
    ax.set_ylim(0, max_val)

    # Format y-axis in millions
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))

    # Value labels
    for bar in bars1:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val*0.01,
                    f'{bar.get_height()/1e6:.2f}M', ha='center', va='bottom',
                    fontsize=10, color=ACCENT2)
    for bar in bars2:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val*0.01,
                    f'{bar.get_height()/1e6:.2f}M', ha='center', va='bottom',
                    fontsize=10, color=PINK)

    fig.tight_layout()
    fig.savefig(f'{outdir}/nodehours_breakdown.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {outdir}/nodehours_breakdown.png')


if __name__ == '__main__':
    os.makedirs(OUTDIR, exist_ok=True)
    plot_utilization_comparison(OUTDIR)
    print("\nDone!")
