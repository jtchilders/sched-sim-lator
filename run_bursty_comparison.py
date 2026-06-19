#!/usr/bin/env python3
"""
Run flat vs bursty Poisson sweeps and generate comparison plots.

Runs the sim at 10%, 25%, 50%, 100% arrival rate scale for both
flat Poisson and AR(1) bursty arrivals, then plots:
  1. Sweep results table comparison
  2. Utilization comparison
  3. Queue depth comparison
  4. Validation: daily rate distribution (bursty vs empirical)
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
from sim import QUEUES, Scheduler, JobGenerator, Job, summarize, CAPACITY_POOL_NODES

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

ACCENT = '#64ffda'
ACCENT2 = '#57cbff'
PINK = '#ff6b8a'
WARN = '#ffb347'
PURPLE = '#c4b5fd'

SCALES = [0.10, 0.25, 0.50, 1.00]


def run_sim(arrival_scale, bursty=False, seed=SEED):
    """Run simulation, return (sched, jobs, duration_h, summary_dict)."""
    rng = np.random.default_rng(seed)
    import random as _random
    _random.seed(seed)

    duration_h = DURATION_DAYS * 24.0
    fitted = trace_sampler.FittedSampler(db_path=DB_PATH, rng=rng, use_cache=True)

    queues = copy.deepcopy(QUEUES)
    for qc in queues:
        rate = fitted.arrival_rate(qc.name)
        if qc.name == "capacity" and arrival_scale != 1.0:
            rate *= arrival_scale
        qc.arrival_rate_per_h = rate

    gen = JobGenerator(queues, rng, sampler=fitted,
                       bursty=bursty, bursty_rho=0.90, bursty_cv=0.74)
    jobs = gen.generate(duration_h)

    sched = Scheduler(total_nodes=TOTAL_NODES, enable_backfill=True,
                      capacity_pool=512, on_demand_nodes=0)
    sched.run(jobs, duration_h=duration_h, sample_dt_h=SAMPLE_DT_H)

    # Compute summary stats
    started = [j for j in jobs if j.start_time_h is not None]
    unstarted = [j for j in jobs if j.start_time_h is None]

    def queue_stats(jobs_list, queue_name):
        qj = [j for j in jobs_list if j.queue == queue_name and j.start_time_h is not None]
        if not qj:
            return {'n': 0, 'wait_med': 0, 'wait_p95': 0, 'wait_max': 0}
        waits = [(j.start_time_h - j.submit_time_h) for j in qj]
        return {
            'n': len(qj),
            'wait_med': np.median(waits),
            'wait_p95': np.percentile(waits, 95),
            'wait_max': np.max(waits),
        }

    # Utilization
    util_samples = pd.DataFrame(sched.utilization_samples, columns=['t', 'busy'])
    avg_util = (util_samples['busy'] / TOTAL_NODES * 100).mean()

    cap_stats = queue_stats(jobs, 'capacity')
    large_stats = queue_stats(jobs, 'large')

    summary = {
        'scale': arrival_scale,
        'bursty': bursty,
        'n_jobs': len(jobs),
        'n_unstarted': len(unstarted),
        'cap_jobs_day': len([j for j in jobs if j.queue == 'capacity']) / DURATION_DAYS,
        'cap_p95_wait': cap_stats['wait_p95'],
        'cap_max_wait': cap_stats['wait_max'],
        'large_max_wait': large_stats['wait_max'],
        'avg_util': avg_util,
    }

    return sched, jobs, duration_h, summary


def run_all_sweeps():
    """Run flat and bursty sweeps at all scales."""
    results = []

    for scale in SCALES:
        for bursty in [False, True]:
            label = f"{'bursty' if bursty else 'flat'} @ {scale:.0%}"
            print(f"\n{'='*60}")
            print(f"Running: {label}")
            print(f"{'='*60}")
            sched, jobs, dur_h, summary = run_sim(scale, bursty=bursty)
            results.append(summary)
            print(f"  Jobs: {summary['n_jobs']}, Unstarted: {summary['n_unstarted']}, "
                  f"Cap p95: {summary['cap_p95_wait']:.1f}h, "
                  f"Cap max: {summary['cap_max_wait']:.1f}h, "
                  f"Large max: {summary['large_max_wait']:.1f}h, "
                  f"Util: {summary['avg_util']:.1f}%")

    return pd.DataFrame(results)


def plot_sweep_comparison(df, outdir):
    """Bar chart comparing flat vs bursty across scales."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Flat Poisson vs Bursty AR(1) Arrivals: Sweep Comparison',
                 fontsize=18, y=0.98)

    flat = df[~df['bursty']]
    bursty = df[df['bursty']]

    x = np.arange(len(SCALES))
    w = 0.35
    scale_labels = [f'{s:.0%}' for s in SCALES]

    metrics = [
        ('cap_p95_wait', 'Capacity p95 Wait (hours)', 'Upper: worse'),
        ('cap_max_wait', 'Capacity Max Wait (hours)', 'Upper: worse'),
        ('large_max_wait', 'Large Max Wait (hours)', 'Upper: worse'),
        ('n_unstarted', 'Unstarted Jobs (at 30d)', 'Upper: worse'),
    ]

    for ax, (col, ylabel, note) in zip(axes.flat, metrics):
        bars1 = ax.bar(x - w/2, flat[col].values, w,
                       color=ACCENT2, alpha=0.8, label='Flat Poisson')
        bars2 = ax.bar(x + w/2, bursty[col].values, w,
                       color=PINK, alpha=0.8, label='Bursty AR(1)')

        ax.set_xticks(x)
        ax.set_xticklabels(scale_labels)
        ax.set_xlabel('Capacity λ Scale')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=10)
        ax.grid(True, axis='y')

        # Value labels
        for bar in bars1:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h,
                        f'{h:.1f}' if isinstance(h, float) else f'{h:.0f}',
                        ha='center', va='bottom', fontsize=9, color=ACCENT2)
        for bar in bars2:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h,
                        f'{h:.1f}' if isinstance(h, float) else f'{h:.0f}',
                        ha='center', va='bottom', fontsize=9, color=PINK)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f'{outdir}/bursty_sweep_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {outdir}/bursty_sweep_comparison.png')


def plot_arrival_validation(outdir):
    """Validate bursty model by comparing daily rate distributions."""
    rng = np.random.default_rng(SEED)
    duration_h = DURATION_DAYS * 24.0

    fitted = trace_sampler.FittedSampler(db_path=DB_PATH, rng=rng, use_cache=True)
    queues = copy.deepcopy(QUEUES)
    for qc in queues:
        qc.arrival_rate_per_h = fitted.arrival_rate(qc.name)

    # Generate flat jobs
    rng_flat = np.random.default_rng(SEED)
    gen_flat = JobGenerator(copy.deepcopy(queues), rng_flat, sampler=fitted, bursty=False)
    jobs_flat = gen_flat.generate(duration_h)

    # Generate bursty jobs
    rng_bursty = np.random.default_rng(SEED)
    fitted2 = trace_sampler.FittedSampler(db_path=DB_PATH, rng=rng_bursty, use_cache=True)
    gen_bursty = JobGenerator(copy.deepcopy(queues), rng_bursty, sampler=fitted2,
                              bursty=True, bursty_rho=0.90, bursty_cv=0.74)
    jobs_bursty = gen_bursty.generate(duration_h)

    # Load empirical daily counts
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    emp_df = pd.read_sql_query("""
        SELECT submit_time FROM jobs
        WHERE submit_time IS NOT NULL AND end_time IS NOT NULL AND nodes > 0
    """, conn)
    conn.close()
    emp_df['dt'] = pd.to_datetime(emp_df['submit_time'], utc=True)
    emp_df['date'] = emp_df['dt'].dt.date
    emp_daily = emp_df.groupby('date').size().values

    # Compute daily counts for sim jobs
    def daily_counts(jobs_list):
        days = [int(j.submit_time_h / 24.0) for j in jobs_list]
        counts = np.bincount(days, minlength=DURATION_DAYS)[:DURATION_DAYS]
        return counts

    flat_daily = daily_counts(jobs_flat)
    bursty_daily = daily_counts(jobs_bursty)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Arrival Rate Validation: Flat vs Bursty vs Empirical', fontsize=18, y=0.98)

    # Left: daily counts time series
    days = np.arange(DURATION_DAYS)
    ax1.plot(days, flat_daily, color=ACCENT2, linewidth=1.5, alpha=0.8, label='Flat Poisson')
    ax1.plot(days, bursty_daily, color=PINK, linewidth=1.5, alpha=0.8, label='Bursty AR(1)')
    ax1.axhline(flat_daily.mean(), color=ACCENT2, linestyle='--', linewidth=1, alpha=0.5)
    ax1.axhline(bursty_daily.mean(), color=PINK, linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Simulation Day')
    ax1.set_ylabel('Jobs Submitted')
    ax1.set_title('Daily Submissions (30-day window)', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True)

    # Annotate CVs
    cv_flat = flat_daily.std() / flat_daily.mean()
    cv_bursty = bursty_daily.std() / bursty_daily.mean()
    cv_emp = emp_daily.std() / emp_daily.mean()
    ax1.text(0.02, 0.95,
             f'CV flat: {cv_flat:.2f}\nCV bursty: {cv_bursty:.2f}\nCV empirical: {cv_emp:.2f}',
             transform=ax1.transAxes, fontsize=12, color=ACCENT,
             verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#112240',
                       edgecolor=ACCENT, alpha=0.8))

    # Right: histogram of daily counts
    bins = np.linspace(0, max(emp_daily.max(), bursty_daily.max()), 40)
    ax2.hist(emp_daily, bins=bins, density=True, histtype='step',
             color=WARN, linewidth=2, label=f'Empirical (365d, CV={cv_emp:.2f})')
    ax2.hist(flat_daily, bins=np.linspace(0, flat_daily.max()*1.2, 30), density=True,
             histtype='step', color=ACCENT2, linewidth=2,
             linestyle='--', label=f'Flat Poisson (CV={cv_flat:.2f})')
    ax2.hist(bursty_daily, bins=bins, density=True, histtype='step',
             color=PINK, linewidth=2, linestyle='--',
             label=f'Bursty AR(1) (CV={cv_bursty:.2f})')
    ax2.set_xlabel('Jobs per Day')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution of Daily Job Counts', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f'{outdir}/bursty_validation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {outdir}/bursty_validation.png')


def print_comparison_table(df):
    """Print a formatted comparison table."""
    print("\n" + "="*100)
    print("SWEEP COMPARISON: FLAT POISSON vs BURSTY AR(1)")
    print("="*100)
    print(f"{'Scale':>8} {'Mode':>8} {'Jobs':>8} {'Unstrt':>8} {'Cap/day':>10} "
          f"{'Cap p95':>10} {'Cap max':>10} {'Lrg max':>10} {'Util':>8}")
    print("-"*100)
    for _, row in df.iterrows():
        mode = 'bursty' if row['bursty'] else 'flat'
        print(f"{row['scale']:>7.0%} {mode:>8} {row['n_jobs']:>8.0f} "
              f"{row['n_unstarted']:>8.0f} {row['cap_jobs_day']:>10.1f} "
              f"{row['cap_p95_wait']:>9.1f}h {row['cap_max_wait']:>9.1f}h "
              f"{row['large_max_wait']:>9.1f}h {row['avg_util']:>7.1f}%")
        if mode == 'bursty':
            print()


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    print("Running sweep comparison...")
    df = run_all_sweeps()
    print_comparison_table(df)

    print("\nGenerating comparison plots...")
    plot_sweep_comparison(df, OUTDIR)

    print("\nGenerating arrival validation plot...")
    plot_arrival_validation(OUTDIR)

    # Save results
    df.to_csv(f'{OUTDIR}/bursty_sweep_results.csv', index=False)
    print(f'  → {OUTDIR}/bursty_sweep_results.csv')

    print("\nDone!")


if __name__ == '__main__':
    main()
