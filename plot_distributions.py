#!/usr/bin/env python3
"""
Generate empirical vs. simulated distribution comparison plots.

Produces:
  1. Node count distributions per bucket (empirical vs simulated)
  2. Walltime distributions per bucket (empirical old caps vs simulated new caps)
  3. Runtime distributions per bucket
  4. Arrival rate comparison (empirical vs fitted)

All saved to results/distributions/
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trace_sampler
from trace_sampler import FittedSampler, _load_raw, _bucket_of, DEFAULT_BUCKETS, OLD_CAPS_H, NEW_CAPS_H, CAP_PACKER_THRESHOLD, PROD_QUEUE_MAP

DB_PATH = "/Users/jchilders/pbs_monitor_aurora.db"
OUTDIR = "results/distributions"
SEED = 42
N_SAMPLES = 50_000  # samples per bucket for simulated distributions

# Plot style
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

EMPIRICAL_COLOR = '#57cbff'
SIMULATED_COLOR = '#ff6b8a'
CAPPACKER_COLOR = '#ffb347'

BUCKET_ORDER = ['capacity', 'small', 'medium', 'large']
BUCKET_LABELS = {
    'capacity': 'Capacity (1–128 nodes)',
    'small': 'Small (129–512 nodes)',
    'medium': 'Medium (513–2,048 nodes)',
    'large': 'Large (2,049+ nodes)',
}


def load_empirical(db_path):
    """Load raw empirical data, bucket by node count."""
    df = _load_raw(db_path)
    df = df[df['nodes'] <= 10624]
    df['bucket'] = df['nodes'].apply(lambda n: _bucket_of(int(n), DEFAULT_BUCKETS))
    df = df.dropna(subset=['bucket'])
    # Filter to production queues only (same as FittedSampler)
    keep = []
    for bucket, queues in PROD_QUEUE_MAP.items():
        sub = df[(df['orig_queue'].isin(queues)) & (df['bucket'] == bucket)]
        keep.append(sub)
    return pd.concat(keep, ignore_index=True)


def generate_simulated(sampler, n_per_bucket):
    """Draw samples from the FittedSampler."""
    rows = []
    for bucket in BUCKET_ORDER:
        for _ in range(n_per_bucket):
            nodes, wt, rt = sampler.sample(bucket)
            rows.append({'bucket': bucket, 'nodes': nodes,
                         'walltime_h': wt, 'runtime_h': rt})
    return pd.DataFrame(rows)


def plot_node_distributions(emp, sim, outdir):
    """Step histograms of node counts per bucket."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Node Count Distributions: Empirical vs Simulated', fontsize=18, y=0.98)

    for ax, bucket in zip(axes.flat, BUCKET_ORDER):
        lo, hi = DEFAULT_BUCKETS[bucket]
        emp_nodes = emp[emp['bucket'] == bucket]['nodes'].to_numpy()
        sim_nodes = sim[sim['bucket'] == bucket]['nodes'].to_numpy()

        bins = np.linspace(lo, hi, min(60, hi - lo + 1))
        ax.hist(emp_nodes, bins=bins, density=True, histtype='step',
                color=EMPIRICAL_COLOR, linewidth=2, label=f'Empirical (n={len(emp_nodes):,})')
        ax.hist(sim_nodes, bins=bins, density=True, histtype='step',
                color=SIMULATED_COLOR, linewidth=2, linestyle='--', label=f'Simulated (n={len(sim_nodes):,})')
        ax.set_title(BUCKET_LABELS[bucket], fontsize=14)
        ax.set_xlabel('Nodes')
        ax.set_ylabel('Density')
        ax.legend(fontsize=10)
        ax.grid(True)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f'{outdir}/node_distributions.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {outdir}/node_distributions.png')


def plot_walltime_distributions(emp, sim, outdir):
    """Walltime distributions showing empirical (old caps) vs simulated (new caps)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Walltime Distributions: Empirical (Old Caps) vs Simulated (New Caps)', fontsize=18, y=0.98)

    for ax, bucket in zip(axes.flat, BUCKET_ORDER):
        old_cap = OLD_CAPS_H[bucket]
        new_cap = NEW_CAPS_H[bucket]
        emp_wt = emp[emp['bucket'] == bucket]['walltime_h'].to_numpy()
        sim_wt = sim[sim['bucket'] == bucket]['walltime_h'].to_numpy()

        max_val = max(emp_wt.max(), sim_wt.max(), new_cap) * 1.1
        bins = np.linspace(0, min(max_val, new_cap * 1.1), 80)

        ax.hist(emp_wt, bins=bins, density=True, histtype='step',
                color=EMPIRICAL_COLOR, linewidth=2, label=f'Empirical (cap={old_cap:.0f}h)')
        ax.hist(sim_wt, bins=bins, density=True, histtype='step',
                color=SIMULATED_COLOR, linewidth=2, linestyle='--', label=f'Simulated (cap={new_cap:.0f}h)')

        ax.axvline(old_cap, color=EMPIRICAL_COLOR, linestyle=':', linewidth=1.5, alpha=0.8,
                   label=f'Old cap ({old_cap:.0f}h)')
        if new_cap != old_cap:
            ax.axvline(new_cap, color=SIMULATED_COLOR, linestyle=':', linewidth=1.5, alpha=0.8,
                       label=f'New cap ({new_cap:.0f}h)')

        ax.set_title(BUCKET_LABELS[bucket], fontsize=14)
        ax.set_xlabel('Walltime (hours)')
        ax.set_ylabel('Density')
        ax.legend(fontsize=9)
        ax.grid(True)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f'{outdir}/walltime_distributions.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {outdir}/walltime_distributions.png')


def plot_walltime_cappacker_detail(emp, sim, sampler, outdir):
    """Show the two-component walltime model: interior vs cap-packer split."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Two-Component Walltime Model: Interior vs Cap-Packer Split', fontsize=18, y=0.98)

    for ax, bucket in zip(axes.flat, BUCKET_ORDER):
        old_cap = OLD_CAPS_H[bucket]
        new_cap = NEW_CAPS_H[bucket]
        threshold = CAP_PACKER_THRESHOLD * old_cap

        emp_sub = emp[emp['bucket'] == bucket]
        emp_wt = emp_sub['walltime_h'].to_numpy()

        is_cp = emp_wt >= threshold
        interior = emp_wt[~is_cp]
        cappack = emp_wt[is_cp]
        cp_frac = sampler.cap_packer_frac(bucket)

        bins_emp = np.linspace(0, old_cap * 1.1, 60)
        bins_sim = np.linspace(0, new_cap * 1.1, 80)

        # Empirical split
        ax.hist(interior, bins=bins_emp, density=True, alpha=0.5,
                color=EMPIRICAL_COLOR, label=f'Interior ({len(interior):,})', edgecolor='none')
        ax.hist(cappack, bins=bins_emp, density=True, alpha=0.5,
                color=CAPPACKER_COLOR, label=f'Cap-packers ({len(cappack):,}, {cp_frac:.0%})', edgecolor='none')

        ax.axvline(threshold, color=CAPPACKER_COLOR, linestyle=':', linewidth=1.5,
                   label=f'Threshold ({threshold:.0f}h = 90% of old cap)')

        ax.set_title(f'{BUCKET_LABELS[bucket]}  —  cap-packer fraction: {cp_frac:.1%}', fontsize=13)
        ax.set_xlabel('Walltime (hours)')
        ax.set_ylabel('Density')
        ax.legend(fontsize=9)
        ax.grid(True)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f'{outdir}/walltime_two_component.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {outdir}/walltime_two_component.png')


def plot_runtime_distributions(emp, sim, outdir):
    """Runtime distributions per bucket."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Runtime Distributions: Empirical vs Simulated', fontsize=18, y=0.98)

    for ax, bucket in zip(axes.flat, BUCKET_ORDER):
        emp_rt = emp[emp['bucket'] == bucket]['runtime_h'].to_numpy()
        sim_rt = sim[sim['bucket'] == bucket]['runtime_h'].to_numpy()

        max_val = np.percentile(np.concatenate([emp_rt, sim_rt]), 99)
        bins = np.linspace(0, max_val, 80)

        ax.hist(emp_rt, bins=bins, density=True, histtype='step',
                color=EMPIRICAL_COLOR, linewidth=2, label=f'Empirical (n={len(emp_rt):,})')
        ax.hist(sim_rt, bins=bins, density=True, histtype='step',
                color=SIMULATED_COLOR, linewidth=2, linestyle='--', label=f'Simulated (n={len(sim_rt):,})')

        ax.set_title(BUCKET_LABELS[bucket], fontsize=14)
        ax.set_xlabel('Runtime (hours)')
        ax.set_ylabel('Density')
        ax.legend(fontsize=10)
        ax.grid(True)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f'{outdir}/runtime_distributions.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {outdir}/runtime_distributions.png')


def plot_arrival_rates(emp, sampler, outdir):
    """Bar chart comparing empirical vs fitted arrival rates."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Arrival Rates: Empirical vs Fitted', fontsize=18)

    span_h = emp['submit_h'].max() - emp['submit_h'].min()
    emp_rates = {}
    fit_rates = {}
    for bucket in BUCKET_ORDER:
        n = (emp['bucket'] == bucket).sum()
        emp_rates[bucket] = n / span_h * 24  # jobs/day
        fit_rates[bucket] = sampler.arrival_rate(bucket) * 24

    x = np.arange(len(BUCKET_ORDER))
    w = 0.35
    bars1 = ax.bar(x - w/2, [emp_rates[b] for b in BUCKET_ORDER], w,
                   color=EMPIRICAL_COLOR, alpha=0.8, label='Empirical')
    bars2 = ax.bar(x + w/2, [fit_rates[b] for b in BUCKET_ORDER], w,
                   color=SIMULATED_COLOR, alpha=0.8, label='Fitted')

    ax.set_xticks(x)
    ax.set_xticklabels([BUCKET_LABELS[b].split('(')[0].strip() for b in BUCKET_ORDER])
    ax.set_ylabel('Jobs / day')
    ax.set_title('Per-Queue Arrival Rates')
    ax.legend()
    ax.grid(True, axis='y')

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=11, color=EMPIRICAL_COLOR)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=11, color=SIMULATED_COLOR)

    fig.tight_layout()
    fig.savefig(f'{outdir}/arrival_rates.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {outdir}/arrival_rates.png')


def plot_summary_stats_table(emp, sim, outdir):
    """Summary statistics comparison table as an image."""
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('off')
    fig.suptitle('Summary Statistics: Empirical vs Simulated', fontsize=18, y=0.95)

    rows = []
    for bucket in BUCKET_ORDER:
        e = emp[emp['bucket'] == bucket]
        s = sim[sim['bucket'] == bucket]
        rows.append([
            BUCKET_LABELS[bucket].split('(')[0].strip(),
            f'{len(e):,}',
            f'{e["nodes"].median():.0f} / {e["nodes"].mean():.0f}',
            f'{s["nodes"].median():.0f} / {s["nodes"].mean():.0f}',
            f'{e["walltime_h"].median():.1f} / {e["walltime_h"].mean():.1f}',
            f'{s["walltime_h"].median():.1f} / {s["walltime_h"].mean():.1f}',
            f'{e["runtime_h"].median():.1f} / {e["runtime_h"].mean():.1f}',
            f'{s["runtime_h"].median():.1f} / {s["runtime_h"].mean():.1f}',
        ])

    col_labels = ['Queue', 'N (emp)', 'Nodes med/mean\n(empirical)',
                  'Nodes med/mean\n(simulated)', 'Walltime med/mean\n(empirical)',
                  'Walltime med/mean\n(simulated)', 'Runtime med/mean\n(empirical)',
                  'Runtime med/mean\n(simulated)']

    table = ax.table(cellText=rows, colLabels=col_labels, loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#8892b0')
        if r == 0:
            cell.set_facecolor('#112240')
            cell.set_text_props(color='#64ffda', fontweight='bold')
        else:
            cell.set_facecolor('#0a1628')
            cell.set_text_props(color='#e6f1ff')

    fig.tight_layout()
    fig.savefig(f'{outdir}/summary_stats.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {outdir}/summary_stats.png')


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("Loading empirical data...")
    emp = load_empirical(DB_PATH)
    print(f"  {len(emp):,} empirical jobs loaded")

    print("Building FittedSampler...")
    sampler = FittedSampler(db_path=DB_PATH, rng=rng, use_cache=True)
    sampler.print_fit_summary()

    print(f"\nGenerating {N_SAMPLES:,} simulated samples per bucket...")
    sim = generate_simulated(sampler, N_SAMPLES)
    print(f"  {len(sim):,} total simulated samples")

    print("\nGenerating plots...")
    plot_node_distributions(emp, sim, OUTDIR)
    plot_walltime_distributions(emp, sim, OUTDIR)
    plot_walltime_cappacker_detail(emp, sim, sampler, OUTDIR)
    plot_runtime_distributions(emp, sim, OUTDIR)
    plot_arrival_rates(emp, sampler, OUTDIR)
    plot_summary_stats_table(emp, sim, OUTDIR)
    print("\nDone!")


if __name__ == '__main__':
    main()
