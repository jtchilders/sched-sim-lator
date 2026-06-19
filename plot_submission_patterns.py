#!/usr/bin/env python3
"""
Analyze and plot temporal submission patterns from Aurora PBS traces.

Produces:
  1. Hourly pattern (hour-of-day, averaged across all days)
  2. Daily pattern (day-of-week)
  3. Weekly pattern (submissions per week over the full trace)
  4. Monthly pattern (submissions per month)

Uses the raw finished_jobs table from the Aurora DB.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timezone

DB_PATH = "/Users/jchilders/pbs_monitor_aurora.db"
OUTDIR = "results/distributions"

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


def load_submissions(db_path):
    """Load job submission timestamps from the Aurora DB."""
    import sqlite3
    conn = sqlite3.connect(db_path)

    df = pd.read_sql_query("""
        SELECT submit_time, nodes, queue
        FROM jobs
        WHERE submit_time IS NOT NULL AND end_time IS NOT NULL AND nodes > 0
        ORDER BY submit_time
    """, conn)
    conn.close()

    # submit_time is a datetime string like '2025-06-04 02:42:59.000000' (UTC)
    df['submit_dt'] = pd.to_datetime(df['submit_time'], utc=True)
    df['submit_dt_ct'] = df['submit_dt'].dt.tz_convert('America/Chicago')
    df['hour'] = df['submit_dt_ct'].dt.hour
    df['dow'] = df['submit_dt_ct'].dt.dayofweek  # 0=Mon, 6=Sun
    df['dow_name'] = df['submit_dt_ct'].dt.day_name()
    df['date'] = df['submit_dt_ct'].dt.date
    df['week'] = df['submit_dt_ct'].dt.isocalendar().week.astype(int)
    df['year_week'] = df['submit_dt_ct'].dt.strftime('%Y-W%W')
    df['month'] = df['submit_dt_ct'].dt.to_period('M')

    print(f"Loaded {len(df):,} submissions")
    print(f"Date range: {df['submit_dt_ct'].min()} to {df['submit_dt_ct'].max()}")
    span_days = (df['submit_dt_ct'].max() - df['submit_dt_ct'].min()).days
    print(f"Span: {span_days} days ({span_days/365:.1f} years)")
    print(f"Mean rate: {len(df)/span_days:.0f} jobs/day")

    return df


def plot_hourly(df, outdir):
    """Hour-of-day submission pattern."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Hourly Submission Pattern (Central Time)', fontsize=18, y=0.98)

    # All queues combined
    hourly = df.groupby('hour').size()
    total_days = df['date'].nunique()
    hourly_rate = hourly / total_days  # avg jobs per hour-slot per day

    ax1.bar(hourly_rate.index, hourly_rate.values, color=ACCENT, alpha=0.8, edgecolor='none')
    ax1.axhline(hourly_rate.mean(), color=WARN, linestyle='--', linewidth=1.5,
                label=f'Mean: {hourly_rate.mean():.1f} jobs/h')
    ax1.set_xlabel('Hour of Day (CT)')
    ax1.set_ylabel('Avg Jobs Submitted per Hour')
    ax1.set_title('All Queues', fontsize=14)
    ax1.set_xticks(range(0, 24, 2))
    ax1.legend(fontsize=11)
    ax1.grid(True, axis='y')

    # Ratio to mean (burstiness multiplier)
    ratio = hourly_rate / hourly_rate.mean()
    ax2.bar(ratio.index, ratio.values, color=ACCENT2, alpha=0.8, edgecolor='none')
    ax2.axhline(1.0, color=WARN, linestyle='--', linewidth=1.5, label='Flat rate (1.0×)')
    ax2.set_xlabel('Hour of Day (CT)')
    ax2.set_ylabel('Rate Multiplier (vs daily mean)')
    ax2.set_title('Hourly Rate Multiplier', fontsize=14)
    ax2.set_xticks(range(0, 24, 2))
    ax2.legend(fontsize=11)
    ax2.grid(True, axis='y')

    # Annotate peak/trough
    peak_h = ratio.idxmax()
    trough_h = ratio.idxmin()
    ax2.annotate(f'{ratio[peak_h]:.1f}×', xy=(peak_h, ratio[peak_h]),
                 xytext=(peak_h+1, ratio[peak_h]+0.15),
                 arrowprops=dict(arrowstyle='->', color=ACCENT),
                 fontsize=12, color=ACCENT)
    ax2.annotate(f'{ratio[trough_h]:.1f}×', xy=(trough_h, ratio[trough_h]),
                 xytext=(trough_h+1, ratio[trough_h]+0.3),
                 arrowprops=dict(arrowstyle='->', color=PINK),
                 fontsize=12, color=PINK)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f'{outdir}/submission_hourly.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {outdir}/submission_hourly.png')


def plot_daily(df, outdir):
    """Day-of-week submission pattern."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Day-of-Week Submission Pattern', fontsize=18, y=0.98)

    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_short = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    daily = df.groupby('dow').size()
    n_weeks = df['date'].nunique() / 7.0
    daily_rate = daily / n_weeks

    colors = [ACCENT2]*5 + [PURPLE]*2  # weekday vs weekend

    ax1.bar(range(7), [daily_rate.get(i, 0) for i in range(7)],
            color=colors, alpha=0.8, edgecolor='none')
    ax1.axhline(daily_rate.mean(), color=WARN, linestyle='--', linewidth=1.5,
                label=f'Mean: {daily_rate.mean():.0f} jobs/day')
    ax1.set_xticks(range(7))
    ax1.set_xticklabels(dow_short)
    ax1.set_ylabel('Avg Jobs Submitted per Day')
    ax1.set_title('Absolute Rate', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, axis='y')

    # Ratio
    ratio = daily_rate / daily_rate.mean()
    ax2.bar(range(7), [ratio.get(i, 0) for i in range(7)],
            color=colors, alpha=0.8, edgecolor='none')
    ax2.axhline(1.0, color=WARN, linestyle='--', linewidth=1.5, label='Flat rate (1.0×)')
    ax2.set_xticks(range(7))
    ax2.set_xticklabels(dow_short)
    ax2.set_ylabel('Rate Multiplier')
    ax2.set_title('Day-of-Week Multiplier', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, axis='y')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f'{outdir}/submission_daily.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {outdir}/submission_daily.png')


def plot_weekly(df, outdir):
    """Weekly submission counts over the full trace period."""
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.suptitle('Weekly Job Submissions Over Time', fontsize=18, y=0.98)

    # Group by actual week
    df['week_start'] = df['submit_dt_ct'].dt.to_period('W').apply(lambda r: r.start_time)
    weekly = df.groupby('week_start').size().reset_index(name='count')
    weekly['week_start'] = pd.to_datetime(weekly['week_start'])

    ax.bar(weekly['week_start'], weekly['count'], width=5,
           color=ACCENT2, alpha=0.7, edgecolor='none')

    # Rolling 4-week average
    rolling = weekly['count'].rolling(4, center=True, min_periods=1).mean()
    ax.plot(weekly['week_start'], rolling, color=WARN, linewidth=2.5,
            label='4-week rolling avg')

    mean_val = weekly['count'].mean()
    ax.axhline(mean_val, color=PINK, linestyle='--', linewidth=1.5,
               label=f'Overall mean: {mean_val:.0f}/week')

    # Annotate peaks
    peak_idx = weekly['count'].idxmax()
    peak_week = weekly.loc[peak_idx]
    ax.annotate(f'Peak: {peak_week["count"]:,}',
                xy=(peak_week['week_start'], peak_week['count']),
                xytext=(peak_week['week_start'], peak_week['count'] + 500),
                arrowprops=dict(arrowstyle='->', color=ACCENT),
                fontsize=12, color=ACCENT, ha='center')

    ax.set_xlabel('Week')
    ax.set_ylabel('Jobs Submitted')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, axis='y')

    # Format x-axis dates
    ax.tick_params(axis='x', rotation=45)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f'{outdir}/submission_weekly.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {outdir}/submission_weekly.png')


def plot_monthly(df, outdir):
    """Monthly submission counts. Marks partial months."""
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle('Monthly Job Submissions', fontsize=18, y=0.98)

    monthly = df.groupby('month').size()
    months = [str(m) for m in monthly.index]

    # Detect partial first/last months
    colors = []
    for i, m in enumerate(monthly.index):
        if i == 0 or i == len(monthly) - 1:
            colors.append('#555555')  # gray for partial
        else:
            colors.append(ACCENT)

    bars = ax.bar(range(len(months)), monthly.values,
                  color=colors, alpha=0.8, edgecolor='none')

    # Mean of complete months only
    complete = monthly.iloc[1:-1] if len(monthly) > 2 else monthly
    mean_val = complete.mean()
    ax.axhline(mean_val, color=WARN, linestyle='--', linewidth=1.5,
               label=f'Mean (complete months): {mean_val:,.0f}/month')

    ax.set_xticks(range(len(months)))
    labels = []
    for i, m in enumerate(months):
        if i == 0 or i == len(months) - 1:
            labels.append(f'{m}\n(partial)')
        else:
            labels.append(m)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Jobs Submitted')
    ax.legend(fontsize=12)
    ax.grid(True, axis='y')

    # Value labels on bars
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 100,
                f'{h:,.0f}', ha='center', va='bottom', fontsize=9, color=ACCENT)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f'{outdir}/submission_monthly.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {outdir}/submission_monthly.png')


def plot_daily_timeseries(df, outdir):
    """Daily submission counts as a time series — shows the raw burstiness."""
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.suptitle('Daily Job Submissions Over Time', fontsize=18, y=0.98)

    daily = df.groupby('date').size().reset_index(name='count')
    daily['date'] = pd.to_datetime(daily['date'])

    ax.fill_between(daily['date'], 0, daily['count'],
                    color=ACCENT2, alpha=0.3, linewidth=0)
    ax.plot(daily['date'], daily['count'], color=ACCENT2, linewidth=0.5, alpha=0.6)

    # 7-day rolling average
    rolling = daily['count'].rolling(7, center=True, min_periods=1).mean()
    ax.plot(daily['date'], rolling, color=WARN, linewidth=2,
            label='7-day rolling avg')

    mean_val = daily['count'].mean()
    ax.axhline(mean_val, color=PINK, linestyle='--', linewidth=1.5,
               label=f'Mean: {mean_val:.0f}/day')

    # Coefficient of variation
    cv = daily['count'].std() / daily['count'].mean()
    ax.text(0.02, 0.95, f'CV = {cv:.2f}  (Poisson would be ~{1/np.sqrt(mean_val):.3f})',
            transform=ax.transAxes, fontsize=13, color=ACCENT,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#112240',
                      edgecolor=ACCENT, alpha=0.8))

    ax.set_xlabel('Date')
    ax.set_ylabel('Jobs Submitted per Day')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, axis='y')
    ax.tick_params(axis='x', rotation=45)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f'{outdir}/submission_daily_ts.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {outdir}/submission_daily_ts.png')


def print_burstiness_stats(df):
    """Print summary statistics about submission burstiness."""
    daily = df.groupby('date').size()
    hourly = df.set_index('submit_dt_ct').resample('1h').size()

    print("\n=== Burstiness Statistics ===")
    print(f"  Daily submissions:")
    print(f"    Mean: {daily.mean():.0f},  Std: {daily.std():.0f},  CV: {daily.std()/daily.mean():.2f}")
    print(f"    Min: {daily.min()},  Max: {daily.max()},  Ratio: {daily.max()/daily.min():.1f}×")
    print(f"    p10: {daily.quantile(0.1):.0f},  p90: {daily.quantile(0.9):.0f}")
    print(f"    Poisson CV (expected): {1/np.sqrt(daily.mean()):.4f}")
    print(f"    Actual / Poisson CV ratio: {(daily.std()/daily.mean()) / (1/np.sqrt(daily.mean())):.1f}×")
    print()
    print(f"  Hourly submissions:")
    print(f"    Mean: {hourly.mean():.1f},  Std: {hourly.std():.1f},  CV: {hourly.std()/hourly.mean():.2f}")
    print(f"    Zero-hours: {(hourly == 0).sum()} ({(hourly == 0).mean()*100:.1f}%)")
    print(f"    Max hourly: {hourly.max()}")

    # Autocorrelation
    from pandas.plotting import autocorrelation_plot
    daily_series = df.set_index('submit_dt_ct').resample('1D').size()
    acf_1 = daily_series.autocorr(lag=1)
    acf_7 = daily_series.autocorr(lag=7)
    acf_30 = daily_series.autocorr(lag=30)
    print(f"\n  Daily autocorrelation:")
    print(f"    Lag 1 day:  {acf_1:.3f}")
    print(f"    Lag 7 days: {acf_7:.3f}")
    print(f"    Lag 30 days: {acf_30:.3f}")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    df = load_submissions(DB_PATH)
    print_burstiness_stats(df)

    print("\nGenerating plots...")
    plot_hourly(df, OUTDIR)
    plot_daily(df, OUTDIR)
    plot_weekly(df, OUTDIR)
    plot_monthly(df, OUTDIR)
    plot_daily_timeseries(df, OUTDIR)
    print("\nDone!")


if __name__ == '__main__':
    main()
