#!/usr/bin/env python3
"""
Monte-Carlo scan reducer + sensitivity/stability analyzer.

Treats the scan as a proper Monte Carlo study: many SEEDS per fixed config. For
each config, aggregates the per-seed results into a distributional summary
(n, mean, 95% CI on the mean, std, CV, and p5/p25/p50/p75/p95), so you can judge
both how GOOD a config is (mean/CI) and how STABLE it is (percentile spread, CV).

Then answers "which configurable drives node-hour delivery?":
  - per-config stability table (config_stats.csv)
  - box plots across configs for headline metrics (distribution per config)
  - a CV/stability bar (fragile configs surface)
  - main-effect plots: objective vs each swept parameter (marginalized)
  - a tornado chart ranking parameters by effect size on a chosen objective

Usage:
  python analyze_scan.py --results results/<scan>/results.parquet \
      --out results/<scan>/analysis [--objective avg_util_pct]
"""
from __future__ import annotations

import argparse
import os
import pathlib

import numpy as np
import pandas as pd

# metrics we summarize distributionally (present-if-in-frame)
CORE_METRICS = [
    "avg_util_pct", "alloc_util", "wait_p50_h", "wait_p95_h", "wait_max_h",
    "big_wait_p95_h", "small_wait_p95_h", "n_big_unstarted",
    "throughput_jobs_per_day", "node_hours_delivered",
]
# per-program delivered node-hours + shares are added dynamically


def _pct(s, q):
    return float(np.percentile(s, q)) if len(s) else float("nan")


def summarize(results_path: str) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    df = pd.read_parquet(results_path)
    ok = df[df["status"] == "OK"].copy()
    # swept-parameter columns = label columns (everything that isn't a metric/meta)
    meta = {"config_hash", "seed", "status", "runtime_s", "note"}
    metric_cols = set(CORE_METRICS) | {c for c in ok.columns
                                       if c.endswith(("_delivered", "_delivered_nh",
                                                      "_wait_p95", "_burn",
                                                      "_thresh_nodes"))}
    metric_cols |= {c for c in ok.columns if c.startswith(("n_", "wait_", "big_",
                                                           "small_", "avg_"))}
    param_cols = [c for c in ok.columns if c not in meta and c not in metric_cols]
    # which metrics actually exist and are numeric
    metrics = [c for c in ok.columns
               if c in metric_cols and pd.api.types.is_numeric_dtype(ok[c])]

    rows = []
    for chash, g in ok.groupby("config_hash"):
        rec = {"config_hash": chash, "n_seeds": len(g)}
        for pc in param_cols:
            rec[pc] = g[pc].iloc[0]
        for m in metrics:
            v = g[m].to_numpy(float)
            v = v[np.isfinite(v)]
            if len(v) == 0:
                continue
            mean = float(v.mean()); sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            ci = 1.96 * sd / np.sqrt(len(v)) if len(v) > 1 else 0.0
            rec[f"{m}__mean"] = round(mean, 4)
            rec[f"{m}__ci95"] = round(ci, 4)
            rec[f"{m}__std"] = round(sd, 4)
            rec[f"{m}__cv"] = round(sd / mean, 4) if mean != 0 else float("nan")
            rec[f"{m}__p5"] = round(_pct(v, 5), 4)
            rec[f"{m}__p25"] = round(_pct(v, 25), 4)
            rec[f"{m}__p50"] = round(_pct(v, 50), 4)
            rec[f"{m}__p75"] = round(_pct(v, 75), 4)
            rec[f"{m}__p95"] = round(_pct(v, 95), 4)
        rows.append(rec)
    stats = pd.DataFrame(rows)
    return ok, stats, (param_cols, metrics)


def _fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def make_plots(ok, stats, param_cols, metrics, objective, outdir):
    plt = _fig()
    outdir = pathlib.Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    obj = objective if objective in metrics else "avg_util_pct"

    # short config label = concatenation of swept params (for axis ticks)
    def clabel(row):
        return " | ".join(f"{c.split('.')[-1]}={row[c]}" for c in param_cols) or row["config_hash"][:6]
    cfg_label = {h: clabel(stats[stats.config_hash == h].iloc[0])
                 for h in stats.config_hash}
    order = stats.sort_values(f"{obj}__mean", ascending=False)["config_hash"].tolist() \
        if f"{obj}__mean" in stats else list(stats.config_hash)

    # (A) box plot: distribution of objective per config (stability)
    data = [ok[ok.config_hash == h][obj].dropna().to_numpy() for h in order]
    labels = [cfg_label[h] for h in order]
    n = len(order)
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * n + 4), 5))
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel(obj); ax.grid(alpha=.3, axis="y")
    ax.set_title(f"{obj}: distribution across seeds per config (Monte-Carlo stability)")
    fig.tight_layout(); fig.savefig(outdir / f"box_{obj}.png", dpi=110); plt.close(fig)

    # (B) CV / stability bar across configs (higher CV = more fragile)
    cvcol = f"{obj}__cv"
    if cvcol in stats:
        s2 = stats.sort_values(cvcol, ascending=False)
        fig, ax = plt.subplots(figsize=(max(8, 0.5 * n + 4), 4.5))
        lbls = [cfg_label[h] for h in s2.config_hash]
        ax.bar(range(len(lbls)), s2[cvcol] * 100)
        ax.set_xticks(range(len(lbls)))
        ax.set_xticklabels(lbls, rotation=40, ha="right", fontsize=7)
        ax.set_ylabel(f"CV of {obj} (%)")
        ax.set_title(f"Config stability: coefficient of variation of {obj} "
                     f"(higher = more seed-fragile)")
        ax.grid(alpha=.3, axis="y")
        fig.tight_layout(); fig.savefig(outdir / f"stability_cv_{obj}.png", dpi=110)
        plt.close(fig)

    # (C) main-effect plots: objective mean vs each swept parameter
    for pc in param_cols:
        try:
            grp = stats.groupby(pc)[f"{obj}__mean"].mean()
            ci = stats.groupby(pc)[f"{obj}__ci95"].mean()
        except KeyError:
            continue
        if len(grp) < 2:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.2))
        x = np.arange(len(grp))
        ax.errorbar(x, grp.values, yerr=ci.values, marker="o", capsize=4, lw=2)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v)[:24] for v in grp.index], rotation=30,
                           ha="right", fontsize=8)
        ax.set_ylabel(f"mean {obj}"); ax.grid(alpha=.3)
        ax.set_title(f"Main effect on {obj}: {pc.split('.')[-1]}")
        fig.tight_layout()
        safe = pc.replace(".", "_").replace("/", "_")
        fig.savefig(outdir / f"maineffect_{safe}_{obj}.png", dpi=110); plt.close(fig)

    # (D) tornado: rank parameters by their effect size (max-min of group means)
    effects = []
    for pc in param_cols:
        try:
            gm = stats.groupby(pc)[f"{obj}__mean"].mean()
        except KeyError:
            continue
        if len(gm) >= 2:
            effects.append((pc.split(".")[-1], float(gm.max() - gm.min())))
    if effects:
        effects.sort(key=lambda x: x[1])
        fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(effects) + 1)))
        ax.barh([e[0] for e in effects], [e[1] for e in effects], color="#c0392b")
        ax.set_xlabel(f"effect on {obj} (max-min of parameter-group means)")
        ax.set_title(f"Parameter sensitivity (tornado): what moves {obj}")
        ax.grid(alpha=.3, axis="x")
        fig.tight_layout(); fig.savefig(outdir / f"tornado_{obj}.png", dpi=110)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="scan results.parquet")
    ap.add_argument("--out", default=None)
    ap.add_argument("--objective", default="avg_util_pct",
                    help="metric for main-effect/tornado/box (e.g. avg_util_pct, "
                         "INCITE_delivered_nh, wait_p95_h)")
    args = ap.parse_args()
    outdir = args.out or (os.path.dirname(args.results) + "/analysis")
    ok, stats, (param_cols, metrics) = summarize(args.results)
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    stats.to_csv(pathlib.Path(outdir) / "config_stats.csv", index=False)
    print(f"{len(stats)} configs, {ok.groupby('config_hash').size().median():.0f} "
          f"median seeds/config. swept params: {[p.split('.')[-1] for p in param_cols]}")
    print(f"metrics summarized: {len(metrics)}")
    # quick stability snapshot on the objective
    obj = args.objective if args.objective in metrics else "avg_util_pct"
    if f"{obj}__mean" in stats:
        top = stats.sort_values(f"{obj}__mean", ascending=False).head(8)
        cols = ["config_hash", "n_seeds", f"{obj}__mean", f"{obj}__ci95",
                f"{obj}__cv", f"{obj}__p5", f"{obj}__p95"]
        print(f"\nTop configs by {obj} (mean, with stability):")
        print(top[[c for c in cols if c in top]].to_string(index=False))
    make_plots(ok, stats, param_cols, metrics, obj, outdir)
    print(f"\nWrote config_stats.csv + plots -> {outdir}/")


if __name__ == "__main__":
    main()
