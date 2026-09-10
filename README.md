# schedsim

A replay-validated, Monte Carlo-ready simulator of PBS scheduling on Aurora,
built to answer one question: **how does the queue menu (node bands, walltime
caps, priorities, limits) change job throughput and system utilisation?**

Status: **stage 1 (replay validation) complete.** The scheduler reproduces
Aurora's spring-2026 behaviour when fed the real trace; the workload model and
menu optimisation layers are next (see `docs/DESIGN.md`).

```
data/trace/*.parquet   <-- python -m schedsim extract --db pbs_monitor_aurora.db
schedsim/              the library (engine, menu, machine, priority, metrics, replay)
configs/replay/        validation windows
tests/                 synthetic-job unit tests (no trace needed)
docs/                  DESIGN (architecture + roadmap), VALIDATION (findings), PRIORITY (formula)
legacy/                the previous v2 generator/scheduler, kept for reference
```

## Quick start

```bash
uv venv .venv && source .venv/bin/activate && uv pip install -e ".[dev]"
python -m pytest -q                                   # 17 synthetic tests, <1 s

# once, on the machine with the 9 GB DB (~3 min): jobs, reservations, queues, node states
python -m schedsim extract --db /path/pbs_monitor_aurora.db --out data/trace

# replay the real trace through the simulated scheduler and compare
python -m schedsim replay --config configs/replay/aurora_2026_spring.yaml
python -m schedsim replay --config configs/replay/aurora_2026_spring.yaml \
    --set scheduler.backfill_depth=3 --set output.dir=results/replay/depth3
```

Outputs per run: `REPORT.md`, `ranking.csv` (priority formulas ranked by fit),
`compare_<formula>.csv` (per-queue observed vs simulated waits), `jobs_*.parquet`
(per-job observed and simulated start), wait-CDF and utilisation plots, and a
`manifest.json` with the science hash of everything that changed the numbers.

## What the replay showed (spring 2026, 60 days, 53k jobs)

| queue | n | obs p50 | sim p50 | obs p90 | sim p90 | obs mean | sim mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| small (256-1024 n) | 4,405 | 2.0 h | 1.5 h | 25.9 h | 21.6 h | 9.0 h | 8.2 h |
| large (1920+ n) | 604 | 3.8 h | 1.6 h | 15.2 h | 16.2 h | 6.7 h | 5.7 h |
| capacity (1-16 n, 7 d) | 5,911 | 0.8 h | 0.7 h | 20.5 h | 27.5 h | 7.4 h | 9.0 h |

Utilisation against 9,600 reportable nodes: observed 51.2%, simulated 50.8%.
Details, what had to be modelled to get there, and the remaining gaps are in
`docs/VALIDATION.md`. The sort formula PBS actually uses was recovered from
recorded job scores (`docs/PRIORITY.md`): it is quadratic in eligible time,
linear in node count, and multiplied by project priority.

## Design in one paragraph

Jobs are rows in a `JobTable` (pandas) whether they come from the trace or a
workload model. `Machine` gives usable nodes over time (node snapshots minus
reservations). `Menu` holds queues with bounds and PBS limits, parsed from the
real `queues` table or from YAML. `Engine` runs PBS-style cycles: score with a
vectorised safe expression, walk in order, start what fits in the free-node
profile, reserve for the blocked top jobs (`backfill_depth`), enforce
per-queue run limits and strict ordering within the prod queue family, and
honour dedicated partitions. `metrics` computes everything exactly from
intervals. `replay` is the validation harness.
