# Design

## Goal

Optimise Aurora's PBS queue menu for throughput and utilisation, with
uncertainty quantification, using a simulator whose scheduler is validated by
replaying the real trace and whose workload can be generated from historical
data plus explicit assumptions where data is lacking.

## Layers

```
 trace DB ──extract──▶ parquet ──┐
                                 ├──▶ replay.build_population ──▶ JobTable ─┐
 (stage 2) workload model ───────┘                                          │
                                                                            ▼
 queues table / YAML ──▶ Menu ──────────────────────────────────────▶  Engine ──▶ start/end per job
 node snapshots + reservations ──▶ Machine ─────────────────────────▶   ▲
 priority expression (fitted) ──▶ ExprPriority ─────────────────────────┘
                                                                            │
                                                          metrics / replay compare / report
```

* **JobTable** (`schedsim/jobs.py`): the only contract between workload and
  scheduler. Fixed columns; extra columns pass through. `submit_h` may be
  negative (queued when the window opens); `initial_start_h` marks jobs already
  running. Scoring parameters travel with the job, as they do in PBS.
* **Machine** (`machine.py`): `available(t) = min(up(t), schedulable - reserved(t))`.
  `up(t)` comes from per-node state snapshots (job-exclusive + free); reservation
  windows are explicit so the backfill profile drains ahead of maintenance.
  `reportable_nodes` (9,600) is only a metrics denominator.
* **Menu** (`menu.py`): queues with node/walltime bounds, PBS limits
  (`max_run_per_user/project`, `max_queued_per_user/project`, aggregate node
  cap) and scoring defaults. Parsed from the real `queues` table for replay.
* **Engine** (`engine.py`): PBS-faithful cycles. Passes run on arrivals and
  finishes (rate-limited by a load-dependent cycle time) and periodically.
  Free-node *profile* over time = availability minus running walltime
  footprints minus placed reservations; a job starts iff its footprint fits;
  the first `backfill_depth` blocked jobs per group reserve the earliest slot;
  `strict_groups` implements PBS strict ordering inside the prod family;
  partitions pin queues (debug) to dedicated nodes; run and queued limits are
  enforced. Actual runtime frees nodes; requested walltime shapes the profile.
* **Priority** (`priority.py`, `safe_expr.py`): a whitelisted arithmetic
  expression over per-job PBS parameters, vectorised with numpy. The default is
  the formula fitted to recorded scores (`docs/PRIORITY.md`).
* **Metrics** (`metrics.py`): exact interval integration for busy nodes,
  censoring-aware wait summaries, KS / quantile / Spearman comparisons.
* **Replay** (`replay.py`): builds the window population, calibrates the
  machine, runs each candidate formula, compares, writes the report.

## Roadmap (stages 2-4)

2. **Workload model.** Fit a menu-independent demand model from the trace:
   arrival intensity by program x project x time-of-week with a per-project
   burst component (check the index of dispersion), joint (nodes, walltime,
   runtime ratio) bootstrap conditioned on program/project/month and a
   data-defined size class (never on queue names), and **censored walltime
   demand**: requested walltimes sitting at the historical cap are treated as
   right-censored so desired walltime can be sampled beyond old caps. Also a
   closed-loop pacing model for workflow users (see VALIDATION limitations).
3. **Response layer.** Intent -> submission under a hypothetical menu: routing,
   truncate / chain / resize with node-hour conservation, held-back submissions
   under `max_queued`. Every behavioural parameter has a prior.
4. **Experiments.** Two-level Monte Carlo (outer: epistemic parameters and
   block-bootstrapped trace months; inner: seeds) with common random numbers
   across menus; explicit objective module (utilisation vs 9,600, node-hours,
   per-class wait, capability-job fraction as a constraint); Pareto reporting;
   variance decomposition by uncertain input.
