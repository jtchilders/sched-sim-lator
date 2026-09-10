# Recovering ALCF's job sort formula

ALCF documents the scheduler only qualitatively (larger jobs gain priority
faster, shorter jobs gain faster, INCITE/ALCC outrank discretionary, negative
balances are demoted). The PBS trace, however, records both the inputs and the
output:

* every job's `Resource_List` carries `base_score`, `score_boost`,
  `project_priority` (25 INCITE/ALCC, 20 some ALCC, 2 discretionary),
  `enable_wfp`, `wfp_factor` (1e5), `enable_fifo`, `fifo_factor` (1800),
  `enable_backfill`, `backfill_factor` (84600), `backfill_max` (50), `total_cpus`;
* `job_history.score` holds the score PBS computed at each monitor snapshot
  (~1M rows), and `etime` gives when a job became eligible.

Fitting on 312 snapshots of queued small/medium/large jobs with continuous
eligible-time accrual (no holds, `eligible_time == start - etime`), in log space:

```
score - base_score - score_boost  ~  eligible^1.95 * nodes^1.08 * walltime^-0.36 * project_priority^0.98
R^2 = 0.966, residual sd 0.53 (factor 1.7)
```

The clean integer form `eligible_h^2 * nodes * project_priority` has a constant
of 1.2e-5 that is stable across project priorities (2, 20, 25) and eligible
times (1-64 h). It equals `project_priority * nodes / total_nodes * (eligible_s / 1e4)^2`
within 2%. The apparent walltime dependence is a small set of snapshots whose
eligible time was shorter than `now - etime` (holds), not a real term.

FIFO queues (debug) score `51 + eligible hours` exactly (e.g. 51.00833 after
30 s). Backfill queues (base 0) score `min(backfill_max, eligible_s /
backfill_factor)` (0.00117 after 120 s vs 0.00142 predicted; stale-by-a-cycle
timing explains the gap). Held-back capacity jobs (over `max_queued`) score
~0: eligible time does not accrue in the routing queue.

Default in `schedsim.priority.ALCF_FITTED`:

```
base_score + score_boost
 + enable_wfp * project_priority * nodes / total_nodes * (eligible_s / 1e4) ** 2
 + enable_fifo * eligible_h
 + enable_backfill * min(backfill_max, eligible_s / backfill_factor)
```

Consequences: project priority is a **multiplier** (a DD job accrues 12.5x
slower than INCITE), the growth is **quadratic** in wait, and **linear in node
count** with no walltime term, so a 1-node capacity job with base_score 0 stays
near zero for days while an 8k-node job overtakes everything within hours.
In replay, this formula outranks the earlier cubic reconstruction and FIFO.
