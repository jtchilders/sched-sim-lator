# Configurable Monte Carlo Queue Simulator (v2)

A config-driven, discrete-event simulation of the Aurora job queue, built to
study how allocation policy, capacity-protection strategy, and the score
function affect **utilization, throughput, and queue-time** — one feature at a
time, across many runs.

This is the v2 architecture (branch `feat/configurable-montecarlo`). The v1
files (`sim.py`, `sim_programs.py`, `program_profiles.py`) remain for reference.

## Design principles

1. **Everything is config.** A run is fully described by one YAML file. No
   hard-coded policy values. Every output is tagged with the config's hash for
   provenance, so a figure always ties back to the exact parameters.
2. **Joint conditional generation.** Jobs are bootstrapped as *whole rows*
   (nodes, walltime, runtime kept together) from real trace cells conditioned
   on `(program, alloc-month-offset, size_tier)` — preserving the real
   correlation between size and duration that independent-marginal sampling
   destroys.
3. **Program calendars are intrinsic.** Seasonal burn curves are keyed on
   *months-since-allocation-year-start*, so INCITE's fast January start, ALCC's
   slow post-July ramp, and DD's flatness are program properties, not calendar
   coincidences.
4. **Discrete-event core; Δt is measurement, not scheduling.** The scheduler is
   event-driven (correct wait-time tails). `run.sample_dt_h` only sets the
   cadence at which metrics are snapshotted — it never delays a job.
5. **Capacity protection is a pluggable, testable strategy** — not a baked-in
   constant. The old 512-node pool cap is now one option among several to A/B.
6. **Configurable score function.** The priority formula is a string in YAML,
   parsed safely (AST-validated, then compiled — no `eval` of arbitrary code).

## Modules

| File | Role |
|------|------|
| `config.py` | `SimConfig` (nested dataclasses ↔ YAML), validation, `config_hash()` |
| `score_expr.py` | Safe, compiled score-expression evaluator |
| `generator.py` | Joint conditional Monte Carlo job generator + DB fit |
| `genesis.py` | Genesis Mission synthesis (assumption-driven scenarios) |
| `scheduler.py` | Discrete-event scheduler, protection strategies, budget, guard |
| `metrics.py` | Decision table + tidy long-format time-series + summary stats |
| `run.py` | `python run.py --config foo.yaml` |

## Running

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
export PBS_SIM_DB=/Users/jchilders/pbs_monitor_aurora.db   # or set trace_db in YAML

python run.py --config configs/validate_baseline.yaml
python run.py --config configs/validate_baseline.yaml --print-only   # resolve + hash only
```

Outputs (to `output.outdir`): `decision_table.csv`, `decision_table_agg.csv`
(mean/std over seeds), `timeseries.csv` (tidy: `t_h, metric, program, value`),
`resolved_config.yaml`, and `manifest.json` (config hash + per-seed summaries).

## The score function

`scheduler.score_expr` is any arithmetic expression over these variables
(per job, at scheduling time):

`base`, `wait`, `nodes`, `walltime`, `aging_rate`, `budget_ratio`,
`budget_damp`, `delivered_share`, `target_share`, `now`, `queue_depth`,
`free_nodes`.

Functions allowed: `min max abs exp log sqrt floor ceil pow`. Anything else
(imports, attribute access, comprehensions, unknown names) is rejected at
config-load time. Examples:

```yaml
score_expr: "base + aging_rate * wait"                                  # FIFO+aging
score_expr: "(base + aging_rate*wait) * budget_damp"                    # budget-damped
score_expr: "base + aging_rate*wait + 20*(target_share - delivered_share)"  # fair-share pull
score_expr: "base + aging_rate*wait - 0.001*nodes"                      # small-job favoring
```

## Capacity-protection strategies

Set via `capacity_protection.strategy`. This is the lever the study exists to
tune (stop low-node long jobs from swamping big jobs):

- `none` — no protection (baseline; shows the swamping problem).
- `running_pool_cap` — cap concurrent nodes used by a protected tier
  (`pool_nodes`, `protected_tier`). The original 512 mechanism.
- `dedicated_partition` — physically reserve `partition_nodes` for big jobs
  (`big_job_min_nodes`); small jobs are limited to the remainder.
- `size_reservation` — (stub; reserved for a big-job-reservation guarantee.)

## Oversubscription guard

Before the event loop, `_preflight_check` compares offered load against the
protection strategy's throughput ceiling. If a config is oversubscribed (the
backlog would grow without bound), the run **aborts in seconds** with an exact
diagnosis rather than hanging. Example (the old 512 cap on the empirical
workload):

```
OVERSUBSCRIBED: protected tier 'capacity' is offered 1,785,628 node-h but the
running_pool_cap of 512 nodes can deliver at most 368,640 node-h over 30d
(4.8x oversubscribed). ...
```

## Validation (why you can trust the generator)

`configs/validate_baseline.yaml` (full year, no protection, blind policy) — the
generator reproduces the observed delivered-share split:

| Program | Simulated (365d) | Real (whole trace) |
|---------|-----------------:|-------------------:|
| INCITE  | 49.6%            | 57.0%              |
| DD      | 35.6%            | 29.8%              |
| ALCC    | 14.8%            | 13.2%              |

ALCC correctly fills in over its allocation year (9.9% in a 60-day Jan window →
14.8% full-year), confirming the program-calendar model. Utilization ~72.6%,
consistent with the historical demand-bound ~66%.

## Sweeps

A sweep is a set of YAML files (or one base + overrides) differing in one field
— e.g. `capacity_protection.pool_nodes ∈ {256, 512, 1024, 2048}` or a set of
`score_expr` strings — each run tagged by `config_hash`.

Use `sweep.py` with a small grid file:

```bash
python sweep.py --grid configs/sweeps/score_sweep.yaml
python sweep.py --grid configs/sweeps/score_sweep.yaml --seeds 3   # multi-seed
```

Grid file format (each key is a dotted path into the config; values are the list
to sweep; runs = cartesian product):

```yaml
base: configs/validate_baseline_30d.yaml
name: score_sweep
grid:
  scheduler.score_expr:
    - "base + aging_rate*wait"
    - "base + aging_rate*wait + 30*(target_share-delivered_share)"
  capacity_protection.pool_nodes: [512, 1024, 2048]
```

Output: `results/<name>/sweep_results.csv` with the three objectives per run
(`avg_util_pct`, `alloc_util`, `wait_p50_h/p95_h`) plus per-program delivered
share and a printed leaderboard. Oversubscribed configs are caught by the
pre-flight guard and recorded as `status=SATURATED` without killing the sweep.

### What the model CAN and CANNOT optimize (read before sweeping)

The `size_tiers` (queue menu) serve two roles, with very different fidelity:

- **`walltime_cap_h` IS a real lever** — it truncates job walltimes in the
  generator, changing the workload and scheduling. Optimize freely.
- **Per-tier `base_priority` / `aging_rate` and the `score_expr` ARE real
  levers** — they re-rank a fixed workload, which is exactly what the model
  does. Optimize freely.
- **`capacity_protection` IS a real lever** — pluggable strategy for the
  small-vs-big-job tradeoff. Optimize freely.
- **Node `min_nodes` / `max_nodes` boundaries are NOT a demand lever.** The
  generator bootstraps real historical job rows, so moving a node boundary only
  *relabels* the same jobs into different tiers — it does **not** change how many
  big/small jobs users submit. A real queue-menu change induces user behavioral
  response (resize, resubmit, migrate) that this model does not represent.
  Optimizing a node-boundary menu against the current model answers "what if we
  re-scored the same jobs," not "what if users adapted to a new menu." Treat
  node-boundary conclusions as invalid until a behavioral-response layer exists.

**Load regime matters.** At ≥1.0× offered load the machine is demand-bound and
utilization pins near its ceiling regardless of score — the score function then
only changes *who waits*, not *how full* the machine is. Score/protection
optimization for utilization only bites at the load regimes the pre-flight
NOTE flags. Sweep across load scales, not just policies.
