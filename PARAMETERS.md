# PARAMETERS.md — every simulator parameter, explained

Complete reference for the `SimConfig` YAML. Each section maps to a dataclass in
`config.py`. For every parameter: what it means, why it matters, and a config
example. A run is **fully described by one YAML file**; anything omitted takes
the default shown.

Legend for the "scan?" column:
- **PRIMARY** — a main optimization axis you'll search over.
- **REGIME** — controls load/demand; policy levers only bite at some values.
- **CONTEXT** — fix per study; defines the scenario, don't sweep blindly.
- **FIDELITY/PERF** — leave at default unless studying it specifically.
- **OBJECTIVE INPUT** — shapes what's measured, not an input to optimize.

---

## `machine` — the hardware being modeled

| Parameter | Default | scan? | Significance |
|---|---|---|---|
| `total_nodes` | 10624 | CONTEXT | Physical node count (Aurora nominal). Hard ceiling on concurrent capacity. |
| `production_nodes` | 9600 | CONTEXT | **DOE-negotiated accountability denominator.** Utilization is measured against this (not `total_nodes`, since some nodes are always down). Also sets the capacity-job threshold = 20% = 1920 nodes. The number you're held accountable to. |

```yaml
machine:
  total_nodes: 10624
  production_nodes: 9600
```

---

## `size_tiers` — node-count bands (a list)

Each tier defines a band and its default scoring/walltime. This is the "queue
menu" in the classic sense. Fields per tier:

| Field | scan? | Significance |
|---|---|---|
| `name` | — | Tier label (e.g. capacity/small/medium/large). Used in per-queue metrics. |
| `min_nodes`, `max_nodes` | see note | The node-count band. **NOTE:** moving boundaries only *relabels* bootstrapped historical jobs — not a demand lever unless `behavior` is enabled. Tiers must cover [1, total_nodes] with no gaps. |
| `walltime_cap_h` | PRIMARY | Max requested walltime for jobs in this band. A *real* lever (truncates the workload). Superseded by `walltime_policy` when that is enabled. |
| `base_priority` | PRIMARY | Static starting priority. Higher = scheduled first, all else equal. The main knob to make big jobs beat small jobs. |
| `aging_rate` | PRIMARY | Priority gained per hour waited. Anti-starvation; higher = wait matters more vs. size. |

```yaml
size_tiers:
  - {name: capacity, min_nodes: 1,    max_nodes: 128,   walltime_cap_h: 168, base_priority: 5,  aging_rate: 0.5}
  - {name: small,    min_nodes: 129,  max_nodes: 512,   walltime_cap_h: 72,  base_priority: 20, aging_rate: 2.0}
  - {name: medium,   min_nodes: 513,  max_nodes: 2048,  walltime_cap_h: 48,  base_priority: 40, aging_rate: 5.0}
  - {name: large,    min_nodes: 2049, max_nodes: 10624, walltime_cap_h: 24,  base_priority: 80, aging_rate: 10.0}
```

---

## `programs` — allocation programs (INCITE / ALCC / DD)

| Parameter | Default | scan? | Significance |
|---|---|---|---|
| `name` | — | CONTEXT | INCITE / ALCC / DD. |
| `target_share` | .50/.25/.10 | CONTEXT | Yearly-average node-hour fraction the program is *meant* to get (policy intent; drives the budget). |
| `overburn` | INCITE .25 | CONTEXT | How far over budget a program may deliver before a hard cap (INCITE → 1.25×). Models ALCF's real over-tolerance. |
| `soft_floor` | DD true | CONTEXT | If true, never hard-capped when capacity is idle (DD is a soft floor, matching its historical ~30% over-run). |
| `alloc_year_start_month` | INCITE 1, ALCC 7, DD 1 | CONTEXT | **Key to calendar behavior.** Burn curves are keyed on months-since-this, so INCITE's fast January start and ALCC's slow post-July ramp are intrinsic, not calendar accidents. |

```yaml
programs:
  - {name: INCITE, target_share: 0.50, overburn: 0.25, alloc_year_start_month: 1}
  - {name: ALCC,   target_share: 0.25, overburn: 0.0,  alloc_year_start_month: 7}
  - {name: DD,     target_share: 0.10, overburn: 0.0,  soft_floor: true, alloc_year_start_month: 1}
```

---

## `generator` — the joint Monte-Carlo job generator

| Parameter | Default | scan? | Significance |
|---|---|---|---|
| `condition_on` | (program, alloc_month_offset, size_tier) | FIDELITY | Cells the joint sampler bootstraps whole rows from — preserves size↔duration correlation. |
| `min_cell_rows` | 20 | FIDELITY | Below this many rows in a cell, fall back to the program-wide pool (avoids sampling outliers). |
| `min_runtime_s` | 30 | FIDELITY | Drop sub-30s noise jobs from the trace fit. |
| `burn_curve_key` | alloc_month_offset | FIDELITY | What the seasonal arrival-rate multiplier is keyed on. |
| `seed` | 42 | — | Generator RNG seed (per-run seed overrides). |
| `load_multiplier` | 1.0 | **REGIME** | **Scales ALL arrival rates uniformly.** The demand knob: <1 lightens, >1 stresses. Sweep this to find the demand-bound → policy-bound transition — most policy levers only bite at certain load. |

```yaml
generator:
  load_multiplier: 0.6
  min_cell_rows: 20
```

---

## `genesis` — Genesis Mission (synthesized; no historical data)

| Parameter | Default | scan? | Significance |
|---|---|---|---|
| `enabled` | true | CONTEXT | Whether Genesis is in the run. |
| `scenario` | ai_default | CONTEXT | Job-mix assumption: `ai_default` (1-node/7-day AI), `incite_like` (big capability), `bursty_campaign` (mid, fast surge). |
| `share` | 0.15 | CONTEXT | Genesis's target node-hour fraction. |
| `genesis_from` | proportional | CONTEXT | **Where Genesis's share comes from:** `proportional` / `incite` / `dd`. The central balancing-act question. |
| `ramp_start_month` / `ramp_full_month` | 7 / 10 | CONTEXT | Ramp window (Jul→Oct) as Genesis spins up. |

```yaml
genesis:
  enabled: true
  scenario: ai_default
  share: 0.15
  genesis_from: incite
```

---

## `projects` — per-project allocation layer (competitive-award realism)

| Parameter | Default | scan? | Significance |
|---|---|---|---|
| `enabled` | false | CONTEXT | Turns on per-project awards / over-allocation / under-use. Off = program-level only. |
| `over_allocation` | 1.15 | CONTEXT | Programs award this multiple of their fraction across projects (most under-use). Reproduces deliberate over-subscription. |
| `project_damp_strength` | 6.0 | CONTEXT | How hard a project's priority damps as it nears its award (heavy burners yield, idle headroom flows to actives). |
| `min_project_jobs` | 20 | FIDELITY | Projects below this fold into a `<PROG>_misc` pseudo-project. |

```yaml
projects:
  enabled: true
  over_allocation: 1.15
  project_damp_strength: 6.0
```

---

## `deadlines` — conference-deadline submission spikes

| Parameter | scan? | Significance |
|---|---|---|
| `enabled` | CONTEXT | Whether deadline bursts apply. |
| per deadline `name`/`month`/`day` | CONTEXT | When the deadline falls. |
| `lead_days` | CONTEXT | Length of the submission-spike window before it. |
| `rate_multiplier` | CONTEXT | Arrival-rate multiplier in the window. |
| `affected_fraction` | CONTEXT | Fraction of projects that "chase" this deadline. |

```yaml
deadlines:
  enabled: true
  deadlines:
    - {name: SC, month: 4, day: 1, lead_days: 28, rate_multiplier: 2.0, affected_fraction: 0.35}
    - {name: NeurIPS, month: 5, day: 20, lead_days: 21, rate_multiplier: 2.2, affected_fraction: 0.25}
```

---

## `walltime_policy` — the queue-menu-as-one-rule (Stage 2)

| Parameter | scan? | Significance |
|---|---|---|
| `enabled` | PRIMARY | Replaces per-tier `walltime_cap_h` with a single `max_walltime(nodes)` rule. |
| `breakpoints` | PRIMARY | List of `(min_nodes, max_walltime_h)`. Encodes ANY size→walltime shape: small-gets-long, big-unlocks-long, or U-shape. This IS the collapsed queue menu. |

```yaml
# small jobs get 7 days (AI), full-machine capped at 24h (MTBF)
walltime_policy:
  enabled: true
  breakpoints:
    - [1, 168]
    - [512, 48]
    - [1920, 24]
```

---

## `behavior` — behavioral size-choice model

| Parameter | Default | scan? | Significance |
|---|---|---|---|
| `enabled` | false | PRIMARY (for menu studies) | Lets jobs **re-choose node count** in response to the walltime incentive — the only honest way to test a menu that would change user behavior. |
| `adapt_fraction` | 0.3 | PRIMARY | Fraction of jobs that resize to chase their desired walltime. |
| `program_adapt` | () | PRIMARY | Per-program overrides, e.g. INCITE more walltime-motivated. |
| `max_resize_nodes` | 1920 | CONTEXT | Ceiling on behavioral resize (a 1-node job can't jump to 10k). |
| `min_desired_walltime_h` | 24.0 | CONTEXT | Only jobs wanting ≥ this bother resizing. |

```yaml
behavior:
  enabled: true
  adapt_fraction: 0.3
  program_adapt:
    - [INCITE, 0.5]
    - [DD, 0.1]
```

---

## `scheduler` — the PBS-faithful scheduling engine

| Parameter | Default | scan? | Significance |
|---|---|---|---|
| `enable_backfill` | true | CONTEXT | EASY backfill on/off. |
| `score_expr` | `base + aging_rate*wait` | **PRIMARY** | **The configurable priority formula** (safe AST-parsed string). See variables below. The primary optimization axis. |
| `budget_damp_strength` | 8.0 | PRIMARY | How sharply a program's priority damps past its pro-rated budget (feeds the `budget_damp` variable). |
| `policy` | budget | CONTEXT | `blind` (program-agnostic) or `budget` (budget-damped priority + overburn cap). |
| `reserve_min_nodes` | 0 → 1920 | **PRIMARY** | **Large-job-starvation lever.** Draining reservation engages only for jobs ≥ this (0 = default capacity threshold). Below it, blocked jobs just wait. This is what protects 10k-node jobs. |
| `sched_cycle_h` | 0.1667 (600s) | REGIME/PERF | **PBS `scheduler_iteration`.** One scheduling pass per cycle, not per event — fidelity match + perf fix. Also a research knob (cycle length vs. wait). |
| `examine_cap` | 4000 | PERF | Beyond this many pending jobs, only the top-K by priority are examined per cycle (mirrors PBS `backfill_depth`). |

### Score-expression variables

`score_expr` is any safe arithmetic expression over these per-job, per-cycle
variables (functions allowed: `min max abs exp log sqrt floor ceil pow`):

| Variable | Meaning |
|---|---|
| `base` | job's size-tier `base_priority` |
| `wait` | hours the job has waited |
| `nodes` | job node count |
| `walltime` | requested walltime (h) |
| `aging_rate` | job's size-tier `aging_rate` |
| `budget_ratio` | program delivered / pro-rated budget |
| `budget_damp` | smooth damping factor in (0,1] as budget is approached |
| `project_damp` | per-project damping in (0,1] (1.0 if project layer off) |
| `delivered_share` | program's delivered share so far |
| `target_share` | program's target share |
| `now` | sim time (h) |
| `queue_depth` | current pending count |
| `free_nodes` | currently free nodes |

```yaml
scheduler:
  enable_backfill: true
  policy: blind
  score_expr: "base + aging_rate*wait"
  reserve_min_nodes: 0          # 0 -> capacity threshold (1920)
  sched_cycle_h: 0.16667        # 600s / 10 min
```

Score-expression examples (see `configs/scores/` for ready presets):

```yaml
score_expr: "base + aging_rate*wait"                                    # FIFO + aging
score_expr: "base + aging_rate*wait - 0.002*nodes"                      # favor small jobs
score_expr: "base + aging_rate*wait + 0.001*nodes"                      # favor big jobs
score_expr: "base + aging_rate*wait + 30*(target_share - delivered_share)"  # pull to allocation targets
score_expr: "(base + aging_rate*wait) * budget_damp"                    # budget-aware
score_expr: "base + aging_rate*wait + 20*(target_share-delivered_share) - 0.5*log(nodes)"  # combined
```

---

## `run` — the simulation window and replication

| Parameter | Default | scan? | Significance |
|---|---|---|---|
| `duration_days` | 365 | CONTEXT | Simulated span. |
| `start_month` | 1 | CONTEXT | Calendar month the sim clock starts (aligns seasonal curves). |
| `sample_dt_h` | 1.0 | OBJECTIVE INPUT | **Measurement** granularity (telemetry cadence) — NOT scheduling granularity. Δt sharpens metrics without biasing waits. Set to 24 for daily util points. |
| `n_seeds` | 1 | OBJECTIVE INPUT | Runs per config for confidence intervals. |
| `base_seed` | 42 | — | First seed; seed i = base_seed + i. |

```yaml
run: {duration_days: 365, start_month: 1, sample_dt_h: 24.0, n_seeds: 5, base_seed: 42}
```

---

## `output` — results and the safety guard

| Parameter | Default | scan? | Significance |
|---|---|---|---|
| `outdir` | results/run | — | Where outputs land. |
| `write_decision_table` / `write_timeseries` / `write_jobs` | T/T/F | — | Which artifacts to emit (`write_jobs` = per-job detail, large). |
| `saturation_abort_pending` | 100000 | PERF | Abort with a clear message if pending exceeds this (guards against oversubscribed configs grinding). |

```yaml
output:
  outdir: results/my_run
  write_timeseries: true
  saturation_abort_pending: 200000
```

---

## Objectives (measured, not set)

These come out of a run; they're what you optimize *toward*, not inputs:

- **System utilization** — busy nodes / `production_nodes`, over time and integrated.
- **Allocation fidelity** — per-program delivered share vs `target_share`, budget burn.
- **Queued time** — wait percentiles overall and split large (≥1920) vs small.
- **Throughput** — completed jobs/day, node-hours/day.
- **Starvation** — unstarted counts, especially large jobs.

See `metrics.py` (per-run) and `analyze.py` (across a scan) for how these are
computed and plotted.
