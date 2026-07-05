# PBS Queue Simulator

Monte Carlo discrete-event simulation of a PBS-style job queue on a large HPC
machine (modeled on Aurora @ ALCF). It exists to answer allocation-policy
questions — most importantly: **how does introducing the Genesis Mission
allocation program, and deciding whose share it comes from, affect queue waits
and delivered node-hours for INCITE / ALCC / Discretionary?**

There are two simulators in this repo:

- **`sim.py`** — the original *size-based* model (jobs bucketed only by node
  count). Good for throughput/backfill studies; program-agnostic.
- **`sim_programs.py`** — the *program-aware* model (the current focus). Adds
  the allocation-program dimension — INCITE, ALCC, DD (Discretionary), and the
  new Genesis Mission — on top of the same discrete-event scheduler. This is the
  one that answers the committee's balancing-act question.

---

## What the program-aware simulation does (`sim_programs.py`)

### 1. Allocation programs

Every job belongs to one of four programs:

- **INCITE** — capability science, Jan–Dec allocation year, +25% overburn
  headroom.
- **ALCC** — Jul–Jun allocation year.
- **DD (Discretionary)** — continuous, flat demand, no fixed budget window
  (soft floor; never hard-capped when capacity is idle).
- **Genesis Mission** — brand new, no historical data, so modeled from explicit
  labeled scenarios (see below).

INCITE / ALCC / DD are **fitted from the real Aurora PBS trace database**
(`pbs_monitor_aurora.db`, ~416K finished jobs). For each program we fit:

- node-size distribution (they differ — INCITE/ALCC capability-leaning, DD
  bimodal with a 1-node spike),
- walltime / runtime distributions,
- a **monthly seasonal burn curve** (e.g. INCITE's big November push, ALCC's
  spring ramp, DD's flatness), so the simulation reproduces *when* each program
  demands time, not just how much.

### 2. Budgets and the scheduling policy

Fair-share is **not** enforced (ALCF doesn't). Instead we model per-program
annual node-hour budgets and a smooth priority damper:

- Each program has an annual budget `B_p = share_p × machine_node_hours/year`.
- A program's scheduling priority **damps smoothly** as its delivered
  node-hours approach its pro-rated budget — so under-using programs keep easy
  access and heavy burners yield, but nothing is blocked until the overburn
  ceiling (INCITE +25%, others 0).

Two policies (`--policy`):

- `blind` — program-agnostic (reproduces `sim.py` behavior; validation
  baseline).
- `budget` — the realistic budget-damped model above.

The scheduler underneath is the same discrete-event engine as `sim.py`:
priority scoring across a single global queue with **EASY backfill** and aging
to prevent starvation.

### 3. Genesis Mission — scenarios and the reallocation knob

Genesis has no history, so it's modeled as **configurable, labeled scenarios**
(`--genesis-scenario`):

- `ai_default` — AI-centric, 1-node / 7-day dominated (Phase-1 description),
  ramps Jul→Oct.
- `incite_like` — capability jobs: large nodes (256–2048), 6–24h walltimes.
- `bursty_campaign` — mid-to-large nodes with a fast Jul→Aug surge.

The central policy question is **where Genesis's share comes from**
(`--genesis-from`):

- `proportional` (a.k.a. `all`) — taken from INCITE/ALCC/DD in proportion to
  their base shares.
- `incite` — deducted entirely from INCITE.
- `dd` — deducted entirely from DD.

All four shares are renormalized to sum to 1.0 and printed at the top of each
run.

### 4. Outputs

- **Program decision table** (the committee table): per program — target vs
  delivered share, budget burn %, wait p50/p95/max, unstarted jobs.
- **CSV export** (`--csv PATH`): the decision table plus two telemetry files —
  `<stem>_telemetry.csv` (cumulative delivered node-hours per program over time)
  and `<stem>_util.csv` (machine utilization samples).
- **Committee figures** via `plot_programs_sim.py`: delivered-vs-target shares,
  stacked delivered node-hours over time, budget burn-down, and wait
  percentiles.

Large artifacts (`*.png`, `*.csv` under `results/`) are gitignored and tracked
via `results/MANIFEST.md`.

---

## Running it

```bash
python -m pip install -r requirements.txt

# Program-aware run: Genesis at 15% taken from INCITE, AI-default scenario.
# NOTE: Genesis ramps Jul->Oct, so use --start-month >= 8 (or a full year)
# or Genesis generates 0 jobs.
python3 sim_programs.py \
    --duration-days 30 --start-month 8 \
    --policy budget --seed 42 \
    --genesis-from incite --genesis-scenario ai_default \
    --csv results/run/decision.csv

# Committee plots from that run's CSVs
python3 plot_programs_sim.py \
    --csv results/run/decision.csv \
    --outdir results/run/figs \
    --title "Genesis 15% from INCITE"
```

Key flags (`sim_programs.py`): `--policy {blind,budget}`,
`--shares I,A,D,G` (default `0.50,0.25,0.10,0.15`),
`--genesis-from {proportional,all,incite,dd}`,
`--genesis-scenario {ai_default,incite_like,bursty_campaign}`,
`--start-month`, `--no-genesis`, `--duration-days`, `--seed`, `--csv`.

---

## Original size-based model (`sim.py`)

Still available for program-agnostic throughput/backfill work.

- **Machine**: configurable node count (default 10,624 ≈ Aurora nominal).
- **Queues** (4 buckets by job size): `tiny` (1–128, 168h cap), `small`
  (129–512, 72h), `medium` (513–2048, 48h), `large` (2049+, 24h).
- **Arrivals**: Poisson, configurable rate per queue.
- **Sizes/walltimes**: log-normal within each queue's bounds.
- **Scheduler**: priority scoring across a single global queue with EASY
  backfill + aging.

```
score(job) = base_priority(queue) + aging_rate(queue) * wait_hours
```

```bash
python3 sim.py --duration-days 7 --seed 42
```

---

## Status

- **Phase 1** ✅ — bursty AR(1) crash fixed; sigma calibration sped up.
- **Phase 2a** ✅ — per-program profiles fitted from the Aurora DB + profile
  plots.
- **Phase 2b** ✅ — Genesis share-reallocation knob (`--genesis-from`) and
  scenario set (`--genesis-scenario`); telemetry CSV export.
- **Phase 2d** ✅ — committee figures (`plot_programs_sim.py`).
- **Phase 3** (next) — multi-seed harness with 95% confidence intervals on all
  wait/share metrics; sensitivity sweeps.

See `REMEDIATION_PLAN.md` for the full plan and `ANALYSIS_REVIEW.md` for the
critique that motivated it.
