# Code Review — sched-sim-lator (feat/program-dimension)

Reviewer: Wesley · 2026-07-14
Workspace: `~/workspaces/sched-sim-lator`, branch `feat/program-dimension` @ e799375
Verified by running against the real Aurora DB (`/Users/jchilders/pbs_monitor_aurora.db`, 416K finished jobs).

---

## TL;DR

The program-aware rework is a genuinely good piece of design. The empirical
per-program fitting, the seasonal burn curves, the budget-damped priority with
overburn ceilings, the Genesis scenario machinery, and the CSV/telemetry
plumbing all closely follow the remediation plan and are cleanly written. The
docs (README / ANALYSIS_REVIEW / REMEDIATION_PLAN) are excellent and honest.

**But there is one blocking bug that invalidates every `sim_programs.py` result
produced so far, and it is not in the new code — it is an old lever from
`sim.py` that the new empirical arrival rates expose:**

> The `capacity` queue (all 1–128-node jobs) is throttled by a **512-node
> running pool cap**, but the real Aurora trace sends ~85% of jobs into that
> bucket. Offered load into the pool is **~4.4× its throughput**, so the
> capacity backlog grows without bound. The pending list balloons, the
> `O(n log n)`-per-pass scheduler degrades to quadratic wall-clock, a 30-day
> run **never terminates in practice**, and the machine sits **~97% idle**
> (running plateaus at ~320 of 10,624 nodes) while thousands of small jobs
> starve behind an artificial cap.

I confirmed this end-to-end and confirmed the fix direction: raising the pool
cap to a non-binding value makes the identical 30-day run **finish in 23.7 s,
start 46,899 / 46,953 jobs, and hit 64.7 % utilization** — right back to the
demand-bound ~66 % the original analysis expected.

Until this is fixed, no decision table or committee figure from
`sim_programs.py` is trustworthy: they were all generated under a silent,
unbounded capacity-queue backlog.

---

## 1. BLOCKER — capacity-pool cap chokes the empirical workload

### Evidence

Reproduced deterministically (seed 42, `--start-month 8`, `ai_default`,
`--genesis-from incite`, `--policy budget`, Aurora node count 10,624):

At simulated day 8 of a 30-day run:
```
pending = 7,225   running = 0   free_nodes = 10,624   (machine EMPTY)
pending by queue: {capacity: 7,214, large: 11}
delivered vs prorated budget: all programs UNDER budget (no ceiling hit)
pending currently over-ceiling: {}   (budget cap is NOT the cause)
```
The pending list keeps climbing (12,297 pending at day 12.5 and rising) and the
per-pass sort cost grows with it, so wall-clock time per simulated hour keeps
increasing — the run decelerates and does not finish.

Root-cause arithmetic (3-day generation, seed 42):
```
capacity-queue jobs offered:  164,004 node-h
512-node pool over 3 days:     36,864 node-h
offered / capacity = 4.4x  -> permanent, unbounded backlog
generated jobs by queue: capacity 4,329 | small 370 | medium 115 | large 28
```
~85–90 % of all generated jobs are 1–128 nodes → all funnel into a bucket
capped at 512 concurrently-running nodes.

### Why it happens

`CAPACITY_POOL_NODES = 512` was a reasonable protective lever in the *original*
`sim.py`, where capacity arrival rates were hand-tuned placeholders
(`arrival_rate_per_h = 25.4`, etc.). `sim_programs.py` correctly replaces those
placeholders with **empirically fitted** rates (INCITE 399/day, DD 644/day,
ALCC 116/day), but the 512-node pool cap was carried over unchanged. Empirical
small-job volume simply overwhelms a 512-node pool.

`ProgramScheduler._can_start` (and the inlined hot-path copy) still enforces the
pool cap via `sim.CAPACITY_QUEUE`, and `_estimate_reservation` still adds the
pool constraint for capacity jobs — so the cap is fully live in the new model.

### Fix options (pick per intent, not just to make it run)

1. **If the 512-node pool is not a real Aurora policy for this study** (most
   likely): raise it to non-binding / remove it in `sim_programs.py`. A run
   with `--capacity-pool 6000` already behaves correctly. Cleanest: make the
   program-aware model *not* inherit the size-bucket pool cap at all — programs,
   not size buckets, are the accounting unit now.
2. **If it *is* a real policy:** then the finding is itself a result ("Aurora's
   small-job pool is 4.4× oversubscribed under current demand") and must be
   surfaced explicitly, not left as a silent backlog. But then the size-bucket
   `capacity` queue and the program dimension are conflating two different
   axes — see §2.
3. Either way: **add a saturation guard** — abort or warn loudly when pending
   grows monotonically past a threshold, so an oversubscribed run can never
   again masquerade as a completed one.

**Regenerate every `sim_programs.py` artifact after the fix.** The current
decision tables report near-zero delivered shares for whoever lands most in the
capacity bucket, purely as an artifact of this chokepoint.

---

## 2. Design tension — size buckets vs. program dimension are conflated

The program-aware model still routes every job through `sim.QUEUES` (the four
size buckets) to get `base_priority`, `aging_rate`, `walltime_cap`, and the
capacity-pool cap. That is what drags the 512-node pool cap into a model whose
real accounting unit is now the *program*, not the size bucket.

This is the deeper version of review item #5's "`tiny`/`capacity` overload":
the size bucket named `capacity` now carries a scheduling constraint
(`CAPACITY_QUEUE` pool cap) that has nothing to do with allocation programs but
silently dominates program outcomes. Recommend separating the two concerns
explicitly:

- Size bucket → only walltime cap + base scoring weights.
- Program → budget, overburn, damping, delivered-NH accounting.
- Any running-pool cap should be an explicit, named, optional lever, not a
  side effect of a job's node count.

---

## 3. Performance — the scheduler is quadratic in a backlog

Even after the pool fix, `_try_schedule` sorts the entire pending list every
pass (`keyed = [...]; keyed.sort()`), and the run coalesces events but still
re-sorts on every finish batch. At healthy load (pending stays small) this is
fine — 30 days ran in 23.7 s. But it means the sim is only ever one
oversubscribed scenario away from the same quadratic wall — which is exactly the
Genesis-surge / elevated-load regime the tool exists to explore (review #6).

Recommendations:
- Keep `pending` in a priority structure (heap keyed on damped score, or a
  per-program deque + a small merge) instead of a full re-sort per pass.
- The `_min_pending_nodes` guard is a good exact early-exit; keep it, but it
  only helps when free_nodes is the binding constraint — under a pool-capped
  backlog it doesn't fire.
- The min-nodes cache is invalidated on every removal and lazily recomputed by
  a full `min()` scan — under a deep pending list that scan is itself O(n) and
  runs often. Track it incrementally on removal too, or use a heap.

---

## 4. What is genuinely good (keep it)

- **Empirical per-program fitting** (`program_profiles.py`) is clean and fast
  (~3 s to fit all three real programs). Node/walltime/rt-ratio arrays sampled
  with replacement, per-program seasonal multipliers fitted from the monthly
  node-hour time series — this directly realizes the plan.
- **Budget-damped priority with overburn ceiling** is a faithful, readable
  implementation of Taylor's stated policy (no fair-share; per-program budget +
  INCITE +25 % overburn; DD as a soft floor via `_ceiling_nh` omission). The
  smooth `exp(-damp_strength·(ratio−1))` damper is a nice touch.
- **Genesis scenario machinery** (`genesis_scenario`, ramp multiplier, share
  derivation from target node-hours) is well-factored and clearly labeled as
  assumption-driven.
- **`--genesis-from` reallocation** with renormalization + printed share table
  is exactly the first-class policy axis the plan called for.
- **Phase 1 bursty fix is real:** `var_log_m = sigma**2/(1-rho**2)` is present;
  `sim.py --bursty` runs clean (verified, exit 0). The sigma calibration is
  cached and vectorized.
- **CSV/telemetry export** (decision table + `_telemetry` + `_util`) is solid
  and idempotent on suffixes.

---

## 5. Correctness / smaller issues

- **Dead code still present:** `sim.Scheduler._start` (sim.py ~L566) is unused
  (the run loop uses `_start_at_index`). Review #7 flagged it; still there.
  Remove before it diverges from `_start_at_index`.
- **`summarize()` still prints** the "capacity pool peak not directly tracked in
  df" caveat (sim.py ~L690) even though `capacity_pool_samples` exists. Wire the
  real peak in.
- **`_month_at` uses fixed 30-day months** (`day // 30`). Over a 365-day run
  this drifts ~5 days by December and the last "month" is short. For seasonal
  burn curves it's probably tolerable, but it means the calendar and the burn
  multipliers are slightly misaligned from real month boundaries — worth a
  comment or a real calendar map, especially since INCITE's November burn is a
  headline behavior.
- **Genesis `full_lambda_per_day` derivation** divides annual NH target by
  `active_frac` of the year, but the ramp multiplier is renormalized to mean 1.0
  over active months — double-check the delivered Genesis share actually lands
  near its target across a full-year run (it's the whole point of the tool). In
  the 30-day Aug-start slice Genesis delivered ~0.47× prorated, which may just
  be the short window, but it needs a full-year validation.
- **Hard-coded DB path** `/Users/jchilders/pbs_monitor_aurora.db` is now behind
  `PBS_SIM_DB` env var in both entry points — good, that closes review #7's
  three-hard-coded-paths item for the new files. `run_bursty_comparison.py` and
  `trace_sampler` should be checked for the same.
- **`min_runtime_s=30` and walltime-repair heuristic** (`walltime = runtime*1.5`
  for bad rows) are reasonable but undocumented as modeling choices; note them.

---

## 6. Still-open items from the prior plan (not yet done)

- **Phase 3 (statistics):** no multi-seed / CI harness yet. Every number is
  seed 42. This is the highest-value next step *after* the pool fix — the p95 /
  max-wait tails driving any starvation claim need CIs, and the committee table
  needs mean±CI per program. The README correctly lists this as "next."
- **Sensitivity sweeps** (`CAP_PACKER_THRESHOLD`, re-inflation fraction, pool
  size, target shares) — not present.
- **Phase 5 realism** (preemption, MTBF derate, topology) — correctly deferred.

---

## Priority-ordered recommendations

1. **Fix the capacity-pool chokepoint** (§1) and add a saturation guard.
   Decouple size-bucket pool caps from the program model (§2). *Blocks
   everything.*
2. **Regenerate all `sim_programs.py` decision tables and figures** after the
   fix; the existing ones are artifacts of the backlog.
3. **Validate delivered-vs-target shares over a full 365-day run** — confirm the
   model reproduces the observed 57/30/13 baseline under `--policy blind`
   (the plan's own validation gate) before trusting `budget`-policy deltas.
4. **Make the scheduler backlog-robust** (heap instead of per-pass full sort)
   so elevated-load / Genesis-surge runs stay tractable (§3).
5. **Add the multi-seed + CI harness** (Phase 3) — no headline number should
   ship on a single seed.
6. Clean up dead `_start`, the capacity-pool-peak caveat, and document the
   30-day-month / walltime-repair modeling choices (§5).

---

## Reproduction (for the fix PR)

```bash
cd ~/workspaces/sched-sim-lator
uv venv .venv && source .venv/bin/activate && uv pip install -r requirements.txt
export PBS_SIM_DB=/Users/jchilders/pbs_monitor_aurora.db

# Reproduces the hang / idle-machine backlog (default pool=512):
python3 sim_programs.py --duration-days 30 --start-month 8 \
    --policy budget --seed 42 --genesis-from incite --genesis-scenario ai_default
# -> pending grows unbounded, machine ~97% idle, does not finish

# Confirms the fix direction (non-binding pool): finishes in ~24s, 99.9% started, 64.7% util
python3 sim_programs.py --duration-days 30 --start-month 8 \
    --policy budget --seed 42 --genesis-from incite --genesis-scenario ai_default \
    --capacity-pool 6000
```
