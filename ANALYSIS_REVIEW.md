# Critical Review — PBS Queue Simulator

Reviewer: Wesley (⚛️) · 2026-06-19
Scope: `~/projects/pbs-queue-sim/` as it exists today.

## TL;DR

The simulator is a competent, well-structured single-machine PBS discrete-event
model with a genuinely thoughtful empirical-fitting pipeline (the two-component
walltime model is the best idea in here). **But it cannot currently answer the
question you actually need it for.** There is *no representation of allocation
programs* (INCITE / ALCC / Genesis Mission) anywhere in the model — jobs are
bucketed purely by node count. The INCITE/ALCC-vs-Genesis balancing act is a
*policy/allocation-share* problem, and the model is a *job-size* model. That is
the headline gap.

Secondary, but important: a confirmed crash bug in the bursty arrival path means
the `bursty_sweep_results.csv` artifact going into the slides **cannot be
reproduced by the current code**, and the project is not under version control.

---

## 1. The headline gap: no allocation/program dimension

Everything in the model keys off node count → one of {capacity, small, medium,
large}. There is no concept of:

- **Allocation program** (INCITE, ALCC, Genesis, Director's Discretionary).
- **Per-program allocation budgets** (node-hours awarded per quarter/year).
- **Burn-rate / fair-share accounting** against those budgets.
- **Program-level priority or quotas** in the scheduler.

To "address the balancing act of INCITE/ALCC vs Genesis," the model needs a
**program tag on every job** plus a scheduling lever that trades the programs
off. Concretely, add at least one of:

1. **Fair-share scheduler term.** Add `program` to `Job`; track delivered
   node-hours per program; add a score term that boosts under-served programs
   and damps over-served ones relative to their target share. This is the
   standard HPC mechanism (PBS `fairshare`, Slurm multifactor) and it's what
   ALCF actually argues about.
2. **Per-program reservations / caps.** Like the existing capacity-pool cap, but
   per program: e.g. "Genesis may not exceed X% of running node-hours" or
   "INCITE+ALCC guaranteed floor of Y%."
3. **Mixed arrival streams.** Drive arrivals per (program × size bucket) so you
   can model "Genesis ramps up to 30% of submitted node-hours over Q3."

Without one of these, the sim can show *that* the machine fills up, but not
*who wins* when INCITE deadlines collide with a Genesis surge — which is the
whole decision.

**Recommended minimal design (DataFrame-friendly):**
- Trace/job row gains a `program` column (default empirical mix from the DB if
  the `account`/`project` field exists; otherwise a configurable mix).
- Scheduler maintains `delivered_nh[program]` and `target_share[program]`.
- `score += fairshare_weight * (target_share[p] - actual_share[p])`.
- Report becomes a per-program table: node-hours delivered vs target, p95 wait,
  starvation — exactly the trade-off table you'd put in front of the committee.

---

## 2. Confirmed bug: `--bursty` is broken (NameError)

In `JobGenerator._generate_daily_multipliers` (sim.py ~L180–200), `var_log_m` is
**used but never defined**:

```python
log_m[0] = self.rng.normal(0, np.sqrt(var_log_m))   # L194 — NameError
...
multipliers = np.exp(log_m - var_log_m / 2)         # L199 — NameError
```

Verified empirically: any `--bursty` run dies with
`NameError: name 'var_log_m' is not defined`. The fix is to compute it from the
calibrated sigma that the function already produces:

```python
sigma = self._calibrate_sigma(n_days, rho, cv)
var_log_m = sigma**2 / (1 - rho**2)   # stationary variance of the AR(1)
```

**Reproducibility consequence:** `results/distributions/bursty_sweep_results.csv`
(and the bursty slides built from it) were produced by code that no longer runs.
You cannot regenerate that figure today. Before this goes in front of anyone,
fix the bug, re-run, and confirm the numbers still hold. The bursty story is
also the one most relevant to Genesis (a new program is exactly a bursty,
autocorrelated demand surge), so this path needs to actually work.

---

## 3. No version control

`pbs-queue-sim/` is not a git repo. For an analysis that's generating slides and
feeding a procurement/policy decision, that's a real risk: no provenance for any
figure, no way to tie a result to the code that made it, easy to silently break
(see #2). Recommend `git init`, commit the code, and `.gitignore` the `.venv/`,
`*.pkl` caches, and large `results/*.png`/`slides.html` (or commit results in a
separate tagged snapshot). Each figure should be reproducible from a known SHA.

---

## 4. Scheduler modeling oversights

These won't crash anything but they bias the conclusions:

- **No node-level topology / fragmentation.** Nodes are a fungible integer pool.
  Real Aurora scheduling is constrained by dragonfly groups, rack/blade
  boundaries, and contiguity for large jobs. A "free_nodes ≥ requested" test
  overstates schedulability for large jobs and *understates* large-queue wait —
  which matters because the INCITE/large-job story leans on those waits.
- **Backfill uses requested walltime, actual runtime is hidden — good — but
  reservations are only ever made for the single top-scored job.** Real EASY
  reservations and the multi-reservation conservative variants behave
  differently under heavy mixed load. The single-reservation model tends to be
  optimistic on backfill throughput. Worth a sensitivity note.
- **No reservations / maintenance / draining.** Real machines lose capacity to
  scheduled maintenance, dedicated time, and node failures. Average utilization
  of ~66% in the runs may be partly a real artifact and partly because demand
  (arrival rate) is the binding constraint, not policy — see #6.
- **No preemption.** The `on-demand` partition is modeled as nodes *removed*
  from the large ceiling, not as genuinely preemptable capacity. If Genesis is
  going to get preemptable/on-demand priority, preemption is the mechanism to
  model, and it's absent.
- **MTBF is mentioned in comments as the rationale for the 24h large cap but is
  not modeled.** No job failures, no resubmits. For large jobs this changes
  effective throughput materially.
- **`_estimate_reservation` ignores backfilled jobs' actual (shorter) runtimes**
  by design (scheduler can't see them) — correct — but it also doesn't account
  for the capacity pool releasing in the non-capacity branch, so reservation
  times for main-queue jobs can be slightly pessimistic. Minor.

---

## 5. Statistical / empirical-fitting issues

The fitting pipeline is the strongest part, but:

- **Two-component walltime model rests on one untested assumption:** that
  cap-packers under the *old* cap will re-inflate to the *new* cap with the
  *capacity-queue's* shape. That's a plausible behavioral model, but it's a
  modeling *choice presented as data*. It should be labeled as an assumption
  with a sensitivity band (e.g. what if only 50% of cap-packers re-inflate?).
  Right now the slides risk presenting a behavioral prior as an empirical fit.
- **`CAP_PACKER_THRESHOLD = 0.90` is a hard knob with no sensitivity sweep.**
  The split between "interior" and "cap-packer" jobs drives the whole walltime
  story. Sweep it (0.8 / 0.9 / 0.95) and show the conclusions are stable.
- **Arrival rate is a single pooled Poisson rate per bucket** computed as
  `n / span_h` over the *whole* source window. This throws away the strong
  diurnal/weekly structure you separately plotted (submission_hourly/weekly).
  Flat Poisson under-represents peak-hour contention. The bursty AR(1) model is
  the intended fix — which makes bug #2 doubly important.
- **`rng.choice` resampling of empirical arrays** is bootstrap-with-replacement
  from finite samples. Fine for the body, but tail behavior (the p95/p99 waits
  you report) is dominated by a handful of real extreme jobs — those tails are
  noisy and should carry confidence intervals from multi-seed runs.
- **No multi-seed / CI reporting anywhere.** Every headline number is a single
  seed (42). p95 and max-wait especially need N seeds with mean±CI, or the
  starvation findings aren't defensible. This is the single highest-value cheap
  improvement after fixing the program dimension.
- **`tiny` vs `capacity` naming is overloaded and confusing.** The DB `tiny`
  queue maps to the sim `capacity` bucket, but `capacity` is *also* a DB queue
  used as the cap-packer donor, *and* `capacity` is the sim bucket name. Three
  meanings for two words. This is a documentation/foot-gun risk; rename sim
  buckets to size labels (xs/s/m/l) and keep program/queue names separate.

---

## 6. Interpretation risk: utilization is demand-bound, not policy-bound

Across the replay (full year) and empirical runs, utilization sits at ~66% with
near-zero unstarted jobs at 1.0× scale. That strongly suggests **the machine is
not saturated by historical demand** — the queue policy is barely being
stressed. Conclusions like "policy X gives 66% utilization" are really saying
"historical Aurora demand fills the machine to 66%." The interesting policy
differences only appear at the 0.5×–1.0× *scaled-up* capacity arrival rates
where unstarted jobs explode (4000 at 1.0×). The slides should be explicit that
**the policy questions only become live under elevated load** — which is exactly
the regime Genesis Mission introduces. Frame Genesis as the demand increment
that moves the system from demand-bound to policy-bound.

---

## 7. Smaller items

- `Scheduler._start` and `_start_at_index` are two code paths that start jobs;
  `_start` looks like dead/legacy code (the run loop uses `_start_at_index` via
  `_try_schedule`). Remove `_start` to avoid divergence.
- `summarize()` prints a "capacity pool peak not directly tracked in df" caveat
  even though `sched.capacity_pool_samples` exists — wire the real peak in.
- `max_wait_h` is keyed off the module-level `QUEUES`, so it silently breaks if
  someone deepcopies QUEUES with renamed buckets (the sweep driver deepcopies).
- Hard-coded absolute DB path `/Users/jchilders/pbs_monitor_aurora.db` in three
  places (sim default, run_bursty_comparison, trace_sampler implicit). Make it
  one config/env var.
- The `O(n log n)` re-sort of all pending jobs on every schedule pass is fine at
  these sizes but will bite if you scale arrival rate or duration much further;
  note it before someone runs a 10× sweep and wonders why it's slow.

---

## Priority-ordered recommendations

1. **Add the program/allocation dimension (INCITE/ALCC/Genesis) + fair-share or
   per-program caps.** Without this the tool doesn't answer the question. (#1)
2. **Fix the `var_log_m` crash and re-run the bursty sweep; confirm slide
   numbers.** (#2)
3. **`git init` + reproducibility hygiene** so every figure ↔ a SHA. (#3)
4. **Multi-seed runs with CIs** on all wait/starvation metrics. (#5)
5. **Sensitivity sweeps** on `CAP_PACKER_THRESHOLD`, cap-packer re-inflation
   fraction, and capacity-pool size. (#4, #5)
6. **Reframe the narrative** around demand-bound vs policy-bound, with Genesis as
   the load that crosses that line. (#6)
7. Model **preemption / on-demand as real preemptable capacity** if Genesis is
   going to get on-demand priority. (#4)
8. Add **topology/fragmentation and a maintenance/MTBF derate** as a second-order
   realism pass, after the above. (#4)
