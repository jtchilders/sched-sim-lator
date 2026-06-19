# Remediation Plan — PBS Queue Simulator

Author: Wesley (⚛️) · 2026-06-19 (rev 2 — adds program profiles & Genesis)
Companion to: `ANALYSIS_REVIEW.md`
Repo: https://github.com/jtchilders/sched-sim-lator (pushed as `jtchilders-ai-assistant`)
Goal: make this tool able to answer the **INCITE/ALCC vs Genesis Mission
balancing** question, with reproducible, defensible results.

## Status (rev 2)

- **Phase 0 DONE:** repo initialized, baseline committed + tagged `v1-baseline`,
  pushed to `jtchilders/sched-sim-lator`. `.gitignore` excludes venv/caches/DB/
  large artifacts. (Remaining Phase 0 items — dep pinning, env-var DB path,
  RUNBOOK — folded into Phase 1.)
- **Key discovery:** the pbs_monitor DB already has the program label we need.
  The `jobs` table has both `allocation_type` (INCITE / ALCC / Discretionary /
  UNKNOWN) and `project`. **Program profiles can be fully data-driven**, not
  guessed. Empirical facts below now anchor Phase 2.

---

## Guiding principles

- **Land foundation before features.** Get version control + reproducibility in
  place first so every later change is traceable and every figure ties to a SHA.
- **Don't disturb existing results.** All new work on branches; current
  `results/` and `slides.html` stay as the "v1" snapshot until superseded.
- **Each phase ends with a runnable artifact** (a figure, a table, or a test),
  not just code.
- **Statistics before polish.** Confidence intervals and sensitivity bands make
  or break credibility in front of a committee; topology realism is second-order.

---

## Empirical program facts (from the Aurora DB, 357 days, 416K finished jobs)

These drive the Phase 2 profiles. All from `allocation_type` on FINISHED jobs
with `nodes>0` and `runtime>=30s`.

**Delivered node-hour share (what actually happened):**

| Program | Jobs | Node-hours | Share | Target |
|--------|------|-----------|-------|--------|
| INCITE | 142,435 | 38.9M | **57.0%** | 60% |
| Discretionary | 230,071 | 20.3M | **29.8%** | 10% |
| ALCC | 41,540 | 9.0M | **13.2%** | 30% |
| UNKNOWN | 1,965 | 0.06M | 0.1% | — |

Note the gap between *delivered* and *target* shares — DD massively over-ran its
10% nominal hold (29.8%), ALCC under-delivered (13.2% vs 30%). That gap is itself
a finding: the current program-blind policy does not steer toward targets. This
is the baseline the fair-share lever must improve on.

**Job-size profile (nodes):**

| Program | mean | p50 | p90 | p99 | character |
|--------|------|-----|-----|-----|-----------|
| INCITE | 110.8 | 16 | 256 | 2048 | largest, capability-leaning |
| ALCC | 92.8 | 12 | 256 | 1800 | similar to INCITE, slightly smaller |
| Discretionary | 87.1 | **1** | 108 | 2048 | bimodal: many 1-node + occasional big |

**Runtime (hours):** ALCC longest (mean 2.55h), INCITE 1.31h, DD shortest
(0.79h) — DD = lots of short experimental jobs. **Arrival rate:** DD 644/day,
INCITE 399/day, ALCC 116/day.

**Temporal / calendar structure (the important behavioral signal):**
- INCITE node-hours **peak Nov 2025 (10.4M)** — year-end burn before the
  Dec deadline — then collapse to 1.6M by Feb 2026. Classic allocation-year
  end-of-cycle rush, consistent with the "13th month" extension behavior.
- ALCC **ramps after its July start**, builds through the spring (peaks
  May 2026 at 1.8M), consistent with a July–June allocation year.
- DD is **steady year-round** — small experimental jobs, no strong cycle.

This confirms the calendars you described and means we can fit per-program
*seasonal burn curves* from data rather than assuming flat demand.

---

## Phase 0 — Foundation & reproducibility  ✅ DONE (repo) / partially deferred

Closes review items **#3** (no VCS) and the reproducibility half of **#2**.

- [x] `git init`; baseline committed + tagged `v1-baseline`; pushed to GitHub
      as `jtchilders-ai-assistant`.
- [x] `.gitignore`: `.venv/`, `*.pkl`, `__pycache__/`, `*.db`, large outputs.
- [ ] *(→ Phase 1)* `results/MANIFEST.md` recording SHA + command per figure.
- [ ] *(→ Phase 1)* Pin dependency versions in `requirements.txt`.
- [ ] *(→ Phase 1)* DB path from env var `PBS_SIM_DB` (kills 3 hard-coded paths).
- [ ] *(→ Phase 1)* `RUNBOOK.md`: exact commands to regenerate every figure.

**Exit criterion:** fresh clone + `pip install -r requirements.txt` + one
documented command reproduces a baseline figure.

---

## Phase 1 — Fix the bursty crash  *(~0.5 day)*

Closes **#2** (the `var_log_m` NameError).

- [ ] Define the missing stationary variance in
      `JobGenerator._generate_daily_multipliers`:
      `var_log_m = sigma**2 / (1 - rho**2)`.
- [ ] Add a unit test that runs `--bursty` end-to-end on a tiny window and
      asserts no exception + multipliers sum ≈ n_days (mean ≈ 1.0).
- [ ] Re-run `run_bursty_comparison.py`; regenerate
      `bursty_sweep_results.csv` + the bursty slides.
- [ ] **Validate against empirical CV** (the script already computes CV
      flat/bursty/empirical) — confirm bursty CV ≈ 0.74 and document any drift
      from the pre-bug numbers in `RUNBOOK.md`.

**Exit criterion:** `--bursty` runs clean; the slide figure is reproducible from
a SHA; CV validation passes.

---

## Phase 2 — Program / allocation dimension  *(the core feature; ~3–4 days)*

Closes **#1** — the reason the tool exists for this decision. Now data-grounded
(see empirical facts above).

### 2a. Data model & program profiles
- [ ] Add `program` to `Job` (`INCITE`, `ALCC`, `DD`, `Genesis`). Map DB
      `allocation_type` → program (Discretionary→DD; UNKNOWN→drop or fold into
      DD behind a flag).
- [ ] **Build a `ProgramProfile` per program**, fitted from the DB:
  - node-size distribution (per program — they differ: INCITE/ALCC
    capability-leaning, DD bimodal with a 1-node spike)
  - walltime/runtime distributions (per program; reuse two-component model but
    fit threshold/shape per program)
  - **arrival rate AND seasonal burn curve** — fit a monthly multiplier from
    each program's node-hour time series so INCITE's Nov burn, ALCC's spring
    ramp, and DD's flatness are reproduced
  - allocation **calendar**: INCITE Jan–Dec (+optional 13th-month extension
    flag through next Jan), ALCC Jul–Jun, DD continuous. The calendar gates
    when a program's budget resets and shapes its burn curve.
- [ ] Annual **budget** per program (node-hours) = `target_share ×
      machine_node_hours_per_year`. Track burn against it; the seasonal curve is
      how fast each program spends down its budget.
- [ ] Extend `trace_sampler` to sample per **(program × size bucket)**.

### 2b. Genesis Mission profile (doesn't exist yet — scenario-driven)
- [ ] Genesis has no history, so model it as **configurable scenarios**, each a
      `ProgramProfile` built from explicit, labeled assumptions. Starter set:
  - **Genesis-as-INCITE-like:** capability jobs, large nodes, Jan–Dec calendar.
  - **Genesis-as-bursty-campaign:** AR(1)-bursty arrivals (high ρ), large
    node-hour surges around mission deadlines, short calendar windows.
  - **Genesis-as-on-demand/preemptable:** latency-sensitive, gets preemption
    priority over a preemptable pool (ties to Phase 5).
  - Each scenario parameterized by: target share, **ramp schedule** (Genesis
    grows 0→full over N months), size profile, burstiness, calendar.
- [ ] **Share-reallocation knob** — where does Genesis's share come from?
      (proportional from all / from INCITE only / from DD only). This is the
      central policy question; make it a first-class CLI sweep axis.

### 2c. Scheduling levers (implement both; they answer different questions)
- [ ] **Fair-share term.** Track `delivered_nh[program]`; add
      `score += fairshare_weight * (target_share[p] − actual_share[p])`.
      Baseline check: confirm the *program-blind* policy reproduces the observed
      57/30/13 vs 60/30/10 gap (validates the model), then show fair-share pulls
      delivered shares toward targets.
- [ ] **Per-program caps / floors.** Generalize the capacity-pool cap: ceiling
      ("Genesis ≤ X% running node-hours") and/or floor ("INCITE+ALCC ≥ Y%").
- [ ] Selectable: `--policy blind|fairshare|caps` to compare against baseline.

### 2d. Reporting
- [ ] Per-program decision table: delivered share vs target, budget burn %,
      p50/p95/max wait, starvation, unstarted — **the committee table.**
- [ ] Plot: stacked node-hours by program over time vs target lines + per-program
      budget burn-down curves against their calendars.
- [ ] Keep per-size-bucket reports (orthogonal view).

**Exit criterion:** a run prints "under Genesis scenario X taking 15% from
INCITE, INCITE p95 wait goes A→B, ALCC floor holds at Y%, delivered shares land
at I/A/G/D" — the tradeoff is quantified per scenario.

---

## Phase 3 — Statistical rigor  *(~1 day)*

Closes the multi-seed / CI and sensitivity gaps in **#5**.

- [ ] **Multi-seed harness.** Run N seeds (default 20), report mean ± 95% CI on
      all wait/starvation/share metrics. Bootstrap CI for the tail (p95/p99)
      since those are driven by few extreme jobs.
- [ ] **Sensitivity sweeps** with the harness:
  - `CAP_PACKER_THRESHOLD` ∈ {0.80, 0.90, 0.95}
  - cap-packer **re-inflation fraction** (new knob: what % of old cap-packers
    actually re-inflate to the new cap) ∈ {0.25, 0.5, 1.0}
  - capacity-pool size, and the new program target shares.
- [ ] Re-label the two-component walltime model in code+slides as an
      **assumption with a sensitivity band**, not an empirical fit.

**Exit criterion:** every headline number in the slides carries a CI; the main
conclusions are shown stable across the sensitivity ranges (or the instability
is documented).

---

## Phase 4 — Narrative reframe  *(~0.5 day, mostly slides/docs)*

Closes interpretation risk **#6**.

- [ ] Add explicit "demand-bound vs policy-bound" framing: historical Aurora
      demand fills the machine to ~66%, so policy differences are invisible at
      1.0× — they only appear under elevated load.
- [ ] Position **Genesis Mission as the demand increment** that crosses the
      system from demand-bound into policy-bound, making the fair-share/caps
      levers actually bite. Run the program sweeps at the load scales where this
      happens (0.5×–1.0×+ and a Genesis-ramp scenario).

**Exit criterion:** the slide story leads with the load regime, then shows the
program tradeoff inside the regime where it matters.

---

## Phase 5 — Scheduler realism (second-order)  *(~2–3 days; do after decision-relevant phases)*

Closes the modeling oversights in **#4**.

- [ ] **Preemption / true on-demand.** Model on-demand as genuinely preemptable
      capacity (checkpoint/requeue cost configurable) rather than nodes removed
      from the large ceiling — required if Genesis gets on-demand priority.
- [ ] **Maintenance / MTBF derate.** Periodic dedicated/maintenance windows +
      large-job failure&resubmit at an MTBF-scaled rate. Will lower effective
      large-queue throughput realistically.
- [ ] **Topology/fragmentation (optional, heaviest).** Approximate dragonfly
      group contiguity for large jobs so large-queue waits aren't optimistic.
      Gate behind a flag; keep the fungible-pool model as default/fast path.

**Exit criterion:** a sensitivity note quantifying how much these change the
large-queue / INCITE conclusions.

---

## Phase 6 — Code hygiene  *(fold into the phases above; ~0.5 day total)*

Closes the smaller items in **#7**.

- [ ] Remove dead `Scheduler._start` (run loop uses `_start_at_index`).
- [ ] Wire real capacity-pool peak into `summarize()` (data already collected).
- [ ] Fix `max_wait_h` keying so it survives deepcopied/renamed `QUEUES`
      (the sweep driver deepcopies).
- [ ] Rename sim buckets to size labels (xs/s/m/l) to end the
      `tiny`/`capacity` triple-overload; keep program names separate.
- [ ] Note the `O(n log n)` per-pass re-sort as a known scaling limit; revisit
      only if a sweep needs ≫ current job counts.

---

## Sequencing & dependencies

```
Phase 0 (git/repro) ──┬─> Phase 1 (bursty fix) ──┐
                      │                           ├─> Phase 3 (stats/CIs) ─> Phase 4 (narrative)
                      └─> Phase 2 (programs) ─────┘                              │
                                                                                 └─> Phase 5 (realism)
Phase 6 hygiene: interleaved, low risk.
```

- **0 done** (repo live; traceability in place).
- **1 and 2 are independent** and can run in parallel; 2 is the critical path
  for the decision.
- **3 depends on 1+2** (you CI the program metrics).
- **4 depends on 3** (narrative uses the CI'd program sweeps).
- **5 is decoupled**; schedule after the decision-relevant deliverable exists.

**Critical path to a decision-ready answer:** 0 → 2 → 3 → 4.
Phase 1 and Phase 5 improve fidelity but are not on that critical path.

---

## Effort summary

| Phase | Item | Est. | On critical path? |
|------|------|------|------|
| 0 | Foundation/repro | ✅ done (repo) | yes (blocker) |
| 1 | Bursty fix + repro hygiene | 0.5d | no |
| 2 | Program dimension + profiles + Genesis + levers | 3–4d | **yes** |
| 3 | Stats/CIs/sensitivity | 1–1.5d | yes |
| 4 | Narrative reframe | 0.5d | yes |
| 5 | Scheduler realism | 2–3d | no |
| 6 | Hygiene | 0.5d | interleaved |

Decision-ready (0+2+3+4): **~5–6 days**. Full hardening incl. 1+5+6: **~8–10 days**.

---

## Open questions for Taylor

These affect Phase 2 modeling choices. None block starting — I'll use the noted
defaults if you don't weigh in.

1. **Genesis share source.** When Genesis takes its share, does it come
   proportionally from all programs, from INCITE only, or from DD's hold?
   *Default:* sweep all three and show the tradeoff (it's the headline result).
2. **Genesis size/character.** Is Genesis expected to be capability
   (huge jobs), capacity (many small), or mixed? Any known target share or
   ramp timeline? *Default:* run the 3 starter scenarios (INCITE-like, bursty
   campaign, on-demand) at a placeholder 15% share ramped over 6 months.
6. **Machine target.** Keep modeling Aurora (10,624 nodes), or is this meant to
   inform an ALCF-4 / future-system allocation policy? Affects the annual
   node-hour budget math. *Default:* Aurora, with node count a CLI knob.
3. **13th-month extension.** Should I model INCITE's January extension as extra
   demand overlapping the new year's INCITE start (double-loading Jan)?
   *Default:* yes, behind a `--incite-13th-month` flag, off by default.
4. **DD treatment.** DD over-delivered (29.8% vs 10% nominal). Is the 10% a hard
   cap that *should* be enforced, or a soft floor it's fine to exceed when
   capacity is idle? This changes whether DD is a cap or a floor in the model.
   *Default:* soft floor (can exceed when idle), since that matches history.
5. **Fair-share target shares.** Use the canonical 60/30/10 (INCITE/ALCC/DD) as
   the fair-share targets, or the *delivered* 57/30/13? *Default:* targets =
   60/30/10 (the policy intent); show delivered vs target as the result.

---

## Proposed first action

Phase 0 is done (repo live, baseline tagged). On your go I'll start Phase 2 on
branch `feat/program-dimension`:
1. Add a `profile_programs.py` that fits per-program `ProgramProfile`s from the
   DB and writes a summary + plots (node-size, walltime, seasonal burn curve,
   calendar) — the empirical foundation, reviewable before any scheduler change.
2. Add `program` to `Job` + the fair-share scheduler term, validate it
   reproduces the 57/30/13 baseline under `--policy blind`.
3. Bring back the first per-program decision table as proof-of-concept.
All without touching current `results/` or `slides.html`.
