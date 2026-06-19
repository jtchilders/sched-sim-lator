# Remediation Plan — PBS Queue Simulator

Author: Wesley (⚛️) · 2026-06-19
Companion to: `ANALYSIS_REVIEW.md`
Goal: make this tool able to answer the **INCITE/ALCC vs Genesis Mission
balancing** question, with reproducible, defensible results.

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

## Phase 0 — Foundation & reproducibility  *(blocks everything; ~0.5 day)*

Closes review items **#3** (no VCS) and the reproducibility half of **#2**.

- [ ] `git init`; commit current code as `v1-baseline` tag (snapshot of what
      produced the existing slides).
- [ ] `.gitignore`: `.venv/`, `*.pkl` (trace/fitted caches), `__pycache__/`,
      and large binary outputs. Decide: keep `results/*.png` out of git, commit
      a `results/MANIFEST.md` that records which SHA + command produced each.
- [ ] Pin dependencies: freeze `requirements.txt` with versions
      (numpy/pandas/matplotlib actually used).
- [ ] Single source of truth for the DB path: one `--trace-db` default read from
      env var `PBS_SIM_DB` (kills the 3 hard-coded `/Users/jchilders/...` paths).
- [ ] Add a `RUNBOOK.md`: exact commands to regenerate every figure.

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

## Phase 2 — Program / allocation dimension  *(the core feature; ~2–3 days)*

Closes **#1** — the reason the tool exists for this decision.

### 2a. Data model
- [ ] Add `program` to `Job` (enum-ish: `INCITE`, `ALCC`, `Genesis`, `DD`).
- [ ] Discover the real field in the pbs_monitor DB: inspect `jobs` for
      `account` / `project` / `allocation` columns; build an empirical
      program → job mapping if it exists. If it doesn't, document that program
      mix is a *configured input*, not fitted, and expose it as CLI/config.
- [ ] Extend `trace_sampler` so fits are per **(program × size bucket)** where
      data supports it; fall back to a configurable program mix per bucket
      otherwise.

### 2b. Scheduling levers (implement both; they answer different questions)
- [ ] **Fair-share term.** Track `delivered_nh[program]`; add
      `score += fairshare_weight * (target_share[p] - actual_share[p])`.
      Targets configurable (e.g. INCITE 40% / ALCC 20% / Genesis 30% / DD 10%).
      This models "steer toward awarded shares" — the ALCF policy mechanism.
- [ ] **Per-program caps / floors.** Generalize the existing capacity-pool cap:
      ceiling ("Genesis ≤ X% running node-hours") and/or floor ("INCITE+ALCC
      guaranteed ≥ Y%"). Models hard contractual guarantees.
- [ ] Make the lever selectable: `--policy fairshare|caps|none` so we can
      compare against the current (program-blind) baseline.

### 2c. Reporting
- [ ] New per-program summary table: node-hours delivered vs target share,
      p50/p95/max wait, starvation flag, jobs unstarted — **this is the decision
      table.**
- [ ] New plot: stacked node-hours by program over time vs target share lines.
- [ ] Keep the existing per-size-bucket reports intact (orthogonal view).

**Exit criterion:** a single run prints "if Genesis target = 30%, INCITE p95
wait goes from A→B and delivered shares land at X/Y/Z" — i.e. the tradeoff is
quantified.

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

- **0 first** (everything else needs traceability).
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
| 0 | Foundation/repro | 0.5d | yes (blocker) |
| 1 | Bursty fix | 0.5d | no |
| 2 | Program dimension + levers | 2–3d | **yes** |
| 3 | Stats/CIs/sensitivity | 1d | yes |
| 4 | Narrative reframe | 0.5d | yes |
| 5 | Scheduler realism | 2–3d | no |
| 6 | Hygiene | 0.5d | interleaved |

Decision-ready (0+2+3+4): **~4–5 days**. Full hardening incl. 1+5+6: **~7–9 days**.

---

## Proposed first action

On your go: `git init` + Phase 0, then branch `feat/program-dimension` and
prototype the program tag + fair-share term against the existing fitted sampler
— without touching current `results/` or `slides.html`. I'll bring back the
first per-program tradeoff table as the proof-of-concept before going further.
