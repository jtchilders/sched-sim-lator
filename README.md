# PBS Queue Simulator

Monte Carlo simulation of a PBS-style job queue on a large HPC machine
(modeled loosely after Aurora @ ALCF).

## Design

- **Machine**: configurable node count (default 10,624 ≈ Aurora nominal)
- **Queues** (4 buckets by job size):
  - `tiny`:    1–128 nodes,    wallclock cap 7d (168h)
  - `small`:   129–512 nodes,  wallclock cap 72h
  - `medium`:  513–2048 nodes, wallclock cap 48h
  - `large`:   2049+ nodes,    wallclock cap 24h
- **Arrivals**: Poisson process, configurable rate per queue
- **Sizes/walltimes**: log-normal within each queue's bounds
- **Scheduler**: priority scoring across a single global queue, with
  EASY backfill + aging to prevent starvation.

## Score function

```
score(job) = base_priority(queue) + aging_rate(queue) * wait_hours
```

- Larger queues get higher `base_priority` (they're harder to schedule,
  so they should be considered first).
- Smaller queues get smaller `base_priority` but the same or smaller
  aging rate — they win via backfill, not via aging.
- Configurable weights so you can tune fairness vs throughput.

## EASY Backfill

- Top-scored job gets a *reservation* for the earliest time it can fit.
- Any other job whose walltime is short enough to finish before that
  reservation is allowed to start now if it fits in currently-free nodes.

## Outputs

- Per-queue wait time distributions (mean, median, p95, max)
- Utilization (fraction of node-hours used)
- Throughput (jobs/day, node-hours/day)
- Matplotlib plots: wait time CDFs, utilization timeline, queue depth

## Run

```bash
python -m pip install -r requirements.txt
python sim.py --duration-days 7 --seed 42
```
