# Crux `parton` setup — one-time bootstrap for scans

Workspace layout (all under `~/schedsim/` on Crux, per the plan):

```
/home/parton-ai/schedsim/
├── sched-sim-lator/     # git clone (code)
├── venv/                # cray-python 3.11 venv + deps
└── data/
    └── profiles_cache.npz   # scp'd from the Mac mini (the 9 GB DB stays home)

/lus/eagle/projects/datascience/parton-ai/schedsim/experiments/  # scan outputs
```

## 0. Compile the profile cache on the Mac mini (where the DB lives)

```bash
cd ~/workspaces/sched-sim-lator && source .venv/bin/activate
python compile_profiles.py --config configs/validate_baseline.yaml \
    --out data/profiles_cache.npz          # ~3 MB, ~4 s
```

Verify it reproduces the DB bit-for-bit (already tested; re-check after any
trace-DB update):

```bash
python - <<'PY'
import numpy as np, dataclasses as dc
from config import SimConfig
from generator import JobGenerator, load_trace
from scheduler import Scheduler
import metrics as M
from compile_profiles import load_cache_df
c=SimConfig.from_yaml("configs/stage2_base.yaml"); c=dc.replace(c, run=dc.replace(c.run, duration_days=15))
def run(df):
    g=JobGenerator(c,df); j=g.generate(np.random.default_rng(42)); s=Scheduler(c)
    if c.projects.enabled: s.attach_projects(g.projects_by_prog)
    s.run(j); return M.summary_stats(j,s,c)
a=run(load_trace(c)); b=run(load_cache_df("data/profiles_cache.npz",c))
print("MATCH" if all(a[k]==b[k] for k in a) else "MISMATCH")
PY
```

## 1. Clone the repo on Crux

```bash
ssh parton-ai@crux.alcf.anl.gov
mkdir -p ~/schedsim && cd ~/schedsim
git clone git@github.com:jtchilders/sched-sim-lator.git   # jtchilders-ai-assistant SSH key
```

## 2. Build the venv (cray-python 3.11)

```bash
module load cray-python/3.11.7
python -m venv ~/schedsim/venv
source ~/schedsim/venv/bin/activate
pip install --upgrade pip
pip install numpy pandas pyyaml pyarrow      # matplotlib only if plotting on-node
```

(No `uv` on Crux; plain venv + pip. `cray-python/3.11.7` is the only Python
module needed — system python3 is 3.6 and too old.)

## 3. Copy the cache from the Mac mini

```bash
# from the Mac mini:
scp ~/workspaces/sched-sim-lator/data/profiles_cache.npz \
    parton-ai@crux.alcf.anl.gov:~/schedsim/data/
```

## 4. Confirm eagle write access (datascience project)

```bash
mkdir -p /lus/eagle/projects/datascience/parton-ai/schedsim/experiments
# if permission denied, request parton-ai be added to the datascience unix group
```

## 5. Smoke test (login node, few tasks)

```bash
cd ~/schedsim/sched-sim-lator && source ~/schedsim/venv/bin/activate
python scan.py --spec experiments/example_scan.yaml \
    --cache ~/schedsim/data/profiles_cache.npz --workers 8 --out /tmp/smoke
```

## 6. Submit a real scan to parton (batch)

```bash
qsub -v SPEC=experiments/<your_scan>.yaml scripts/run_scan_crux.pbs
qstat -u parton-ai
# results -> /lus/eagle/projects/datascience/parton-ai/schedsim/experiments/<name>/results.parquet
```

Idempotent: if the job hits walltime, just `qsub` again — it resumes, skipping
completed (config_hash, seed) cells.

## 7. Pull results back to the Mac mini for analysis

```bash
scp parton-ai@crux.alcf.anl.gov:/lus/eagle/projects/datascience/parton-ai/schedsim/experiments/<name>/results.parquet .
```

## Throughput sizing

One `parton` node = 128 cores. At ~3 s/run (short window, ≤0.6× load) that's
~40 runs/s → ~140k runs/hr. Full-year runs are ~2 min each → ~1 run/core/2min →
~3,800 runs/hr. A scan of thousands of points × seeds fits comfortably in a
single multi-hour batch job.
