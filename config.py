"""
SimConfig — single source of truth for a simulation run.

Everything the simulator does is driven by this config; there are NO hard-coded
policy values elsewhere. A run is fully described by one YAML file, so a
parameter sweep is just a set of YAML files (or one base + overrides), and every
output can be traced back to the exact config that produced it via config_hash().

Load with:  SimConfig.from_yaml("path.yaml")
Dump with:  cfg.to_yaml() / cfg.config_hash()
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict, fields, is_dataclass
from typing import Any, Optional

import yaml


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MachineConfig:
    total_nodes: int = 10_624              # physical node count (Aurora nominal)
    # Production nodes: the DOE-negotiated accountability denominator. Some
    # physical nodes are always down/out-of-service, so utilization and the
    # capacity-job threshold are measured against production_nodes, not
    # total_nodes. Aurora: 9600. Defaults to total_nodes when unset (<=0).
    production_nodes: int = 9_600

    def prod(self) -> int:
        return self.production_nodes if self.production_nodes and self.production_nodes > 0 else self.total_nodes


@dataclass(frozen=True)
class SizeTier:
    """A node-count band, used both for generator conditioning and scoring."""
    name: str
    min_nodes: int
    max_nodes: int
    walltime_cap_h: float
    base_priority: float = 0.0
    aging_rate: float = 0.0


@dataclass(frozen=True)
class ProgramConfig:
    """Per-program allocation policy. Distributions are FITTED from the DB;
    these are the policy knobs, not the empirical shape."""
    name: str
    target_share: float                    # yearly-average node-hour fraction
    overburn: float = 0.0                   # may deliver up to (1+overburn)*budget
    soft_floor: bool = False               # True => never hard-capped (DD)
    alloc_year_start_month: int = 1        # 1=Jan (INCITE); ALCC=7; drives burn-curve keying


@dataclass(frozen=True)
class WalltimePolicyConfig:
    """Queue-menu-as-a-rule: max walltime allowed as a function of node count.

    This REPLACES per-tier walltime caps when enabled, collapsing the
    small/medium/large queue menu into ONE rule that is trivially explainable:
    'a job of N nodes may request up to max_walltime(N) hours'.

    `breakpoints` is a list of (min_nodes, max_walltime_h) pairs, sorted by
    min_nodes. For a job of N nodes, the cap is the max_walltime_h of the
    highest breakpoint whose min_nodes <= N. This can encode ANY monotonic (up
    or down) or non-monotonic size->walltime relationship — the direction is a
    policy choice, not baked in.

    Example (big-unlocks-long): [[1,6],[512,12],[1920,168]]
    Example (small-gets-long):  [[1,168],[512,48],[1920,24]]
    """
    enabled: bool = False
    breakpoints: tuple = ()   # tuple of (min_nodes:int, max_walltime_h:float)

    def cap_for(self, nodes: int, fallback_cap_h: float) -> float:
        if not self.enabled or not self.breakpoints:
            return fallback_cap_h
        cap = fallback_cap_h
        for min_n, wt in sorted(self.breakpoints, key=lambda b: b[0]):
            if nodes >= min_n:
                cap = wt
        return cap


@dataclass(frozen=True)
class BehaviorConfig:
    """Behavioral size-choice model.

    Historical jobs chose their node count under the OLD queue menu. To honestly
    test a NEW walltime_policy, a fraction of jobs must be allowed to RE-CHOOSE
    their size in response to the new walltime incentive: a user who wants a long
    run, under a policy where long walltime requires >= K nodes, may bring K
    nodes to unlock it (and vice-versa). Without this the sim only re-scores the
    old job mix and cannot show the behavioral equilibrium.

    Model (deliberately simple + bounded, so it is explainable and sweepable):
      - With probability `adapt_fraction`, a job is 'walltime-motivated': it has
        a desired walltime (its originally-sampled walltime). If the new policy
        would cap it below that desire, the job's owner resizes UP to the
        smallest node count whose max_walltime(nodes) >= desired walltime
        (bounded by max_resize_nodes), trading size for the runtime they want.
      - `program_adapt` optionally overrides adapt_fraction per program (e.g.
        INCITE users are more walltime-motivated than DD).
      - resize is capped so a 1-node job can't jump to 10k nodes unrealistically.
    When disabled, jobs keep their historical sizes (walltime just gets capped).
    """
    enabled: bool = False
    adapt_fraction: float = 0.3            # fraction of jobs that resize to chase walltime
    program_adapt: tuple = ()              # tuple of (program, fraction) overrides
    max_resize_nodes: int = 1920           # ceiling on behavioral resize
    min_desired_walltime_h: float = 24.0   # only jobs wanting >= this bother resizing


@dataclass(frozen=True)
class GeneratorConfig:
    """Joint conditional Monte Carlo generator settings."""
    # Conditioning dimensions for the joint sampler. Jobs are bootstrapped from
    # real trace rows grouped by these keys; a job draw keeps its (nodes,
    # walltime, runtime) TOGETHER (joint), not as independent marginals.
    condition_on: tuple = ("program", "alloc_month_offset", "size_tier")
    # Minimum rows in a cell before we fall back to the parent (program-only)
    # pool. Prevents sampling from 1-2 outlier rows.
    min_cell_rows: int = 20
    min_runtime_s: int = 30                 # drop sub-30s noise jobs
    # Burn curve: fit per-program monthly multiplier keyed on months-since
    # alloc_year_start (so ALCC's slow-July-start vs INCITE's fast-Jan-start is
    # intrinsic, not a calendar coincidence).
    burn_curve_key: str = "alloc_month_offset"
    seed: int = 42
    # Scales ALL arrival rates uniformly. 1.0 = historical volume; <1 lightens
    # load (study policy at non-saturated demand), >1 stresses it. Lets you
    # sweep the demand-bound -> policy-bound transition.
    load_multiplier: float = 1.0


@dataclass(frozen=True)
class ProjectsConfig:
    """Project-level allocation layer.

    ALCF awards hours per PROJECT (via competitive review), not per program.
    Programs deliberately OVER-allocate (award more than their fraction) because
    most projects UNDER-use their award. This layer partitions each program's
    historical jobs by the real `project` column and gives each project a
    notional award; per-project budget damping then makes over-allocation safe
    (idle projects' headroom flows to active ones).

    When disabled, allocation is modeled at the program level only (v2 behavior).
    """
    enabled: bool = False
    # Over-allocation factor: a program awards this multiple of its target share
    # across its projects (e.g. 1.15 = award 115% of the fraction, expecting
    # ~87% utilization). > 1.0 reproduces the deliberate over-subscription.
    over_allocation: float = 1.15
    # Per-project budget damping: priority damps as a project nears its award.
    # 0 = no per-project steering (projects only limited by program ceiling).
    project_damp_strength: float = 6.0
    # Projects with fewer than this many historical jobs are folded into a
    # program-wide "misc" pseudo-project (avoids thousands of 1-job projects).
    min_project_jobs: int = 20


@dataclass(frozen=True)
class Deadline:
    """A conference/paper deadline that drives a bounded submission spike."""
    name: str
    month: int                     # calendar month of the deadline (1-12)
    day: int = 15                  # day of month
    lead_days: int = 21            # spike window length before the deadline
    rate_multiplier: float = 2.0   # arrival-rate multiplier in the window
    affected_fraction: float = 0.4 # fraction of projects that chase this deadline


@dataclass(frozen=True)
class DeadlinesConfig:
    """2-3 conference deadlines sprinkled through the year."""
    enabled: bool = False
    deadlines: tuple = ()


@dataclass(frozen=True)
class GenesisConfig:
    """Genesis has no history => synthesized from explicit assumptions."""
    enabled: bool = True
    scenario: str = "ai_default"           # ai_default | incite_like | bursty_campaign
    share: float = 0.15
    genesis_from: str = "proportional"     # proportional | incite | dd
    ramp_start_month: int = 7
    ramp_full_month: int = 10


@dataclass(frozen=True)
class CapacityProtectionConfig:
    """PLUGGABLE strategy to stop low-node long jobs swamping big jobs.
    The 512-pool is now ONE testable strategy among several — this is the
    lever the whole study is about."""
    strategy: str = "running_pool_cap"     # none | running_pool_cap | dedicated_partition | size_reservation
    pool_nodes: int = 512                  # for running_pool_cap
    protected_tier: str = "capacity"       # which size tier is throttled
    partition_nodes: int = 0               # for dedicated_partition (reserve for big jobs)
    big_job_min_nodes: int = 2049          # what counts as a "big job" to protect


@dataclass(frozen=True)
class SchedulerConfig:
    enable_backfill: bool = True
    # Fully configurable score expression (AST-restricted, no arbitrary code).
    # Available variables: base, wait, nodes, walltime, aging_rate, program,
    # budget_ratio, budget_damp, delivered_share, target_share, now,
    # queue_depth, free_nodes. See scheduler.SCORE_VARS for the full list.
    score_expr: str = "base + aging_rate * wait"
    # Budget-damped priority knobs (used by budget_damp variable).
    budget_damp_strength: float = 8.0
    policy: str = "budget"                 # blind | budget
    # Draining reservation engages only for jobs >= this node count (protects
    # large capability jobs from small-job starvation). 0 = default to the
    # capacity-job threshold (20% of production nodes).
    reserve_min_nodes: int = 0
    # PBS-faithful scheduling cycle: the scheduler recomputes priorities, sorts
    # the queue, and runs one greedy+reservation+backfill pass once per cycle —
    # NOT on every job event. Matches PBS Pro's scheduler_iteration (Aurora ~600s
    # = 10 min). Between cycles, arrivals queue and finishes free nodes, but no
    # (re)scheduling happens until the next cycle (so a job may wait up to one
    # cycle after nodes free — real PBS behavior). Also the key perf lever: cost
    # is O(cycles x P log P), independent of event count / load depth.
    sched_cycle_h: float = 600.0 / 3600.0   # 10 minutes
    # Beyond this many pending jobs, each cycle examines only the top-K by
    # priority (partial select) — bounds per-cycle sort cost. PBS-like bounded
    # per-iteration work. Large enough not to affect scheduling decisions.
    examine_cap: int = 4000


@dataclass(frozen=True)
class RunConfig:
    duration_days: float = 365.0
    start_month: int = 1                   # calendar month the sim clock starts
    sample_dt_h: float = 1.0               # MEASUREMENT granularity (NOT scheduling)
    n_seeds: int = 1                       # multi-seed -> CIs
    base_seed: int = 42


@dataclass(frozen=True)
class OutputConfig:
    outdir: str = "results/run"
    write_decision_table: bool = True
    write_timeseries: bool = True
    write_jobs: bool = False               # per-job detail (large)
    saturation_abort_pending: int = 100_000  # guard: abort if pending exceeds this


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

DEFAULT_SIZE_TIERS = (
    SizeTier("capacity", 1,    128,    168.0, base_priority=5.0,  aging_rate=0.5),
    SizeTier("small",    129,  512,    72.0,  base_priority=20.0, aging_rate=2.0),
    SizeTier("medium",   513,  2048,   48.0,  base_priority=40.0, aging_rate=5.0),
    SizeTier("large",    2049, 10_624, 24.0,  base_priority=80.0, aging_rate=10.0),
)

DEFAULT_PROGRAMS = (
    ProgramConfig("INCITE", target_share=0.50, overburn=0.25, alloc_year_start_month=1),
    ProgramConfig("ALCC",   target_share=0.25, overburn=0.0,  alloc_year_start_month=7),
    ProgramConfig("DD",     target_share=0.10, overburn=0.0,  soft_floor=True, alloc_year_start_month=1),
)


@dataclass(frozen=True)
class SimConfig:
    trace_db: str = "/Users/jchilders/pbs_monitor_aurora.db"
    machine: MachineConfig = field(default_factory=MachineConfig)
    size_tiers: tuple = DEFAULT_SIZE_TIERS
    programs: tuple = DEFAULT_PROGRAMS
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    genesis: GenesisConfig = field(default_factory=GenesisConfig)
    projects: ProjectsConfig = field(default_factory=ProjectsConfig)
    deadlines: DeadlinesConfig = field(default_factory=DeadlinesConfig)
    walltime_policy: WalltimePolicyConfig = field(default_factory=WalltimePolicyConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    capacity_protection: CapacityProtectionConfig = field(default_factory=CapacityProtectionConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    run: RunConfig = field(default_factory=RunConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # -- construction -------------------------------------------------------

    @staticmethod
    def from_yaml(path: str) -> "SimConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return SimConfig.from_dict(raw)

    @staticmethod
    def from_dict(d: dict) -> "SimConfig":
        d = dict(d or {})
        kw: dict[str, Any] = {}
        if "trace_db" in d:
            kw["trace_db"] = d["trace_db"]
        if "machine" in d:
            kw["machine"] = MachineConfig(**d["machine"])
        if "size_tiers" in d:
            kw["size_tiers"] = tuple(SizeTier(**t) for t in d["size_tiers"])
        if "programs" in d:
            kw["programs"] = tuple(ProgramConfig(**p) for p in d["programs"])
        if "generator" in d:
            g = dict(d["generator"])
            if "condition_on" in g:
                g["condition_on"] = tuple(g["condition_on"])
            kw["generator"] = GeneratorConfig(**g)
        if "genesis" in d:
            kw["genesis"] = GenesisConfig(**d["genesis"])
        if "projects" in d:
            kw["projects"] = ProjectsConfig(**d["projects"])
        if "deadlines" in d:
            dd = dict(d["deadlines"])
            if "deadlines" in dd:
                dd["deadlines"] = tuple(Deadline(**x) for x in dd["deadlines"])
            kw["deadlines"] = DeadlinesConfig(**dd)
        if "walltime_policy" in d:
            wp = dict(d["walltime_policy"])
            if "breakpoints" in wp:
                wp["breakpoints"] = tuple(tuple(b) for b in wp["breakpoints"])
            kw["walltime_policy"] = WalltimePolicyConfig(**wp)
        if "behavior" in d:
            bh = dict(d["behavior"])
            if "program_adapt" in bh:
                bh["program_adapt"] = tuple(tuple(x) for x in bh["program_adapt"])
            kw["behavior"] = BehaviorConfig(**bh)
        if "capacity_protection" in d:
            kw["capacity_protection"] = CapacityProtectionConfig(**d["capacity_protection"])
        if "scheduler" in d:
            kw["scheduler"] = SchedulerConfig(**d["scheduler"])
        if "run" in d:
            kw["run"] = RunConfig(**d["run"])
        if "output" in d:
            kw["output"] = OutputConfig(**d["output"])
        cfg = SimConfig(**kw)
        cfg.validate()
        return cfg

    # -- validation ---------------------------------------------------------

    def validate(self) -> None:
        errs = []
        if self.machine.total_nodes <= 0:
            errs.append("machine.total_nodes must be > 0")
        share_sum = sum(p.target_share for p in self.programs)
        gshare = self.genesis.share if self.genesis.enabled else 0.0
        # programs + genesis need not sum to 1 pre-reallocation; reallocation
        # (genesis_from) fixes it. Just sanity-check bounds.
        for p in self.programs:
            if not (0.0 <= p.target_share <= 1.0):
                errs.append(f"program {p.name} target_share out of [0,1]")
        if not (0.0 <= gshare <= 1.0):
            errs.append("genesis.share out of [0,1]")
        if self.capacity_protection.strategy not in (
                "none", "running_pool_cap", "dedicated_partition", "size_reservation"):
            errs.append(f"unknown capacity_protection.strategy "
                        f"{self.capacity_protection.strategy!r}")
        if self.scheduler.policy not in ("blind", "budget"):
            errs.append(f"unknown scheduler.policy {self.scheduler.policy!r}")
        if self.genesis.genesis_from not in ("proportional", "all", "incite", "dd"):
            errs.append(f"unknown genesis.genesis_from {self.genesis.genesis_from!r}")
        if self.run.sample_dt_h <= 0:
            errs.append("run.sample_dt_h must be > 0")
        # tier coverage: tiers must cover [1, total_nodes] without gaps
        tiers = sorted(self.size_tiers, key=lambda t: t.min_nodes)
        if tiers and tiers[0].min_nodes != 1:
            errs.append("size_tiers must start at min_nodes=1")
        for a, b in zip(tiers, tiers[1:]):
            if b.min_nodes != a.max_nodes + 1:
                errs.append(f"size_tier gap/overlap between {a.name} and {b.name}")
        if errs:
            raise ValueError("SimConfig invalid:\n  - " + "\n  - ".join(errs))

    # -- serialization / provenance ----------------------------------------

    def to_dict(self) -> dict:
        return _asdict_tuples(self)

    def to_yaml(self, path: Optional[str] = None) -> str:
        s = yaml.safe_dump(self.to_dict(), sort_keys=False)
        if path:
            with open(path, "w") as f:
                f.write(s)
        return s

    def config_hash(self) -> str:
        """Stable 12-char hash of the full config for provenance tagging."""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    # -- convenience --------------------------------------------------------

    def tier_for_nodes(self, nodes: int) -> SizeTier:
        for t in self.size_tiers:
            if t.min_nodes <= nodes <= t.max_nodes:
                return t
        return self.size_tiers[-1]


def _asdict_tuples(obj) -> Any:
    """asdict that turns tuples of dataclasses into lists of dicts (YAML-clean)."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _asdict_tuples(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (tuple, list)):
        return [_asdict_tuples(v) for v in obj]
    return obj
