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
    total_nodes: int = 10_624              # Aurora nominal


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
