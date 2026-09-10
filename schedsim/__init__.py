"""schedsim: a Monte Carlo / replay simulator for PBS-style HPC scheduling on Aurora.

Layers (see docs/DESIGN.md):
  trace/    extract the PBS trace DB into parquet; build replay job tables
  jobs      the JobTable contract (a DataFrame with fixed columns)
  menu      queue menu: node/walltime bounds, limits, scoring defaults
  machine   node availability (schedulable nodes + downtime/reservation windows)
  priority  vectorised job-scoring models (ALCF WFP reconstruction, expressions)
  engine    the event-driven scheduler (cycles, profile backfill, reservations)
  metrics   exact utilisation from intervals, wait statistics, comparisons
  replay    validation harness: real trace through the simulated scheduler
"""
__version__ = "0.1.0"
