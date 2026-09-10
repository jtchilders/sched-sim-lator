"""Extract the PBS monitor SQLite DB into compact parquet tables.

Run once on the machine that holds the DB:

    python -m schedsim extract --db /path/pbs_monitor_aurora.db --out data/trace

Produces:
  jobs.parquet          one row per finished job with the columns the simulator
                        needs, including the per-job ALCF scoring parameters
                        (base_score, score_boost, project_priority, wfp/fifo/
                        backfill switches and factors) pulled from raw_pbs_data.
  reservations.parquet  PBS reservations (maintenance 'pm' windows, R-queues).
  queues.parquet        queue definitions parsed from the queues table JSON.
  node_availability.parquet  per-snapshot usable/reserved/down node counts
                        (from node_snapshots; Dec 2025 onward).

Everything downstream reads parquet; the 9 GB DB never needs to move.
"""
from __future__ import annotations

import json
import pathlib
import sqlite3
import time

import numpy as np
import pandas as pd

# Resource_List fields that drive ALCF's job sort formula. Pulled per job so a
# replay uses exactly the parameters PBS saw, not menu defaults.
_RL_FIELDS = {
    "base_score": "REAL", "score_boost": "REAL", "project_priority": "REAL",
    "enable_wfp": "REAL", "enable_fifo": "REAL", "enable_backfill": "REAL",
    "wfp_factor": "REAL", "fifo_factor": "REAL", "backfill_factor": "REAL",
    "backfill_max": "REAL", "burn_ratio": "REAL",
    "award_category": "TEXT", "total_allocation": "REAL",
    "current_allocation": "REAL",
}

_JOBS_SQL = """
SELECT job_id, owner, project, allocation_type, queue, nodes, walltime,
       submit_time, start_time, end_time,
       actual_runtime_seconds, queue_time_seconds, exit_status, state,
       json_extract(raw_pbs_data, '$.eligible_time') AS eligible_time,
       json_extract(raw_pbs_data, '$.etime')         AS etime,
       json_extract(raw_pbs_data, '$.depend')        AS depend,
       json_extract(raw_pbs_data, '$.Hold_Types')    AS hold_types,
       json_extract(raw_pbs_data, '$.run_count')     AS run_count,
       json_extract(raw_pbs_data, '$.Priority')      AS pbs_priority,
       {rl}
FROM jobs
WHERE state IN ('FINISHED') AND nodes IS NOT NULL AND nodes > 0
  AND submit_time IS NOT NULL
"""


def parse_hms_hours(s: pd.Series) -> pd.Series:
    """'HH:MM:SS' (hours may exceed 24) -> float hours. NaN on failure."""
    parts = s.astype("string").str.split(":", expand=True)
    if parts.shape[1] < 3:
        return pd.Series(np.nan, index=s.index)
    h = pd.to_numeric(parts[0], errors="coerce")
    m = pd.to_numeric(parts[1], errors="coerce")
    sec = pd.to_numeric(parts[2], errors="coerce")
    return (h + m / 60.0 + sec / 3600.0).astype(float)


def extract_jobs(con: sqlite3.Connection) -> pd.DataFrame:
    rl = ",\n       ".join(
        f"json_extract(raw_pbs_data, '$.Resource_List.{k}') AS {k}" for k in _RL_FIELDS)
    df = pd.read_sql_query(_JOBS_SQL.format(rl=rl), con)
    for c in ("submit_time", "start_time", "end_time"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    # PBS etime: when the job became eligible to run (after holds / dependencies)
    df["eligible_time_ts"] = pd.to_datetime(df["etime"], format="%a %b %d %H:%M:%S %Y", errors="coerce")
    df["has_depend"] = df["depend"].notna()
    df = df.drop(columns=["etime", "depend"])
    df["walltime_h"] = parse_hms_hours(df["walltime"])
    df["eligible_h"] = parse_hms_hours(df["eligible_time"])
    df["runtime_h"] = df["actual_runtime_seconds"].astype(float) / 3600.0
    df["wait_h"] = df["queue_time_seconds"].astype(float) / 3600.0
    # where PBS did not record queue_time, derive from timestamps
    m = df["wait_h"].isna() & df["start_time"].notna()
    df.loc[m, "wait_h"] = (df.loc[m, "start_time"] - df.loc[m, "submit_time"]).dt.total_seconds() / 3600.0
    for k, t in _RL_FIELDS.items():
        if t == "REAL":
            df[k] = pd.to_numeric(df[k], errors="coerce")
    df["run_count"] = pd.to_numeric(df["run_count"], errors="coerce")
    df["pbs_priority"] = pd.to_numeric(df["pbs_priority"], errors="coerce")
    df = df.drop(columns=["walltime", "eligible_time", "actual_runtime_seconds",
                          "queue_time_seconds"])
    df["nodes"] = df["nodes"].astype(np.int32)
    for c in ("owner", "project", "allocation_type", "queue", "award_category", "state", "hold_types"):
        df[c] = df[c].astype("string")
    return df


def extract_reservations(con: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT reservation_id, reservation_name, owner, state, queue, nodes, "
        "start_time, end_time, duration_seconds FROM reservations", con)
    for c in ("start_time", "end_time"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["nodes"] = pd.to_numeric(df["nodes"], errors="coerce").fillna(0).astype(np.int32)
    return df


def _parse_limit(s, prefix: str):
    """PBS limit strings like '[u:PBS_GENERIC=2]' / '[p:PBS_GENERIC=10]' /
    '[o:PBS_ALL=128]'. Returns the integer for the given entity prefix, else NaN."""
    if not isinstance(s, str):
        return np.nan
    for chunk in s.replace("]", "").split("["):
        chunk = chunk.strip()
        if chunk.startswith(prefix + ":") and "=" in chunk:
            try:
                return float(chunk.split("=", 1)[1])
            except ValueError:
                return np.nan
    return np.nan


def extract_queues(con: sqlite3.Connection) -> pd.DataFrame:
    raw = pd.read_sql_query("SELECT name, queue_type, is_active, raw_pbs_data FROM queues", con)
    rows = []
    for _, r in raw.iterrows():
        d = json.loads(r["raw_pbs_data"]) if r["raw_pbs_data"] else {}
        rmax = d.get("resources_max", {}) or {}
        rmin = d.get("resources_min", {}) or {}
        rdef = d.get("resources_default", {}) or {}
        mrr = d.get("max_run_res", {}) or {}
        rows.append({
            "name": r["name"],
            "queue_type": d.get("queue_type", r["queue_type"]),
            "enabled": str(d.get("enabled", "True")) == "True",
            "started": str(d.get("started", "True")) == "True",
            "queue_priority": float(d.get("Priority", 0) or 0),
            "min_nodes": float(rmin.get("nodect", np.nan)),
            "max_nodes": float(rmax.get("nodect", np.nan)),
            "min_walltime_h": parse_hms_hours(pd.Series([rmin.get("walltime")])).iloc[0],
            "max_walltime_h": parse_hms_hours(pd.Series([rmax.get("walltime")])).iloc[0],
            "max_run_per_user": _parse_limit(d.get("max_run"), "u"),
            "max_run_per_project": _parse_limit(d.get("max_run"), "p"),
            "max_queued_per_user": _parse_limit(d.get("max_queued"), "u"),
            "max_queued_per_project": _parse_limit(d.get("max_queued"), "p"),
            "max_nodes_total": _parse_limit(mrr.get("nodect"), "o"),
            "base_score": float(rdef.get("base_score", np.nan)) if rdef.get("base_score") is not None else np.nan,
            "enable_wfp": float(rdef.get("enable_wfp", np.nan)) if rdef.get("enable_wfp") is not None else np.nan,
            "enable_fifo": float(rdef.get("enable_fifo", np.nan)) if rdef.get("enable_fifo") is not None else np.nan,
            "enable_backfill": float(rdef.get("enable_backfill", np.nan)) if rdef.get("enable_backfill") is not None else np.nan,
            "from_route_only": str(d.get("from_route_only", "False")) == "True",
            "route_destinations": d.get("route_destinations", ""),
        })
    return pd.DataFrame(rows)


def extract(db_path: str, out_dir: str) -> dict:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    try:
        t0 = time.time()
        jobs = extract_jobs(con)
        jobs.to_parquet(out / "jobs.parquet", index=False)
        print(f"jobs: {len(jobs):,} rows -> {out/'jobs.parquet'} ({time.time()-t0:.0f}s)")
        res = extract_reservations(con)
        res.to_parquet(out / "reservations.parquet", index=False)
        print(f"reservations: {len(res):,} rows")
        q = extract_queues(con)
        q.to_parquet(out / "queues.parquet", index=False)
        print(f"queues: {len(q):,} rows")
        na = extract_node_availability(con)
        na.to_parquet(out / "node_availability.parquet", index=False)
        print(f"node availability: {len(na):,} snapshots "
              f"({na.timestamp.min()} -> {na.timestamp.max()})")
    finally:
        con.close()
    return {"jobs": len(jobs), "reservations": len(res), "queues": len(q), "node_snapshots": len(na)}


# Node-state letters used by the pbs_monitor snapshot encoding (verified against
# the `nodes` table): E job-exclusive, A free, L resv-exclusive, B offline,
# I down+offline, J state-unknown+down, N state-unknown+offline. Others (C, H,
# K, M, ...) are rare transitional/maintenance states and are treated as down.
USABLE_LETTERS = ("E", "A")


def extract_node_availability(con: sqlite3.Connection) -> pd.DataFrame:
    """Per-snapshot counts of node states -> the machine's real usable-node
    timeline. `up_nodes` = job-exclusive + free (what the general scheduler
    could use); `resv_nodes` = inside PBS reservations; rest = down/offline."""
    cur = con.execute("SELECT timestamp, snapshot_data FROM node_snapshots ORDER BY timestamp")
    rows = []
    for ts, s in cur:
        if not s:
            continue
        c = {}
        for ch in set(s):
            c[ch] = s.count(ch)
        rows.append({"timestamp": ts, "n": len(s),
                     "busy_nodes": c.get("E", 0), "free_nodes": c.get("A", 0),
                     "resv_nodes": c.get("L", 0),
                     "up_nodes": c.get("E", 0) + c.get("A", 0),
                     "down_nodes": len(s) - c.get("E", 0) - c.get("A", 0) - c.get("L", 0)})
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.dropna(subset=["timestamp"]).reset_index(drop=True)
