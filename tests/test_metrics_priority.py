import numpy as np
import pandas as pd
import pytest

from schedsim import metrics as M
from schedsim.jobs import JobArrays
from schedsim.priority import ExprPriority, ALCF_FITTED
from schedsim.safe_expr import compile_expr, ExprError
from schedsim.menu import Menu
from schedsim.config import ReplayConfig
from tests.conftest import make_jobs


def test_busy_timeline_exact():
    tl = M.busy_timeline(np.array([0.0, 1.0]), np.array([2.0, 3.0]), np.array([10, 20]), 0, 4, 1.0)
    assert tl.busy_nodes.tolist() == [10.0, 30.0, 20.0, 0.0]
    # fractional overlap: 10 nodes over [0.5, 1.25) in 1 h bins -> 5, 2.5
    tl = M.busy_timeline(np.array([0.5]), np.array([1.25]), np.array([10]), 0, 2, 1.0)
    assert tl.busy_nodes.tolist() == pytest.approx([5.0, 2.5])
    assert M.utilization(np.array([0.0]), np.array([4.0]), np.array([50]), 0, 4, 100) == pytest.approx(0.5)


def test_ks_and_compare():
    a = np.arange(100.0); b = np.arange(100.0) + 50
    assert M.ks_statistic(a, a) == 0.0
    assert M.ks_statistic(a, b) == pytest.approx(0.5, abs=0.02)
    c = M.compare_distributions(a, a)
    assert c["ks"] == 0.0 and c["log2_ratio_p50"] == 0.0 and c["spearman"] == pytest.approx(1.0)


def test_wait_summary_censoring():
    s = M.wait_summary(np.array([1.0, 2.0, 3.0]), censored_lower_h=np.array([10.0]))
    assert s["n_started"] == 3 and s["n_censored"] == 1
    assert s["lb_p95_h"] > s["started_p95_h"]
    e = M.wait_summary(np.array([]))
    assert np.isnan(e["started_p50_h"])   # never report 0.0 for "nothing started"


def test_safe_expr_rejects_unsafe():
    for bad in ["__import__('os')", "a.b", "x[0]", "lambda: 1", "1 if a else 2", "a and b", "'s'*3"]:
        with pytest.raises(ExprError):
            compile_expr(bad, {"a", "b", "x"})
    fn = compile_expr("where(a > 1, a, b) + min(a, b)", {"a", "b"})
    out = fn({"a": np.array([0.0, 2.0]), "b": np.array([5.0, 5.0])})
    assert out.tolist() == [5.0, 4.0]


def test_alcf_fitted_formula_values():
    # 1024-node INCITE job (pp=25) eligible 10 h: WFP term = 25*1024/10624*(36000/1e4)^2
    jobs = make_jobs([dict(nodes=1024, walltime_h=12, runtime_h=12, submit_h=0.0, project_priority=25.0),
                      dict(nodes=1, walltime_h=1, runtime_h=1, submit_h=0.0, enable_wfp=0.0, enable_fifo=1.0)])
    J = JobArrays.from_table(jobs)
    s = ExprPriority(ALCF_FITTED, 10624).score(10.0, J, np.array([0, 1]), np.array([0.0, 0.0]))
    assert s[0] == pytest.approx(51 + 25 * 1024 / 10624 * 3.6 ** 2)
    assert s[1] == pytest.approx(51 + 10.0)      # fifo queue: base + eligible hours
    # eligible time is measured from the eligible timestamp, not submit
    s2 = ExprPriority(ALCF_FITTED, 10624).score(10.0, J, np.array([1]), np.array([8.0]))
    assert s2[0] == pytest.approx(53.0)


def test_menu_from_queue_table_and_route():
    df = pd.DataFrame([
        dict(name="small", queue_type="Execution", min_nodes=256, max_nodes=1024, max_walltime_h=12,
             queue_priority=10, max_run_per_user=np.nan, max_run_per_project=np.nan,
             max_queued_per_user=np.nan, max_queued_per_project=10, max_nodes_total=np.nan,
             base_score=51, enable_wfp=1, enable_fifo=0, enable_backfill=0, from_route_only=True),
        dict(name="R123", queue_type="Execution", min_nodes=np.nan, max_nodes=np.nan, max_walltime_h=np.nan,
             queue_priority=0, max_run_per_user=np.nan, max_run_per_project=np.nan,
             max_queued_per_user=np.nan, max_queued_per_project=np.nan, max_nodes_total=np.nan,
             base_score=np.nan, enable_wfp=np.nan, enable_fifo=np.nan, enable_backfill=np.nan, from_route_only=False),
    ])
    m = Menu.from_queue_table(df)
    assert "small" in m and "R123" not in m
    assert m["small"].max_queued_per_project == 10 and m["small"].max_run_per_user == float("inf")
    assert m.route(512, 6.0) == "small" and m.route(512, 24.0) is None
    d = m.to_dict(); assert Menu.from_dict(d)["small"].max_nodes == 1024


def test_config_is_strict_and_hash_ignores_output():
    with pytest.raises(ValueError):
        ReplayConfig.from_dict({"sheduler": {}})
    with pytest.raises((ValueError, TypeError)):
        ReplayConfig.from_dict({"scheduler": {"cycle": 1}})
    a = ReplayConfig.from_dict({"output": {"dir": "x"}})
    b = ReplayConfig.from_dict({"output": {"dir": "y"}})
    assert a.science_hash() == b.science_hash()
    assert a.science_hash() != ReplayConfig.from_dict({"scheduler": {"backfill_depth": 3}}).science_hash()
