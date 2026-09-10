import numpy as np
import pytest

from schedsim.machine import Machine, DownWindow
from schedsim.menu import Menu, Queue
from tests.conftest import make_jobs, run


def test_easy_reservation_and_backfill(machine100, menu_q):
    # FIFO order: j0 (60 nodes) runs; j1 (80) cannot -> reservation at t=10 when j0
    # ends; j2 (40 nodes, 20 h) would delay j1 -> must wait; j3 (40 nodes, 8 h)
    # ends before t=10 -> backfills now.
    jobs = make_jobs([
        dict(nodes=60, walltime_h=10, runtime_h=10, submit_h=0.00),
        dict(nodes=80, walltime_h=5, runtime_h=5, submit_h=0.01),
        dict(nodes=40, walltime_h=20, runtime_h=20, submit_h=0.02),
        dict(nodes=40, walltime_h=8, runtime_h=8, submit_h=0.03),
    ])
    r = run(machine100, menu_q, jobs)
    # passes are event-triggered: j3 starts the moment it arrives (0.03 h)
    assert r.start_h.tolist() == [0.0, 10.0, 15.0, 0.03]


def test_backfill_uses_walltime_not_runtime(machine100, menu_q):
    # j0 requests 10 h but finishes at 2 h. The reservation for j1 is placed at the
    # WALLTIME end (10 h); when j0 actually ends at 2 h the next pass starts j1.
    jobs = make_jobs([
        dict(nodes=60, walltime_h=10, runtime_h=2, submit_h=0.0),
        dict(nodes=80, walltime_h=5, runtime_h=5, submit_h=0.01),
        dict(nodes=40, walltime_h=9, runtime_h=9, submit_h=0.02),   # 9 h < 10 h: backfills
    ])
    r = run(machine100, menu_q, jobs)
    assert r.start_h[2] == 0.02
    assert r.start_h[1] == pytest.approx(9.02)  # j2 (40) + j1 (80) > 100 until j2 ends


def test_maintenance_window_drains(menu_q):
    m = Machine(total_nodes=100, reportable_nodes=100, schedulable_nodes=100,
                windows=[DownWindow(10, 14, 100, "pm")])
    jobs = make_jobs([dict(nodes=10, walltime_h=5, runtime_h=5, submit_h=6.0),
                      dict(nodes=10, walltime_h=1, runtime_h=1, submit_h=6.0)])
    r = run(m, menu_q, jobs, priority="nodes")
    assert r.start_h[1] == 6.0        # fits before the window
    assert r.start_h[0] == 14.0       # must wait for the window to end


def test_availability_series_is_honoured(menu_q):
    m = Machine(total_nodes=100, reportable_nodes=100, schedulable_nodes=100,
                series_t=np.array([0.0, 5.0]), series_up=np.array([30.0, 100.0]))
    jobs = make_jobs([dict(nodes=50, walltime_h=1, runtime_h=1, submit_h=0.0)])
    r = run(m, menu_q, jobs)
    assert r.start_h[0] == 5.0


def test_run_limits_per_user_and_aggregate(machine100):
    menu = Menu([Queue("q", 1, 100, 200, max_run_per_user=1)])
    jobs = make_jobs([dict(nodes=10, walltime_h=2, runtime_h=2, submit_h=0.0)] * 3)
    r = run(machine100, menu, jobs)
    assert sorted(r.start_h.tolist()) == [0.0, 2.0, 4.0]
    menu2 = Menu([Queue("q", 1, 100, 200, max_nodes_total=25)])
    r2 = run(machine100, menu2, jobs)
    assert sorted(r2.start_h.tolist()) == [0.0, 0.0, 2.0]


def test_queued_limit_holdback_and_eligible_time(machine100):
    # only 2 of a project's jobs may sit in the queue; the 3rd waits in routing and
    # does not accrue eligible time until admitted.
    menu = Menu([Queue("q", 1, 100, 200, max_queued_per_project=2, max_run_per_project=1)])
    jobs = make_jobs([dict(nodes=10, walltime_h=2, runtime_h=2, submit_h=0.0)] * 3)
    r = run(machine100, menu, jobs, enforce_queued_limits=True)
    assert sorted(r.start_h.tolist()) == [0.0, 2.0, 4.0]
    assert sorted(r.eligible_h.tolist()) == [0.0, 0.0, 2.0]


def test_dedicated_partition(menu_q):
    # debug jobs can only use their 8-node partition; the general pool loses 8.
    m = Machine(total_nodes=100, reportable_nodes=100, schedulable_nodes=100)
    menu = Menu([Queue("q", 1, 100, 200), Queue("debug", 1, 8, 1)])
    jobs = make_jobs([dict(nodes=92, walltime_h=5, runtime_h=5, submit_h=0.0),          # fills general pool
                      dict(nodes=1, walltime_h=5, runtime_h=5, submit_h=0.0),           # general: no room
                      dict(nodes=8, walltime_h=1, runtime_h=1, submit_h=0.0, queue="debug"),
                      dict(nodes=1, walltime_h=1, runtime_h=1, submit_h=0.0, queue="debug")])
    r = run(m, menu, jobs, partitions=(("debug", ("debug",), 8),))
    assert r.start_h[0] == 0.0 and r.start_h[2] == 0.0
    assert r.start_h[1] == 5.0            # general pool full (100 - 8 dedicated = 92)
    assert r.start_h[3] == 1.0            # partition full until the 8-node debug job ends


def test_strict_groups_blocks_only_own_group(machine100):
    menu = Menu([Queue("small", 1, 100, 200), Queue("other", 1, 100, 200)])
    jobs = make_jobs([
        dict(nodes=70, walltime_h=10, runtime_h=10, submit_h=0.00, queue="small"),
        dict(nodes=50, walltime_h=5, runtime_h=5, submit_h=0.01, queue="small"),   # blocked -> reservation
        dict(nodes=20, walltime_h=1, runtime_h=1, submit_h=0.02, queue="small"),   # blocked strictly? no: 1 res allowed
        dict(nodes=20, walltime_h=1, runtime_h=1, submit_h=0.03, queue="small"),   # 2nd blocked small job -> group stops
        dict(nodes=10, walltime_h=1, runtime_h=1, submit_h=0.04, queue="other"),   # other queue backfills
    ])
    # With depth 1: j1 reserved at t=10 (needs 50, only 30 free). j2 fits now (20 <= 30)
    # and ends before 10 -> starts. j3 (20) would need 20 of the 10 remaining -> blocked;
    # it is the 2nd blocked small job so the small group stops. j4 (other) still runs.
    r = run(machine100, menu, jobs, ordering="strict_groups", backfill_depth=1,
            strict_groups=(("small",),))
    assert r.start_h[0] == 0.0 and r.start_h[2] == 0.02 and r.start_h[4] == 0.04
    assert np.isnan(r.start_h[3]) or r.start_h[3] >= 1.0   # blocked by strict ordering, not backfilled at 0.03
    assert r.start_h[1] == pytest.approx(10.0)


def test_dispatch_latency_and_preplaced(machine100, menu_q):
    jobs = make_jobs([dict(nodes=10, walltime_h=2, runtime_h=2, submit_h=0.0),
                      dict(nodes=50, walltime_h=2, runtime_h=2, submit_h=-1.0)])
    jobs.loc[1, "initial_start_h"] = -1.0
    r = run(machine100, menu_q, jobs, dispatch_latency_h=0.05)
    assert r.start_h[0] == pytest.approx(0.05)
    assert r.start_h[1] == -1.0 and np.isnan(r.sim_wait_h[1])


def test_unstarted_jobs_are_nan(machine100, menu_q):
    jobs = make_jobs([dict(nodes=150, walltime_h=1, runtime_h=1, submit_h=0.0)])
    r = run(machine100, menu_q, jobs, t_end=10)
    assert np.isnan(r.start_h[0]) and not r.started[0]
