"""
Parity / plumbing tests for the ROSified PhysicsAMM module.

STATUS: scaffold. L1/L2/L6 are written against the node API that does not exist yet -
fill in the marked hooks once the package is implemented. The assertions themselves are
the specification and should not need to change.

Design rationale (see tests/README.md): ANSR is asynchronous and therefore NOT
bit-reproducible, so we never compare expressions or loss values between a reference run
and a ROS run. We assert the *contract* (argv, workspace, output paths) and *invariants*
(structure exists, model is finite, no orphaned processes).

Run:  pytest -v test_ros_parity.py
"""

import math
import os
import re
import signal
import subprocess
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT.parent / "source_code"
FIXTURES = HERE / "fixtures"

# --------------------------------------------------------------------------
# Node modules (importable without ROS via conftest.py sys.path shim)
# --------------------------------------------------------------------------
from physics_amm.session import SessionWorkspace, build_ansr_argv  # noqa: E402
from physics_amm.supervisor import AnsrSupervisor  # noqa: E402

requires_node = pytest.mark.node
requires_ros = pytest.mark.ros


# ==========================================================================
# L1 - Argv contract: the node must invoke ANSR exactly as a human would
# ==========================================================================
@requires_node
def test_argv_contains_only_the_five_agreed_switches():
    argv = build_ansr_argv(  # noqa: F821
        topology="smoke_topology.txt",
        train_data="smoke_train.csv",
        valid_data="smoke_valid.csv",
        outfolder="results",
        constraints=None,
    )
    flags = {a for a in argv if a.startswith("-")}
    assert flags == {"-t", "--train_data", "--valid_data", "-o"}, (
        "only the agreed parameters may be passed; everything else takes SRConfig defaults"
    )


@requires_node
def test_constraints_switch_omitted_entirely_when_absent():
    argv = build_ansr_argv(  # noqa: F821
        topology="t.txt", train_data="tr.csv", valid_data="va.csv",
        outfolder="results", constraints=None,
    )
    assert "-c" not in argv, "-c must be omitted, not passed empty (parser defaults it to None)"


@requires_node
def test_paths_are_bare_filenames_not_absolute():
    """The core prepends topologies/ and data/ - absolute paths would produce 'data//abs/path'."""
    argv = build_ansr_argv(  # noqa: F821
        topology="/abs/path/smoke_topology.txt",
        train_data="/abs/path/smoke_train.csv",
        valid_data="/abs/path/smoke_valid.csv",
        outfolder="results",
        constraints=None,
    )
    for flag in ("-t", "--train_data", "--valid_data"):
        value = argv[argv.index(flag) + 1]
        assert not os.path.isabs(value), f"{flag} must be a bare filename, got {value!r}"
        assert "/" not in value, f"{flag} must be a bare filename, got {value!r}"


@requires_node
def test_subprocess_is_launched_in_its_own_session_with_thread_limits():
    """start_new_session=True is what makes process-group cancellation possible."""
    spec = AnsrSupervisor.build_popen_kwargs(session_dir="/tmp/sess1")  # noqa: F821
    assert spec["start_new_session"] is True, "required for os.killpg cancellation"
    assert spec["cwd"] == "/tmp/sess1", "cwd must be the session workspace"
    assert spec["env"].get("OMP_NUM_THREADS") == "1", "limit PyTorch intra-op threads"


# ==========================================================================
# L2 - Workspace staging: satisfy the core's hard-coded path prefixes
# ==========================================================================
@requires_node
def test_workspace_layout_matches_required_prefixes(tmp_path):
    ws = SessionWorkspace.create(  # noqa: F821
        root=tmp_path,
        train_data=FIXTURES / "data" / "smoke_train.csv",
        valid_data=FIXTURES / "data" / "smoke_valid.csv",
        topology=FIXTURES / "topologies" / "smoke_topology.txt",
        constraints=None,
    )
    assert (Path(ws.path) / "data" / "smoke_train.csv").exists()
    assert (Path(ws.path) / "data" / "smoke_valid.csv").exists()
    assert (Path(ws.path) / "topologies" / "smoke_topology.txt").exists()


@requires_node
def test_two_sessions_get_disjoint_workspaces(tmp_path):
    a = SessionWorkspace.create(root=tmp_path, **_fixture_kwargs())  # noqa: F821
    b = SessionWorkspace.create(root=tmp_path, **_fixture_kwargs())  # noqa: F821
    assert a.path != b.path, "concurrent sessions must not share a workspace or -o directory"


# ==========================================================================
# L3 - Reference smoke (no ROS). Mirrors run_reference_cli.sh.
# ==========================================================================
@pytest.mark.slow
def test_reference_cli_produces_documented_output_structure(tmp_path):
    """Baseline: run ANSR directly and assert the tree the ROS service will read."""
    ws = tmp_path / "ws"
    (ws / "data").mkdir(parents=True)
    (ws / "topologies").mkdir()
    for f in ("smoke_train.csv", "smoke_valid.csv"):
        (ws / "data" / f).write_bytes((FIXTURES / "data" / f).read_bytes())
    (ws / "topologies" / "smoke_topology.txt").write_bytes(
        (FIXTURES / "topologies" / "smoke_topology.txt").read_bytes()
    )

    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    proc = subprocess.run(
        ["python", str(SRC / "Main.py"),
         "-t", "smoke_topology.txt",
         "--train_data", "smoke_train.csv",
         "--valid_data", "smoke_valid.csv",
         "-o", "results",
         "--maxTotalBackprops", "2000"],
        cwd=ws, env=env, capture_output=True, text=True, timeout=1800,
    )
    assert proc.returncode == 0, f"ANSR failed:\n{proc.stderr[-4000:]}"

    best = _latest_current_best(ws / "results" / "seed=1" / "archiveIndividuals")
    assert best is not None, "no current_best_<version>/ folder produced"
    assert (best / "overview.txt").exists()
    assert list(best.glob("*.m")), "expected MATLAB exports"


# ==========================================================================
# L4/L5 - ROS smoke + extended mode (need a running node)
# ==========================================================================
# How long to wait for the first current_best mirror: the run must get through
# the initial-population learning phases first (~1-2 min on the fixture).
MIRROR_TIMEOUT = 300.0


@pytest.fixture
def harness(tmp_path):
    from ros_harness import RosHarness
    h = RosHarness(workspace_root=tmp_path / "sessions")
    yield h
    h.shutdown()


def _send_fixture_goal(harness, session_id):
    return harness.send_goal(
        session_id,
        topology=FIXTURES / "topologies" / "smoke_topology.txt",
        train_data=FIXTURES / "data" / "smoke_train.csv",
        valid_data=FIXTURES / "data" / "smoke_valid.csv",
    )


def _archive_dir(workspace_root: Path) -> Path:
    """The single session's archive mirror dir under the test workspace root."""
    sessions = list(Path(workspace_root).glob("session_*"))
    assert len(sessions) == 1, f"expected one session workspace, got {sessions}"
    return sessions[0] / "results" / "seed=1" / "archiveIndividuals"


@requires_ros
def test_ros_run_produces_same_structure_as_reference(harness, tmp_path):
    """Compare STRUCTURE with L3, never expression or loss values (async EA is not reproducible)."""
    from ros_harness import wait_for

    session_id = harness.start_session()
    goal_handle = _send_fixture_goal(harness, session_id)
    assert goal_handle.accepted, "goal was rejected"

    wait_for(lambda: harness.feedback, timeout=60, message="action feedback")

    archive = _archive_dir(tmp_path / "sessions")
    wait_for(lambda: _latest_current_best(archive) is not None,
             timeout=MIRROR_TIMEOUT, message="current_best mirror")

    # Same tree as L3: overview.txt + >=1 individual .txt + >=1 .m
    best = _latest_current_best(archive)
    wait_for(lambda: (best / "overview.txt").exists()
             and [p for p in best.glob("*.txt") if p.name != "overview.txt"]
             and list(best.glob("*.m")),
             timeout=30, message="mirror folder contents")

    # Retrieval service returns a non-empty, finite model.
    resp = wait_for(
        lambda: (r := harness.get_model(session_id)).models and r or None,
        timeout=60, message="get_model returning a model")
    assert resp.success
    assert resp.models[0].expression, "model has no expression"
    assert math.isfinite(resp.models[0].valid_loss), "model loss not finite"

    # The budget parameters are not exposed over ROS: terminate by cancelling.
    result = harness.cancel_and_wait(goal_handle)
    assert result.result.code == result.result.CODE_CANCELLED
    harness.end_session(session_id)


@requires_ros
def test_data_swap_midrun_and_midrun_query(harness, tmp_path):
    """L5: SetSessionData swaps the staged train CSV -> current_best_2/ appears;
    get_model answers while the action is still running (MultiThreadedExecutor)."""
    from ros_harness import wait_for

    session_id = harness.start_session()
    goal_handle = _send_fixture_goal(harness, session_id)
    assert goal_handle.accepted

    archive = _archive_dir(tmp_path / "sessions")
    wait_for(lambda: _latest_current_best(archive) is not None,
             timeout=MIRROR_TIMEOUT, message="current_best_1 mirror")

    # Mid-run query while the action runs (proves services stay responsive).
    resp = harness.get_model(session_id)
    assert resp.success, resp.message

    # Swap to the B target (x0^2 + 2*x1): reload_data() picks it up by mtime
    # and advances the dataset version -> a current_best_2/ folder appears.
    resp = harness.set_session_data(
        session_id, FIXTURES / "data" / "smoke_train_B.csv")
    assert resp.success, resp.message

    # Mirror writes are throttled (saveCurrentBestMinIntervalS = 5 s default)
    # and individuals must be re-measured first: poll, don't assert instantly.
    wait_for(
        lambda: (v := _latest_current_best(archive)) is not None
        and int(v.name.rsplit("_", 1)[1]) >= 2,
        timeout=MIRROR_TIMEOUT, message="current_best_2 after data swap")

    result = harness.cancel_and_wait(goal_handle)
    assert result.result.code == result.result.CODE_CANCELLED
    harness.end_session(session_id)


# ==========================================================================
# L6 - Cancellation must leave no orphaned Dask processes  (CRITICAL)
# ==========================================================================
@requires_ros
def test_cancel_reaps_the_entire_process_group(harness, tmp_path):
    """
    ANSR has no signal handling; a plain terminate() orphans ~8 Dask processes.
    This test fails loudly if process-group cancellation regresses.
    """
    session_id = harness.start_session()
    goal_handle = _send_fixture_goal(harness, session_id)
    assert goal_handle.accepted

    pgid = harness.find_ansr_pgid(timeout=60)

    # Wait until the Dask tree is up: Main.py + scheduler-side threads spawn
    # nannies and workers as separate processes in the same group.
    deadline = time.time() + 120
    while time.time() < deadline:
        if harness.count_group_processes(pgid) >= 5:
            break
        time.sleep(0.5)
    assert harness.count_group_processes(pgid) >= 5, \
        "Dask workers never came up"

    harness.cancel_and_wait(goal_handle, timeout=60)

    deadline = time.time() + 20
    while time.time() < deadline:
        if not _pgid_alive(pgid):
            return
        time.sleep(0.5)
    pytest.fail(f"orphaned processes remain in group {pgid} after cancellation")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _latest_current_best(population_dir: Path):
    """Highest-numbered current_best_<version>/ - version increments on every data reload."""
    if not population_dir.is_dir():
        return None
    folders = [
        d for d in population_dir.iterdir()
        if d.is_dir() and re.fullmatch(r"current_best_\d+", d.name)
    ]
    if not folders:
        return None
    return max(folders, key=lambda d: int(d.name.rsplit("_", 1)[1]))


def _pgid_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _fixture_kwargs():
    return dict(
        train_data=FIXTURES / "data" / "smoke_train.csv",
        valid_data=FIXTURES / "data" / "smoke_valid.csv",
        topology=FIXTURES / "topologies" / "smoke_topology.txt",
        constraints=None,
    )
