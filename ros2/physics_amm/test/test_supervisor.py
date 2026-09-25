"""Supervisor lifecycle tests using a stub process tree (no ANSR, runs in ms).

The stub child mimics the property of ANSR that matters here: it spawns
grandchildren (like Dask nannies/workers) and installs no signal handling of
its own, so only process-group signalling reaps the whole tree.
"""

import os
import textwrap
import time
from pathlib import Path

import pytest

from physics_amm.session_manager import SessionManager, SessionState
from physics_amm.supervisor import AnsrSupervisor, CANCELLED, FAILED, FINISHED

# A fake Main.py: spawns 3 grandchildren, then sleeps. Ignores nothing —
# exactly like ANSR, which has no signal handlers at all.
STUB_TREE = textwrap.dedent("""
    import subprocess, sys, time
    kids = [subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
            for _ in range(3)]
    time.sleep(600)
""")

STUB_OK = "import sys; sys.exit(0)"
STUB_FAIL = "import sys; sys.stderr.write('boom: bad topology\\n'); sys.exit(3)"


def _make_supervisor(tmp_path: Path, code: str) -> AnsrSupervisor:
    src_dir = tmp_path / "src"
    src_dir.mkdir(exist_ok=True)
    (src_dir / "Main.py").write_text(code)
    session_dir = tmp_path / "session"
    session_dir.mkdir(exist_ok=True)
    return AnsrSupervisor(ansr_source_dir=src_dir), session_dir


def _pgid_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def test_terminate_reaps_whole_process_group(tmp_path):
    """The L6 failure mode in miniature: grandchildren must die with the child."""
    sup, session_dir = _make_supervisor(tmp_path, STUB_TREE)
    sup.start([], session_dir)
    deadline = time.time() + 10
    while time.time() < deadline:  # wait for the 3 grandchildren
        kids = os.popen(f"pgrep -g {sup.pgid}").read().split()
        if len(kids) >= 4:
            break
        time.sleep(0.05)
    assert len(kids) >= 4, "stub tree did not come up"

    assert sup.terminate(grace_s=5.0) is True
    assert not _pgid_alive(sup.pgid), "processes left in the group"
    status, _ = sup.classify_exit()
    assert status == CANCELLED


def test_terminate_escalates_to_sigkill(tmp_path):
    """A child that ignores SIGTERM must still die within grace + kill window."""
    stub = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(600)"
    sup, session_dir = _make_supervisor(tmp_path, stub)
    sup.start([], session_dir)
    time.sleep(0.3)  # let it install the handler
    assert sup.terminate(grace_s=0.5) is True
    assert not _pgid_alive(sup.pgid)


def test_clean_exit_classified_finished(tmp_path):
    sup, session_dir = _make_supervisor(tmp_path, STUB_OK)
    sup.start([], session_dir)
    deadline = time.time() + 10
    while sup.poll() is None and time.time() < deadline:
        time.sleep(0.05)
    assert sup.poll() == 0
    status, _ = sup.classify_exit()
    assert status == FINISHED


def test_failure_classified_with_stderr_tail(tmp_path):
    sup, session_dir = _make_supervisor(tmp_path, STUB_FAIL)
    sup.start([], session_dir)
    deadline = time.time() + 10
    while sup.poll() is None and time.time() < deadline:
        time.sleep(0.05)
    assert sup.poll() == 3
    status, message = sup.classify_exit()
    assert status == FAILED
    assert "boom: bad topology" in message


def test_terminate_after_natural_exit_is_safe(tmp_path):
    sup, session_dir = _make_supervisor(tmp_path, STUB_OK)
    sup.start([], session_dir)
    while sup.poll() is None:
        time.sleep(0.05)
    assert sup.terminate(grace_s=1.0) is True  # no ProcessLookupError leak


def test_subprocess_gets_thread_limits_and_own_session(tmp_path):
    stub = textwrap.dedent("""
        import os, sys
        assert os.environ["OMP_NUM_THREADS"] == "1"
        assert os.getpgid(0) == os.getpid(), "not a session leader"
        sys.exit(0)
    """)
    sup, session_dir = _make_supervisor(tmp_path, stub)
    sup.start([], session_dir)
    while sup.poll() is None:
        time.sleep(0.05)
    assert sup.poll() == 0


# -- session manager -------------------------------------------------------

def test_run_slot_serializes_runs():
    mgr = SessionManager(max_concurrent_runs=1)
    assert mgr.try_acquire_run_slot() is True
    assert mgr.try_acquire_run_slot() is False
    mgr.release_run_slot()
    assert mgr.try_acquire_run_slot() is True


def test_session_lifecycle_and_limits():
    mgr = SessionManager(max_sessions=2)
    a = mgr.create()
    b = mgr.create()
    with pytest.raises(RuntimeError, match="session limit"):
        mgr.create()
    mgr.end(a.id)
    assert a.state is SessionState.ENDED
    with pytest.raises(KeyError):
        mgr.get(a.id)
    mgr.create()  # slot freed by ending a
    assert mgr.get(b.id) is b


def test_end_refuses_running_session_without_force(tmp_path):
    mgr = SessionManager()
    s = mgr.create()
    s.state = SessionState.RUNNING
    with pytest.raises(RuntimeError, match="force"):
        mgr.end(s.id)


def test_end_with_force_terminates_the_run(tmp_path):
    mgr = SessionManager()
    s = mgr.create()
    sup, session_dir = _make_supervisor(tmp_path, STUB_TREE)
    sup.start([], session_dir)
    s.state = SessionState.RUNNING
    s.supervisor = sup
    mgr.end(s.id, force=True, grace_s=5.0)
    assert s.state is SessionState.ENDED
    assert not _pgid_alive(sup.pgid)
