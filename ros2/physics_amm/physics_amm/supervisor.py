"""Subprocess supervision for an ANSR learning run.

ROS-agnostic. ANSR spawns a Dask process tree (scheduler in-process, 4 nannies
+ 4 workers, transient sympy_worker.py children) and has no signal handling
anywhere; workers are respawned mid-run (client.restart every 5000 epochs, five
LocalCluster lifetimes per run). The only safe way to cancel is therefore to
put the child in its own session (process group) at launch and signal the
whole group, escalating SIGTERM -> SIGKILL. Tracking individual child PIDs
would lose them on every worker respawn.
"""

import os
import signal
import sys
import time
from pathlib import Path

import subprocess

# PyTorch in each Dask worker spawns its own intra-op thread pool by default;
# on a shared robot CPU that multiplies the fixed 4-worker footprint.
THREAD_LIMIT_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

FINISHED = "FINISHED"
CANCELLED = "CANCELLED"
FAILED = "FAILED"

_STDERR_TAIL_BYTES = 4096


class AnsrSupervisor:
    """Launches Main.py in a session workspace and manages its process group."""

    def __init__(self, ansr_source_dir, python_executable=None):
        self.main_py = str(Path(ansr_source_dir) / "Main.py")
        self.python_executable = python_executable or sys.executable
        self.proc = None
        self.pgid = None
        self.session_dir = None
        self.started_at = None
        self.cancel_requested = False

    @staticmethod
    def build_popen_kwargs(*, session_dir, stdout=None, stderr=None) -> dict:
        """Popen kwargs for an ANSR run in the given session workspace.

        start_new_session=True puts the child in its own process group so the
        whole Dask tree can be signalled with os.killpg; cwd must be the
        session workspace so the core's hard-coded path prefixes (topologies/,
        data/, ./configs/) resolve against the staged files.
        """
        return {
            "start_new_session": True,
            "cwd": session_dir,
            "env": {**os.environ, **THREAD_LIMIT_ENV},
            "stdout": stdout if stdout is not None else subprocess.DEVNULL,
            "stderr": stderr if stderr is not None else subprocess.DEVNULL,
        }

    def start(self, ansr_args, session_dir):
        if self.proc is not None:
            raise RuntimeError("supervisor already started")
        self.session_dir = str(session_dir)
        logs = Path(self.session_dir) / "logs"
        logs.mkdir(exist_ok=True)
        self._stdout_f = open(logs / "ansr.out.log", "ab")
        self._stderr_f = open(logs / "ansr.err.log", "ab")
        kwargs = self.build_popen_kwargs(
            session_dir=self.session_dir,
            stdout=self._stdout_f, stderr=self._stderr_f,
        )
        cmd = [self.python_executable, self.main_py] + list(ansr_args)
        self.proc = subprocess.Popen(cmd, **kwargs)
        self.pgid = os.getpgid(self.proc.pid)
        self.started_at = time.monotonic()

    def poll(self):
        """Return the exit code, or None while running."""
        if self.proc is None:
            return None
        code = self.proc.poll()
        if code is not None:
            self._close_logs()
        return code

    @property
    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @property
    def elapsed(self) -> float:
        return 0.0 if self.started_at is None else time.monotonic() - self.started_at

    def terminate(self, grace_s: float = 10.0) -> bool:
        """Kill the whole process group: SIGTERM, grace period, SIGKILL.

        Never waits for the core's normal-path 30 s shutdown sleep — the
        signals interrupt it. Returns True once no process remains in the
        group. Safe to call if the run already exited.
        """
        self.cancel_requested = True
        if self.pgid is None:
            return True
        self._signal_group(signal.SIGTERM)
        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline:
            if self._reap() and not self._group_alive():
                return True
            time.sleep(0.2)
        self._signal_group(signal.SIGKILL)
        # SIGKILL cannot be ignored; give the kernel a moment to reap.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if self._reap() and not self._group_alive():
                return True
            time.sleep(0.1)
        return not self._group_alive()

    def classify_exit(self):
        """Map the finished run to (status, message).

        exit 0 -> FINISHED; any exit after a cancel request -> CANCELLED;
        anything else -> FAILED with the stderr tail.
        """
        code = self.proc.poll() if self.proc else None
        if self.cancel_requested:
            return CANCELLED, "run cancelled; process group reaped"
        if code == 0:
            return FINISHED, "ANSR run completed"
        return FAILED, (
            f"ANSR exited with code {code}\n{self._stderr_tail()}"
        )

    # -- internals ---------------------------------------------------------

    def _signal_group(self, sig):
        try:
            os.killpg(self.pgid, sig)
        except ProcessLookupError:
            pass

    def _group_alive(self) -> bool:
        try:
            os.killpg(self.pgid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _reap(self) -> bool:
        """Collect the direct child if it exited; True when it is gone."""
        if self.proc is None:
            return True
        if self.proc.poll() is None:
            return False
        self._close_logs()
        return True

    def _stderr_tail(self) -> str:
        if self.session_dir is None:
            return ""
        path = Path(self.session_dir) / "logs" / "ansr.err.log"
        try:
            data = path.read_bytes()
        except OSError:
            return ""
        return data[-_STDERR_TAIL_BYTES:].decode("utf-8", errors="replace")

    def _close_logs(self):
        for attr in ("_stdout_f", "_stderr_f"):
            f = getattr(self, attr, None)
            if f is not None and not f.closed:
                f.close()
