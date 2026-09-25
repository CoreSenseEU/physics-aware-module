"""Session registry and run-concurrency control.

ROS-agnostic. A session is a UUID plus (once the first learning goal arrives)
a staged workspace and a supervised subprocess. Learning runs are serialized
by default (max_concurrent_runs=1): each ANSR run spawns a fixed 4-worker Dask
tree, so concurrent runs oversubscribe a shared robot CPU and collide on the
Dask dashboard port. Goals arriving while the slot is taken are rejected
(explicit and retryable) rather than silently queued for hours.
"""

import threading
import uuid
from enum import Enum


class SessionState(Enum):
    OPEN = "open"        # created; may accept a learning goal
    RUNNING = "running"  # a learning run is active
    ENDED = "ended"      # closed; workspace removed (terminal)


class Session:
    def __init__(self, session_id: str):
        self.id = session_id
        self.state = SessionState.OPEN
        self.workspace = None        # SessionWorkspace, set by the first goal
        self.supervisor = None       # AnsrSupervisor of the current/last run
        self.results_root = None     # absolute -o dir of the current/last run
        self.last_result_code = None
        self.lock = threading.Lock()


class SessionManager:
    def __init__(self, max_sessions: int = 8, max_concurrent_runs: int = 1):
        self.max_sessions = max_sessions
        self._sessions = {}
        self._registry_lock = threading.Lock()
        self._run_slots = threading.BoundedSemaphore(max_concurrent_runs)

    def create(self) -> Session:
        with self._registry_lock:
            active = sum(1 for s in self._sessions.values()
                         if s.state is not SessionState.ENDED)
            if active >= self.max_sessions:
                raise RuntimeError(
                    f"session limit reached ({self.max_sessions}); "
                    "end an existing session first"
                )
            session = Session(str(uuid.uuid4()))
            self._sessions[session.id] = session
            return session

    def get(self, session_id: str) -> Session:
        try:
            session = self._sessions[session_id]
        except KeyError:
            raise KeyError(f"unknown session_id {session_id!r}")
        if session.state is SessionState.ENDED:
            raise KeyError(f"session {session_id!r} has ended")
        return session

    def try_acquire_run_slot(self) -> bool:
        """Non-blocking: False means another learning run is active."""
        return self._run_slots.acquire(blocking=False)

    def release_run_slot(self):
        self._run_slots.release()

    def end(self, session_id: str, force: bool = False,
            grace_s: float = 10.0, cleanup_workspace: bool = True) -> str:
        """Close a session. Refuses while RUNNING unless force=True, in which
        case the run's whole process group is terminated first."""
        session = self.get(session_id)
        with session.lock:
            if session.state is SessionState.RUNNING:
                if not force:
                    raise RuntimeError(
                        "a learning run is active; cancel it or pass force=true"
                    )
                if session.supervisor is not None:
                    session.supervisor.terminate(grace_s)
            if cleanup_workspace and session.workspace is not None:
                session.workspace.remove()
            session.state = SessionState.ENDED
        return session.id
