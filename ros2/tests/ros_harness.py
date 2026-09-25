"""Test harness for the L4-L6 integration tests.

Spawns the physics_amm node as a subprocess (like a launch file would) and
drives it through an in-process rclpy client node. Requires a sourced ROS
environment and a built workspace (see tests/README.md); tests that use this
module are marked `ros` and deselected by default.
"""

import os
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from physics_amm_msgs.action import LearnModel
from physics_amm_msgs.srv import (
    EndSession,
    GetModel,
    SetSessionData,
    StartSession,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VENV_PYTHON = ROOT.parent / ".venv" / "bin" / "python"
ANSR_SOURCE = ROOT.parent / "source_code"

SERVICE_TIMEOUT = 10.0


class RosHarness:
    """One node subprocess + one client node, torn down deterministically."""

    def __init__(self, workspace_root: Path, feedback_period_s: float = 1.0,
                 sigterm_grace_s: float = 5.0):
        ns = f"test_{uuid.uuid4().hex[:8]}"
        self.namespace = ns
        self.node_proc = subprocess.Popen(
            [
                "ros2", "run", "physics_amm", "physics_amm_node",
                "--ros-args", "-r", f"__ns:=/{ns}",
                "-p", f"workspace_root:={workspace_root}",
                "-p", f"ansr_source_dir:={ANSR_SOURCE}",
                "-p", f"python_executable:={VENV_PYTHON}",
                "-p", f"feedback_period_s:={feedback_period_s}",
                "-p", f"sigterm_grace_s:={sigterm_grace_s}",
            ],
            start_new_session=True,
        )

        rclpy.init()
        self.client = Node("harness_client", namespace=ns)
        self.executor = MultiThreadedExecutor(num_threads=2)
        self.executor.add_node(self.client)
        self._spin_thread = threading.Thread(
            target=self.executor.spin, daemon=True)
        self._spin_thread.start()

        prefix = f"/{ns}"
        self.start_session_cli = self.client.create_client(
            StartSession, f"{prefix}/start_session")
        self.end_session_cli = self.client.create_client(
            EndSession, f"{prefix}/end_session")
        self.get_model_cli = self.client.create_client(
            GetModel, f"{prefix}/get_model")
        self.set_data_cli = self.client.create_client(
            SetSessionData, f"{prefix}/set_session_data")
        self.learn_cli = ActionClient(
            self.client, LearnModel, f"{prefix}/learn_model")

        for cli in (self.start_session_cli, self.end_session_cli,
                    self.get_model_cli, self.set_data_cli):
            assert cli.wait_for_service(timeout_sec=30.0), \
                f"service {cli.srv_name} never appeared"
        assert self.learn_cli.wait_for_server(timeout_sec=30.0), \
            "learn_model action server never appeared"

        self.feedback = []

    # -- services ----------------------------------------------------------

    def _call(self, cli, request):
        future = cli.call_async(request)
        deadline = time.time() + SERVICE_TIMEOUT
        while not future.done():
            if time.time() > deadline:
                raise TimeoutError(f"service call {cli.srv_name} timed out")
            time.sleep(0.02)
        return future.result()

    def start_session(self) -> str:
        resp = self._call(self.start_session_cli, StartSession.Request())
        assert resp.success, resp.message
        return resp.session_id

    def end_session(self, session_id: str, force: bool = False):
        req = EndSession.Request(session_id=session_id, force=force)
        return self._call(self.end_session_cli, req)

    def get_model(self, session_id: str, population: str = ""):
        req = GetModel.Request(session_id=session_id, population=population)
        return self._call(self.get_model_cli, req)

    def set_session_data(self, session_id: str, train_data_path: str,
                         valid_data_path: str = ""):
        req = SetSessionData.Request(
            session_id=session_id, train_data_path=str(train_data_path),
            valid_data_path=str(valid_data_path))
        return self._call(self.set_data_cli, req)

    # -- action ------------------------------------------------------------

    def send_goal(self, session_id: str, *, topology, train_data, valid_data,
                  outfolder="results", constraints=None):
        goal = LearnModel.Goal(
            session_id=session_id,
            topology_path=str(topology),
            train_data_path=str(train_data),
            valid_data_path=str(valid_data),
            constraints_path=str(constraints) if constraints else "",
            outfolder=outfolder,
        )
        future = self.learn_cli.send_goal_async(
            goal, feedback_callback=lambda fb: self.feedback.append(fb.feedback))
        deadline = time.time() + 30.0
        while not future.done():
            if time.time() > deadline:
                raise TimeoutError("goal was never accepted/rejected")
            time.sleep(0.05)
        goal_handle = future.result()
        return goal_handle

    def cancel_and_wait(self, goal_handle, timeout: float = 60.0):
        goal_handle.cancel_goal_async()
        result_future = goal_handle.get_result_async()
        deadline = time.time() + timeout
        while not result_future.done():
            if time.time() > deadline:
                raise TimeoutError("no action result after cancellation")
            time.sleep(0.1)
        return result_future.result()

    # -- process inspection -------------------------------------------------

    def find_ansr_pgid(self, timeout: float = 120.0) -> int:
        """PGID of the ANSR Main.py subprocess spawned by the node."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            out = subprocess.run(
                ["pgrep", "-f", str(ANSR_SOURCE / "Main.py")],
                capture_output=True, text=True).stdout.split()
            if out:
                return os.getpgid(int(out[0]))
            time.sleep(0.25)
        raise TimeoutError("ANSR Main.py process never appeared")

    @staticmethod
    def count_group_processes(pgid: int) -> int:
        out = subprocess.run(["pgrep", "-g", str(pgid)],
                             capture_output=True, text=True).stdout.split()
        return len(out)

    # -- teardown -----------------------------------------------------------

    def shutdown(self):
        try:
            os.killpg(os.getpgid(self.node_proc.pid), signal.SIGINT)
            self.node_proc.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(self.node_proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.executor.shutdown()
        self.client.destroy_node()
        rclpy.shutdown()


def wait_for(predicate, timeout: float, interval: float = 0.5,
             message: str = "condition"):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise TimeoutError(f"timed out waiting for {message}")
