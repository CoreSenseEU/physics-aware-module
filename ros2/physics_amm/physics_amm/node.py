"""ROS 2 node wrapping ANSR: learning as an action, retrieval as services.

Thin by design (D5.1 guideline #1): all logic lives in the ROS-agnostic
modules (session, supervisor, results, session_manager); this file only
translates between ROS interfaces and those modules.

A MultiThreadedExecutor plus split callback groups keep the retrieval services
responsive while a learning run occupies the action execute callback
(guideline #6: no heavy compute in callbacks — the compute is a subprocess,
and the execute callback only polls it).
"""

import math
import os
import sys
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import (
    MutuallyExclusiveCallbackGroup,
    ReentrantCallbackGroup,
)
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from physics_amm_msgs.action import LearnModel
from physics_amm_msgs.msg import AnalyticModel
from physics_amm_msgs.srv import (
    EndSession,
    GetModel,
    SetSessionData,
    StartSession,
)

from physics_amm.results import MirrorReader
from physics_amm.session import SessionWorkspace, build_ansr_argv
from physics_amm.session_manager import SessionManager, SessionState
from physics_amm.supervisor import CANCELLED, FINISHED, AnsrSupervisor


class PhysicsAmmNode(Node):

    def __init__(self):
        super().__init__("physics_amm")

        self.declare_parameter("workspace_root", "~/.ros/physics_amm/sessions")
        self.declare_parameter("ansr_source_dir", "")
        self.declare_parameter("python_executable", "")
        self.declare_parameter("max_concurrent_runs", 1)
        self.declare_parameter("max_sessions", 8)
        self.declare_parameter("sigterm_grace_s", 10.0)
        self.declare_parameter("feedback_period_s", 2.0)
        self.declare_parameter("max_run_seconds", 36000.0)
        self.declare_parameter("default_population", "archive")
        self.declare_parameter("mirror_read_retries", 3)
        self.declare_parameter("cleanup_workspace_on_end", True)

        self.workspace_root = os.path.expanduser(
            self.get_parameter("workspace_root").value)
        self.ansr_source_dir = self._resolve_ansr_source_dir()
        configured_python = self.get_parameter("python_executable").value
        self.python_executable = (
            os.path.expanduser(configured_python) if configured_python else None)
        self.sigterm_grace_s = self.get_parameter("sigterm_grace_s").value
        self.feedback_period_s = self.get_parameter("feedback_period_s").value
        self.max_run_seconds = self.get_parameter("max_run_seconds").value
        self.default_population = self.get_parameter("default_population").value
        self.mirror_read_retries = self.get_parameter("mirror_read_retries").value
        self.cleanup_workspace_on_end = self.get_parameter(
            "cleanup_workspace_on_end").value

        self.sessions = SessionManager(
            max_sessions=self.get_parameter("max_sessions").value,
            max_concurrent_runs=self.get_parameter("max_concurrent_runs").value,
        )

        # Services are quick file reads / registry updates: they share a
        # reentrant group so they stay available while a run occupies the
        # action group.
        service_group = ReentrantCallbackGroup()
        action_group = MutuallyExclusiveCallbackGroup()

        self.create_service(StartSession, "start_session",
                            self._on_start_session, callback_group=service_group)
        self.create_service(EndSession, "end_session",
                            self._on_end_session, callback_group=service_group)
        self.create_service(GetModel, "get_model",
                            self._on_get_model, callback_group=service_group)
        self.create_service(SetSessionData, "set_session_data",
                            self._on_set_session_data, callback_group=service_group)

        self._action_server = ActionServer(
            self, LearnModel, "learn_model",
            execute_callback=self._execute_learn,
            goal_callback=self._on_goal_request,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=action_group,
        )

        self.get_logger().info(
            f"physics_amm node up; ANSR core: {self.ansr_source_dir}; "
            f"ANSR interpreter: {self.python_executable or sys.executable}; "
            f"workspaces: {self.workspace_root}")

    # -- services ----------------------------------------------------------

    def _on_start_session(self, request, response):
        try:
            session = self.sessions.create()
        except RuntimeError as e:
            response.success = False
            response.message = str(e)
            return response
        response.success = True
        response.message = "session created"
        response.session_id = session.id
        self.get_logger().info(f"session {session.id} created")
        return response

    def _on_end_session(self, request, response):
        try:
            self.sessions.end(
                request.session_id, force=request.force,
                grace_s=self.sigterm_grace_s,
                cleanup_workspace=self.cleanup_workspace_on_end,
            )
        except (KeyError, RuntimeError) as e:
            response.success = False
            response.message = str(e)
            return response
        response.success = True
        response.message = "session ended"
        self.get_logger().info(f"session {request.session_id} ended")
        return response

    def _on_get_model(self, request, response):
        try:
            session = self.sessions.get(request.session_id)
        except KeyError as e:
            response.success = False
            response.message = str(e)
            return response
        if session.results_root is None:
            response.success = False
            response.message = "no learning run has been started in this session"
            return response

        population = request.population or self.default_population
        try:
            reader = MirrorReader(session.results_root)
            version, records = reader.read_best(
                population, retries=self.mirror_read_retries)
        except ValueError as e:
            response.success = False
            response.message = str(e)
            return response

        if request.max_models:
            records = records[:request.max_models]
        response.success = True
        response.dataset_version = version
        response.models = [self._to_msg(r) for r in records]
        response.message = (
            f"{len(records)} model(s) from current_best_{version} "
            "(mirror may lag live state by up to ~5 s)"
            if records else
            "no models mirrored yet (mirror writes are throttled; retry shortly)"
        )
        return response

    def _on_set_session_data(self, request, response):
        try:
            session = self.sessions.get(request.session_id)
        except KeyError as e:
            response.success = False
            response.message = str(e)
            return response
        if session.workspace is None:
            response.success = False
            response.message = "no learning run has staged data in this session"
            return response
        try:
            session.workspace.swap_train_data(request.train_data_path)
            swapped = ["train"]
            if request.valid_data_path:
                session.workspace.swap_valid_data(request.valid_data_path)
                swapped.append("valid")
        except (OSError, FileNotFoundError) as e:
            response.success = False
            response.message = str(e)
            return response
        response.success = True
        response.message = (
            f"{'+'.join(swapped)} data swapped; the core reloads on mtime and "
            "advances the dataset version (new current_best_<v+1> mirror)"
        )
        self.get_logger().info(
            f"session {session.id}: data swap ({response.message})")
        return response

    # -- action ------------------------------------------------------------

    def _on_goal_request(self, goal):
        try:
            session = self.sessions.get(goal.session_id)
        except KeyError as e:
            self.get_logger().warning(f"goal rejected: {e}")
            return GoalResponse.REJECT
        if session.state is SessionState.RUNNING:
            self.get_logger().warning(
                f"goal rejected: session {goal.session_id} already running")
            return GoalResponse.REJECT
        for label, path in (("topology_path", goal.topology_path),
                            ("train_data_path", goal.train_data_path),
                            ("valid_data_path", goal.valid_data_path)):
            if not os.path.isfile(path):
                self.get_logger().warning(
                    f"goal rejected: {label} {path!r} is not a file")
                return GoalResponse.REJECT
        if goal.constraints_path and not os.path.isfile(goal.constraints_path):
            self.get_logger().warning(
                f"goal rejected: constraints_path {goal.constraints_path!r} "
                "is not a file")
            return GoalResponse.REJECT
        if not self.sessions.try_acquire_run_slot():
            self.get_logger().warning(
                "goal rejected: another learning run is active "
                "(runs are serialized; retry after it ends)")
            return GoalResponse.REJECT
        # Slot is held from here; _execute_learn releases it.
        return GoalResponse.ACCEPT

    def _execute_learn(self, goal_handle):
        goal = goal_handle.request
        result = LearnModel.Result()
        try:
            session = self.sessions.get(goal.session_id)
            with session.lock:
                session.state = SessionState.RUNNING
            try:
                return self._run_learn(goal_handle, goal, session, result)
            finally:
                with session.lock:
                    if session.state is SessionState.RUNNING:
                        session.state = SessionState.OPEN
        finally:
            self.sessions.release_run_slot()

    def _run_learn(self, goal_handle, goal, session, result):
        try:
            workspace = SessionWorkspace.create(
                root=self.workspace_root,
                train_data=goal.train_data_path,
                valid_data=goal.valid_data_path,
                topology=goal.topology_path,
                constraints=goal.constraints_path or None,
            )
        except (OSError, FileNotFoundError) as e:
            result.code = LearnModel.Result.CODE_FAILED
            result.message = f"workspace staging failed: {e}"
            goal_handle.abort()
            return result

        session.workspace = workspace
        outfolder = goal.outfolder or "results"
        session.results_root = workspace.results_root(outfolder)
        reader = MirrorReader(session.results_root)

        argv = build_ansr_argv(
            topology=goal.topology_path,
            train_data=goal.train_data_path,
            valid_data=goal.valid_data_path,
            outfolder=outfolder,
            constraints=goal.constraints_path or None,
        )
        supervisor = AnsrSupervisor(
            self.ansr_source_dir, python_executable=self.python_executable)
        session.supervisor = supervisor
        try:
            supervisor.start(argv, workspace.path)
        except OSError as e:
            result.code = LearnModel.Result.CODE_FAILED
            result.message = f"failed to start ANSR: {e}"
            goal_handle.abort()
            return result

        self.get_logger().info(
            f"session {session.id}: ANSR started (pgid {supervisor.pgid}) "
            f"in {workspace.path}")

        watchdog_fired = False
        while supervisor.poll() is None:
            if goal_handle.is_cancel_requested:
                self._publish_feedback(goal_handle, supervisor, reader,
                                       status="FINALIZING")
                supervisor.terminate(self.sigterm_grace_s)
                break
            if (self.max_run_seconds > 0
                    and supervisor.elapsed > self.max_run_seconds):
                watchdog_fired = True
                self.get_logger().warning(
                    f"session {session.id}: wall-time watchdog "
                    f"({self.max_run_seconds:.0f}s) — terminating run")
                supervisor.terminate(self.sigterm_grace_s)
                break
            self._publish_feedback(goal_handle, supervisor, reader)
            time.sleep(self.feedback_period_s)

        status, message = supervisor.classify_exit()
        session.last_result_code = status

        version, records = reader.read_best(
            "archive", retries=self.mirror_read_retries)
        result.models = [self._to_msg(r) for r in records]
        result.results_path = os.path.join(session.results_root, "seed=1")

        if status == FINISHED:
            result.code = LearnModel.Result.CODE_FINISHED
            result.message = message
            goal_handle.succeed()
        elif status == CANCELLED:
            result.code = LearnModel.Result.CODE_CANCELLED
            result.message = ("wall-time watchdog cancelled the run"
                              if watchdog_fired else message)
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
            else:
                goal_handle.abort()  # watchdog: no client cancel to honour
        else:
            result.code = LearnModel.Result.CODE_FAILED
            result.message = message
            goal_handle.abort()

        self.get_logger().info(
            f"session {session.id}: run ended ({status}), "
            f"{len(records)} model(s) in current_best_{version}")
        return result

    def _publish_feedback(self, goal_handle, supervisor, reader,
                          status=None):
        version, count, best_loss, best_complexity = reader.summary("archive")
        fb = LearnModel.Feedback()
        fb.elapsed_seconds = supervisor.elapsed
        fb.status = status or ("RUNNING" if version else "STARTING")
        fb.dataset_version = version
        fb.num_models = count
        fb.best_valid_loss = best_loss
        fb.best_complexity = best_complexity
        goal_handle.publish_feedback(fb)

    # -- helpers -----------------------------------------------------------

    def _to_msg(self, rec) -> AnalyticModel:
        msg = AnalyticModel()
        msg.expression = rec.expression
        msg.simplified_expression = rec.simplified_expression
        msg.valid_loss = rec.valid_loss
        msg.complexity = rec.complexity
        msg.num_active_nodes = rec.num_active_nodes
        msg.rmse_constr = rec.rmse_constr
        msg.dataset_version = rec.dataset_version
        msg.population = rec.population
        msg.txt_path = rec.txt_path
        msg.m_path = rec.m_path
        return msg

    def _resolve_ansr_source_dir(self) -> str:
        configured = self.get_parameter("ansr_source_dir").value
        if configured:
            path = os.path.expanduser(configured)
        else:
            from ament_index_python.packages import get_package_share_directory
            path = os.path.join(
                get_package_share_directory("physics_amm"), "ansr_source")
        if not os.path.isfile(os.path.join(path, "Main.py")):
            raise RuntimeError(
                f"ANSR core not found: no Main.py in {path!r} "
                "(set the ansr_source_dir parameter)")
        return path


def main(args=None):
    rclpy.init(args=args)
    node = PhysicsAmmNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
