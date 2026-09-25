"""Minimal launch_testing case: the launch file brings the node up and its
session services answer. Requires a built+sourced workspace (colcon test)."""

import time
import unittest

import launch
import launch_ros.actions
import launch_testing.actions
import launch_testing.markers
import pytest
import rclpy

from physics_amm_msgs.srv import StartSession


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    node = launch_ros.actions.Node(
        package="physics_amm",
        executable="physics_amm_node",
        namespace="physics_amm",
        name="physics_amm",
        output="screen",
    )
    return launch.LaunchDescription([
        node,
        launch_testing.actions.ReadyToTest(),
    ]), {"physics_amm_node": node}


class TestNodeComesUp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.client_node = rclpy.create_node("launch_test_client")

    @classmethod
    def tearDownClass(cls):
        cls.client_node.destroy_node()
        rclpy.shutdown()

    def test_start_session_answers(self):
        cli = self.client_node.create_client(
            StartSession, "/physics_amm/start_session")
        self.assertTrue(cli.wait_for_service(timeout_sec=30.0),
                        "start_session service never appeared")
        future = cli.call_async(StartSession.Request())
        deadline = time.time() + 10.0
        while not future.done() and time.time() < deadline:
            rclpy.spin_once(self.client_node, timeout_sec=0.1)
        self.assertTrue(future.done(), "start_session call timed out")
        resp = future.result()
        self.assertTrue(resp.success)
        self.assertTrue(resp.session_id)
