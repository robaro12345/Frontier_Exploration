"""Opt-in real DDS/action integration; no physical or Gazebo motion is simulated."""

import os
import time
from types import SimpleNamespace

from fronter_exploration.cluster import FrontierCluster
from fronter_exploration.main_node import main_node
from fronter_exploration.state import State
from fronter_exploration.utils import Utils
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from nav_msgs.msg import OccupancyGrid
import numpy as np
import pytest
import rclpy
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node


@pytest.mark.skipif(os.environ.get('FRONTER_ROS_TEST') != '1',
                    reason='Set FRONTER_ROS_TEST=1 to enable DDS integration')
@pytest.mark.parametrize('first_navigation_success', [True, False])
def test_real_ros_training_and_deployment(tmp_path, first_navigation_success):
    """Run two actions, real map callbacks, optimization and model deployment."""
    pytest.importorskip('torch')
    rclpy.init(args=['--ros-args', '-p', 'rl:=true', '-p', 'training:=true',
                     '-p', 'rl.batch_size:=1', '-p', 'rl.target_update:=1',
                     '-p', 'rl.checkpoint_dir:=' + str(tmp_path)], domain_id=187)
    node = None
    simulator = None
    executor = None
    planner_server = None
    navigation_server = None
    deployed = None
    try:
        node = main_node()
        simulator = Node('synthetic_map_and_nav2')
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.add_node(simulator)
        group = ReentrantCallbackGroup()
        stage = [0]
        map_callbacks_during_navigation = [0]
        requested_goals = []
        original_map_callback = node.subscriber.callback

        def observe_map(message):
            if node.state == State.NAVIGATING:
                map_callbacks_during_navigation[0] += 1
            original_map_callback(message)

        node.subscriber.callback = observe_map

        def plan(goal_handle):
            result = ComputePathToPose.Result()
            result.path.poses = [PoseStamped(), goal_handle.request.goal]
            goal_handle.succeed()
            return result

        def navigate(goal_handle):
            requested_goals.append(goal_handle.request.pose)
            time.sleep(.3)  # Server worker only; main_node callbacks must keep running.
            stage[0] += 1
            if stage[0] == 1 and not first_navigation_success:
                goal_handle.abort()
            else:
                goal_handle.succeed()
            return NavigateToPose.Result()

        planner_server = ActionServer(simulator, ComputePathToPose,
                                      'compute_path_to_pose', plan,
                                      callback_group=group)
        navigation_server = ActionServer(simulator, NavigateToPose,
                                         'navigate_to_pose', navigate,
                                         callback_group=group)
        publisher = simulator.create_publisher(OccupancyGrid, 'map', 10)
        message = OccupancyGrid()
        message.header.frame_id = 'map'
        message.info.width = message.info.height = 60
        message.info.resolution = .05
        message.info.origin.orientation.w = 1.

        def publish_map():
            grid = np.full((60, 60), -1, dtype=np.int8)
            if stage[0] == 0:
                grid[15:45, 15:45] = 0
            elif stage[0] == 1:
                grid[10:50, 10:50] = 0
            else:
                grid[:] = 0
            message.header.stamp = simulator.get_clock().now().to_msg()
            message.data = grid.ravel().tolist()
            publisher.publish(message)

        simulator.create_timer(.05, publish_map, callback_group=group)
        deadline = time.monotonic() + 15.
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=.05)
            if node.trainer.finished and node._rl_future is None:
                break
        assert node.trainer.finished, 'Training did not finish within 15 seconds'
        assert len(requested_goals) == 2
        assert map_callbacks_during_navigation[0] > 0
        assert node.trainer.agent.steps == 2 and node.trainer.agent.updates == 2
        assert node.trainer.successes == 1 + int(first_navigation_success)
        assert node.trainer.explored_area == pytest.approx(2700 * .05 ** 2)
        assert node.trainer.agent.replay[-1][3]
        row = node.trainer.report.records[0]
        assert row['success'] and row['coverage'] == 1.
        assert row['steps'] == 2 and row['navigation_successes'] == node.trainer.successes
        assert row['cumulative_reward'] == pytest.approx(node.trainer.episode_reward)
        checkpoint = tmp_path / 'latest.pt'
        assert checkpoint.is_file()
        node.trainer.close()
        executor.remove_node(node)
        node.destroy_node()
        node = None

        planner_server.destroy()
        navigation_server.destroy()
        planner_server = navigation_server = None
        executor.shutdown(timeout_sec=2.)
        executor = None
        simulator.destroy_node()
        simulator = None

        # Exercise the actual deployment flag path and the same navigation workflow.
        rclpy.shutdown()
        rclpy.init(args=['--ros-args', '-p', 'rl:=true', '-p', 'training:=false',
                         '-p', 'rl.checkpoint:=' + str(checkpoint)], domain_id=188)
        deployed = main_node()
        deployed.planner.client.server_is_ready = lambda: True
        deployed.nav.client.server_is_ready = lambda: True
        recorded = []
        deployed.nav.goal = lambda *args: recorded.append(args)
        message.data = np.zeros((60, 60), dtype=np.int8).ravel().tolist()
        # Feed the previously observed candidate estimates to the deployment decision.
        deployed.map = message
        message.header.stamp = deployed.get_clock().now().to_msg()
        deployed.latest_map = message
        deployed.grid = np.zeros((60, 60), dtype=np.int8)
        cluster = FrontierCluster(23, [(15, 15)] * 21)
        cluster.information_gain = 21.
        cluster.path_cost = 1.
        deployed.frontier = {23: cluster}
        deployed.scoring(SimpleNamespace(estimation=lambda *args: None),
                         Utils(0., 0., .05))
        assert deployed.trainer is None and deployed.deployment.agent.epsilon == 0.
        assert deployed.best_node.id == 23 and len(recorded) == 1
        np.testing.assert_allclose(deployed.env.get_state()[23], [.21, .1])
        assert deployed.deployment.agent.updates == 2
        assert not deployed.deployment.agent.replay
    finally:
        if deployed is not None:
            deployed.destroy_node()
        if node is not None:
            node.trainer.close()
            node.destroy_node()
        if planner_server is not None:
            planner_server.destroy()
        if navigation_server is not None:
            navigation_server.destroy()
        if executor is not None:
            executor.shutdown(timeout_sec=2.)
        if simulator is not None:
            simulator.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


@pytest.mark.skipif(os.environ.get('FRONTER_ROS_TEST') != '1',
                    reason='Set FRONTER_ROS_TEST=1 to enable DDS integration')
def test_real_ros_shutdown_discovery_wait_beyond_ten_seconds():
    """Do not restart while a real old map publisher remains visible beyond 10 seconds."""
    import time
    from fronter_exploration.RL.run_training import _wait_for_shutdown
    rclpy.init(domain_id=190)
    observer = Node('shutdown_observer', use_global_arguments=False)
    old_stack = Node('old_map_stack', use_global_arguments=False)
    publisher = old_stack.create_publisher(OccupancyGrid, 'map', 10)
    started = time.monotonic()
    stopped = [False]

    def expire():
        if not stopped[0] and time.monotonic() - started >= 12.:
            old_stack.destroy_publisher(publisher)
            stopped[0] = True

    timer = observer.create_timer(.05, expire)
    wrapped = SimpleNamespace(
        count_publishers=observer.count_publishers,
        nav=SimpleNamespace(client=SimpleNamespace(server_is_ready=lambda: False)),
        planner=SimpleNamespace(client=SimpleNamespace(server_is_ready=lambda: False)),
        get_logger=observer.get_logger)
    try:
        # Wait for discovery so the test cannot pass before seeing the old publisher.
        deadline = time.monotonic() + 5.
        while observer.count_publishers('map') == 0:
            rclpy.spin_once(observer, timeout_sec=.05)
            assert time.monotonic() < deadline
        _wait_for_shutdown(wrapped, lambda current, timeout_sec: rclpy.spin_once(
            observer, timeout_sec=timeout_sec), rclpy.ok, timeout=30.)
        assert stopped[0] and time.monotonic() - started >= 12.
        assert observer.count_publishers('map') == 0
    finally:
        observer.destroy_timer(timer)
        observer.destroy_node()
        old_stack.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


@pytest.mark.skipif(os.environ.get('FRONTER_ROS_TEST') != '1',
                    reason='Set FRONTER_ROS_TEST=1 to enable DDS integration')
@pytest.mark.parametrize('startup_failure,alternative_goal',
                         [(False, False), (True, False), (False, True)])
def test_world_loop_real_ros_reset_and_clock_jump(
        tmp_path, monkeypatch, startup_failure, alternative_goal):
    """Keep real ROS training alive while replacing two synthetic simulator stacks."""
    pytest.importorskip('torch')
    import json
    from pathlib import Path
    import fronter_exploration.RL.run_training as runner
    from rosgraph_msgs.msg import Clock

    rclpy.init(args=['--ros-args', '-p', 'rl:=true', '-p', 'training:=true',
                     '-p', 'use_sim_time:=true', '-p', 'rl.episode_limit:=2',
                     '-p', 'rl.batch_size:=1', '-p', 'rl.target_update:=1',
                     '-p', 'rl.checkpoint_dir:=' + str(tmp_path)], domain_id=189)
    node = None
    executor = None
    launched = []
    stopped = []
    try:
        node = main_node()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)

        def launch(command, **kwargs):
            simulator = Node('synthetic_episode', use_global_arguments=False)
            executor.add_node(simulator)
            group = ReentrantCallbackGroup()
            stage = [0]
            ticks = [0]
            planned_goals = []
            maps = simulator.create_publisher(OccupancyGrid, 'map', 10)
            clocks = simulator.create_publisher(Clock, 'clock', 10)

            def plan(handle):
                result = ComputePathToPose.Result()
                planned_goals.append((handle.request.goal.pose.position.x,
                                      handle.request.goal.pose.position.y))
                if alternative_goal and len(planned_goals) == 1:
                    handle.abort()
                    return result
                result.path.poses = [PoseStamped(), handle.request.goal]
                handle.succeed()
                return result

            def navigate(handle):
                stage[0] += 1
                handle.succeed()
                return NavigateToPose.Result()

            servers = [ActionServer(simulator, ComputePathToPose, 'compute_path_to_pose',
                                    plan, callback_group=group),
                       ActionServer(simulator, NavigateToPose, 'navigate_to_pose',
                                    navigate, callback_group=group)]
            if startup_failure and not launched:
                for server in servers:
                    server.destroy()
                servers = []  # Map/clock arrive, but Nav2 never activates on this launch.

            def publish():
                ticks[0] += 1
                clock = Clock()
                clock.clock.sec, clock.clock.nanosec = divmod(ticks[0] * 100000000, 10**9)
                clocks.publish(clock)
                message = OccupancyGrid()
                message.header.frame_id = 'map'
                message.header.stamp = clock.clock
                message.info.width = message.info.height = 60
                message.info.resolution = .05
                message.info.origin.orientation.w = 1.
                if alternative_goal and stage[0] > 0:
                    message.info.origin.position.x = message.info.origin.position.y = .0125
                grid = np.full((60, 60), -1, dtype=np.int8)
                if stage[0] == 0:
                    grid[15:45, 15:45] = 0
                else:
                    grid[:] = 0
                message.data = grid.ravel().tolist()
                maps.publish(message)

            simulator.create_timer(.05, publish, callback_group=group)
            process = SimpleNamespace(simulator=simulator, servers=servers,
                                      poll=lambda: None, pid=100 + len(launched),
                                      planned_goals=planned_goals)
            launched.append((command, process))
            return process

        def stop(process):
            for server in process.servers:
                server.destroy()
            executor.remove_node(process.simulator)
            process.simulator.destroy_node()
            stopped.append(process)

        monkeypatch.setattr(runner.subprocess, 'Popen', launch)
        monkeypatch.setattr(runner, 'stop_simulation', stop)
        root = Path(__file__).resolve().parents[3]
        metadata = json.loads((root / 'maps.json').read_text())
        episodes = runner.episode_schedule(metadata, root / 'src/fronter_exploration/worlds',
                                           maps=['map01_open_obstacles', 'map02_corridor_loop'],
                                           pose_index=0)
        original_agent = node.trainer.agent
        runner.run_episodes(node, episodes,
                            lambda current, timeout_sec: executor.spin_once(timeout_sec),
                            rclpy.ok, startup_timeout=3. if startup_failure else 10.,
                            episode_timeout=15., startup_retries=int(startup_failure),
                            continue_on_failure=True)
        assert len(launched) == len(stopped) == 2 + int(startup_failure)
        assert node.trainer.agent is original_agent
        assert node.trainer.completed_episodes == original_agent.episodes == 2
        assert original_agent.steps >= 2 and original_agent.updates >= 2
        assert len(original_agent.replay) == original_agent.steps
        assert len(node.trainer.report.records) == 2 + int(startup_failure)
        if startup_failure:
            failed = node.trainer.report.records[0]
            assert failed['outcome'] == 'aborted' and not failed['success']
            assert failed['steps'] == 0 and 'No exploration started' in failed['reason']
            assert 'world:=map01_open_obstacles.sdf' in launched[1][0]
        if alternative_goal:
            for _, process in launched:
                assert len(process.planned_goals) >= 2
                assert process.planned_goals[0] != process.planned_goals[1]
            assert node.env.last_info['map_alignment'] == 'conservative_overlap'
            assert node.env.last_info['explored_cells'] > 0
        for row, episode in zip(node.trainer.report.records[int(startup_failure):], episodes):
            assert row['world'] == episode.world and row['pose_index'] == episode.pose_index
            assert row['spawn_pose'] == list(episode.pose)
            assert row['success'] and row['coverage'] == 1.
        assert 'world:=map01_open_obstacles.sdf' in launched[0][0]
        assert 'world:=map02_corridor_loop.sdf' in launched[-1][0]
        assert 'x_pose:=-5.15' in launched[-1][0]
        assert (tmp_path / 'latest.pt').is_file()
    finally:
        if executor is not None:
            executor.shutdown(timeout_sec=2.)
        for _, process in launched:
            if process not in stopped:
                for server in process.servers:
                    server.destroy()
                process.simulator.destroy_node()
        if node is not None:
            node.trainer.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
