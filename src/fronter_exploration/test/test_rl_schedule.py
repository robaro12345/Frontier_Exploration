"""Validate world/spawn scheduling and simulator restart orchestration."""

import json
from pathlib import Path
from types import SimpleNamespace as NS

from fronter_exploration.RL.run_training import episode_schedule, launch_command, run_episodes
import pytest


ROOT = Path(__file__).resolve().parents[3]


def test_actual_world_schedule():
    """Use the real metadata and every supplied SDF, including nonzero spawn poses."""
    metadata = json.loads((ROOT / 'maps.json').read_text())
    worlds = ROOT / 'src/fronter_exploration/worlds'
    schedule = episode_schedule(metadata, worlds)
    assert len(schedule) == 120
    assert len({episode.world for episode in schedule}) == 10
    assert schedule[0].pose == (3.05, 8.55, -1.54)
    assert schedule[11].pose_index == 11
    assert schedule[12].world == 'map02_corridor_loop.sdf'
    command = launch_command(schedule[0])
    assert command[:4] == ['ros2', 'launch', 'fronter_exploration', 'sim_launch.py']
    assert 'x_pose:=3.05' in command and 'y_pose:=8.55' in command
    assert 'yaw:=-1.54' in command and 'headless:=true' in command
    chosen = episode_schedule(metadata, worlds, ['map02_corridor_loop'], rounds=2,
                              pose_index=4)
    assert len(chosen) == 2 and chosen[0].pose == (-1.55, -4.15, .18)
    assert len(episode_schedule(metadata, worlds, rounds=2, limit=3)) == 3


@pytest.mark.parametrize('kwargs', [{'rounds': 0}, {'limit': 0}, {'pose_index': 99},
                                    {'maps': ['missing']}, {'start_episode': 0},
                                    {'start_episode': 121}])
def test_invalid_schedule(kwargs):
    """Fail before starting ROS or creating any simulator process."""
    metadata = json.loads((ROOT / 'maps.json').read_text())
    with pytest.raises(ValueError):
        episode_schedule(metadata, ROOT / 'src/fronter_exploration/worlds', **kwargs)


def test_resume_schedule_before_episode_limit():
    """Retry original entry 22, then continue through every remaining spawn."""
    metadata = json.loads((ROOT / 'maps.json').read_text())
    worlds = ROOT / 'src/fronter_exploration/worlds'
    original = episode_schedule(metadata, worlds)
    remaining = episode_schedule(metadata, worlds, start_episode=22)
    assert remaining == original[21:] and len(remaining) == 99
    assert remaining[0].world == 'map02_corridor_loop.sdf'
    assert remaining[0].pose_index == 9 and remaining[0].pose == (-6.15, -2.05, -2.14)
    assert episode_schedule(metadata, worlds, start_episode=22, limit=1) == [original[21]]
    assert episode_schedule(metadata, worlds, rounds=2, start_episode=121) == original


def test_restart_loop_retains_agent_and_replay(tmp_path, monkeypatch):
    """Use controlled processes to verify one restart per pose, retaining one node."""
    import fronter_exploration.RL.run_training as runner
    starts, stops, resets, contexts = [], [], [], []
    replay = []
    logger = NS(info=lambda message: None)
    node = NS(trainer=NS(directory=tmp_path, completed_episodes=0, replay=replay),
              exploration_complete=False, _rl_future=None, _episode_started=False,
              get_logger=lambda: logger, count_publishers=lambda topic: 0,
              nav=NS(client=NS(server_is_ready=lambda: False)),
              planner=NS(client=NS(server_is_ready=lambda: False)))
    node.trainer.prepare_episode = lambda *args: contexts.append(args)

    def reset(request, response):
        resets.append(True)
        node.exploration_complete = False
        node._episode_started = False
        response.success = True
        return response

    def start(command, **kwargs):
        starts.append((command, kwargs))
        return NS(poll=lambda: None, pid=1000 + len(starts))

    def spin(current, timeout_sec):
        assert current is node and current.trainer.replay is replay
        if not current.exploration_complete:
            current._episode_started = True
            replay.append(len(starts))
            current.trainer.completed_episodes += 1
            current.exploration_complete = True

    node._reset_episode = reset
    monkeypatch.setattr(runner.subprocess, 'Popen', start)
    monkeypatch.setattr(runner, 'stop_simulation', lambda process: stops.append(process.pid))
    metadata = json.loads((ROOT / 'maps.json').read_text())
    episodes = episode_schedule(metadata, ROOT / 'src/fronter_exploration/worlds', limit=2)
    run_episodes(node, episodes, spin, lambda: True)
    assert len(starts) == len(stops) == 2 and len(resets) == 1
    assert replay == [1, 2] and node.trainer.completed_episodes == 2
    assert starts[0][1]['start_new_session']
    assert starts[0][1]['env']['GZ_PARTITION'] == starts[1][1]['env']['GZ_PARTITION']
    assert 'x_pose:=3.05' in starts[0][0] and 'x_pose:=-5.95' in starts[1][0]
    assert contexts == [(episode.world, episode.pose_index, episode.pose) for episode in episodes]


def test_launch_failure_stops_process(tmp_path, monkeypatch):
    """A dead simulator cannot silently advance the curriculum."""
    import fronter_exploration.RL.run_training as runner
    stopped, failures = [], []
    node = NS(trainer=NS(directory=tmp_path, completed_episodes=0),
              exploration_complete=False, _rl_future=None, _episode_started=False,
              get_logger=lambda: NS(info=lambda message: None))
    node.trainer.prepare_episode = lambda *args: None
    node.trainer.abort_episode = lambda *args: failures.append(args)
    node._report_info = lambda: {}
    node._abort_rl = lambda reason: node.trainer.abort_episode(reason, None)
    process = NS(poll=lambda: 1, returncode=1, pid=1)
    directory = tmp_path / 'episodes'
    directory.mkdir()
    existing_log = directory / '0001_map01_open_obstacles_0.log'
    existing_log.write_text('previous launch evidence')
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *args, **kwargs: process)
    monkeypatch.setattr(runner, 'stop_simulation', lambda proc: stopped.append(proc))
    with pytest.raises(RuntimeError, match='Simulation launch exited'):
        run_episodes(node, [runner.Episode('map01_open_obstacles.sdf', 0, (1., 2., 3.))],
                     lambda *args, **kwargs: None, lambda: True)
    assert stopped == [process] and node.trainer.completed_episodes == 0
    assert 'Simulation launch exited' in failures[0][0]
    assert existing_log.read_text() == 'previous launch evidence'
    assert len(list(directory.glob('0001_map01_open_obstacles_0*.log'))) == 2


@pytest.mark.parametrize('mode', ['startup_timeout', 'launch_exit', 'navigation_abort',
                                  'episode_timeout'])
def test_failed_attempt_recovery_keeps_learning_and_reports(tmp_path, monkeypatch, mode):
    """Real node/worker reset after failure; retry startup or advance the schedule."""
    import fronter_exploration.RL.run_training as runner
    from fronter_exploration.RL.training import PARAMETERS, Training
    from test_rl import Logger
    from test_rl_integration import orchestration_node
    import numpy as np

    node = orchestration_node()
    node.rl = node.training = True
    node.episode_aborted = False
    node.subscriber = object()
    node.destroy_subscription = lambda subscription: None
    node.create_subscription = lambda *args: object()
    node.count_publishers = lambda topic: 0
    for client in (node.nav, node.planner):
        client.client = NS(server_is_ready=lambda: False)
    node.trainer = Training(dict(PARAMETERS, episode_limit=2, batch_size=1,
                                 checkpoint_dir=str(tmp_path)), Logger())
    agent = node.trainer.agent
    replay = agent.replay
    clock = [0.]
    starts, stops = [], []
    state = {3: np.array([.2, .4], dtype=np.float32)}

    def start(command, **kwargs):
        assert len(stops) == len(starts)  # Every old stack stopped before a new launch.
        starts.append(command)
        index = len(starts)
        return NS(pid=index, returncode=1,
                  poll=lambda: 1 if index == 1 and mode == 'launch_exit' else None)

    def spin(current, timeout_sec):
        clock[0] += .1
        if current.exploration_complete:
            current._poll_rl()
            return
        if len(starts) == 1:
            if mode in ('startup_timeout', 'launch_exit'):
                return
            current._episode_started = True
            if mode == 'navigation_abort':
                current._abort_rl('No reachable frontier paths before unreachable_timeout')
                current._rl_future.result(5)
                current._poll_rl()
            return
        current._episode_started = True
        current.trainer.decide({}, (state, 3, 2., {}, True), terminal=True).result(5)
        current.exploration_complete = True

    monkeypatch.setattr(runner.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(runner.subprocess, 'Popen', start)
    monkeypatch.setattr(runner, 'stop_simulation', lambda process: stops.append(process.pid))
    episodes = [runner.Episode('first.sdf', 0, (1., 2., 3.)),
                runner.Episode('second.sdf', 1, (4., 5., 6.))]
    try:
        run_episodes(node, episodes, spin, lambda: True, startup_timeout=.15,
                     episode_timeout=.3, startup_retries=1, continue_on_failure=True)
        rows = node.trainer.report.records
        startup = mode in ('startup_timeout', 'launch_exit')
        assert [row['success'] for row in rows] == ([False, True, True] if startup
                                                  else [False, True])
        assert [row['world'] for row in rows] == (['first.sdf', 'first.sdf', 'second.sdf']
                                                 if startup else ['first.sdf', 'second.sdf'])
        assert node.trainer.agent is agent and agent.replay is replay
        assert agent.updates == agent.steps == node.trainer.completed_episodes == (
            2 if startup else 1)
        assert len(starts) == len(stops) == len(rows)
        assert rows[0]['outcome'] == 'aborted' and rows[0]['steps'] == 0
    finally:
        node.trainer.close()


def test_startup_retry_exhaustion_is_bounded(tmp_path, monkeypatch):
    """Each startup failure is retained; exhaust retries then advance exactly once."""
    import fronter_exploration.RL.run_training as runner
    from concurrent.futures import Future
    starts, failures, resets = [], [], []
    logger = NS(info=lambda message: None, warn=lambda message: None)
    node = NS(trainer=NS(directory=tmp_path, completed_episodes=0),
              exploration_complete=False, episode_aborted=False,
              _rl_future=None, _episode_started=False, get_logger=lambda: logger,
              count_publishers=lambda topic: 0,
              nav=NS(client=NS(server_is_ready=lambda: False)),
              planner=NS(client=NS(server_is_ready=lambda: False)))
    node.trainer.prepare_episode = lambda *args: None

    def abort(reason):
        failures.append(reason)
        node.exploration_complete = node.episode_aborted = True
        node._rl_future = Future()
        node._rl_future.set_result(None)

    def reset(request, response, *, simulation_stopped=False):
        assert simulation_stopped
        resets.append(True)
        node.exploration_complete = node.episode_aborted = False
        response.success = True
        return response

    node._abort_rl = abort
    node._poll_rl = lambda: setattr(node, '_rl_future', None)
    node._reset_episode = reset
    monkeypatch.setattr(runner.subprocess, 'Popen',
                        lambda command, **kwargs: starts.append(command) or NS(
                            poll=lambda: 1, returncode=1))
    monkeypatch.setattr(runner, 'stop_simulation', lambda process: None)
    episodes = [runner.Episode('first.sdf', 0, (0., 0., 0.)),
                runner.Episode('second.sdf', 0, (0., 0., 0.))]
    run_episodes(node, episodes, lambda *args, **kwargs: None, lambda: True,
                 startup_retries=1, continue_on_failure=True)
    assert len(starts) == len(failures) == 4 and len(resets) == 3
    assert node.trainer.completed_episodes == 0


def test_shutdown_wait_handles_delayed_discovery_without_ignoring_endpoints(monkeypatch):
    """Old endpoints can disappear after 10 seconds; no restart happens while they exist."""
    import fronter_exploration.RL.run_training as runner
    from test_rl import Logger
    clock = [0.]
    logger = Logger()
    node = NS(count_publishers=lambda topic: int(clock[0] < 20.),
              nav=NS(client=NS(server_is_ready=lambda: clock[0] < 25.)),
              planner=NS(client=NS(server_is_ready=lambda: clock[0] < 15.)),
              get_logger=lambda: logger)
    monkeypatch.setattr(runner.time, 'monotonic', lambda: clock[0])
    runner._wait_for_shutdown(node, lambda *args, **kwargs: clock.__setitem__(0, clock[0] + 1),
                              lambda: True)
    assert clock[0] == 25.
    assert any('map publishers=1' in message for message in logger.messages)
    assert any('navigation ready=True' in message for message in logger.messages)


def test_shutdown_wait_still_rejects_persistent_publishers(monkeypatch):
    """Do not bypass a genuinely live stack just to continue training."""
    import fronter_exploration.RL.run_training as runner
    from test_rl import Logger
    clock = [0.]
    node = NS(count_publishers=lambda topic: 1,
              nav=NS(client=NS(server_is_ready=lambda: False)),
              planner=NS(client=NS(server_is_ready=lambda: False)),
              get_logger=lambda: Logger())
    monkeypatch.setattr(runner.time, 'monotonic', lambda: clock[0])
    with pytest.raises(RuntimeError, match='map publishers=1.*waited 3s'):
        runner._wait_for_shutdown(
            node, lambda *args, **kwargs: clock.__setitem__(0, clock[0] + 1),
            lambda: True, timeout=3.)


@pytest.mark.parametrize('timeout', [0., -1., float('nan'), float('inf')])
def test_invalid_shutdown_timeout_fails_before_launch(timeout):
    """A configured shutdown wait is always finite and bounded."""
    with pytest.raises(ValueError, match='shutdown timeouts'):
        run_episodes(None, [], None, None, shutdown_timeout=timeout)
