"""Tests for candidate replay, map rewards and nonblocking episode transitions."""

from types import SimpleNamespace as NS

from fronter_exploration.RL.env import Env, explored_cells
import numpy as np
import pytest


class Logger:
    """Collect messages without requiring a running ROS graph."""

    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)

    warn = info
    error = info


def make_node():
    """Build the fields actually consumed by the environment."""
    origin = NS(position=NS(x=0., y=0.), orientation=NS(x=0., y=0., z=0., w=1.))
    logger = Logger()
    grid = np.array([[-1, -1, 0], [100, -1, 0]])
    message = NS(info=NS(origin=origin, resolution=.5, width=3, height=2),
                 header=NS(frame_id='map', stamp=NS(sec=1, nanosec=0)),
                 data=grid.ravel().tolist())
    node = NS(grid=grid, map=message, latest_map=message, now_ns=10**9,
              frontier={3: NS(id=3, information_gain=20., path_cost=4.),
                        7: NS(id=7, information_gain=40., path_cost=2.)},
              get_logger=lambda: logger)
    node.get_clock = lambda: NS(now=lambda: NS(nanoseconds=node.now_ns))
    return node


def post_map(node, env):
    """Supply an acquired outcome after the configured settling interval."""
    from copy import deepcopy
    message = deepcopy(node.map)
    node.now_ns += 10**9
    message.header.stamp.sec, message.header.stamp.nanosec = divmod(node.now_ns, 10**9)
    message.data = node.grid.ravel().tolist()
    node.latest_map = message
    assert env.capture_post_map(message)


def test_unknown_to_known_only():
    old = np.array([[-1, -1, 0, 100, -1]])
    new = np.array([[0, 100, 100, 0, -1]])
    assert explored_cells(old, new) == 2


def test_map_expansion_and_origin_shift():
    old = np.array([[-1, 0], [-1, 100]])
    new = np.full((4, 4), -1)
    new[1:3, 1:3] = np.array([[0, 100], [-1, 0]])
    assert explored_cells(old, new, (0., 0.), (-.5, -.5), .5) == 1
    assert explored_cells(old, new, (10., 10.), (0., 0.), .5) == 0
    assert explored_cells(old, new, (0., 0.), (.1, 0.), .5) == 0


@pytest.mark.parametrize('origin,expected', [
    ((.25, .25), 9), ((-.25, -.25), 9), ((.25, 0.), 12), ((0., -.25), 12)])
def test_fractional_map_shift_counts_only_fully_unknown_footprints(origin, expected):
    """Partial cells outside the previous grid cannot earn invented exploration gain."""
    old = np.full((4, 4), -1)
    new = np.zeros((4, 4), dtype=int)
    assert explored_cells(old, new, (0., 0.), origin) == expected


def test_fractional_shift_excludes_known_boundary_and_unknown_outcomes():
    """A raster shift across existing known space is not new exploration."""
    old = np.array([[-1, -1, 0, 100]])
    new = np.array([[-1, 0, 100, 0]])
    assert explored_cells(old, new, (0., 0.), (.25, 0.)) == 0
    old[:] = -1
    assert explored_cells(old, new, (0., 0.), (.25, 0.)) == 2
    new[:] = -1
    assert explored_cells(old, new, (0., 0.), (.25, 0.)) == 0
    with pytest.raises(ValueError, match='do not overlap'):
        explored_cells(old, new, (0., 0.), (10.25, 0.), require_overlap=True)


def test_fractional_shift_known_cell_overlap_is_conservative():
    """One previously known cell disqualifies every new footprint crossing it."""
    old = np.full((4, 4), -1)
    old[1, 1] = 100
    assert explored_cells(old, np.zeros((4, 4)), (0., 0.), (.25, .25)) == 5


def test_fractional_map_reward_is_stored_and_optimized(agent):
    """Real transition/replay/update retains positive gain after a fractional shift."""
    node = make_node()
    node.grid = np.full((4, 4), -1)
    node.map.info.width = node.map.info.height = 4
    node.map.data = node.grid.ravel().tolist()
    env = Env(node)
    env.step(3)
    node.grid[:] = 0
    node.map.info.origin.position.x = node.map.info.origin.position.y = .125
    env.navigation_finished(True)
    post_map(node, env)
    transition = env.complete_transition({})
    expected = .7 * 9 * .5 ** 2 - .2 * 4
    assert transition[2] == pytest.approx(expected)
    assert env.last_info['explored_cells'] == 9 and env.last_info['map_comparable']
    assert env.last_info['map_alignment'] == 'conservative_overlap'
    assert env.last_info['stagnation_penalty'] == 0
    agent.remember(transition)
    agent.remember(transition)
    assert agent.replay[-1][1] == pytest.approx(expected)
    assert np.isfinite(agent.optimize()) and agent.updates == 1


def test_state_features_and_zero_cost():
    node = make_node()
    env = Env(node)
    state = env.get_state()
    np.testing.assert_allclose(state[3], [.2, .4])
    np.testing.assert_allclose(state[7], [.4, .2])
    for cluster in node.frontier.values():
        cluster.information_gain = 0.
        cluster.path_cost = 0.
    assert all(np.isfinite(value).all() for value in env.get_state().values())
    node.frontier.clear()
    assert env.get_state() == {}


def test_nonblocking_transition_and_reward_snapshot():
    node = make_node()
    env = Env(node, .4, .2, .2, .2, collision_enabled=True)
    state = env.reset()
    cluster = env.step(3, state)
    assert cluster is node.frontier[3]
    assert not env.awaiting_next_state  # step returns without any navigation wait.
    state[3][:] = 99
    with pytest.raises(RuntimeError):
        env.step(7)
    with pytest.raises(RuntimeError):
        env.reset()
    with pytest.raises(RuntimeError):
        env.complete_transition({})
    env.record_collision()
    env.navigation_finished(True)
    node.grid[0, 0] = 0
    node.grid[0, 2] = 100  # known-to-known must not receive gain.
    post_map(node, env)
    node.frontier = {12: NS(id=12, information_gain=30., path_cost=5.)}
    transition = env.complete_transition(env.get_state())
    np.testing.assert_allclose(transition[0][3], [.2, .4])
    assert transition[1] == 3
    assert transition[2] == pytest.approx(.4 * .25 - .2 * 4. - .2)
    assert set(transition[3]) == {12}
    assert not transition[4]
    assert env.last_info['explored_cells'] == 1
    assert env.current_action is None
    assert env.step(12) is node.frontier[12]
    env.navigation_finished(False)
    post_map(node, env)
    terminal = env.complete_transition({})
    assert terminal[4] and terminal[2] == pytest.approx(-.2 * 5. - .2)
    assert not env.last_info['navigation_success']


def test_invalid_action():
    env = Env(make_node())
    env.reset()
    with pytest.raises(ValueError):
        env.step(999)


@pytest.fixture
def agent():
    """Use a tiny batch to exercise actual gradient updates."""
    pytest.importorskip('torch')
    from fronter_exploration.RL.dqn import DQNAgent, DQNConfig
    return DQNAgent(DQNConfig(batch_size=2, target_update=1))


def test_dynamic_selection_replay_and_optimization(agent):
    import torch
    state = {3: np.array([.5, 1.], dtype=np.float32),
             7: np.array([1., .5], dtype=np.float32)}
    assert agent.select_action({}) is None
    assert agent.select_action(state) in state
    agent.remember((state, 3, 1., {12: state[7]}, False))
    agent.remember((state, 7, -1., {}, False))
    state[3][:] = 99
    np.testing.assert_allclose(agent.replay[0][0], [.5, 1.])
    assert agent.replay[1][3]  # Empty next candidates are terminal.
    weights = [p.detach().clone() for p in agent.policy.parameters()]
    assert np.isfinite(agent.optimize())
    assert any(not torch.equal(old, new)
               for old, new in zip(weights, agent.policy.parameters()))
    assert all(torch.equal(p, t) for p, t in
               zip(agent.policy.parameters(), agent.target.parameters()))
    assert agent.updates == 1 and agent.epsilon < 1.
    assert agent.select_action({91: np.zeros(2, dtype=np.float32)}, False) == 91


def test_bellman_targets(agent):
    import torch
    with torch.no_grad():
        for parameter in agent.target.parameters():
            parameter.zero_()
        agent.target.layers[-1].bias.fill_(2.)
    candidates = np.array([[1., .2], [.5, 1.]], dtype=np.float32)
    samples = [(np.zeros(2), 3., candidates, False),
               (np.zeros(2), 4., candidates, True),
               (np.zeros(2), 5., np.empty((0, 2), dtype=np.float32), False)]
    np.testing.assert_allclose(agent.bellman_targets(samples).numpy(),
                               [3. + .99 * 2., 4., 5.])


def test_checkpoint_and_deployment(agent, tmp_path):
    import torch
    from fronter_exploration.RL.deploy import Deployment
    from fronter_exploration.RL.dqn import DQNAgent
    path = tmp_path / 'nested' / 'model.pt'
    state = {3: np.array([.5, 1.], dtype=np.float32),
             7: np.array([1., .5], dtype=np.float32)}
    agent.remember((state, 3, 1., state, False))
    agent.remember((state, 7, 2., {}, True))
    agent.optimize()
    agent.save(path)
    restored = DQNAgent(agent.config)
    restored.load(path, training=True)
    assert restored.epsilon == agent.epsilon and restored.updates == agent.updates
    assert all(torch.equal(p, r) for p, r in
               zip(agent.policy.parameters(), restored.policy.parameters()))
    deploy = Deployment(path)
    assert deploy.agent.epsilon == 0.
    assert deploy.select_action(state) == agent.select_action(state, explore=False)
    assert deploy.select_action({}) is None
    assert len(deploy.agent.replay) == 0
    with pytest.raises(ValueError):
        Deployment('')


def test_training_worker_and_episode_limit(tmp_path):
    pytest.importorskip('torch')
    from fronter_exploration.RL.training import PARAMETERS, Training
    settings = dict(PARAMETERS, batch_size=1, checkpoint_every=1,
                    checkpoint_dir=str(tmp_path))
    logger = Logger()
    training = Training(settings, logger)
    training.start_episode()
    state = {7: np.array([1., .5], dtype=np.float32)}
    assert training.decide(state).result(timeout=5) == 7
    assert training.decide({}, (state, 7, -.5, {}, True),
                           {'navigation_success': False}, terminal=True).result(5) is None
    assert training.finished and training.episode_steps == 1
    assert training.successes == 0 and training.agent.updates == 1
    assert (tmp_path / 'latest.pt').is_file()
    assert (tmp_path / 'step_1.pt').is_file()
    with pytest.raises(RuntimeError):
        training.start_episode()
    training.close()
