"""Regression tests for the audited reward, freshness and checkpoint defects."""

from copy import deepcopy
import time
from types import SimpleNamespace as NS

from fronter_exploration.RL.env import Env, FEATURE_SCALING
import numpy as np
import pytest
from test_rl import Logger, make_node, post_map
from test_rl_integration import action_node, Handle, orchestration_node, Transport


@pytest.mark.parametrize('weights', [
    (-.1, .7, .4, 0.), (.7, .2, .2, 0.),
    (float('nan'), .2, .1, 0.), (.7, float('inf'), .1, 0.)])
def test_invalid_reward_weights_rejected(weights):
    with pytest.raises(ValueError, match='Reward weights'):
        Env(make_node(), *weights)


def test_weight_tolerance_and_yaml_names():
    from pathlib import Path
    from fronter_exploration.RL.training import PARAMETERS
    import yaml
    Env(make_node(), .7, .2, .10000001, 0.)
    configured = yaml.safe_load((Path(__file__).parents[1] / 'config/rl.yaml').read_text())
    values = configured['main_node']['ros__parameters']
    assert {key[3:]: value for key, value in values.items() if key.startswith('rl.')} == PARAMETERS
    assert sum(values['rl.reward_' + term]
               for term in ('gain', 'distance', 'stagnation', 'collision')) == pytest.approx(1.)


@pytest.mark.parametrize('weights,threshold,contacts,gain,expected', [
    ((1., 0., 0., 0.), .05, 0, True, .25),
    ((0., 1., 0., 0.), .05, 0, True, -4.),
    ((0., 0., 1., 0.), .05, 0, False, -1.),
    ((0., 0., 0., 1.), .05, 2, True, -2.),
    ((.4, .2, .2, .2), .3, 2, True, .4 * .25 - .2 * 4 - .2 - .2 * 2),
])
def test_all_reward_components_once(weights, threshold, contacts, gain, expected):
    node = make_node()
    env = Env(node, *weights, stagnation_threshold=threshold, collision_enabled=True)
    env.step(3)
    for _ in range(contacts):
        env.record_collision()  # Explicit simulated sensor input, not a Nav2 failure.
    env.navigation_finished(False)
    if gain:
        node.grid[0, 0] = 100
    node.grid[0, 2] = 100  # Known-to-known excluded.
    post_map(node, env)
    transition = env.complete_transition({})
    assert transition[2] == pytest.approx(expected)
    assert env.last_info['explored_cells'] == int(gain)
    assert env.last_info['collisions'] == contacts


def test_collision_disabled_and_stagnation_at_threshold():
    node = make_node()
    env = Env(node, stagnation_threshold=.25)
    env.step(3)
    env.record_collision()
    assert env.collision_count == 0
    env.navigation_finished(True)
    env.record_collision()
    node.grid[0, 0] = 0
    post_map(node, env)
    transition = env.complete_transition({})
    assert env.last_info['stagnation_penalty'] == 0
    assert transition[2] == pytest.approx(.7 * .25 - .2 * 4.)


def test_latest_pre_action_map_and_timestamp_gates():
    node = make_node()
    env = Env(node)
    newest = deepcopy(node.map)
    newest.data[0] = 0  # SLAM update while planning, absent from node.grid.
    newest.header.stamp.sec = 2
    node.latest_map = newest
    node.now_ns = 2 * 10**9
    env.step(3)
    assert env.previous_map[0, 0] == 0 and node.grid[0, 0] == -1
    assert not env.capture_post_map(newest)  # Navigation still pending.
    env.navigation_finished(True)
    assert not env.capture_post_map(newest)
    delayed = deepcopy(newest)
    delayed.header.stamp.nanosec = 100000000
    node.now_ns += 100000000
    assert not env.capture_post_map(delayed)  # Within the SLAM settling interval.
    node.grid[0, 0] = 0
    post_map(node, env)
    transition = env.complete_transition({})
    assert env.last_info['explored_cells'] == 0
    assert transition[2] == pytest.approx(-.2 * 4 - .1)


def test_stale_pre_map_rejected_before_action_is_committed():
    node = make_node()
    env = Env(node)
    node.now_ns = 8 * 10**9
    with pytest.raises(RuntimeError, match='stale'):
        env.step(3)
    assert env.current_action is None


def test_out_of_order_maps_are_not_latest():
    node = orchestration_node()
    node.rl = True
    node.planner.client = NS(server_is_ready=lambda: False)
    node.nav.client = NS(server_is_ready=lambda: False)
    newer = deepcopy(node.map)
    newer.header.stamp.sec = 2
    node.mapCallback(newer)
    node.mapCallback(node.map)
    assert node.latest_map is newer


@pytest.mark.parametrize('change', ['resolution', 'no_overlap',
                                    'rotation', 'frame'])
def test_incompatible_geometry_does_not_fabricate_gain_or_stagnation(change):
    node = make_node()
    env = Env(node)
    env.step(3)
    env.navigation_finished(True)
    if change == 'resolution':
        node.map.info.resolution = .25
    elif change == 'no_overlap':
        node.map.info.origin.position.x = 10.
    elif change == 'rotation':
        node.map.info.origin.orientation.z = 1.
    else:
        node.map.header.frame_id = 'different_map'
    node.grid[:] = 0
    post_map(node, env)
    assert env.complete_transition({})[2] == pytest.approx(-.2 * 4)
    assert not env.last_info['map_comparable']
    assert env.last_info['stagnation_penalty'] == 0


def test_missing_post_map_aborts_without_episode_completion():
    node = orchestration_node()
    node.rl = True
    node.env.step(3)
    node.env.navigation_finished(False)
    node.env.wait_started = time.monotonic() - 31.
    node._poll_rl()
    assert node.exploration_complete and node._rl_future is None
    assert node.env.current_action == 3  # No fabricated transition/counter increment.


def test_planning_failure_is_not_completion_and_can_retry():
    from fronter_exploration.cluster import Cluster, FrontierCluster
    from fronter_exploration.state import State
    from fronter_exploration.utils import Utils
    node = orchestration_node()
    node.rl = True
    node.env.step(3)
    node.env.navigation_finished(False)
    post_map(node, node.env)
    pending_action = node.env.current_action
    clusters = Cluster(1, 1, np.zeros((1, 1)))
    node.frontier = clusters.frontier = {9: FrontierCluster(9, [(0, 0)] * 21)}
    node.planner.path = {9: None}
    node.scoring(clusters, Utils(0., 0., .5))
    assert not node.exploration_complete and node.state == State.IDLE
    assert node.env.current_action == pending_action and not node.env.episode_done
    # A subsequent successful estimate finalizes the pending failure as nonterminal.
    node.frontier = {12: NS(id=12, information_gain=30., path_cost=2.)}
    transition = node.env.complete_transition(node.env.get_state())
    assert not transition[4] and set(transition[3]) == {12}


def test_unconfirmed_navigation_cancellation_aborts(monkeypatch):
    import fronter_exploration.nav2_client as module
    monkeypatch.setattr(module, 'ActionClient', Transport)
    from fronter_exploration.utils import Utils
    node = action_node()
    nav = module.Nav2Client(node)
    nav.goal(0, 1, Utils(0., 0., .5))
    handle = Handle()
    nav.client.requests[-1][1].set_result(handle)
    node.timers[0]()
    assert handle.cancelled and not node.env.awaiting_next_state
    node.timers[0]()
    assert node.exploration_complete and not node.env.awaiting_next_state
    handle.result.set_result(NS(status=4))
    assert not node.env.awaiting_next_state  # Late result cannot revive aborted episode.


def test_navigation_transport_error_does_not_finalize_unknown_motion(monkeypatch):
    import fronter_exploration.nav2_client as module
    monkeypatch.setattr(module, 'ActionClient', Transport)
    from fronter_exploration.utils import Utils
    node = action_node()
    nav = module.Nav2Client(node)
    nav.goal(0, 1, Utils(0., 0., .5))
    handle = Handle()
    nav.client.requests[-1][1].set_result(handle)
    handle.result.set_exception(RuntimeError('transport lost'))
    assert node.exploration_complete and not node.env.awaiting_next_state
    assert not node.timers


def test_reward_replayed_and_bootstrapped_without_change():
    from fronter_exploration.RL.dqn import DQNAgent, DQNConfig
    import torch
    node = make_node()
    env = Env(node, .4, .2, .2, .2, collision_enabled=True)
    env.step(3)
    env.record_collision()
    env.navigation_finished(True)
    post_map(node, env)
    transition = env.complete_transition(env.get_state())
    agent = DQNAgent(DQNConfig(batch_size=1))
    agent.remember(transition)
    expected = -.2 * 4 - .2 - .2
    assert agent.replay[0][1] == transition[2] == pytest.approx(expected)
    with torch.no_grad():
        for parameter in agent.target.parameters():
            parameter.zero_()
        agent.target.layers[-1].bias.fill_(3.)
    assert agent.bellman_targets(list(agent.replay)).item() == pytest.approx(expected + .99 * 3)


def trained_agent():
    """Produce at least one real optimizer update for checkpoint tests."""
    from fronter_exploration.RL.dqn import DQNAgent, DQNConfig
    agent = DQNAgent(DQNConfig(batch_size=1, target_update=2))
    state = {3: np.array([.2, .4], dtype=np.float32)}
    agent.remember((state, 3, 2., {}, True))
    agent.optimize()
    return agent


def test_untrained_shutdown_preserves_existing_trained_checkpoint(tmp_path):
    from fronter_exploration.RL.training import PARAMETERS, Training
    from fronter_exploration.RL.dqn import DQNAgent
    checkpoint = tmp_path / 'latest.pt'
    trained_agent().save(checkpoint)
    before = checkpoint.read_bytes()
    trainer = Training(dict(PARAMETERS, checkpoint_dir=str(tmp_path)), Logger())
    trainer.close()
    assert checkpoint.read_bytes() == before
    fresh = DQNAgent()
    assert not fresh.save(checkpoint)
    with pytest.raises(ValueError, match='cannot overwrite'):
        fresh.save(checkpoint, allow_untrained=True)
    assert checkpoint.read_bytes() == before


def test_zero_update_deployment_requires_explicit_override(tmp_path):
    from fronter_exploration.RL.dqn import DQNAgent
    from fronter_exploration.RL.deploy import Deployment
    checkpoint = tmp_path / 'testing.pt'
    agent = DQNAgent()
    assert not agent.save(checkpoint) and not checkpoint.exists()
    assert agent.save(checkpoint, allow_untrained=True)
    with pytest.raises(ValueError, match='zero optimizer updates'):
        Deployment(checkpoint)
    deploy = Deployment(checkpoint, allow_untrained=True)
    assert deploy.agent.updates == 0


def test_feature_scaling_checkpoint_and_inference_consistency(tmp_path):
    from fronter_exploration.RL.deploy import Deployment
    import torch
    node = make_node()
    node.frontier = {3: node.frontier[3]}
    env = Env(node)
    node.frontier[3].path_cost = 2.
    short = env.get_state()[3]
    node.frontier[3].path_cost = 20.
    long = env.get_state()[3]
    np.testing.assert_allclose(short, [.2, .2])
    np.testing.assert_allclose(long, [.2, 2.])
    agent = trained_agent()
    path = tmp_path / 'model.pt'
    agent.save(path)
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint['feature_scaling'] == FEATURE_SCALING
    assert checkpoint['updates'] == 1
    assert set(checkpoint) == {'format_version', 'feature_scaling', 'policy', 'target',
                               'optimizer', 'epsilon', 'steps', 'updates', 'episodes'}
    deploy = Deployment(path)
    for key, value in agent.policy.state_dict().items():
        assert torch.equal(value, deploy.agent.policy.state_dict()[key])
    for key, value in agent.target.state_dict().items():
        assert torch.equal(value, checkpoint['target'][key])
    assert any(not torch.equal(p, t) for p, t in
               zip(agent.policy.parameters(), agent.target.parameters()))
    features = torch.as_tensor(np.stack([short, long]))
    assert torch.equal(agent.policy(features), deploy.agent.policy(features))
    updates = deploy.agent.updates
    assert deploy.select_action({3: short, 7: long}) in (3, 7)
    assert deploy.agent.updates == updates and not deploy.agent.replay
    # Older tensor payloads remain readable, but unsafe feature semantics fail clearly.
    checkpoint['format_version'] = 1
    del checkpoint['feature_scaling']
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match='legacy feature scaling'):
        Deployment(path)
