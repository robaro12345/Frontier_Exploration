"""Exercise real orchestration and action callbacks with controlled transport."""

from concurrent.futures import Future
from types import SimpleNamespace as NS

from fronter_exploration.cluster import Cluster, FrontierCluster
from fronter_exploration.main_node import main_node
from fronter_exploration.RL.env import Env
from fronter_exploration.state import State
from fronter_exploration.utils import Utils
import numpy as np
import pytest
from test_rl import Logger, make_node, post_map


class Transport:
    """Return unresolved Futures; callbacks are completed explicitly by tests."""

    def __init__(self, *args):
        self.requests = []
        self.ready = True

    def server_is_ready(self):
        return self.ready

    def send_goal_async(self, goal):
        future = Future()
        self.requests.append((goal, future))
        return future


class Handle:
    """Represent accepted/rejected goals and expose a result future."""

    def __init__(self, accepted=True):
        self.accepted = accepted
        self.result = Future()
        self.cancelled = False

    def get_result_async(self):
        return self.result

    def cancel_goal_async(self):
        self.cancelled = True
        future = Future()
        future.set_result(None)
        return future


def action_node():
    """Supply node timers and flags without opening DDS sockets."""
    node = make_node()
    node.rl = True
    node.state = State.NAVIGATING
    node.planner = NS(path={3: 'old-path'})
    node.env = Env(node)
    node.env.reset()
    node.env.step(3)
    node.get_clock = lambda: NS(now=lambda: NS(nanoseconds=node.now_ns, to_msg=lambda: __import__(
        'builtin_interfaces.msg', fromlist=['Time']).Time()))
    node.timers = []
    node.create_timer = lambda period, callback, **kwargs: node.timers.append(callback) or callback
    node.destroy_timer = lambda timer: node.timers.remove(timer)
    node._abort_rl = lambda reason: setattr(node, 'exploration_complete', True)
    return node


@pytest.mark.parametrize('status,success', [(4, True), (5, False), (6, False)])
def test_navigation_status_and_stale_result(monkeypatch, status, success):
    import fronter_exploration.nav2_client as module
    monkeypatch.setattr(module, 'ActionClient', Transport)
    node = action_node()
    nav = module.Nav2Client(node)
    nav.goal(0, 1, Utils(0., 0., .5))
    assert node.state == State.NAVIGATING and not node.navigation_done
    handle = Handle()
    nav.client.requests[-1][1].set_result(handle)
    handle.result.set_result(NS(status=status))
    assert node.navigation_done and node.navigation_success == success
    assert node.state == State.IDLE and node.env.awaiting_next_state
    assert not node.planner.path and not node.timers


def test_rejected_navigation_and_timeout(monkeypatch):
    import fronter_exploration.nav2_client as module
    monkeypatch.setattr(module, 'ActionClient', Transport)
    node = action_node()
    nav = module.Nav2Client(node)
    nav.goal(0, 1, Utils(0., 0., .5))
    nav.client.requests[-1][1].set_result(Handle(accepted=False))
    assert node.navigation_done and not node.navigation_success
    assert node.state == State.IDLE and node.env.awaiting_next_state
    post_map(node, node.env)
    node.env.complete_transition(node.env.get_state())
    node.env.step(3)
    nav.goal(0, 1, Utils(0., 0., .5))
    handle = Handle()
    nav.client.requests[-1][1].set_result(handle)
    node.timers[0]()
    assert handle.cancelled and not node.navigation_done
    assert not node.env.awaiting_next_state
    handle.result.set_result(NS(status=5))
    assert node.navigation_done and not node.navigation_success


def test_navigation_server_unavailable(monkeypatch):
    import fronter_exploration.nav2_client as module
    monkeypatch.setattr(module, 'ActionClient', Transport)
    node = action_node()
    nav = module.Nav2Client(node)
    nav.client.ready = False
    nav.goal(0, 1, Utils(0., 0., .5))
    assert node.navigation_done and node.env.awaiting_next_state
    assert not nav.client.requests


def test_planner_rejection_advances_queue(monkeypatch):
    import fronter_exploration.planner_client as module
    monkeypatch.setattr(module, 'ActionClient', Transport)
    node = action_node()
    advanced = []
    node.send_to_queue = lambda utils, clusters: advanced.append(True)
    planner = module.Planner(node)
    planner.cost_planner(NS(id=3, nearest_point=(0, 1)), Utils(0., 0., .5), None)
    assert not advanced
    planner.client.requests[-1][1].set_result(Handle(accepted=False))
    assert advanced == [True] and planner.path[3] is None
    assert not planner.goal_futures and not node.timers


def test_planner_timeout_and_empty_path(monkeypatch):
    import fronter_exploration.planner_client as module
    monkeypatch.setattr(module, 'ActionClient', Transport)
    node = action_node()
    advanced = []
    node.send_to_queue = lambda utils, clusters: advanced.append(True)
    planner = module.Planner(node)
    args = (NS(id=3, nearest_point=(0, 1)), Utils(0., 0., .5), None)
    planner.cost_planner(*args)
    handle = Handle()
    planner.client.requests[-1][1].set_result(handle)
    node.timers[0]()
    handle.result.set_result(NS(status=4, result=NS(path=NS(poses=[NS()]))))
    assert handle.cancelled and advanced == [True] and planner.path[3] is None
    planner.cost_planner(*args)
    empty = Handle()
    planner.client.requests[-1][1].set_result(empty)
    empty.result.set_result(NS(status=4, result=NS(path=NS(poses=[]))))
    assert advanced == [True, True] and planner.path[3] is None


def test_cluster_pruning_and_zero_path():
    clusters = Cluster(1, 1, np.zeros((1, 1)))
    clusters.frontier = {i: FrontierCluster(i, [(0, 0)]) for i in range(3)}
    planner = NS(path={0: None, 1: None, 2: NS()},
                 path_cost_calculator=lambda key: 0.)
    clusters.estimation(planner, None)
    assert set(clusters.frontier) == {2}
    assert np.isfinite(clusters.frontier[2].score)


def planner_result(handle, start, end):
    """Deliver an actual ROS path through the asynchronous planner callbacks."""
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Path
    path = Path()
    for x, y in (start, end):
        pose = PoseStamped()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        path.poses.append(pose)
    handle.result.set_result(NS(status=4, result=NS(path=path)))


@pytest.mark.parametrize('replanned_end,usable', [((.6, 0.), True), ((.2, 0.), False)])
def test_rl_near_frontier_replans_once(monkeypatch, replanned_end, usable):
    """A near-goal success cannot repeatedly navigate within Nav2's tolerance."""
    import fronter_exploration.planner_client as module
    monkeypatch.setattr(module, 'ActionClient', Transport)
    node = action_node()
    advanced = []
    node.send_to_queue = lambda *args: advanced.append(True)
    cluster = FrontierCluster(3, [(0, 2), (0, 6), (0, 10)])
    cluster.nearest_point = (0, 2)
    clusters = NS(frontier={3: cluster})
    planner = module.Planner(node)
    utils = Utils(-.05, -.05, .1)
    planner.cost_planner(cluster, utils, clusters)
    first = Handle()
    planner.client.requests[-1][1].set_result(first)
    planner_result(first, (0., 0.), (.2, 0.))
    assert len(planner.client.requests) == 2 and not advanced
    assert cluster.nearest_point == (0, 6)
    assert planner.client.requests[-1][0].goal.pose.position.x == pytest.approx(.6)
    second = Handle()
    planner.client.requests[-1][1].set_result(second)
    planner_result(second, (0., 0.), replanned_end)
    assert advanced == [True] and not node.timers
    assert len(planner.client.requests) == 2
    assert (planner.path[3] is not None) == usable
    if usable:
        assert planner.path_cost_calculator(3) == pytest.approx(.6)


@pytest.mark.parametrize('rl,cells,end,usable', [
    (True, [(0, 1), (0, 2)], (.2, 0.), False),
    (False, [(0, 1), (0, 2)], (.2, 0.), True),
    (True, [(0, 6), (0, 10)], (.6, 0.), True),
])
def test_near_goal_filter_preserves_heuristic_and_distant_goals(
        monkeypatch, rl, cells, end, usable):
    """Unavailable farther targets are pruned without declaring completion."""
    import fronter_exploration.planner_client as module
    monkeypatch.setattr(module, 'ActionClient', Transport)
    node = action_node()
    node.rl = rl
    advanced = []
    node.send_to_queue = lambda *args: advanced.append(True)
    cluster = FrontierCluster(3, cells)
    original = cluster.nearest_point
    planner = module.Planner(node)
    planner.cost_planner(cluster, Utils(-.05, -.05, .1), NS(frontier={3: cluster}))
    handle = Handle()
    planner.client.requests[-1][1].set_result(handle)
    planner_result(handle, (0., 0.), end)
    assert advanced == [True] and len(planner.client.requests) == 1
    assert (planner.path[3] is not None) == usable
    assert cluster.nearest_point == original and not node.timers
    assert not getattr(node, 'exploration_complete', False)


def test_failed_frontier_goal_tries_another_cell_before_pruning(monkeypatch):
    """A failed representative must not discard a cluster with a reachable alternative."""
    import fronter_exploration.planner_client as module
    monkeypatch.setattr(module, 'ActionClient', Transport)
    node = action_node()
    advanced = []
    node.send_to_queue = lambda *args: advanced.append(True)
    cluster = FrontierCluster(3, [(0, 0), (0, 5), (0, 10), (0, 15), (0, 20)])
    clusters = NS(frontier={3: cluster})
    planner = module.Planner(node)
    utils = Utils(-.05, -.05, .1)
    initial = cluster.nearest_point
    planner.cost_planner(cluster, utils, clusters)
    first = Handle()
    planner.client.requests[-1][1].set_result(first)
    first.result.set_result(NS(status=6, result=NS(path=NS(poses=[]))))
    assert len(planner.client.requests) == 2 and not advanced
    assert cluster.nearest_point != initial
    target = utils.grid_to_world(*cluster.nearest_point)
    second = Handle()
    planner.client.requests[-1][1].set_result(second)
    planner_result(second, (-1., 0.), tuple(target))
    assert advanced == [True] and not node.timers
    assert planner.path[3] is not None and planner.path_cost_calculator(3) > 0
    assert len(planner.client.requests) == 2


@pytest.mark.parametrize('rl,maximum', [(True, 4), (False, 1)])
def test_frontier_alternative_trials_are_bounded_and_rl_only(monkeypatch, rl, maximum):
    """Exhausting reachable-point attempts advances the queue once; heuristics stay intact."""
    import fronter_exploration.planner_client as module
    monkeypatch.setattr(module, 'ActionClient', Transport)
    node = action_node()
    node.rl = rl
    node.frontier_goal_attempts = 4
    advanced = []
    node.send_to_queue = lambda *args: advanced.append(True)
    cluster = FrontierCluster(3, [(0, index) for index in range(50)])
    planner = module.Planner(node)
    planner.cost_planner(cluster, Utils(0., 0., .1), NS(frontier={3: cluster}))
    for _ in range(maximum):
        handle = Handle()
        planner.client.requests[-1][1].set_result(handle)
        handle.result.set_result(NS(status=6, result=NS(path=NS(poses=[]))))
    assert len(planner.client.requests) == len(planner.attempted_points) == maximum
    assert advanced == [True] and planner.path[3] is None and not node.timers


def test_near_goal_replan_respects_frontier_attempt_budget(monkeypatch):
    """Even a successful but unusably short path cannot exceed the configured budget."""
    import fronter_exploration.planner_client as module
    monkeypatch.setattr(module, 'ActionClient', Transport)
    node = action_node()
    node.frontier_goal_attempts = 1
    advanced = []
    node.send_to_queue = lambda *args: advanced.append(True)
    cluster = FrontierCluster(3, [(0, 2), (0, 6)])
    cluster.nearest_point = (0, 2)
    planner = module.Planner(node)
    planner.cost_planner(cluster, Utils(-.05, -.05, .1), NS(frontier={3: cluster}))
    handle = Handle()
    planner.client.requests[-1][1].set_result(handle)
    planner_result(handle, (0., 0.), (.2, 0.))
    assert len(planner.client.requests) == 1 and advanced == [True]
    assert planner.path[3] is None


def orchestration_node():
    """Construct main_node methods with only their existing data dependencies."""
    node = main_node.__new__(main_node)
    base = make_node()
    node.grid, node.map, node.frontier = base.grid, base.map, base.frontier
    node.latest_map = base.latest_map
    node.now_ns = base.now_ns
    node.get_clock = lambda: NS(now=lambda: NS(nanoseconds=node.now_ns))
    node.get_logger = lambda: Logger()
    node.rl = False
    node.training = False
    node.trainer = None
    node._rl_future = None
    node._episode_started = False
    node.exploration_complete = False
    node.state = State.IDLE
    node.env = Env(node)
    node.planner = NS(path={})
    node.requests = []
    node.nav = NS(goal=lambda *args: node.requests.append(args))
    for cluster in node.frontier.values():
        cluster.nearest_point = (0, 1)
        cluster.r = cluster.g = cluster.b = 0.
        cluster.score = cluster.information_gain / cluster.path_cost
    return node


def test_heuristic_selection_unchanged():
    node = orchestration_node()
    main_node.scoring(node, NS(estimation=lambda *args: None), Utils(0., 0., .5))
    assert node.best_node.id == 7  # Original gain/path score is greatest.
    assert node.state == State.NAVIGATING and len(node.requests) == 1
    assert node.env.current_action is None


def test_main_training_transition_through_next_scoring(tmp_path):
    pytest.importorskip('torch')
    from fronter_exploration.RL.training import PARAMETERS, Training
    node = orchestration_node()
    node.rl = node.training = True
    node.trainer = Training(dict(PARAMETERS, batch_size=1,
                                 checkpoint_dir=str(tmp_path)), Logger())
    utils = Utils(0., 0., .5)
    clus = NS(estimation=lambda *args: None)
    node.scoring(clus, utils)
    node._rl_future.result(5)
    node._poll_rl()
    old_action = node.best_node.id
    assert old_action in (3, 7) and node.state == State.NAVIGATING
    node.env.navigation_finished(True)
    node.grid[0, 0] = 0
    post_map(node, node.env)
    next_cluster = NS(id=12, information_gain=30., path_cost=2.,
                      nearest_point=(0, 1), r=0., g=0., b=0.)
    node.frontier = {12: next_cluster}
    node.scoring(clus, utils)
    node._rl_future.result(5)
    node._poll_rl()
    assert node.best_node.id == 12 and node.trainer.agent.steps == 1
    assert node.trainer.agent.updates == 1 and node.trainer.successes == 1
    node.env.navigation_finished(False)
    post_map(node, node.env)
    node.frontier = {}
    node._finish_exploration(utils)
    node._rl_future.result(5)
    node._poll_rl()
    assert node.exploration_complete and node.trainer.finished
    assert node.trainer.agent.steps == 2 and node.trainer.successes == 1
    assert node.trainer.agent.replay[-1][3]
    response = node._reset_episode(None, NS())
    assert not response.success
    node.trainer.close()


def test_episode_reset_service(tmp_path):
    pytest.importorskip('torch')
    from fronter_exploration.RL.training import PARAMETERS, Training
    node = orchestration_node()
    node.rl = node.training = True
    old_subscription = node.subscriber = object()
    discarded = []
    node.destroy_subscription = lambda subscription: discarded.append(subscription)
    node.create_subscription = lambda *args: object()
    node.trainer = Training(dict(PARAMETERS, episode_limit=2,
                                 checkpoint_dir=str(tmp_path)), Logger())
    assert not node._reset_episode(None, NS()).success
    node._finish_exploration(Utils(0., 0., .5))
    assert not node._reset_episode(None, NS()).success  # Worker result still pending.
    node._rl_future.result(5)
    node._poll_rl()
    assert node._reset_episode(None, NS()).success
    assert not node.exploration_complete and not node._episode_started
    assert discarded == [old_subscription] and node.subscriber is not old_subscription
    assert node.latest_map is None and node.env.last_map_stamp == -1
    node.trainer.close()


def test_unreachable_watchdog_records_completed_action_as_failed_episode(tmp_path, monkeypatch):
    """A completed navigation's real reward survives a bounded planning failure."""
    from fronter_exploration.RL.training import PARAMETERS, Training
    import fronter_exploration.main_node as module
    node = orchestration_node()
    node.rl = node.training = True
    node.trainer = Training(dict(PARAMETERS, batch_size=1, episode_limit=2,
                                 checkpoint_dir=str(tmp_path)), Logger())
    node.unreachable_timeout = 60.
    clock = [10.]
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    node.trainer.start_episode()
    node.env.reset()
    node.env.step(3)
    node.env.navigation_finished(True)
    node.grid[0, 0] = 0
    post_map(node, node.env)
    expected_reward = node.env.calculate_reward()
    node.frontier = {}
    clus = NS(estimation=lambda *args: None)
    utils = Utils(0., 0., .5)
    try:
        node.scoring(clus, utils)
        assert node._unreachable_since == 10.
        clock[0] = 50.
        node.scoring(clus, utils)
        node._poll_rl()
        assert not node.exploration_complete  # New failed cycles do not restart the deadline.
        clock[0] = 71.
        node._poll_rl()
        if node._rl_future is not None:
            node._rl_future.result(5)
            node._poll_rl()
        row = node.trainer.report.records[0]
        assert not row['success'] and row['steps'] == row['navigation_successes'] == 1
        assert row['cumulative_reward'] == pytest.approx(expected_reward)
        assert 'unreachable_timeout' in row['reason']
        assert node.trainer.agent.updates == 1 and node.trainer.agent.replay[-1][3]
        assert node.trainer.completed_episodes == node.trainer.agent.episodes == 0
        assert node.env.current_action is None and not node.requests
        assert not list(node.trainer.report.directory.glob('*.png'))
        # The public service still refuses to reset an aborted stack that could be moving.
        assert not node._reset_episode(None, NS()).success
        node.subscriber = object()
        node.destroy_subscription = lambda subscription: None
        node.create_subscription = lambda *args: object()
        assert node._reset_episode(None, NS(), simulation_stopped=True).success
        assert not node.episode_aborted and node._unreachable_since is None
        assert node.env.post_map is None and not node.env.awaiting_next_state
    finally:
        node.trainer.close()


def test_reachable_candidate_clears_planning_failure_deadline():
    """Recovery can resume ordinary selection before the watchdog expires."""
    node = orchestration_node()
    node._unreachable_since = 1.
    node.scoring(NS(estimation=lambda *args: None), Utils(0., 0., .5))
    assert node._unreachable_since is None and len(node.requests) == 1


def test_abort_invalidates_planner_callbacks(monkeypatch):
    """A late accepted goal from the old attempt cannot rebuild its queue."""
    import fronter_exploration.planner_client as module
    monkeypatch.setattr(module, 'ActionClient', Transport)
    node = orchestration_node()
    node.rl = True
    node._test_timers = []
    from builtin_interfaces.msg import Time
    node.get_clock = lambda: NS(now=lambda: NS(to_msg=lambda: Time()))
    node.create_timer = lambda period, callback, **kwargs: (
        node._test_timers.append(callback) or callback)
    node.destroy_timer = lambda timer: node._test_timers.remove(timer)
    node.planner = module.Planner(node)
    node.planner.cost_planner(NS(id=3, nearest_point=(0, 1)), Utils(0., 0., .5), None)
    node._abort_rl('Simulation failure')
    handle = Handle()
    node.planner.client.requests[-1][1].set_result(handle)
    assert handle.cancelled and not node._test_timers and not node.requests
    assert not node.planner.goal_futures and not node.planner.path
