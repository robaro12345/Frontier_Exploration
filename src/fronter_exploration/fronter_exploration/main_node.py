import math
import time

import rclpy
import numpy as np
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from fronter_exploration.detector import Detector
from fronter_exploration.cluster import Cluster, FrontierCluster
from fronter_exploration.visualization import Visulatization
from fronter_exploration.utils import Utils
from fronter_exploration.nav2_client import Nav2Client
from fronter_exploration.planner_client import Planner
from fronter_exploration.RL.env import Env
from nav_msgs.msg import OccupancyGrid
from collections import deque
from visualization_msgs.msg import MarkerArray
from fronter_exploration.state import State

class main_node(Node):
    def __init__(self):
        super().__init__("main_node")
        self.map = None
        self.latest_map = None
        self.declare_parameter("rl", False)
        self.rl = self.get_parameter("rl").value
        self.declare_parameter("training", False)
        self.training = self.get_parameter("training").value
        self.subscriber = self.create_subscription(OccupancyGrid, "map", self.mapCallback, 10)
        self.get_logger().info("Main node started")
        self.frontier_pub = self.create_publisher(OccupancyGrid, "frontier_map", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "frontier_markers", 10)
        self.nav = Nav2Client(self)
        self.planner = Planner(self)
        self.request_queue = deque()
        self.current_cluster = None
        self.state = State.IDLE
        self.best_node = None
        self.navigation_done = False
        self.navigation_success = False
        self.env = Env(self)
        self.exploration_complete = False
        self.episode_aborted = False
        self._episode_started = False
        self._unreachable_since = None
        self._rl_future = None
        self.trainer = None
        if self.training and not self.rl:
            self.get_logger().warn('training is ignored when rl=False')
        if self.rl:
            # Keep PyTorch entirely optional for the original heuristic mode.
            from fronter_exploration.RL.training import PARAMETERS, Training
            settings = {}
            for name, default in PARAMETERS.items():
                settings[name] = self.declare_parameter('rl.' + name, default).value
            self.navigation_timeout = settings['navigation_timeout']
            self.planner_timeout = settings['planner_timeout']
            self.frontier_goal_min_distance = settings['frontier_goal_min_distance']
            self.unreachable_timeout = settings['unreachable_timeout']
            self.frontier_goal_attempts = settings['frontier_goal_attempts']
            if self.frontier_goal_attempts < 1:
                raise ValueError('Frontier goal attempts must be positive')
            if not math.isfinite(self.unreachable_timeout) or self.unreachable_timeout <= 0:
                raise ValueError('Unreachable frontier timeout must be finite and positive')
            if (not math.isfinite(self.frontier_goal_min_distance)
                    or self.frontier_goal_min_distance <= 0):
                raise ValueError('Frontier goal minimum distance must be finite and positive')
            if self.navigation_timeout <= 0 or self.planner_timeout <= 0:
                raise ValueError('Action timeouts must be positive')
            self.env = Env(self, settings['reward_gain'], settings['reward_distance'],
                           settings['reward_stagnation'], settings['reward_collision'],
                           settings['stagnation_threshold'], settings['collision_enabled'],
                           settings['map_max_age'], settings['map_settle_time'],
                           settings['map_wait_timeout'])
            if not settings['collision_enabled']:
                self.get_logger().info(
                    'Collision contribution disabled: no contact sensor connected')
            if self.training:
                from std_srvs.srv import Trigger
                self.trainer = Training(settings, self.get_logger())
                self.reset_service = self.create_service(
                    Trigger, '~/reset_episode', self._reset_episode)
            else:
                from fronter_exploration.RL.deploy import Deployment
                self.deployment = Deployment(settings['checkpoint'],
                                             settings['allow_untrained_checkpoint'])
                if self.deployment.agent.updates == 0:
                    self.get_logger().warn('TESTING OVERRIDE: deploying zero-update weights')
            self.rl_timer = self.create_timer(
                .05, self._poll_rl, clock=Clock(clock_type=ClockType.STEADY_TIME))

    def scoring(self,clus,utils):
        clus.estimation(self.planner,self)
        if not self.frontier:
            if self.rl and getattr(self, '_unreachable_since', None) is None:
                self._unreachable_since = time.monotonic()
            self.get_logger().warn('Planning found no reachable candidates; retrying on next map')
            self.state = State.IDLE
            self.planner.path.clear()
            return
        self._unreachable_since = None
        if not self.rl:
            self.best_node = max(self.frontier.values(),key=lambda x:x.score)
        else:
            self.rl_state = self.env.get_state()
            if not self._episode_started:
                self.env.reset()
                if self.training:
                    self.trainer.start_episode(self._report_info().get('known_fraction'))
                self._episode_started = True
            transition = self.env.complete_transition(self.rl_state)
            if self.training:
                self.state = State.SCORING
                self._rl_utils = utils
                self._rl_future = self.trainer.decide(
                    self.rl_state, transition, dict(self.env.last_info))
                return
            action = self.deployment.select_action(self.rl_state)
            try:
                self.best_node = self.env.step(action, self.rl_state)
            except (RuntimeError, ValueError) as error:
                self._abort_rl(str(error))
                return
        self._navigate(utils)

    def _navigate(self, utils):
        self.get_logger().info(f"Best cluster: {self.best_node.id}, Color : {self.best_node.r},{self.best_node.g},{self.best_node.b}")
        self.state = State.NAVIGATING
        self.nav.goal(self.best_node.nearest_point[0],self.best_node.nearest_point[1],utils)

    def _poll_rl(self):
        unreachable_since = getattr(self, '_unreachable_since', None)
        if (not self.exploration_complete and unreachable_since is not None
                and time.monotonic() - unreachable_since > self.unreachable_timeout):
            self._abort_rl('No reachable frontier paths before unreachable_timeout')
        if (not self.exploration_complete and self.env.wait_started is not None
                and time.monotonic() - self.env.wait_started > self.env.map_wait_timeout):
            self._abort_rl('No fresh post-navigation map before map_wait_timeout')
        if self._rl_future is None or not self._rl_future.done():
            return
        future, self._rl_future = self._rl_future, None
        try:
            action = future.result()
            if self.exploration_complete:
                self.state = State.IDLE
                if getattr(self, 'episode_aborted', False):
                    return
                self.get_logger().info(
                    'Training finished' if self.trainer.finished else
                    'Episode ended. Reset Gazebo and SLAM, then call ~/reset_episode')
                return
            self.best_node = self.env.step(action, self.rl_state)
            self._navigate(self._rl_utils)
        except Exception as error:
            self._abort_rl(str(error))

    def _abort_rl(self, reason):
        if getattr(self, 'episode_aborted', False):
            return
        self.episode_aborted = True
        self.exploration_complete = True
        self.state = State.IDLE
        self._discard_action_callbacks()
        self.get_logger().error(f'RL episode aborted (not completed): {reason}')
        if self.training and self.trainer is not None:
            transition = None
            if (self.env.current_action is not None and self.env.awaiting_next_state
                    and self.env.post_map is not None):
                transition = self.env.complete_transition({}, done=True)
            self._rl_future = self.trainer.abort_episode(
                reason, self._report_info().get('known_fraction'),
                transition=transition, info=dict(self.env.last_info))

    def _discard_action_callbacks(self):
        """Invalidate old action results before shutdown or a simulator restart."""
        for client in (self.nav, self.planner):
            if hasattr(client, 'generation'):
                client.generation += 1
            timer = getattr(client, 'timeout_timer', None)
            if timer is not None:
                self.destroy_timer(timer)
                client.timeout_timer = None
            for name in ('goal_futures', 'result_futures'):
                getattr(client, name, {}).clear()
        getattr(self, 'request_queue', deque()).clear()

    def _report_info(self):
        """Report observed map coverage independently of reward snapshot timing."""
        message = self.latest_map
        if message is None or not len(message.data):
            return {}
        return {'known_fraction': float(np.mean(np.asarray(message.data) != -1))}

    def _finish_exploration(self, utils, reason='No frontier clusters remain'):
        self.get_logger().info('All Done: ' + reason)
        self.exploration_complete = True
        if self.rl:
            self.episode_aborted = False
            transition = self.env.complete_transition({}, done=True)
            if self.training:
                self.state = State.SCORING
                info = dict(self.env.last_info, **self._report_info(), reason=reason)
                self._rl_future = self.trainer.decide(
                    {}, transition, info, terminal=True)
                return
        return_goal = utils.world_to_grid(0, 0)
        self.state = State.NAVIGATING
        self.nav.goal(return_goal[0], return_goal[1], utils)
        if not self.rl:
            # Preserve the existing heuristic termination behavior.
            raise SystemExit(0)

    def _reset_episode(self, request, response, *, simulation_stopped=False):
        if getattr(self, 'episode_aborted', False) and not simulation_stopped:
            response.success = False
            response.message = 'Episode aborted; restart the exploration node and simulation'
        elif self._rl_future is not None or not self.exploration_complete:
            response.success = False
            response.message = 'Wait for the current exploration episode to end'
        elif self.trainer.finished:
            response.success = False
            response.message = 'Training episode limit reached; restart with a checkpoint'
        else:
            self._discard_action_callbacks()
            # Discovery removal does not discard unread samples in the old subscription.
            self.destroy_subscription(self.subscriber)
            self.subscriber = self.create_subscription(
                OccupancyGrid, 'map', self.mapCallback, 10)
            self._episode_started = False
            self.episode_aborted = False
            self._unreachable_since = None
            self.exploration_complete = False
            self.env.episode_done = False
            self.env.last_map_stamp = -1  # New simulator may reset its clock.
            self.env.post_map = None
            self.env.wait_started = None
            self.env.current_action = None
            self.env.current_state = None
            self.env.awaiting_next_state = False
            self.latest_map = None
            self.state = State.IDLE
            self.planner.path.clear()
            response.success = True
            response.message = 'New episode starts on next map; simulator/SLAM reset is external'
        return response

    def send_to_queue(self,utils,clus):
        if not self.request_queue:
            self.scoring(clus,utils)
            return
        cluster = self.request_queue.popleft()
        self.current_cluster = cluster
        self.planner.cost_planner(self.current_cluster,utils,clus)

    def mapCallback(self, msg):
        if self.exploration_complete:
            return
        if self.rl and not self.env.observe_map(msg):
            return
        self.latest_map = msg
        if self.state==State.IDLE:
            if (self.rl and self.env.current_action is not None
                    and (not self.env.awaiting_next_state
                         or not self.env.capture_post_map(msg))):
                return
            if not self.planner.client.server_is_ready() or not self.nav.client.server_is_ready():
                return
            self.best_node = None
            self.map = msg
            self.grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))
            self.height = msg.info.height
            self.width = msg.info.width
            self.frontier_mask = np.zeros((self.height, self.width), dtype=np.int8)
            map_util = Utils(self.map.info.origin.position.x,self.map.info.origin.position.y,self.map.info.resolution)
            visual = Visulatization(msg,map_util)
            detector = Detector(self.height,self.width,self.grid,self.frontier_mask)
            cluster = Cluster(self.height,self.width,self.frontier_mask)
            self.frontier_mask = detector.find_frontiers()
            self.frontier = cluster.clustering()
            if not self.frontier:
                self._finish_exploration(map_util)
                return
            visual.frontier_msg(self.frontier_mask,self.frontier_pub)
            visual.frontier_visual(self.marker_pub,self.frontier,self.get_clock())
            self.state = State.PLANNING
            self.request_queue.clear()
            for clust in self.frontier.values():
                self.request_queue.append(clust)
            self.send_to_queue(map_util,cluster)

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = main_node()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            if node.trainer is not None:
                node.trainer.close(coverage=node._report_info().get('known_fraction'))
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
