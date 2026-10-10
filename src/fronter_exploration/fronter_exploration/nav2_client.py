from functools import partial

from action_msgs.msg import GoalStatus
from fronter_exploration.state import State
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.clock import Clock, ClockType


class Nav2Client:
    def __init__(self, node):
        self.node = node
        self.client = ActionClient(node, NavigateToPose, 'navigate_to_pose')
        self.generation = 0
        self.goal_handle = None
        self.timeout_timer = None
        self.timed_out = False

    def goal(self, x, y, utils):
        self.generation += 1
        generation = self.generation
        self.node.navigation_done = False
        self.node.navigation_success = False
        self.goal_handle = None
        self.timed_out = False
        goal_pos = utils.grid_to_world(x, y)
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.node.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = goal_pos[0]
        goal_msg.pose.pose.position.y = goal_pos[1]
        goal_msg.pose.pose.orientation.w = 1.0
        if not self.client.server_is_ready():
            self._finish(False, generation, 'Navigation server unavailable')
            return
        try:
            self.timeout_timer = self.node.create_timer(
                getattr(self.node, 'navigation_timeout', 300.),
                partial(self._timeout, generation),
                clock=Clock(clock_type=ClockType.STEADY_TIME)
                if getattr(self.node, 'rl', False) else self.node.get_clock())
            future = self.client.send_goal_async(goal_msg)
            future.add_done_callback(partial(self.goal_response_callback, generation))
        except Exception as error:
            self._error(generation, f'Navigation request failed: {error}')

    def goal_response_callback(self, generation, future):
        try:
            goal_handle = future.result()
            if generation != self.generation:
                if goal_handle.accepted:
                    goal_handle.cancel_goal_async()
                return
            if not goal_handle.accepted:
                self._finish(False, generation, 'Goal rejected')
                return
            self.goal_handle = goal_handle
            if self.timed_out:
                goal_handle.cancel_goal_async()
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(partial(self.result_callback, generation))
        except Exception as error:
            self._error(generation, f'Navigation response failed: {error}')

    def result_callback(self, generation, future):
        try:
            wrapped = future.result()
            success = wrapped.status == GoalStatus.STATUS_SUCCEEDED and not self.timed_out
            self._finish(success, generation,
                         'Goal reached' if success else f'Navigation status={wrapped.status}')
        except Exception as error:
            self._error(generation, f'Navigation result failed: {error}')

    def _error(self, generation, message):
        if generation != self.generation:
            return
        if not getattr(self.node, 'rl', False):
            self._finish(False, generation, message)
            return
        self.generation += 1
        if self.timeout_timer is not None:
            self.node.destroy_timer(self.timeout_timer)
            self.timeout_timer = None
        self.node._abort_rl(message)  # Transport failure cannot confirm motion stopped.

    def _timeout(self, generation):
        if generation != self.generation:
            return
        if getattr(self.node, 'rl', False):
            if self.timed_out:
                # A cancel request is not proof that physical navigation stopped.
                self._error(generation, 'Navigation cancellation has no terminal result')
                return
            self.timed_out = True
            self.node.destroy_timer(self.timeout_timer)
            self.timeout_timer = self.node.create_timer(
                10., partial(self._timeout, generation),
                clock=Clock(clock_type=ClockType.STEADY_TIME))
            if self.goal_handle is not None:
                try:
                    self.goal_handle.cancel_goal_async()
                except Exception as error:
                    self._error(generation, f'Navigation cancellation failed: {error}')
                    return
            self.node.get_logger().warn(
                'Navigation timed out; awaiting terminal cancellation result')
            return
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
        self._finish(False, generation, 'Navigation timed out; cancellation requested')

    def _finish(self, success, generation, message):
        if generation != self.generation:
            return
        self.generation += 1  # Ignore late results from a timed-out/cancelled goal.
        if self.timeout_timer is not None:
            self.node.destroy_timer(self.timeout_timer)
            self.timeout_timer = None
        self.node.get_logger().info(message)
        self.node.navigation_done = True
        self.node.navigation_success = success
        self.node.planner.path.clear()
        if getattr(self.node, 'rl', False):
            self.node.env.navigation_finished(success)
        self.node.state = State.IDLE
