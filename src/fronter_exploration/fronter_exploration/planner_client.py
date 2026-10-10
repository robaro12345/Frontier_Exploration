"""Estimate frontier paths asynchronously, with bounded RL goal alternatives."""

import math
from functools import partial

from action_msgs.msg import GoalStatus

from nav2_msgs.action import ComputePathToPose

from rclpy.action import ActionClient


class Planner:
    """Process one cluster at a time without blocking map or action callbacks."""

    def __init__(self, node):
        """Initialize the action transport, result cache and generation guards."""
        self.node = node
        self.client = ActionClient(node, ComputePathToPose, 'compute_path_to_pose')
        self.path = {}
        self.goal_futures = {}
        self.result_futures = {}
        self.generation = 0
        self.timeout_timer = None
        self.goal_handle = None
        self.goal_adjusted = False
        self.attempted_points = set()

    def cost_planner(self, cluster, utils, clus, adjusted=False, alternate=False):
        """Request a path to the current cluster target and track attempted cells."""
        if not alternate:
            self.attempted_points.clear()
        self.attempted_points.add(tuple(cluster.nearest_point))
        self.generation += 1
        generation = self.generation
        self.goal_handle = None
        self.goal_adjusted = adjusted
        goal = ComputePathToPose.Goal()
        goal.goal.header.frame_id = 'map'
        goal.goal.header.stamp = self.node.get_clock().now().to_msg()
        goal_pos = utils.grid_to_world(cluster.nearest_point[0], cluster.nearest_point[1])
        goal.goal.pose.position.x = goal_pos[0]
        goal.goal.pose.position.y = goal_pos[1]
        goal.goal.pose.orientation.w = 1.0
        goal.use_start = False
        goal.planner_id = 'GridBased'
        callback_args = (generation, cluster.id, clus, utils)
        try:
            self.timeout_timer = self.node.create_timer(
                getattr(self.node, 'planner_timeout', 30.),
                partial(self._timeout, *callback_args))
            future = self.client.send_goal_async(goal)
            self.goal_futures[cluster.id] = future
            future.add_done_callback(partial(self.goal_response_callback, *callback_args))
        except Exception as error:
            self.node.get_logger().warn(f'Planner request failed: {error}')
            self._finish(*callback_args, None)

    def goal_response_callback(self, generation, cluster_id, clus, utils, future):
        """Attach result handling only to the current accepted planning request."""
        try:
            goal_handle = future.result()
            if generation != self.generation:
                if goal_handle.accepted:
                    goal_handle.cancel_goal_async()
                return
            self.goal_futures.pop(cluster_id, None)
            if not goal_handle.accepted:
                self.node.get_logger().info('Planner rejected goal')
                self._finish(generation, cluster_id, clus, utils, None)
                return
            self.goal_handle = goal_handle
            result_future = goal_handle.get_result_async()
            self.result_futures[cluster_id] = result_future
            result_future.add_done_callback(partial(
                self.result_callback, generation, cluster_id, clus, utils))
        except Exception as error:
            self.node.get_logger().warn(f'Planner response failed: {error}')
            self._finish(generation, cluster_id, clus, utils, None)

    def result_callback(self, generation, cluster_id, clus, utils, future):
        """Handle successful paths and ordinary planning failures separately."""
        retry_candidates = False
        try:
            wrapped = future.result()
            path = (wrapped.result.path
                    if wrapped.status == GoalStatus.STATUS_SUCCEEDED else None)
            if path is not None and not path.poses:
                path = None
            retry_candidates = wrapped.status in (
                GoalStatus.STATUS_ABORTED, GoalStatus.STATUS_SUCCEEDED)
        except Exception as error:
            self.node.get_logger().warn(f'Planner result failed: {error}')
            path = None
        self._finish(generation, cluster_id, clus, utils, path,
                     retry_candidates=retry_candidates)

    def _timeout(self, generation, cluster_id, clus, utils):
        if generation != self.generation:
            return
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
        self.node.get_logger().warn(f'Planner timed out for frontier {cluster_id}')
        self._finish(generation, cluster_id, clus, utils, None)

    def _finish(self, generation, cluster_id, clus, utils, path, retry_candidates=False):
        if generation != self.generation:
            return
        self.generation += 1
        if self.timeout_timer is not None:
            self.node.destroy_timer(self.timeout_timer)
            self.timeout_timer = None
        self.goal_futures.pop(cluster_id, None)
        self.result_futures.pop(cluster_id, None)
        if path is not None and getattr(self.node, 'rl', False):
            minimum = getattr(self.node, 'frontier_goal_min_distance', .5)
            start = path.poses[0].pose.position
            cluster = clus.frontier[cluster_id]
            target = utils.grid_to_world(*cluster.nearest_point)
            endpoint = path.poses[-1].pose.position
            if (math.hypot(target[0] - start.x, target[1] - start.y) < minimum
                    or math.hypot(endpoint.x - start.x, endpoint.y - start.y) < minimum):
                candidates = [cell for cell in cluster.cells
                              if math.dist(utils.grid_to_world(*cell),
                                           (start.x, start.y)) >= minimum
                              and tuple(cell) not in self.attempted_points]
                if (candidates and not self.goal_adjusted and len(self.attempted_points)
                        < getattr(self.node, 'frontier_goal_attempts', 4)):
                    cluster.nearest_point = min(
                        candidates, key=lambda cell: math.dist(cell, cluster.centroid))
                    target = utils.grid_to_world(*cluster.nearest_point)
                    self.node.get_logger().info(
                        f'Frontier {cluster_id} goal too close; replanning to '
                        f'({target[0]:.2f}, {target[1]:.2f})')
                    self.cost_planner(cluster, utils, clus, adjusted=True, alternate=True)
                    return
                self.node.get_logger().warn(
                    f'Frontier {cluster_id} has no usable goal beyond '
                    f'{minimum:.2f} m; retrying on a later map')
                path = None
                retry_candidates = False  # Preserve the bounded near-goal adjustment.
        if (path is None and retry_candidates and getattr(self.node, 'rl', False)
                and clus is not None):
            cluster = clus.frontier[cluster_id]
            candidates = [cell for cell in cluster.cells
                          if tuple(cell) not in self.attempted_points]
            maximum = getattr(self.node, 'frontier_goal_attempts', 4)
            if candidates and len(self.attempted_points) < maximum:
                # Spread trials across the cluster instead of retrying adjacent cells.
                cluster.nearest_point = max(candidates, key=lambda cell: min(
                    math.dist(cell, tried) for tried in self.attempted_points))
                target = utils.grid_to_world(*cluster.nearest_point)
                self.node.get_logger().info(
                    f'Frontier {cluster_id} path failed; trying alternative '
                    f'{len(self.attempted_points) + 1}/{maximum} at '
                    f'({target[0]:.2f}, {target[1]:.2f})')
                self.cost_planner(cluster, utils, clus, alternate=True)
                return
        self.path[cluster_id] = path
        self.node.send_to_queue(utils, clus)

    def path_cost_calculator(self, cluster_id):
        """Sum segment lengths of the accepted path in metres."""
        path_cost = 0
        current_path = self.path[cluster_id].poses
        for i in range(len(current_path) - 1):
            path_cost += math.hypot(
                current_path[i + 1].pose.position.x - current_path[i].pose.position.x,
                current_path[i + 1].pose.position.y - current_path[i].pose.position.y)
        return path_cost
