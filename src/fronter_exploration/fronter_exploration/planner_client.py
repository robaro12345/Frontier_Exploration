from rclpy.action import ActionClient
from nav2_msgs.action import ComputePathToPose
from functools import partial
import math

class planner:
    def __init__(self,node):
        self.node = node
        self.client = ActionClient(node,ComputePathToPose,"compute_path_to_pose")
        self.path = {}

    def cost_planner(self,cluster,utils):
        goal = ComputePathToPose.Goal()
        goal.goal.header.frame_id = 'map'
        goal.goal.header.stamp = self.node.get_clock().now().to_msg()
        goal_pos = utils.grid_to_world(cluster.nearest_point[0],cluster.nearest_point[1])
        goal.goal.pose.position.x = goal_pos[0]
        goal.goal.pose.position.y = goal_pos[1]
        goal.goal.pose.position.z = 0.0
        goal.goal.pose.orientation.w = 1.0
        goal.use_start = False
        goal.planner_id = "GridBased"
        self.node.get_logger().info(
            f"Cluster {cluster.id}: "
            f"goal = ({goal.goal.pose.position.x:.3f}, "
            f"{goal.goal.pose.position.y:.3f})"
        )
        self.client.wait_for_server()
        future = self.client.send_goal_async(goal)
        future.add_done_callback(partial(self.goal_response_callback,cluster.id))
        

    def goal_response_callback(self, cluster_id, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.node.get_logger().info("Planner rejected goal")
            self.path[cluster_id] = 0
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(partial(self.result_callback,cluster_id))
    
    def result_callback(self,cluster_id, future):
        result = future.result().result
        path = result.path
        result = future.result().result
        self.path[cluster_id] = path
        if len(path.poses):
            end = path.poses[-1].pose.position
            self.node.get_logger().info(
                f"Returned end = ({end.x:.3f}, {end.y:.3f})"
            )

    def path_cost_calculator(self,cluster_id):
        path_cost = 0
        current_path = self.path[cluster_id].poses
        for i in range(len(current_path)-1):
            path_cost += math.sqrt((current_path[i+1].pose.position.x-current_path[i].pose.position.x)**2+(current_path[i+1].pose.position.y-current_path[i].pose.position.y)**2)
        self.node.get_logger().info(
            f"Cluster {cluster_id}: "
            f"{len(current_path)} poses, "
            f"path length = {path_cost}"
        )
        return path_cost