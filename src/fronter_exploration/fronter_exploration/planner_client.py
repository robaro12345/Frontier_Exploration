from rclpy.action import ActionClient
from nav2_msgs.action import ComputePathToPose
from functools import partial
import math


class Planner:
    def __init__(self,node):
        self.node = node
        self.client = ActionClient(node,ComputePathToPose,"compute_path_to_pose")
        self.path = {}
        self.goal_futures = {}
        self.result_futures = {}

    def cost_planner(self,cluster,utils,clus):
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
        future = self.client.send_goal_async(goal)
        self.goal_futures[cluster.id] = future
        future.add_done_callback(partial(self.goal_response_callback,cluster.id,clus,utils))
        

    def goal_response_callback(self, cluster_id,clus,utils, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.node.get_logger().info("Planner rejected goal")
            self.path[cluster_id] = None
            return
        del self.goal_futures[cluster_id]
        result_future = goal_handle.get_result_async()
        self.result_futures[cluster_id] = result_future
        result_future.add_done_callback(partial(self.result_callback,cluster_id,clus,utils))
    
    def result_callback(self, cluster_id,clus,utils, future):
        wrapped = future.result()
        if wrapped.status != 4:
            self.path[cluster_id] = None
        else:
            self.path[cluster_id] = wrapped.result.path
        self.node.send_to_queue(utils,clus)

    def path_cost_calculator(self,cluster_id):
        path_cost = 0
        current_path = self.path[cluster_id].poses
        for i in range(len(current_path)-1):
            path_cost += math.sqrt((current_path[i+1].pose.position.x-current_path[i].pose.position.x)**2+(current_path[i+1].pose.position.y-current_path[i].pose.position.y)**2)
        return path_cost