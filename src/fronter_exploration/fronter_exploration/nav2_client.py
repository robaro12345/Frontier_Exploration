from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose

class nav2_client:
    def __init__(self,node):
        self.node = node
        self.client = ActionClient(node,NavigateToPose,"navigate_to_pose")
    
    def goal(self,x,y,utils):
        goal_pos = utils.grid_to_world(x,y)
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.node.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = goal_pos[0]
        goal_msg.pose.pose.position.y = goal_pos[1]
        goal_msg.pose.pose.position.z = 0.0
        goal_msg.pose.pose.orientation.w = 1.0
        self.client.wait_for_server()
        future = self.client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)
    
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.node.get_logger().warn("Goal rejected")
            self.node.is_navigating = False
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result().result
        self.node.get_logger().info("Goal reached")
        self.node.is_navigating = False
        self.node.planner.path = {}