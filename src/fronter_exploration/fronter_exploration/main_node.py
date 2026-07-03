import rclpy
import numpy as np
from rclpy.node import Node
from fronter_exploration.detector import detector
from fronter_exploration.cluster import cluster, FrontierCluster
from fronter_exploration.visualization import visulatization
from fronter_exploration.utils import utils
from fronter_exploration.nav2_client import nav2_client
from nav_msgs.msg import OccupancyGrid
from tf2_ros import Buffer, TransformListener,TransformException
from visualization_msgs.msg import MarkerArray


class main_node(Node):
    def __init__(self):
        super().__init__("main_node")
        self.map = None
        self.subscriber = self.create_subscription(OccupancyGrid, "map", self.mapCallback, 10)
        self.get_logger().info("Main node started")
        self.frontier_pub = self.create_publisher(OccupancyGrid, "frontier_map", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "frontier_markers", 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav = nav2_client(self)
        self.is_navigating = False
        self.goal_Failed = False

    def get_robot_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform("map","base_link",rclpy.time.Time())
            self.robot_pose = (transform.transform.translation.x,transform.transform.translation.y)
            self.get_logger().info(f"Robot pose: {self.robot_pose}")
        except TransformException as ex:
            self.get_logger().info(f"TF lookup failed: {ex}")

    def mapCallback(self, msg):
        self.map = msg
        self.grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.height = msg.info.height
        self.width = msg.info.width
        self.frontier_mask = np.zeros((self.height, self.width), dtype=np.int8)
        det = detector(self.height,self.width,self.grid,self.frontier_mask)
        self.frontier_mask = det.find_frontiers()
        clus = cluster(self.height,self.width,self.frontier_mask)
        self.frontier = clus.clustering()
        until = utils(self.map.info.origin.position.x,self.map.info.origin.position.y,self.map.info.resolution)
        visual = visulatization(msg,until)
        visual.frontier_msg(self.frontier_mask,self.frontier_pub)
        visual.frontier_visual(self.marker_pub,self.frontier,self.get_clock())
        self.get_robot_pose()
        clus.estimation(self.robot_pose,until)
        best_cluster = max(self.frontier.values(),key=lambda x:x.score)
        self.get_logger().debug(f"Best cluster: {best_cluster.id}, Color : {best_cluster.r},{best_cluster.g},{best_cluster.b}")
        if not self.is_navigating:
            self.is_navigating = True
            self.nav.goal(best_cluster.nearest_point[0],best_cluster.nearest_point[1],until)
        
        


def main(args=None):
    rclpy.init(args=args)
    node = main_node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
