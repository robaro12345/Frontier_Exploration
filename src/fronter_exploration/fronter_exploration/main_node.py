import rclpy
import numpy as np
from rclpy.node import Node
from fronter_exploration.detector import detector
from fronter_exploration.cluster import cluster, FrontierCluster
from fronter_exploration.visualization import visulatization
from fronter_exploration.utils import utils
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import MarkerArray


class main_node(Node):
    def __init__(self):
        super().__init__("main_node")
        self.map = None
        self.subscriber = self.create_subscription(OccupancyGrid, "map", self.mapCallback, 10)
        self.get_logger().info("Main node started")
        self.frontier_pub = self.create_publisher(OccupancyGrid, "frontier_map", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "frontier_markers", 10)

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

        


def main(args=None):
    rclpy.init(args=args)
    node = main_node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
