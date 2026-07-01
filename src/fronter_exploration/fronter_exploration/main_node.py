import rclpy
import numpy as np
from rclpy.node import Node
from fronter_exploration.detector import detector
from fronter_exploration.cluster import cluster, FrontierCluster
from nav_msgs.msg import OccupancyGrid

class main_node(Node):
    def __init__(self):
        super().__init__("main_node")
        self.map = None
        self.subscriber = self.create_subscription(OccupancyGrid, "map", self.mapCallback, 10)
        self.get_logger().info("Main node started")
        self.frontier_pub = self.create_publisher(OccupancyGrid, "frontier_map", 10)
    
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
        frontier_msg = OccupancyGrid()
        frontier_msg.header = msg.header
        frontier_msg.info = msg.info
        frontier_msg.data = []
        for value in self.frontier_mask.flatten():
            if value == 1:
                frontier_msg.data.append(100)
            else:
                frontier_msg.data.append(0)
        self.frontier_pub.publish(frontier_msg)

        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock.now().to_msg()
        


def main(args=None):
    rclpy.init(args=args)
    node = main_node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
