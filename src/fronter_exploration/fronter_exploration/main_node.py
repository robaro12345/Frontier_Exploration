import rclpy
import numpy as np
from rclpy.node import Node
from fronter_exploration.detector import Detector
from fronter_exploration.cluster import Cluster, FrontierCluster
from fronter_exploration.visualization import Visulatization
from fronter_exploration.utils import Utils
from fronter_exploration.nav2_client import Nav2Client
from fronter_exploration.planner_client import Planner
from nav_msgs.msg import OccupancyGrid
from collections import deque
from visualization_msgs.msg import MarkerArray
from fronter_exploration.state import State

class main_node(Node):
    def __init__(self):
        super().__init__("main_node")
        self.map = None
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

    def scoring(self,clus,utils):
        clus.estimation(self.planner,self)
        self.best_node = max(self.frontier.values(),key=lambda x:x.score)
        self.get_logger().info(f"Best cluster: {self.best_node.id}, Color : {self.best_node.r},{self.best_node.g},{self.best_node.b}")
        self.state = State.NAVIGATING
        self.nav.goal(self.best_node.nearest_point[0],self.best_node.nearest_point[1],utils)

    def send_to_queue(self,utils,clus):
        if not self.request_queue:
            self.scoring(clus,utils)
            return
        cluster = self.request_queue.popleft()
        self.current_cluster = cluster
        self.planner.cost_planner(self.current_cluster,utils,clus)

    def mapCallback(self, msg):
        if self.state==State.IDLE:
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
            if self.frontier == {}:
                self.get_logger().info("All Done:")
                return_goal = map_util.world_to_grid(0,0)
                self.nav.goal(return_goal[0],return_goal[1],map_util)
                exit(0)
            visual.frontier_msg(self.frontier_mask,self.frontier_pub)
            visual.frontier_visual(self.marker_pub,self.frontier,self.get_clock())
            self.state = State.PLANNING
            self.request_queue.clear()
            for clust in self.frontier.values():
                self.request_queue.append(clust)
            self.send_to_queue(map_util,cluster)
            
        
        


def main(args=None):
    rclpy.init(args=args)
    node = main_node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
