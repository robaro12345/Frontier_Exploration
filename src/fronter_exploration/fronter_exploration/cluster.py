from collections import deque
import numpy as np
import math
MIN_CLUSTER_SIZE = 20

COLORS = [
    (1.0,0.0,0.0),      # Red
    (0.0,1.0,0.0),      # Green
    (0.0,0.0,1.0),      # Blue
    (1.0,1.0,0.0),      # Yellow
    (1.0,0.0,1.0),      # Magenta
    (0.0,1.0,1.0),      # Cyan
    (1.0,0.5,0.0),    # Orange
    (0.5,0.0,1.0),    # Violet
]

class FrontierCluster:
    def __init__(self,id,cells):
        self.id = id
        self.cells = cells
        self.centroid = (sum(r for r,c in cells)/len(cells),sum(c for r,c in cells)/len(cells))
        self.size = len(cells)
        self.r , self.g , self.b = COLORS[id%len(COLORS)]
        self.nearest_point = min(self.cells,key=lambda p:((p[0]-self.centroid[0])**2+(p[1]-self.centroid[1])**2) ) 
        self.path_cost = None
        self.information_gain = None
        self.score = None

class cluster:
    def __init__(self,height,width,frontier_mask):
        self.height = height
        self.width = width
        self.frontier_mask = frontier_mask
        self.visited_status = np.zeros((self.height, self.width), dtype=bool)
        self.frontier = {}
        self.cluster_id = 0

    def clustering(self):
        NEIGHBOR_8 = [(0,1),(0,-1),(-1,1),(1,1),(1,0),(-1,0),(-1,-1),(1,-1)]        
        for i in range(self.height):
            for j in range(self.width):
                if self.frontier_mask[i,j] == 0 or self.visited_status[i,j]:
                    continue
                else:
                    queue = deque([(i,j)])
                    self.visited_status[i,j] = True
                    cluster = []
                    while queue:
                        x,y = queue.popleft()
                        cluster.append((x,y))
                        for dx,dy in NEIGHBOR_8:
                            nx,ny = x+dx,y+dy
                            if 0<=nx<self.height and 0<=ny<self.width and self.frontier_mask[nx,ny] == 1 and not self.visited_status[nx,ny]:
                                self.visited_status[nx,ny] = True
                                queue.append((nx,ny))
                    if len(cluster) > MIN_CLUSTER_SIZE:
                        frontier_cluster = FrontierCluster(self.cluster_id,cluster)
                        self.frontier[self.cluster_id] = frontier_cluster
                        self.cluster_id += 1
        return self.frontier

    def estimation(self,robot_pos,until):
        grid_robot_pos = until.world_to_grid(robot_pos[0],robot_pos[1])
        for _,cluster in self.frontier.items():
            cluster.path_cost = math.sqrt((cluster.nearest_point[0]-grid_robot_pos[0])**2+(cluster.nearest_point[1]-grid_robot_pos[1])**2)
            cluster.information_gain = cluster.size
            cluster.score = cluster.information_gain/cluster.path_cost