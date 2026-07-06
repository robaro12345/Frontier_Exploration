class Detector:
    def __init__(self,height,width,grid,frontier_mask):
        self.height = height
        self.width = width
        self.grid = grid
        self.frontier_mask = frontier_mask
    
    def find_frontiers(self):
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i,j] == 0:
                    frontier_state = False
                    neighbor = [(0,1),(0,-1),(1,0),(-1,0)]
                    for di,dj in neighbor:
                        ni,nj = i+di ,j+dj
                        if (0<=ni<self.height and 0<=nj<self.width and self.grid[ni,nj] == -1):
                            frontier_state = True
                            break
                    self.frontier_mask[i,j] = 1 if frontier_state else 0
        return self.frontier_mask
        