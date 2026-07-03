class utils:
    def __init__(self,origin_x,origin_y,resolution):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.resolution = resolution

    def grid_to_world(self, row, col):
        return [self.origin_x + (col + 0.5) * self.resolution, self.origin_y + (row + 0.5) * self.resolution]

    def world_to_grid(self,x,y):
        return [(y-self.origin_y)/self.resolution - 0.5,(x-self.origin_x)/self.resolution - 0.5]
