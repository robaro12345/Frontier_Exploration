from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point


class visulatization:
    def __init__(self,map_msg,utilt):
        self.msg = map_msg
        self.utilt = utilt

    def frontier_msg(self,frontier_mask,frontier_pub):
        frontier_msg = OccupancyGrid()
        frontier_msg.header = self.msg.header
        frontier_msg.info = self.msg.info
        frontier_msg.data = []
        for value in frontier_mask.flatten():
            if value == 1:
                frontier_msg.data.append(100)
            else:
                frontier_msg.data.append(0)
        frontier_pub.publish(frontier_msg)

    def marker_creator(self, marker_header, ns, marker_id, m_type, grid_point, z, s_x, s_y, s_z, c_r, c_g, c_b, c_a):
        marker = Marker()
        marker.header = marker_header
        marker.ns = ns
        marker.id = marker_id
        marker.type = m_type
        marker.action = Marker.ADD
        x, y = self.utilt.grid_to_world(grid_point[0], grid_point[1])
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.scale.x = s_x
        marker.scale.y = s_y
        marker.scale.z = s_z
        marker.color.r = c_r
        marker.color.g = c_g
        marker.color.b = c_b
        marker.color.a = c_a
        return marker

    def frontier_visual(self,marker_pub,frontier,stamp):
        marker_array = MarkerArray()
        for _,cluster in frontier.items():
            marker = Marker()
            marker.header.frame_id = self.msg.header.frame_id
            marker.header.stamp = stamp.now().to_msg()
            marker.ns = "frontier"
            marker.id = cluster.id
            marker.type = Marker.CUBE_LIST
            marker.action = Marker.ADD
            marker.scale.x = self.utilt.resolution
            marker.scale.y = self.utilt.resolution
            marker.scale.z = self.utilt.resolution
            marker.color.r = cluster.r
            marker.color.g = cluster.g
            marker.color.b = cluster.b
            marker.color.a = 1.0
            for row,col in cluster.cells:
                x,y = self.utilt.grid_to_world(row,col)
                p = Point()
                p.x = x
                p.y = y
                p.z = 0.05
                marker.points.append(p)
            marker_array.markers.append(marker)

            centroid_marker = self.marker_creator(
                marker.header, "Centroids", 1000 + cluster.id, Marker.SPHERE, 
                cluster.centroid, 0.1, 0.15, 0.15, 0.15, 1.0, 1.0, 1.0, 1.0
            )
            marker_array.markers.append(centroid_marker)
            
            nearest_marker = self.marker_creator(
                marker.header, "nearest", 100 + cluster.id, Marker.SPHERE, 
                cluster.nearest_point, 0.1, 0.2, 0.2, 0.2, 1.0, 0.5, 0.5, 1.0
            )
            marker_array.markers.append(nearest_marker)
        marker_pub.publish(marker_array)