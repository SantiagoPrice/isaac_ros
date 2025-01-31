#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion, PoseStamped
from sensor_msgs_py import point_cloud2
import numpy as np
import cv2
from builtin_interfaces.msg import Duration
from rclpy.duration import Duration as RclpyDuration  # For timeouts and durations
import tf2_ros
import tf_transformations
from scipy.spatial import cKDTree
import time
import tf2_geometry_msgs  # Necessary for transforming PoseStamped messages
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose_stamped

# import sys
# Example path: adjust for your ROS version / Python version / install paths
# sys.path.append('/opt/ros/humble/lib/python3.10/site-packages')
# import tf2_sensor_msgs.tf2_sensor_msgs as tf2_sensor

class PlaneCloudProcessorNode(Node):
    def __init__(self):
        super().__init__('plane_cloud_processor_node')
        
        # Publishers
        self.polygon_pub = self.create_publisher(Marker, 'table_polygon', 1)
        self.padded_line_marker_pub = self.create_publisher(Marker, '/padded_line', 1)
        self.placements_marker_pub = self.create_publisher(MarkerArray, '/possible_placements', 1)
        self.filtered_placements_marker_pub = self.create_publisher(MarkerArray, '/filtered_placements', 1)
        self.best_point_marker_pub = self.create_publisher(Marker, "/best_point_marker", 1)
        self.best_point_pub = self.create_publisher(Point, "/best_point_topic", 1)

        # Subscribers
        self.create_subscription(PointCloud2, 'plane_cloud', self.plane_cloud_callback, 1)
        self.create_subscription(PointCloud2, '/lidar_filtered', self.lidar_filtered_callback, 1)

        # Scene data for collision checking
        self.latest_lidar_cloud = None
        self.scene_tree = None

        self.transformed_pose_previous = None
        self.transform_hesai_to_map = None
        self.last_transform_time = None
        self.transform_update_interval = RclpyDuration(seconds=1.0)

        # Trackers for previous markers
        self.previous_marker_count = 0
        self.previous_locations_count = 0
        self.previous_filtered_marker_count = 0  # For filtered locations

        # TF Buffer and Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Example wheelchair dimensions (adjust these as needed)
        self.wheelchair_dimensions = [0.7, 0.7, 0.8]

        self.marker_base = 'base_link'

    # def lidar_filtered_callback(self, msg: PointCloud2):
    #     num_points = msg.width * msg.height
    #     if num_points == 0:
    #         self.get_logger().warn("No points in '/lidar_filtered'")
    #         return

    #     # Convert raw data to float32 array
    #     raw_array = np.frombuffer(msg.data, dtype=np.float32, count=num_points*4)
    #     # Reshape to Nx4: [x, y, z, rgb_as_float]
    #     points_array = raw_array.reshape(num_points, 4)

    #     # Extract xyz
    #     xyz = points_array[:, :3]

    #     # Extract the rgb float and reinterpret as uint32 for bit manipulation
    #     rgb_float = points_array[:, 3]
    #     rgb_uint32 = rgb_float.view(np.uint32)

    #     # Extract individual channels (r,g,b)
    #     r = (rgb_uint32 >> 16) & 0xFF
    #     g = (rgb_uint32 >> 8) & 0xFF
    #     b = rgb_uint32 & 0xFF

    #     # Now xyz is an Nx3 float32 array of coordinates
    #     # and (r, g, b) are Nx1 arrays of uint8 channel values
    #     self.latest_lidar_cloud = xyz

    #     valid_mask = np.isfinite(xyz).all(axis=1)
    #     xyz = xyz[valid_mask]

    #     # If needed, build KD-tree
    #     if xyz.size > 0:
    #         self.scene_tree = cKDTree(xyz)
    #     else:
    #         self.scene_tree = None

    def lidar_filtered_callback(self, msg: PointCloud2):
        """
        Manually transform the incoming '/lidar_filtered' cloud from
        msg.header.frame_id -> 'map' without tf2_sensor_msgs.
        Then build a KD-tree in the map frame.
        """

        # 1) Basic checks
        num_points = msg.width * msg.height
        if num_points == 0:
            self.get_logger().warn("No points in '/lidar_filtered'")
            return

        # 2) Extract raw (x,y,z,rgb) from msg
        raw_array = np.frombuffer(msg.data, dtype=np.float32, count=num_points * 4)
        points_array = raw_array.reshape(num_points, 4)  # shape is (N, 4)
        xyz = points_array[:, :3]                       # Extract just x,y,z
        rgb_float = points_array[:, 3]                  # If you need colors

        # 3) Filter out NaNs or infinities
        valid_mask = np.isfinite(xyz).all(axis=1)
        xyz = xyz[valid_mask]  # shape (M, 3)

        if xyz.size == 0:
            self.get_logger().warn("All points are invalid in '/lidar_filtered'")
            return

        # 4) Lookup transform from cloud's frame to 'map'
        #    Use the same timestamp as the incoming cloud if you have time-synced TF.
        #    Otherwise, rclpy.time.Time() = 'now' can be used as fallback.
        try:
            transform_stamped = self.tf_buffer.lookup_transform(
                "map", 
                msg.header.frame_id, 
                rclpy.time.Time()
            )
        except Exception as ex:
            self.get_logger().warn(f"Could not lookup transform {msg.header.frame_id} -> 'map': {ex}")
            return

        # 5) Convert the geometry_msgs transform -> 4x4 matrix
        t = transform_stamped.transform.translation
        r = transform_stamped.transform.rotation
        transform_matrix = tf_transformations.quaternion_matrix([r.x, r.y, r.z, r.w])
        transform_matrix[0, 3] = t.x
        transform_matrix[1, 3] = t.y
        transform_matrix[2, 3] = t.z

        # 6) Manually apply the transform to each point
        ones_col = np.ones((xyz.shape[0], 1), dtype=np.float32)
        xyz_hom = np.hstack((xyz, ones_col))  # shape (M, 4)
        xyz_in_map_hom = (transform_matrix @ xyz_hom.T).T  # shape (M, 4)
        xyz_in_map = xyz_in_map_hom[:, :3]                 # drop the homogeneous coord

        # 7) Build or update the KD-tree in the 'map' frame
        if xyz_in_map.size > 0:
            self.scene_tree = cKDTree(xyz_in_map)
            self.latest_lidar_cloud = xyz_in_map
            self.get_logger().info(f"Updated scene_tree with {xyz_in_map.shape[0]} points in 'map' frame.")
        else:
            self.scene_tree = None
            self.latest_lidar_cloud = None
            self.get_logger().warn("No valid points remain after transform/filter in 'map' frame.")

    def plane_cloud_callback(self, plane_cloud_msg):
        self.get_logger().info("Received plane_cloud")
        plane_cloud = self.convert_cloud_to_array(plane_cloud_msg)

        if plane_cloud is None or len(plane_cloud) == 0:
            self.get_logger().warn("Empty plane cloud.")
            return

        self.table_center = np.mean(plane_cloud, axis=0)

        polygon_3d = self.findMinimumAreaShape(plane_cloud)
        self.get_logger().info(f"Detected polygon with {len(polygon_3d)} vertices.")

        # Publish the polygon
        self.publish_polygon_marker(polygon_3d, plane_cloud_msg.header)

        # Compute and publish padded polygon lines (as separate edges)
        padding_offset = 0.9
        padded_edges = self.calculate_padded_edges(polygon_3d, padding_offset)
        self.publish_padded_lines(padded_edges, plane_cloud_msg.header)

        # Find possible placements (sampling along the padded edges)
        location_sampling_distance = 0.1
        possible_locations = self.find_possible_placements(padded_edges, location_sampling_distance)

        # Filter for collisions if scene_tree is available
        if self.scene_tree is not None and self.scene_tree.data.size > 0:
            filtered_locations = self.filter_for_collision(possible_locations, self.scene_tree, self.wheelchair_dimensions)
        else:
            self.get_logger().warn("No scene data available for collision checking. Using raw possible locations.")
            filtered_locations = possible_locations

        best_point, risk_score = self.select_least_collision_risk(filtered_locations)
                        
        if best_point is not None:
            # Publish the best point (also uses padded edges for orientation)
            self.publish_best_point(best_point, padded_edges)
        else:
            self.get_logger().warn("No valid placement point found.")

        # Publish placements before filtering
        self.publish_possible_placements(possible_locations)
        # Now publish the filtered locations
        self.publish_filtered_locations(filtered_locations)

    def convert_cloud_to_array(self, cloud_msg):
        points_gen = point_cloud2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
        points_list = np.array([[p[0], p[1], p[2]] for p in points_gen], dtype=np.float32)
        return points_list

    def findMinimumAreaShape(self, cloud):
        points_2d = cloud[:, :2].astype(np.float32)
        avg_height = np.mean(cloud[:, 2])
        hull_points = cv2.convexHull(points_2d)
        epsilon = 0.05 * cv2.arcLength(hull_points, True)
        approx_polygon = cv2.approxPolyDP(hull_points, epsilon, True)
        polygon_3d = [np.array([pt[0][0], pt[0][1], avg_height]) for pt in approx_polygon]
        return polygon_3d

    def publish_polygon_marker(self, polygon_3d, header):
        marker = Marker()
        marker.header = header
        marker.ns = 'table_polygon'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        for point in polygon_3d:
            p = Point(x=float(point[0]), y=float(point[1]), z=float(point[2]))
            marker.points.append(p)

        # Close the loop visually
        if polygon_3d:
            first_pt = Point(x=float(polygon_3d[0][0]), 
                             y=float(polygon_3d[0][1]), 
                             z=float(polygon_3d[0][2]))
            marker.points.append(first_pt)

        marker.lifetime = Duration(sec=0, nanosec=0)
        self.polygon_pub.publish(marker)

    ############################################################################
    # 1) Calculate padded edges as separate line segments
    ############################################################################
    def calculate_padded_edges(self, polygon, padding_offset):
        """
        Returns a list of edges, each edge is [ [x1, y1, z1], [x2, y2, z2] ].
        """
        # Copy + close loop
        polygon_copy = polygon[:]
        polygon_copy.append(polygon_copy[0])

        # 2D center for deciding which side is 'outward'
        contour_2d = [np.array([pt[0], pt[1]], dtype=np.float32) for pt in polygon_copy[:-1]]
        contour_center = np.mean(contour_2d, axis=0)

        padded_edges = []

        for i in range(len(polygon_copy) - 1):
            p1_2d = np.array([polygon_copy[i][0], polygon_copy[i][1]], dtype=np.float32)
            p2_2d = np.array([polygon_copy[i+1][0], polygon_copy[i+1][1]], dtype=np.float32)
            z_val = polygon_copy[i][2]

            line_vector = p2_2d - p1_2d
            line_length = np.linalg.norm(line_vector)
            if line_length == 0:
                continue

            line_dir = line_vector / line_length
            perp_dir = np.array([-line_dir[1], line_dir[0]])

            # 4 "shifted" corners
            p1_plus  = p1_2d + padding_offset * perp_dir
            p1_minus = p1_2d - padding_offset * perp_dir
            p2_plus  = p2_2d + padding_offset * perp_dir
            p2_minus = p2_2d - padding_offset * perp_dir

            # Decide which side is outward by measuring distance to center
            mid_plus  = (p1_plus + p2_plus)/2.0
            mid_minus = (p1_minus + p2_minus)/2.0
            dist_plus  = np.linalg.norm(mid_plus  - contour_center)
            dist_minus = np.linalg.norm(mid_minus - contour_center)

            if dist_plus > dist_minus:
                # "Plus" side is outward
                padded_edges.append([
                    [p1_plus[0], p1_plus[1], z_val],
                    [p2_plus[0], p2_plus[1], z_val]
                ])
            else:
                # "Minus" side is outward
                padded_edges.append([
                    [p1_minus[0], p1_minus[1], z_val],
                    [p2_minus[0], p2_minus[1], z_val]
                ])

        return padded_edges

    ############################################################################
    # 2) Publish each edge separately using Marker.LINE_LIST
    ############################################################################
    def publish_padded_lines(self, padded_edges, header):
        """
        Publish each edge as a separate line segment (Marker.LINE_LIST).
        Each pair of points appended to marker.points forms one segment.
        """
        marker = Marker()
        marker.header = header
        marker.ns = 'padded_polygon'
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        for edge in padded_edges:
            if len(edge) != 2:
                continue
            pt1, pt2 = edge[0], edge[1]
            marker.points.append(Point(x=float(pt1[0]), y=float(pt1[1]), z=float(pt1[2])))
            marker.points.append(Point(x=float(pt2[0]), y=float(pt2[1]), z=float(pt2[2])))

        marker.lifetime = Duration(sec=0, nanosec=0)
        self.padded_line_marker_pub.publish(marker)

    ############################################################################
    # 3) Sampling each edge to find possible placements
    ############################################################################
    def find_possible_placements(self, padded_edges, location_sampling_distance):
        """
        We iterate over each edge [ptA, ptB]. For each edge, we skip
        `skip_distance` from both ends, then sample points along the remaining segment.
        """
        skip_distance = 0.35  # <-- Distance to skip from each end, half of the wheelchair width
        collision_points = []

        for edge in padded_edges:
            if len(edge) != 2:
                # Not a valid edge; just append empty and continue
                collision_points.append([])
                continue

            p1 = np.array(edge[0])
            p2 = np.array(edge[1])
            line_vec = p2 - p1
            line_len = np.linalg.norm(line_vec)

            if line_len == 0:
                # Degenerate edge
                collision_points.append([])
                continue

            # If the entire edge is shorter than 2x skip_distance, there is no valid region
            if line_len <= 2 * skip_distance:
                collision_points.append([])
                continue

            # Normalize direction
            line_dir = line_vec / line_len

            # Calculate how far along the line we can start/end
            # (e.g., skip first 0.2 m and last 0.2 m)
            start_ratio = skip_distance / line_len            # fraction along the edge
            end_ratio   = 1.0 - (skip_distance / line_len)    # fraction from the other end

            # Number of samples along that "middle" portion
            # We base this on the length of the valid portion
            valid_length = (end_ratio - start_ratio) * line_len
            num_samples = int(valid_length / location_sampling_distance)
            if num_samples < 1:
                # If there's not enough length to sample even once, skip
                collision_points.append([])
                continue

            # Sample from start_ratio to end_ratio
            points_on_line = []
            for j in range(num_samples + 1):
                ratio = start_ratio + j * (end_ratio - start_ratio) / num_samples
                sampled_pt = p1 + ratio * line_vec
                points_on_line.append(sampled_pt.tolist())

            collision_points.append(points_on_line)
        
        return collision_points


    def transform_to_matrix(self, transform):
        translation = [transform.transform.translation.x,
                       transform.transform.translation.y,
                       transform.transform.translation.z]
        rotation = [transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w]
        matrix = tf_transformations.quaternion_matrix(rotation)
        matrix[:3, 3] = translation
        return matrix

    def transform_point(self, point, transform):
        point_hom = np.array([point[0], point[1], point[2], 1.0])
        transform_matrix = self.transform_to_matrix(transform)
        transformed_point_hom = transform_matrix @ point_hom
        return transformed_point_hom[:3]

    def publish_possible_placements(self, placements):
        marker_array = MarkerArray()
        marker_id = 0

        # 'placements' is list-of-lists (clusters). Transform + publish.
        for cluster in placements:
            for point in cluster:
                transformed_point = point #self.transform_point(point, transform)

                marker = Marker()
                marker.header.frame_id = 'map'
                marker.ns = 'possible_placements'
                marker.id = marker_id
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.scale.x = 0.05
                marker.scale.y = 0.05
                marker.scale.z = 0.05

                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0

                marker.pose.position.x = transformed_point[0]
                marker.pose.position.y = transformed_point[1]
                marker.pose.position.z = transformed_point[2]

                marker.pose.orientation.w = 1.0

                marker_array.markers.append(marker)
                marker_id += 1

        # Remove old markers if any
        for old_marker_id in range(marker_id, self.previous_marker_count):
            delete_marker = Marker()
            delete_marker.header.frame_id = 'map'
            delete_marker.ns = 'possible_placements'
            delete_marker.id = old_marker_id
            delete_marker.action = Marker.DELETE
            marker_array.markers.append(delete_marker)

        self.previous_marker_count = marker_id
        self.placements_marker_pub.publish(marker_array)

    def publish_filtered_locations(self, placements):
        # Similar to publish_possible_placements but uses 'filtered_placements' namespace
        timeout = rclpy.duration.Duration(seconds=1.0)
        if not self.tf_buffer.can_transform('map', 'hesai_lidar', rclpy.time.Time(), timeout):
            self.get_logger().warn("No transform available after waiting 1 second.")
            return

        transform = self.tf_buffer.lookup_transform('map', 'hesai_lidar', rclpy.time.Time())
        marker_array = MarkerArray()
        marker_id = 0

        for cluster in placements:
            for point in cluster:
                transformed_point = point #self.transform_point(point, transform)

                marker = Marker()
                marker.header.frame_id = 'map'
                marker.ns = 'filtered_placements'
                marker.id = marker_id
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.scale.x = 0.15
                marker.scale.y = 0.15
                marker.scale.z = 0.15

                # Use a different color for filtered placements (e.g. green)
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0

                marker.pose.position.x = transformed_point[0]
                marker.pose.position.y = transformed_point[1]
                marker.pose.position.z = transformed_point[2]

                marker.pose.orientation.w = 1.0

                marker_array.markers.append(marker)
                marker_id += 1

        # Remove old markers if any
        for old_marker_id in range(marker_id, self.previous_filtered_marker_count):
            delete_marker = Marker()
            delete_marker.header.frame_id = 'map'
            delete_marker.ns = 'filtered_placements'
            delete_marker.id = old_marker_id
            delete_marker.action = Marker.DELETE
            marker_array.markers.append(delete_marker)

        self.previous_filtered_marker_count = marker_id
        self.filtered_placements_marker_pub.publish(marker_array)

    def filter_for_collision(self, locations, scene_tree, wheelchair_dimensions):
        """
        Filters out locations that would result in collisions with the environment.

        :param locations: List of possible placement locations (clusters of points)
        :param scene_tree: KD-Tree of the scene cloud points
        :param wheelchair_dimensions: [width, length, depth] (unused in simple approach)
        :return: List of non-colliding locations (in same list-of-lists structure).
        """
        desirable_locations = []
        collision_radius = 0.8  # Example radius for collision check

        self.get_logger().info(f"Filtering collision with wheelchair dimensions: {wheelchair_dimensions}")

        for ii, cluster in enumerate(locations):
            cluster_array = np.array(cluster)
            non_colliding_points = []
            for point in cluster_array:
                indices = scene_tree.query_ball_point(point[:3], collision_radius)
                if not indices:
                    non_colliding_points.append(point.tolist())

            self.get_logger().info(f"Cluster {ii}: Found {len(non_colliding_points)} non-colliding points.")
            desirable_locations.append(non_colliding_points)

        total_non_colliding_points = sum(len(cluster) for cluster in desirable_locations)
        self.get_logger().info(f"Total non-colliding points: {total_non_colliding_points}")
        return desirable_locations

    def select_least_collision_risk(self, desirable_locations):
        """
        Selects the location with the "lowest cost" (some heuristic).
        Currently, we measure cost as sqrt(x^2 + y^2) but adapt as needed.
        Returns (best_point, best_cost).
        """
        if not desirable_locations or not any(len(cluster) for cluster in desirable_locations):
            self.get_logger().warn("No desirable locations available.")
            return None, float('inf')  # No locations available

        best_point = None
        best_cost = float('inf')

        for cluster in desirable_locations:
            for point in cluster:
                point_array = np.array(point)
                # Simple cost: distance from origin, or from (0,0).
                cost = np.hypot(point_array[0], point_array[1])
                self.get_logger().info(f"-------------- cost: {cost}")

                if cost < best_cost:
                    best_cost = cost
                    best_point = point_array.tolist()

        if best_point:
            self.get_logger().info(f"Selected best point: {best_point} with cost: {best_cost}")
        else:
            self.get_logger().warn("No valid location found with a calculated cost.")

        return best_point, best_cost

    def update_transform(self):
        current_time = self.get_clock().now()
        if (self.last_transform_time is None or
            current_time - self.last_transform_time > self.transform_update_interval):
            try:
                self.transform_hesai_to_map = self.tf_buffer.lookup_transform(
                    'map',
                    'hesai_lidar',
                    rclpy.time.Time()
                )
                self.last_transform_time = current_time
            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.TransformException) as ex:
                self.get_logger().warn(f"Transform error: {ex}")
                self.transform_hesai_to_map = None

    ############################################################################
    # 4) Publish best point using the new edges
    ############################################################################
    def publish_best_point(self, best_point, padded_edges, marker_topic="/best_point_marker"):
        """
        Publishes the best point and orientation perpendicular to the nearest edge.
        Because padded_edges is a list of edges, we flatten it to find a center,
        then find the nearest edge for orientation.
        """
        start_time = time.time()
        if best_point is None:
            self.get_logger().warn("No best point to publish.")
            return

        # Flatten all points for a "center" calculation
        all_points = [pt for edge in padded_edges for pt in edge]
        if not all_points:
            self.get_logger().warn("Padded edges are empty; cannot publish best point.")
            return

        # Compute 2D center of all padded points
        contour = [np.array([pt[0], pt[1]], dtype=np.float32) for pt in all_points]
        table_center = np.mean(contour, axis=0)

        # Direction from center to best_point
        direction_vector = table_center - best_point[:2]
        norm = np.linalg.norm(direction_vector)
        if norm > 1e-9:
            direction_vector /= norm

        # Find the nearest edge to compute the perpendicular orientation
        closest_distance = float('inf')
        perpendicular_vector = None
        best_pt_2d = np.array(best_point[:2])

        # Each edge is exactly two points: [ptA, ptB]
        for edge in padded_edges:
            if len(edge) != 2:
                continue
            seg_start = np.array(edge[0][:2])
            seg_end   = np.array(edge[1][:2])

            line_vec = seg_end - seg_start
            line_len = np.linalg.norm(line_vec)
            if line_len < 1e-9:
                continue
            line_dir = line_vec / line_len

            # Projection of (best_pt - seg_start) onto line_dir
            proj_length = np.dot(best_pt_2d - seg_start, line_dir)
            proj_point_2d = seg_start + proj_length * line_dir

            dist = np.linalg.norm(best_pt_2d - proj_point_2d)
            if dist < closest_distance:
                closest_distance = dist
                # Perpendicular to the line
                perpendicular_vector = np.array([-line_dir[1], line_dir[0], 0.0])

        if perpendicular_vector is None:
            self.get_logger().warn("No valid perpendicular vector found.")
            return

        # Orientation from perpendicular vector
        yaw_angle = np.arctan2(perpendicular_vector[1], perpendicular_vector[0])
        quaternion = tf_transformations.quaternion_from_euler(0, 0, yaw_angle)

        # Create a PoseStamped in 'hesai_lidar' frame
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = 'hesai_lidar'
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose.position = Point(
            x=best_point[0],
            y=best_point[1],
            z=best_point[2]
        )
        pose_stamped.pose.orientation = Quaternion(
            x=quaternion[0],
            y=quaternion[1],
            z=quaternion[2],
            w=quaternion[3]
        )

        # Transform to 'base_link' or 'map', depending on your usage
        self.update_transform()
        if self.transform_hesai_to_map is None:
            self.get_logger().warn("Transform not available, cannot publish best point.")
            return

        transformed_pose = pose_stamped #do_transform_pose_stamped(pose_stamped, self.transform_hesai_to_map)
        if transformed_pose:
            # Potential logic to hold previous pose if table is too far, etc.
            # Just store it or override, as in your original code
            self.transformed_pose_previous = transformed_pose
            self.get_logger().info(f"Selected best transformed_pose: {transformed_pose}")
        else:
            self.get_logger().warn("Could not transform best pose.")
            return

        # Publish arrow marker
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "best_point_vector"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose = transformed_pose.pose
        marker.scale.x = 0.3  # Length of arrow
        marker.scale.y = 0.06
        marker.scale.z = 0.06
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.best_point_marker_pub.publish(marker)

        # Publish point on /best_point_topic
        best_point_msg = transformed_pose.pose.position
        self.best_point_pub.publish(best_point_msg)

        end_time = time.time()
        duration = end_time - start_time
        self.get_logger().info(
            f"Published best point in 'map' frame: ({best_point_msg.x}, {best_point_msg.y}, {best_point_msg.z}) with orientation."
        )

def main(args=None):
    rclpy.init(args=args)
    node = PlaneCloudProcessorNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
