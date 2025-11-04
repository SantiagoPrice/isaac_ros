# Revised code for RAL25 demo

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
from rclpy.duration import Duration as RclpyDuration
import tf2_ros
import tf_transformations
from scipy.spatial import cKDTree
import time
import tf2_geometry_msgs
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose_stamped
from tf2_ros import LookupException, ExtrapolationException, TransformException
import math
from rcl_interfaces.srv import GetParameters
# <-- Import your custom service interface
from multi_goal_planner_interfaces.srv import MultiGoalPlan


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
        self.previous_filtered_marker_count = 0

        # TF Buffer and Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Example wheelchair dimensions
        self.wheelchair_dimensions = [0.7, 0.7, 0.8]

        # For convenience in marker frames
        self.marker_base = 'map'

        # Time indicator that ensures minimun time interval between succesive publications
        self.last_publication_time = self.get_clock().now()
        self.time_interval = 2.0  # seconds

        # ------------------------------------------------------------
        # Create a service client to the C++ "MultiGoalPlanner" node
        # ------------------------------------------------------------
        self.plan_cli = self.create_client(MultiGoalPlan, 'multi_goal_plan')
        while not self.plan_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for 'multi_goal_plan' service...")
        self.get_logger().info("'multi_goal_plan' service client ready!")


    ###########################################################################
    # Lidar callback -> transform points to map -> build cKDTree
    ###########################################################################
    def lidar_filtered_callback(self, msg: PointCloud2):
        # 1) Basic checks
        num_points = msg.width * msg.height
        if num_points == 0:
            self.get_logger().warn("No points in '/lidar_filtered'")
            return

        # 2) Extract raw (x,y,z,rgb) from msg
        raw_array = np.frombuffer(msg.data, dtype=np.float32, count=num_points * 4)
        points_array = raw_array.reshape(num_points, 4)
        xyz = points_array[:, :3]

        # 3) Filter out NaNs/infs
        valid_mask = np.isfinite(xyz).all(axis=1)
        xyz = xyz[valid_mask]
        if xyz.size == 0:
            self.get_logger().warn("All points are invalid in '/lidar_filtered'")
            return

        # 4) Lookup transform from msg.frame -> 'map'
        try:
            transform_stamped = self.tf_buffer.lookup_transform(
                "map",  # target
                msg.header.frame_id,  # source
                rclpy.time.Time()
            )
        except Exception as ex:
            self.get_logger().warn(f"Could not lookup transform {msg.header.frame_id} -> 'map': {ex}")
            return

        # 5) Build a 4x4 transform matrix
        t = transform_stamped.transform.translation
        r = transform_stamped.transform.rotation
        transform_matrix = tf_transformations.quaternion_matrix([r.x, r.y, r.z, r.w])
        transform_matrix[0, 3] = t.x
        transform_matrix[1, 3] = t.y
        transform_matrix[2, 3] = t.z

        # 6) Apply transform to each point
        ones_col = np.ones((xyz.shape[0], 1), dtype=np.float32)
        xyz_hom = np.hstack((xyz, ones_col))  # shape (N,4)
        xyz_map_hom = (transform_matrix @ xyz_hom.T).T
        xyz_in_map = xyz_map_hom[:, :3]

        # 7) Build/update KD-tree
        if xyz_in_map.size > 0:
            self.scene_tree = cKDTree(xyz_in_map)
            self.latest_lidar_cloud = xyz_in_map
            self.get_logger().info(f"Updated scene_tree with {xyz_in_map.shape[0]} points.")
        else:
            self.scene_tree = None
            self.latest_lidar_cloud = None
            self.get_logger().warn("No valid points remain after transform.")


    ###########################################################################
    # Plane callback -> polygon -> padded edges -> sample -> collision check
    ###########################################################################
    def plane_cloud_callback(self, plane_cloud_msg):
        self.get_logger().info("Received plane_cloud")
        plane_cloud = self.convert_cloud_to_array(plane_cloud_msg)

        if plane_cloud is None or len(plane_cloud) == 0:
            self.get_logger().warn("Empty plane cloud.")
            return

        self.table_center = np.mean(plane_cloud, axis=0)

        try:
            # We want the transform that transforms points from base_link frame to map frame
            base_link_map_tf = self.tf_buffer.lookup_transform(
                "map",          # target_frame
                "base_link",    # source_frame
                rclpy.time.Time()  # time
            )
        except Exception as ex:
            self.get_logger().warn(f"Could not get base_link->map transform: {ex}")
            return

        # Extract robot position (rx, ry, rz) in map frame
        rx = base_link_map_tf.transform.translation.x
        ry = base_link_map_tf.transform.translation.y
        rz = base_link_map_tf.transform.translation.z

        robot_position = np.array([rx, ry, rz])
        distance_vector = self.table_center - robot_position
        distance = np.linalg.norm(distance_vector)

        if distance> 5.1:
            self.get_logger().info(
            f"Distance={distance:.2f}m > 2.5m. Generating circle of radius=3.0m around table."
            )

            # 3a) Visualize the 3D circle around the bed center
            circle_radius = 3.0
            
            # 3b) Approximate that circle as a polygon (or boundary) for sampling
            #     We'll assume you have a helper like: create_circle_polygon(center, radius, num_segments)
            circle_polygon_3d = self.create_circle_polygon(self.table_center, circle_radius, 36)
            self.publish_padded_lines(circle_polygon_3d, plane_cloud_msg.header)

            # 3c) Generate possible placements around the circle
            location_sampling_distance = 0.2
            possible_poses_circle = self.find_possible_placements_circle(circle_polygon_3d, location_sampling_distance, distance_vector)

            # 3d) Filter them by collision
            if self.scene_tree is not None:
                filtered_poses_circle = self.filter_for_collision_circle(
                    possible_poses_circle, 
                    self.scene_tree, 
                    self.wheelchair_dimensions
                )
            else:
                self.get_logger().warn("No scene data for collision. Using all possible placements.")
                filtered_poses_circle = possible_poses_circle

            # 3e) Publish them for visualization
            self.publish_possible_placements(possible_poses_circle)
            self.publish_filtered_locations(filtered_poses_circle)

            best_pose, best_cost = self.select_least_collision_risk(filtered_poses_circle, circle_polygon_3d)

            if best_pose is not None:
                self.publish_best_point(best_pose, circle_polygon_3d)
            else:
                self.get_logger().warn("No valid placement pose found.")

        else:
            # 0) Compute poses only if the time interval has passed
            if (self.get_clock().now() - self.last_publication_time).nanoseconds / 1e9 >= self.time_interval:
                
                # 1) Filter outliers (example: local density)
                filtered_cloud = self.remove_outliers_radius(plane_cloud)

                # 1) Get 2D polygon
                polygon_3d = self.findMinimumAreaShape(filtered_cloud) #plane_cloud
                self.get_logger().info(f"Detected polygon with {len(polygon_3d)} vertices.")

                # 2) Publish polygon
                self.publish_polygon_marker(polygon_3d, plane_cloud_msg.header)

                # 3) Build padded edges
                padding_offset = 0.7 #0.85
                padded_edges = self.calculate_padded_edges(polygon_3d, padding_offset)
                self.publish_padded_lines(padded_edges, plane_cloud_msg.header)

                # 4) Find possible placements (PoseStamped)
                location_sampling_distance = 0.1
                possible_poses = self.find_possible_placements(padded_edges, location_sampling_distance)

                # 5) Collision filtering
                if self.scene_tree is not None:
                    filtered_poses = self.filter_for_collision(possible_poses, self.scene_tree, self.wheelchair_dimensions)
                else:
                    self.get_logger().warn("No scene data for collision. Using all possible placements.")
                    filtered_poses = possible_poses

                # 6) Pick best pose (calls your C++ node via the service)
                # best_pose, best_cost = self.select_least_collision_risk(filtered_poses)
                best_pose, best_cost = self.select_least_collision_risk(filtered_poses, padded_edges)

                if best_pose is not None:
                    self.publish_best_point(best_pose, padded_edges)
                else:
                    self.get_logger().warn("No valid placement pose found.")

                # 7) Publish all possible & filtered
                self.publish_possible_placements(possible_poses)
                self.publish_filtered_locations(filtered_poses)

                #) 8) Update last publication time
                self.last_publication_time = self.get_clock().now()


    def convert_cloud_to_array(self, cloud_msg):
        points_gen = point_cloud2.read_points(cloud_msg, field_names=("x","y","z"), skip_nans=True)
        points_list = np.array([[p[0], p[1], p[2]] for p in points_gen], dtype=np.float32)
        return points_list


    def remove_outliers_radius(self, cloud, radius=0.15, min_neighbors=5):
        points_2d = cloud[:, :2]
        tree_2d = cKDTree(points_2d)
        keep_mask = []
        for pt in points_2d:
            idxs = tree_2d.query_ball_point(pt, radius)
            keep_mask.append(len(idxs) >= min_neighbors)
        keep_mask = np.array(keep_mask)
        return cloud[keep_mask]

    def findMinimumAreaShape(self, cloud):
        points_2d = cloud[:, :2].astype(np.float32)
        avg_height = np.mean(cloud[:, 2])
        # hull_points = cv2.convexHull(points_2d)
        best_fitting_box= cv2.minAreaRect(points_2d)
        hull_points = cv2.boxPoints(best_fitting_box)
        epsilon = 0.05 * cv2.arcLength(hull_points, True)
        approx_polygon = cv2.approxPolyDP(hull_points, epsilon, True)
        polygon_3d = [np.array([pt[0][0], pt[0][1], avg_height]) for pt in approx_polygon]
        return polygon_3d

    def create_circle_polygon(self, center, radius, num_segments=36):
        """
        Create a list of line segments forming a circle
        in the horizontal plane at z = center[2], 
        each segment is [[x1, y1, z1], [x2, y2, z2]].

        :param center: [cx, cy, cz], the center of the circle
        :param radius: float, the desired radius of the circle
        :param num_segments: int, how many segments to approximate the circle
        :return: List of edges, where each edge is [[x1, y1, z1], [x2, y2, z2]]
        """
        circle_edges = []
        cx, cy, cz = center[0], center[1], center[2]

        # Generate the points around the circle
        angle_step = 2.0 * math.pi / num_segments
        circle_points = []
        for i in range(num_segments):
            theta = i * angle_step
            x = cx + radius * math.cos(theta)
            y = cy + radius * math.sin(theta)
            z = cz
            circle_points.append([x, y, z])

        # Convert the consecutive points into edges
        # e.g. [pt0->pt1, pt1->pt2, ..., pt(n-1)->pt0]
        for i in range(num_segments):
            p1 = circle_points[i]
            p2 = circle_points[(i + 1) % num_segments]  # wrap around
            circle_edges.append([p1, p2])

        return circle_edges

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

        # close loop
        if polygon_3d:
            p0 = Point(x=float(polygon_3d[0][0]),
                       y=float(polygon_3d[0][1]),
                       z=float(polygon_3d[0][2]))
            marker.points.append(p0)

        self.polygon_pub.publish(marker)


    ###########################################################################
    # 1) Padded edges
    ###########################################################################
    def calculate_padded_edges(self, polygon, padding_offset):
        """
        Returns a list of edges, each edge is [ [x1, y1, z1], [x2, y2, z2] ].
        """
        # Copy + close
        polygon_copy = polygon[:]
        polygon_copy.append(polygon_copy[0])

        # 2D center
        contour_2d = [np.array([pt[0], pt[1]]) for pt in polygon_copy[:-1]]
        contour_center = np.mean(contour_2d, axis=0)

        padded_edges = []
        for i in range(len(polygon_copy) - 1):
            p1_2d = np.array([polygon_copy[i][0], polygon_copy[i][1]], dtype=np.float32)
            p2_2d = np.array([polygon_copy[i+1][0], polygon_copy[i+1][1]], dtype=np.float32)
            z_val = polygon_copy[i][2]

            line_vec = p2_2d - p1_2d
            line_len = np.linalg.norm(line_vec)
            if line_len < 1e-9:
                continue
            line_dir = line_vec / line_len
            perp_dir = np.array([-line_dir[1], line_dir[0]])

            # Shift corners
            p1_plus  = p1_2d + padding_offset * perp_dir
            p1_minus = p1_2d - padding_offset * perp_dir
            p2_plus  = p2_2d + padding_offset * perp_dir
            p2_minus = p2_2d - padding_offset * perp_dir

            # Decide outward side
            mid_plus = (p1_plus + p2_plus)/2.0
            mid_minus= (p1_minus+ p2_minus)/2.0
            dist_plus = np.linalg.norm(mid_plus - contour_center)
            dist_minus= np.linalg.norm(mid_minus- contour_center)
            if dist_plus > dist_minus:
                padded_edges.append([
                    [p1_plus[0], p1_plus[1], z_val],
                    [p2_plus[0], p2_plus[1], z_val],
                ])
            else:
                padded_edges.append([
                    [p1_minus[0], p1_minus[1], z_val],
                    [p2_minus[0], p2_minus[1], z_val],
                ])
        return padded_edges


    def publish_padded_lines(self, padded_edges, header):
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
            ptA, ptB = edge

            marker.points.append(Point(
                x=float(ptA[0]),
                y=float(ptA[1]),
                z=float(ptA[2])
            ))
            marker.points.append(Point(
                x=float(ptB[0]),
                y=float(ptB[1]),
                z=float(ptB[2])
            ))

        self.padded_line_marker_pub.publish(marker)


    ###########################################################################
    # 2) Compute orientation from nearest edge
    ###########################################################################
    def compute_orientation_from_nearest_edge(self, point_3d, padded_edges):
        best_dist = float('inf')
        best_line_dir = None
        p_xy = np.array(point_3d[:2], dtype=np.float32)

        for edge in padded_edges:
            if len(edge) != 2:
                continue
            A = np.array(edge[0][:2], dtype=np.float32)
            B = np.array(edge[1][:2], dtype=np.float32)
            vec = B - A
            length = np.linalg.norm(vec)
            if length < 1e-9:
                continue
            dir_ = vec / length
            diff = p_xy - A
            proj_len = np.dot(diff, dir_)
            proj_pt = A + proj_len * dir_
            dist = np.linalg.norm(p_xy - proj_pt)
            if dist < best_dist:
                best_dist = dist
                best_line_dir = dir_

        if best_line_dir is None:
            return Quaternion(w=1.0)

        perp = np.array([-best_line_dir[1], best_line_dir[0]])
        yaw = np.arctan2(perp[1], perp[0])
        q = tf_transformations.quaternion_from_euler(0, 0, yaw)
        return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])


    ###########################################################################
    # 3) Sample edges -> list-of-lists of PoseStamped
    ###########################################################################
    # def find_possible_placements(self, padded_edges, location_sampling_distance):
    #     skip_distance = 0.3
    #     all_clusters = []

    #     for edge in padded_edges:
    #         if len(edge) != 2:
    #             all_clusters.append([])
    #             continue

    #         p1 = np.array(edge[0])
    #         p2 = np.array(edge[1])
    #         vec = p2 - p1
    #         length = np.linalg.norm(vec)
    #         if length < 1e-9:
    #             all_clusters.append([])
    #             continue

    #         if length <= 2 * skip_distance:
    #             all_clusters.append([])
    #             continue

    #         dir_ = vec / length
    #         start_ratio = skip_distance / length
    #         end_ratio = 1.0 - start_ratio
    #         valid_len = (end_ratio - start_ratio) * length
    #         num_samples = int(valid_len / location_sampling_distance)
    #         if num_samples < 1:
    #             all_clusters.append([])
    #             continue

    #         cluster_list = []
    #         for i in range(num_samples+1):
    #             ratio = start_ratio + i * (end_ratio - start_ratio)/num_samples
    #             sample_xyz = p1 + ratio * vec

    #             q = self.compute_orientation_from_nearest_edge(sample_xyz, padded_edges)

    #             pose_stamped = PoseStamped()
    #             pose_stamped.header.frame_id = 'map'  # these are in map frame
    #             pose_stamped.pose.position.x = float(sample_xyz[0])
    #             pose_stamped.pose.position.y = float(sample_xyz[1])
    #             pose_stamped.pose.position.z = float(sample_xyz[2])
    #             pose_stamped.pose.orientation = q

    #             cluster_list.append(pose_stamped)
    #         all_clusters.append(cluster_list)

    #     return all_clusters

    def quaternion_from_yaw(self, yaw_degrees):
        """
        Helper function to convert a yaw (in degrees) about Z-axis
        into a quaternion in (x, y, z, w) order.
        """
        yaw_radians = math.radians(yaw_degrees)
        # Roll = 0, Pitch = 0, Yaw = yaw_radians
        q = tf_transformations.quaternion_from_euler(0, 0, yaw_radians)
        return q

    def find_possible_placements_circle(self, padded_edges, location_sampling_distance, distance_vector):
        skip_distance = 0.0
        all_clusters = []

        for edge in padded_edges:
            if len(edge) != 2:
                all_clusters.append([])
                continue

            p1 = np.array(edge[0])
            p2 = np.array(edge[1])
            vec = p2 - p1
            length = np.linalg.norm(vec)
            if length < 1e-9:
                all_clusters.append([])
                continue

            if length <= 2 * skip_distance:
                all_clusters.append([])
                continue

            dir_ = vec / length
            start_ratio = skip_distance / length
            end_ratio = 1.0 - start_ratio
            valid_len = (end_ratio - start_ratio) * length
            num_samples = int(valid_len / location_sampling_distance)
            if num_samples < 1:
                all_clusters.append([])
                continue

            cluster_list = []
            for i in range(num_samples+1):
                ratio = start_ratio + i * (end_ratio - start_ratio)/num_samples
                sample_xyz = p1 + ratio * vec

                # judge if the candidate is in the closer half circle
                candidate_vector = sample_xyz - self.table_center
                dot_product = np.dot(candidate_vector, distance_vector)
                cos_alfa= dot_product/(np.linalg.norm(candidate_vector)* np.linalg.norm(distance_vector))
                
                print(f"Cosine of angle {cos_alfa:.2f}")

                if (dot_product < 0) and (abs(cos_alfa) > 0.965):  # cos(15 degrees) = 0.965

                    q = self.compute_orientation_from_nearest_edge(sample_xyz, padded_edges)

                    pose_stamped = PoseStamped()
                    pose_stamped.header.frame_id = 'map'  # these are in map frame
                    pose_stamped.pose.position.x = float(sample_xyz[0])
                    pose_stamped.pose.position.y = float(sample_xyz[1])
                    pose_stamped.pose.position.z = float(sample_xyz[2])
                    pose_stamped.pose.orientation = q

                    cluster_list.append(pose_stamped)
                
                else:
                    continue
            all_clusters.append(cluster_list)

        return all_clusters

    def find_possible_placements(self, padded_edges, location_sampling_distance):
        """
        Generate poses at 30° bearing to each edge, limited only to edges
        whose length is between 1.5m and 2.0m. For each edge, we sample
        positions around 0.3m to 0.5m from each extreme end, oriented
        in the direction of the 'longer side' (from p1 to p2 if p2 is
        farther along the edge).
        """
        all_clusters = []

        # Filter only edges of interest: length between 1.5m and 2.0m
        MIN_EDGE_LENGTH = 1.5
        MAX_EDGE_LENGTH = 3.0

        # Distances (meters) near each edge extreme in which we want to generate poses
        NEAR_START_DIST = 0.1
        NEAR_END_DIST   = 0.6

        for edge in padded_edges:
            # If the edge isn’t exactly two endpoints, skip
            if len(edge) != 2:
                all_clusters.append([])
                self.logger.warn("Edge does not have exactly two endpoints.")
                continue

            p1 = np.array(edge[0])
            p2 = np.array(edge[1])
            vec = p2 - p1
            length = np.linalg.norm(vec)

            # Only consider edges in [1.5, 2.0]
            if not (MIN_EDGE_LENGTH <= length <= MAX_EDGE_LENGTH):
                all_clusters.append([])
                continue

            # Direction vector (p1 -> p2); we'll use atan2 for orientation
            direction = vec / length
            edge_yaw = math.degrees(math.atan2(direction[1], direction[0]))

            cluster_list = []

            # ----------------------------------------------------------------------
            # Sample near the first extreme (p1), from 0.3m to 0.5m
            # ----------------------------------------------------------------------
            start_ratio = NEAR_START_DIST / length
            end_ratio   = min(NEAR_END_DIST / length, 1.0)  # clamp if edge < 0.5m

            if end_ratio > start_ratio:
                # Decide how many samples we want
                valid_len   = (end_ratio - start_ratio) * length
                num_samples = int(valid_len / location_sampling_distance)
                num_samples = max(num_samples, 1)  # at least 1 pose

                for i in range(num_samples):
                    alpha = start_ratio + i * (end_ratio - start_ratio) / (num_samples - 1)
                    sample_xyz = p1 + alpha * vec

                    # Yaw = edge_yaw + 30° bearing
                    yaw_30 = edge_yaw + 30.0
                    q = self.quaternion_from_yaw(yaw_30)

                    pose_stamped = PoseStamped()
                    pose_stamped.header.frame_id = 'map'
                    pose_stamped.pose.position.x = float(sample_xyz[0])
                    pose_stamped.pose.position.y = float(sample_xyz[1])
                    pose_stamped.pose.position.z = float(sample_xyz[2])
                    pose_stamped.pose.orientation.x = q[0]
                    pose_stamped.pose.orientation.y = q[1]
                    pose_stamped.pose.orientation.z = q[2]
                    pose_stamped.pose.orientation.w = q[3]

                    cluster_list.append(pose_stamped)

            # ----------------------------------------------------------------------
            # Sample near the second extreme (p2), from 0.3m to 0.5m
            # but measured backward from p2 --> p1
            # ----------------------------------------------------------------------
            start_ratio_2 = 1.0 - (NEAR_END_DIST / length)
            end_ratio_2   = 1.0 - (NEAR_START_DIST / length)

            # Make sure these bounds make sense
            start_ratio_2 = max(start_ratio_2, 0.0)
            end_ratio_2   = min(end_ratio_2, 1.0)

            if end_ratio_2 > start_ratio_2:
                valid_len_2 = (end_ratio_2 - start_ratio_2) * length
                num_samples_2 = int(valid_len_2 / location_sampling_distance)
                num_samples_2 = max(num_samples_2, 1)

                for i in range(num_samples_2):
                    alpha = start_ratio_2 + i * (end_ratio_2 - start_ratio_2) / (num_samples_2 - 1)
                    sample_xyz = p1 + alpha * vec

                    # Same 30° offset to remain consistent with "pointing toward the longer side" (p1->p2)
                    yaw_30 = edge_yaw + 150.0
                    q = self.quaternion_from_yaw(yaw_30)

                    pose_stamped = PoseStamped()
                    pose_stamped.header.frame_id = 'map'
                    pose_stamped.pose.position.x = float(sample_xyz[0])
                    pose_stamped.pose.position.y = float(sample_xyz[1])
                    pose_stamped.pose.position.z = float(sample_xyz[2])
                    pose_stamped.pose.orientation.x = q[0]
                    pose_stamped.pose.orientation.y = q[1]
                    pose_stamped.pose.orientation.z = q[2]
                    pose_stamped.pose.orientation.w = q[3]

                    cluster_list.append(pose_stamped)

            all_clusters.append(cluster_list)

        return all_clusters
    
    ###########################################################################
    # 4) Filter for collision
    ###########################################################################
    def filter_for_collision(self, pose_clusters, scene_tree, wheelchair_dims):
        collision_radius = 0.6
        self.get_logger().info(f"Filtering collisions using radius {collision_radius}")

        filtered_clusters = []
        for c_idx, cluster in enumerate(pose_clusters):
            valid_poses = []
            for pose_stamped in cluster:
                x = pose_stamped.pose.position.x
                y = pose_stamped.pose.position.y
                z = pose_stamped.pose.position.z

                idxs = scene_tree.query_ball_point([x,y,z], collision_radius)
                if not idxs and x < 4.4 and y > -0.7:
                    valid_poses.append(pose_stamped)
            filtered_clusters.append(valid_poses)

            self.get_logger().info(f"Cluster {c_idx}: {len(valid_poses)} valid out of {len(cluster)}")
        return filtered_clusters

    def filter_for_collision_circle(self, pose_clusters, scene_tree, wheelchair_dims):
        collision_radius = 0.4
        self.get_logger().info(f"Filtering collisions using radius {collision_radius}")

        filtered_clusters = []
        for c_idx, cluster in enumerate(pose_clusters):
            valid_poses = []
            for pose_stamped in cluster:
                x = pose_stamped.pose.position.x
                y = pose_stamped.pose.position.y
                z = pose_stamped.pose.position.z

                idxs = scene_tree.query_ball_point([x,y,z], collision_radius)
                if not idxs:
                    valid_poses.append(pose_stamped)
            filtered_clusters.append(valid_poses)

            self.get_logger().info(f"Cluster {c_idx}: {len(valid_poses)} valid out of {len(cluster)}")
        return filtered_clusters

    ###########################################################################
    # 5) Choose best from filtered using the C++ node's /multi_goal_plan service
    ###########################################################################
    def select_least_collision_risk(self, filtered_clusters, padded_edges):
        """
        This function no longer tries to process the result immediately.
        Instead, it collects all the info, calls the service asynchronously,
        and returns. The real 'post-service' logic moves into plan_response_callback.
        """

        import math
        from multi_goal_planner_interfaces.srv import MultiGoalPlan

        # 1) Flatten + keep track of corners
        candidate_goals = []
        goal_corners_map = []

        for edge_index, cluster in enumerate(filtered_clusters):
            if edge_index >= len(padded_edges):
                break

            edge_corners = padded_edges[edge_index]  # e.g. [[x1, y1, z1],[x2, y2, z2]]
            for pose_stamped in cluster:
                candidate_goals.append(pose_stamped)
                goal_corners_map.append(edge_corners)

        if not candidate_goals:
            self.get_logger().warn("No candidate goals available for planning.")
            return None, float('inf')

        # 2) Lookup start pose
        try:
            transform_stamped = self.tf_buffer.lookup_transform(
                'map',         # target frame
                'base_link',   # source frame
                rclpy.time.Time()
            )
        except (LookupException, ExtrapolationException, TransformException) as ex:
            self.get_logger().error(f"Could not lookup transform 'base_link'->'map': {ex}")
            return None, float('inf')

        start_pose = PoseStamped()
        start_pose.header.frame_id = 'map'
        start_pose.header.stamp = transform_stamped.header.stamp

        start_pose.pose.position.x = transform_stamped.transform.translation.x
        start_pose.pose.position.y = transform_stamped.transform.translation.y
        start_pose.pose.position.z = transform_stamped.transform.translation.z

        q = transform_stamped.transform.rotation
        start_pose.pose.orientation = Quaternion(x=q.x, y=q.y, z=q.z, w=q.w)

        self.get_logger().info(
            f"Start pose: ({start_pose.pose.position.x:.2f}, {start_pose.pose.position.y:.2f})"
        )

        # Debug print
        for i, g in enumerate(candidate_goals):
            self.get_logger().info(
                f"Goal {i}: ({g.pose.position.x:.2f},{g.pose.position.y:.2f})"
            )

        # 3) Build service request
        request = MultiGoalPlan.Request()
        request.start = start_pose
        request.goals = candidate_goals

        # 4) Async call
        self.get_logger().info(f"Calling /multi_goal_plan with {len(candidate_goals)} goals (async).")
        future = self.plan_cli.call_async(request)

        # We need to pass data (candidate_goals, corners, etc.) to the callback.
        # We'll use a lambda to "capture" these values.
        future.add_done_callback(
            lambda fut: self.handle_plan_response(fut, candidate_goals, goal_corners_map)
        )

        # Return a placeholder since the real result will come in the callback
        return None, float('inf')

    def handle_plan_response(self, fut, candidate_goals, goal_corners_map):
        """
        This callback is triggered once the service call is done.
        Here, we finish the logic that used to be 'below' the line
        that caused None in select_least_collision_risk.
        """
        try:
            response = fut.result()
            if not response or not response.success:
                self.get_logger().warn("No valid path found (success=False).")
                return

            in_path_costs = response.costs
            if len(in_path_costs) != len(candidate_goals):
                self.get_logger().error(
                    f"Mismatch: got {len(in_path_costs)} costs, but {len(candidate_goals)} goals."
                )
                return

            # Now compute total cost = in_path_cost + near_goal_cost
            best_index = None
            best_total_cost = float('inf')

            for i, in_path_cost in enumerate(in_path_costs):
                goal_pose = candidate_goals[i]
                corners = goal_corners_map[i]

                near_goal_cost = self.compute_near_goal_cost(goal_pose, corners)
                total_cost = in_path_cost + near_goal_cost

                self.get_logger().info(
                    f"Goal {i}: ({goal_pose.pose.position.x:.2f},{goal_pose.pose.position.y:.2f}), in_path_cost={in_path_cost:.2f}, near_goal_cost={near_goal_cost:.2f}, "
                    f"total_cost={total_cost:.2f}"
                )

                if total_cost < best_total_cost:
                    best_total_cost = total_cost
                    best_index = i

            if best_index is None:
                self.get_logger().warn("No valid minimum total cost found among candidate goals.")
                return

            best_pose = candidate_goals[best_index]
            self.get_logger().info(
                f"Selected best goal => index={best_index}, total_cost={best_total_cost:.2f}. "
                f"Pose=({best_pose.pose.position.x:.2f}, {best_pose.pose.position.y:.2f})"
            )

            # Finally, you can publish or store the best pose here:
            self.publish_best_point(best_pose, ...)

        except Exception as e:
            self.get_logger().error(f"Service call failed: {str(e)}")


    def compute_near_goal_cost(self, goal_pose, corners):
        import math

        gx = goal_pose.pose.position.x
        gy = goal_pose.pose.position.y

        cost = 0.0
        epsilon = 0.01
        for corner in corners:
            cx, cy, _cz = corner
            dist = math.sqrt((gx - cx)**2 + (gy - cy)**2)
            cost += 1.0 / (dist + epsilon)
        return cost * 0

    ###########################################################################
    # 6) Publish all possible placements
    ###########################################################################
    def publish_possible_placements(self, pose_clusters):
        marker_array = MarkerArray()
        marker_id = 0

        for cluster in pose_clusters:
            for pose_stamped in cluster:
                # No transform needed if these are already in map frame
                transformed_pose = pose_stamped

                marker = Marker()
                marker.header.frame_id = 'map'
                marker.ns = 'possible_placements'
                marker.id = marker_id
                marker.type = Marker.ARROW
                marker.action = Marker.ADD

                marker.scale.x = 0.2
                marker.scale.y = 0.08
                marker.scale.z = 0.08

                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0

                marker.pose = transformed_pose.pose

                marker_array.markers.append(marker)
                marker_id += 1

        # Remove old markers if needed
        for old_marker_id in range(marker_id, self.previous_marker_count):
            delete_marker = Marker()
            delete_marker.header.frame_id = 'map'
            delete_marker.ns = 'possible_placements'
            delete_marker.id = old_marker_id
            delete_marker.action = Marker.DELETE
            marker_array.markers.append(delete_marker)

        self.previous_marker_count = marker_id
        self.placements_marker_pub.publish(marker_array)


    ###########################################################################
    # 7) Publish filtered placements
    ###########################################################################
    def publish_filtered_locations(self, pose_clusters):
        marker_array = MarkerArray()
        marker_id = 0

        for cluster in pose_clusters:
            for pose_stamped in cluster:
                transformed_pose = pose_stamped

                marker = Marker()
                marker.header.frame_id = 'map'
                marker.ns = 'filtered_placements'
                marker.id = marker_id
                marker.type = Marker.ARROW
                marker.action = Marker.ADD

                marker.scale.x = 0.2
                marker.scale.y = 0.08
                marker.scale.z = 0.08

                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0

                marker.pose = transformed_pose.pose

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


    ###########################################################################
    # TF management
    ###########################################################################
    def update_transform(self):
        now = self.get_clock().now()
        if (self.last_transform_time is None
            or now - self.last_transform_time > self.transform_update_interval):
            try:
                self.transform_hesai_to_map = self.tf_buffer.lookup_transform(
                    'map',
                    'hesai_lidar',
                    rclpy.time.Time()
                )
                self.last_transform_time = now
            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException,
                    tf2_ros.TransformException) as ex:
                self.get_logger().warn(f"Transform error: {ex}")
                self.transform_hesai_to_map = None


    ###########################################################################
    # 8) Publish best pose
    ###########################################################################
    def publish_best_point(self, best_pose_stamped, padded_edges):
        """
        best_pose_stamped is presumably in 'map' frame already.
        We'll just publish an arrow marker in 'map'.
        """
        start_time = time.time()
        if best_pose_stamped is None:
            self.get_logger().warn("No best pose to publish.")
            return

        transformed_pose = best_pose_stamped  # already in map

        self.transformed_pose_previous = transformed_pose
        self.get_logger().info(f"Best pose: {transformed_pose}")

        # Publish as arrow
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.ns = "best_point_vector"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose = transformed_pose.pose
        marker.scale.x = 0.5  # arrow length
        marker.scale.y = 0.06
        marker.scale.z = 0.06
        marker.color.r = 1.0
        marker.color.a = 1.0

        

        # Also publish /best_point_topic (position only)
        best_point_msg = Point(
            x=transformed_pose.pose.position.x,
            y=transformed_pose.pose.position.y,
            z=transformed_pose.pose.position.z
        )

        # Publish both 
        self.best_point_marker_pub.publish(marker)
        self.best_point_pub.publish(best_point_msg)

        end_time = time.time()
        duration = end_time - start_time
        self.get_logger().info(
            f"Published best point in map frame: ({best_point_msg.x}, {best_point_msg.y}, {best_point_msg.z})"
        )


def main(args=None):
    rclpy.init(args=args)
    node = PlaneCloudProcessorNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
