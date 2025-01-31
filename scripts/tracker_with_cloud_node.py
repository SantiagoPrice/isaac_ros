#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, CameraInfo
from vision_msgs.msg import Detection3DArray
from vision_msgs.msg import Detection3D, BoundingBox3D
from geometry_msgs.msg import Pose, Point, Quaternion
from ultralytics_ros.msg import YoloResult, DesiredLocation, LocationArray
from visualization_msgs.msg import MarkerArray, Marker
from sensor_msgs_py import point_cloud2
import numpy as np
import message_filters
import tf2_ros
import tf_transformations
import open3d as o3d
from cv_bridge import CvBridge
from sensor_msgs.msg import PointCloud2, PointField
from builtin_interfaces.msg import Duration
from rclpy.duration import Duration as RclpyDuration  # For timeouts and durations
from geometry_msgs.msg import PoseStamped
import tf2_geometry_msgs  # Necessary for transforming PoseStamped messages
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose_stamped

import cv2
from scipy.spatial import cKDTree
import time

class PlaneCloudPublisherNode(Node):

    def __init__(self):
        super().__init__('plane_cloud_publisher_node')

        # Declare and get parameters
        self.declare_parameter('camera_info_topic', '/yolov8_encoder/resize/camera_info')
        self.declare_parameter('lidar_topic', '/lidar_filtered')
        self.declare_parameter('yolo_result_topic', 'yolo_result')
        self.declare_parameter('yolo_3d_result_topic', '/yolo_3d_result')
        self.declare_parameter('cluster_tolerance', 0.3)
        self.declare_parameter('voxel_leaf_size', 1.0)
        self.declare_parameter('min_cluster_size', 20)
        self.declare_parameter('max_cluster_size', 10000)

        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        lidar_topic = self.get_parameter('lidar_topic').get_parameter_value().string_value
        yolo_result_topic = self.get_parameter('yolo_result_topic').get_parameter_value().string_value

        # Message filters to synchronize topics
        self.camera_info_sub = message_filters.Subscriber(self, CameraInfo, camera_info_topic)
        self.lidar_sub = message_filters.Subscriber(self, PointCloud2, lidar_topic)
        self.yolo_result_sub = message_filters.Subscriber(self, YoloResult, yolo_result_topic)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.camera_info_sub, self.lidar_sub, self.yolo_result_sub], 500, 2.0)
        self.sync.registerCallback(self.sync_callback)

        # Publishers
        yolo_3d_result_topic = self.get_parameter('yolo_3d_result_topic').get_parameter_value().string_value
        self.detection3d_pub = self.create_publisher(Detection3DArray, yolo_3d_result_topic, 1)
        self.detection_cloud_pub = self.create_publisher(PointCloud2, 'detection_cloud', 1)
        self.marker_pub = self.create_publisher(MarkerArray, 'detection_marker', 1)
        self.plane_cloud_pub = self.create_publisher(PointCloud2, 'plane_cloud', 1)  # Publisher for detected plane clouds

        self.polygon_pub = self.create_publisher(Marker, 'table_polygon', 1)
        self.padded_line_marker_pub = self.create_publisher(Marker, '/padded_line', 1)
        self.placements_marker_pub = self.create_publisher(MarkerArray, '/possible_placements', 1)

        self.pub_locations = self.create_publisher(LocationArray, "/desirable_locations", 1)
        self.create_subscription(LocationArray, '/desirable_locations', self.desirable_locations_callback, 1)
        self.desirable_locations_marker_pub = self.create_publisher(MarkerArray, '/desirable_locations_markers', 1)

        self.best_point_marker_pub = self.create_publisher(Marker, "/best_point_marker", 1)
        self.best_point_pub = self.create_publisher(Point, "/best_point_topic", 1)

        self.transformed_pose_previous = None

        self.transform_hesai_to_map = None
        self.last_transform_time = None
        self.transform_update_interval = RclpyDuration(seconds=1.0)

        self.previous_marker_count = 0  # Track the number of markers published previously
        self.previous_locations_count = 0

        # TF Buffer and Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.bridge = CvBridge()
        self.last_call_time = self.get_clock().now()

    def sync_callback(self, camera_info_msg, cloud_msg, yolo_result_msg):
        self.get_logger().info("sync_callback")
        current_call_time = self.get_clock().now()
        callback_interval = current_call_time - self.last_call_time
        self.last_call_time = current_call_time

        # Downsample and transform the point cloud
        downsampled_cloud = self.downsample_cloud(cloud_msg)

        cam_model = self.get_camera_model(camera_info_msg)
        transformed_cloud = self.transform_cloud(
            downsampled_cloud, cloud_msg.header.frame_id, cam_model['frame_id'], cloud_msg.header.stamp)
        self.get_logger().info(f"Transformed cloud size: {len(transformed_cloud)}")

        # Initialize message objects
        detection3d_array_msg = Detection3DArray()
        detection3d_array_msg.header = cloud_msg.header
        detection3d_array_msg.header.stamp = yolo_result_msg.header.stamp

        # Process detections
        detection_cloud_msg = self.process_detections(
            transformed_cloud, yolo_result_msg, cloud_msg.header, detection3d_array_msg, cam_model)

        marker_array_msg = self.create_marker_array(detection3d_array_msg, callback_interval.nanoseconds / 1e9)

        # Publish the results
        self.detection3d_pub.publish(detection3d_array_msg)
        if detection_cloud_msg is not None:
            self.detection_cloud_pub.publish(detection_cloud_msg)
        else:
            self.get_logger().warn("detection_cloud_msg is None, skipping publish.")
        self.marker_pub.publish(marker_array_msg)

    def process_detections(self, cloud, yolo_result_msg, header, detection3d_array_msg, cam_model):
        self.get_logger().info(f"Number of detections: {len(yolo_result_msg.detections.detections)}")

        all_detection_clouds = []
        detected_planes_cloud = []

        for detection in yolo_result_msg.detections.detections:
            detection_cloud_raw = self.extract_points_in_bbox(cloud, detection, cam_model)

            if detection_cloud_raw.size > 0:
                transformed_points = self.transform_cloud(
                    detection_cloud_raw, cam_model['frame_id'], header.frame_id, header.stamp)

                # for collision check
                downsampled_cloud = self.downsample_cloud_high(cloud)
                original_cloud = self.cloud_to_transformed_cloud(downsampled_cloud, cam_model['frame_id'], header.frame_id, header.stamp)
                scene_tree = cKDTree(original_cloud[:, :3]) # Build KD-Tree for the scene cloud

                closest_detection_cloud = self.cluster_extraction(transformed_points)

                detection_id = None
                if detection.results:
                    detection_id = detection.results[0].hypothesis.class_id
                    self.get_logger().info(f"detection_id is {detection_id}.")

                # Process plane for 'dining table' detection
                if detection_id == '60':  # Assuming '60' is the class ID for 'dining table'
                    self.get_logger().info("Processing plane for 'dining table' detection.")
                    plane_cloud = self.detect_plane(closest_detection_cloud)

                    if plane_cloud is not None:
                        detected_planes_cloud.extend(plane_cloud)
                        self.publish_plane_cloud(plane_cloud, header)

                    # Convert detected plane to minimum area shape (polygon)
                        table_polygon_3d = self.findMinimumAreaShape(np.array(plane_cloud))
                        self.get_logger().info(f"Detected table polygon has {len(table_polygon_3d)} vertices.")

                        # Publish the polygon as a visualization marker in RViz
                        self.publish_polygon_marker(table_polygon_3d, header)

                        # --- New Code Block: Placement Detection, Collision Filtering, and Publishing ---
                        
                        # Step 1: Calculate possible placements around the table
                        padding_offset = 0.9  # Example value, adjust as needed
                        padded_polygon = self.calculate_padded_lines(table_polygon_3d, padding_offset)

                        location_sampling_distance = 0.2  # Example value, adjust as needed
                        possible_locations = self.find_possible_placements(table_polygon_3d, padding_offset, location_sampling_distance)
                        
                        # Step 2: Filter for collisions with the scene cloud
                        wheelchair_dimensions = [0.7, 0.7, 0.8]  # Example wheelchair dimensions
                        cube_dimensions = [1.0, 1.0, 1.0] 
                        filtered_locations = self.filter_for_collision(possible_locations, scene_tree, wheelchair_dimensions)
                        best_point, risk_score = self.select_least_collision_risk(filtered_locations, original_cloud, cube_dimensions)
                        
                        if best_point is not None:
                            # table_center = [0.0, 0.0, 0.0]  # Replace with the actual table center if available
                            self.publish_best_point(best_point, padded_polygon)
                        else:
                            self.get_logger().warn("No valid placement point found.")

                        # Step 3: Calculate weights (for simplicity, set equal weights here)
                        weights = [[1.0 for _ in cluster] for cluster in filtered_locations]
                        
                        # Step 4: Calculate heading (assuming quaternion identity for simplicity)
                        heading = [[Quaternion(x=0.0, y=0.0, z=0.0, w=1.0) for _ in cluster] for cluster in filtered_locations]

                        # Publish padded polygon lines to RViz on '/padded_line'
                        self.publish_padded_lines(padded_polygon, header)

                        # Step 5: Publish the locations
                        self.publish_locations(filtered_locations, weights, heading)


                all_detection_clouds.extend(transformed_points)
                self.create_bounding_box(detection3d_array_msg, closest_detection_cloud, detection.results)

        if all_detection_clouds:
            combined_cloud_array = np.array(all_detection_clouds, dtype=np.float32)
            detection_cloud_msg = self.create_pointcloud2(combined_cloud_array, header)
            return detection_cloud_msg
        else:
            self.get_logger().warn("No valid points in detection clouds.")
            return None

    def downsample_cloud(self, cloud_msg):
        # Read points from the PointCloud2 message
        points_list = list(point_cloud2.read_points(
            cloud_msg, field_names=("x", "y", "z"), skip_nans=True))

        # Check if points_list is empty
        if not points_list:
            self.get_logger().warn("No valid points found in the input cloud.")
            return np.empty((0, 3), dtype=np.float32)

        # Extract x, y, z values and convert to a NumPy array
        points_array = np.array([[p[0], p[1], p[2]] for p in points_list], dtype=np.float32)

        voxel_size = self.get_parameter('voxel_leaf_size').get_parameter_value().double_value

        # Create Open3D point cloud and downsample
        o3d_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_array))
        downsampled_cloud = o3d_cloud.voxel_down_sample(0.05)

        return np.asarray(downsampled_cloud.points)


    def transform_cloud(self, cloud, source_frame, target_frame, stamp):
        try:
            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, stamp)
            transform_matrix = self.transform_to_matrix(transform)
            ones = np.ones((cloud.shape[0], 1))
            cloud_hom = np.hstack([cloud, ones])
            transformed_cloud_hom = (transform_matrix @ cloud_hom.T).T
            return transformed_cloud_hom[:, :3]
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as ex:
            self.get_logger().warn(f"Transform error: {ex}")
            return cloud

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

    def get_camera_model(self, camera_info_msg):
        camera_matrix = np.array(camera_info_msg.k).reshape((3, 3))
        distortion_coeffs = np.array(camera_info_msg.d)
        frame_id = camera_info_msg.header.frame_id
        return {'camera_matrix': camera_matrix, 'distortion_coeffs': distortion_coeffs, 'frame_id': frame_id}

    def extract_points_in_bbox(self, cloud, detection2d_msg, cam_model):
        # Filter points where z > 0 (points in front of the camera)
        valid_indices = cloud[:, 2] > 0  # Assuming z is the third column in cloud array

        # Apply the mask to get only valid points
        cloud_in_front = cloud[valid_indices]

        # Project 3D points to 2D pixel coordinates
        pixel_coords = self.project_points(cloud_in_front, cam_model)

        # Get bounding box limits
        bbox = detection2d_msg.bbox
        u_min = bbox.center.position.x - bbox.size_x / 2
        u_max = bbox.center.position.x + bbox.size_x / 2
        v_min = bbox.center.position.y - bbox.size_y / 2
        v_max = bbox.center.position.y + bbox.size_y / 2

        # Create mask for points inside the bounding box
        inside_mask = (
            (pixel_coords[:, 0] >= u_min) & (pixel_coords[:, 0] <= u_max) &
            (pixel_coords[:, 1] >= v_min) & (pixel_coords[:, 1] <= v_max)
        )

        return cloud_in_front[inside_mask]

    def project_points(self, points_3d, cam_model):
        camera_matrix = cam_model['camera_matrix']
        pixel_coords_hom = points_3d @ camera_matrix.T
        pixel_coords = pixel_coords_hom[:, :2] / pixel_coords_hom[:, 2:3]
        return pixel_coords

    def cluster_extraction(self, cloud):
        if cloud.size == 0:
            self.get_logger().warn("Cloud is empty, skipping clustering.")
            return np.array([])

        # Create Open3D point cloud
        o3d_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud))

        # Perform DBSCAN clustering
        labels = np.array(o3d_cloud.cluster_dbscan(
            eps=self.get_parameter("cluster_tolerance").get_parameter_value().double_value,
            min_points=self.get_parameter("min_cluster_size").get_parameter_value().integer_value,
            print_progress=True))

        # Find the cluster with the closest centroid
        min_distance = float('inf')
        closest_cluster_points = np.array([])

        for cluster_id in np.unique(labels):
            if cluster_id == -1:
                continue  # Ignore noise points

            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_points = np.asarray(o3d_cloud.select_by_index(cluster_indices).points)
            centroid = np.mean(cluster_points, axis=0)
            distance = np.linalg.norm(centroid)

            if distance < min_distance:
                min_distance = distance
                closest_cluster_points = cluster_points

        return closest_cluster_points

    def detect_plane(self, cloud):
        if cloud.size == 0:
            self.get_logger().warn("Input cloud is empty. Skipping plane detection.")
            return None

        # Create Open3D point cloud
        o3d_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud))

        # Perform RANSAC plane segmentation
        plane_model, inliers = o3d_cloud.segment_plane(
            distance_threshold=0.01, ransac_n=3, num_iterations=1000)

        if len(inliers) == 0:
            self.get_logger().info("No plane detected in the point cloud.")
            return None

        plane_cloud = o3d_cloud.select_by_index(inliers)
        plane_points = np.asarray(plane_cloud.points)
        self.table_center = np.mean(plane_points, axis=0)
        self.get_logger().info(f"Calculated table center at: {self.table_center.tolist()}")

        return plane_points

    def publish_plane_cloud(self, plane_cloud, header):
        plane_cloud_msg = self.create_pointcloud2(plane_cloud, header)
        self.plane_cloud_pub.publish(plane_cloud_msg)

    def create_pointcloud2(self, points_array, header):
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud_data = [tuple(point) for point in points_array]
        return point_cloud2.create_cloud(header, fields, cloud_data)

    def create_bounding_box(self, detection3d_array_msg, cloud, results):
        if cloud.size == 0:
            self.get_logger().warn("Cloud is empty, skipping bounding box creation.")
            return

        centroid = np.mean(cloud, axis=0)
        theta = -np.arctan2(centroid[1], np.sqrt(centroid[0]**2 + centroid[2]**2))

        rotation_matrix = tf_transformations.rotation_matrix(theta, [0, 0, 1])
        transformed_cloud = (rotation_matrix[:3, :3] @ cloud.T).T + rotation_matrix[:3, 3]

        min_pt = np.min(transformed_cloud, axis=0)
        max_pt = np.max(transformed_cloud, axis=0)
        bbox_center_transformed = (min_pt + max_pt) / 2

        bbox_center = (tf_transformations.inverse_matrix(rotation_matrix)[:3, :3] @ bbox_center_transformed) + \
                      tf_transformations.inverse_matrix(rotation_matrix)[:3, 3]

        rotation_quat = tf_transformations.quaternion_from_matrix(tf_transformations.inverse_matrix(rotation_matrix))

        detection3d_msg = Detection3D()
        detection3d_msg.bbox = BoundingBox3D()
        detection3d_msg.bbox.center = Pose()
        detection3d_msg.bbox.center.position = Point(
            x=bbox_center[0], y=bbox_center[1], z=bbox_center[2])
        detection3d_msg.bbox.center.orientation = Quaternion(
            x=rotation_quat[0], y=rotation_quat[1], z=rotation_quat[2], w=rotation_quat[3])
        detection3d_msg.bbox.size.x = max_pt[0] - min_pt[0]
        detection3d_msg.bbox.size.y = max_pt[1] - min_pt[1]
        detection3d_msg.bbox.size.z = max_pt[2] - min_pt[2]
        detection3d_msg.results = results

        detection3d_array_msg.detections.append(detection3d_msg)

    def findMinimumAreaShape(self,cloud):
        """
        Function to find the minimal 2D polygonal shape (convex hull approximation) of a 3D cloud projected onto a 2D plane.
        :param cloud: A NumPy array (N, 3) representing points in the point cloud (x, y, z).
        :return: A list of vertices (as NumPy arrays) representing the 2D minimal polygon in 3D space (with averaged height).
        """
        # Project the 3D points onto a 2D plane by ignoring the z-coordinate
        points_2d = np.array([(point[0], point[1]) for point in cloud], dtype=np.float32)
        
        # Compute the average height (z-coordinate)
        avg_height = np.mean(cloud[:, 2])

        # Compute the convex hull of the projected 2D points
        hull_points = cv2.convexHull(points_2d)

        # Approximate the convex hull with a polygon (simplify the shape)
        epsilon = 0.01 * cv2.arcLength(hull_points, True)
        approx_polygon = cv2.approxPolyDP(hull_points, epsilon, True)

        # Convert the 2D polygon back to 3D by adding the average height to each point
        polygon_3d = [np.array([pt[0][0], pt[0][1], avg_height]) for pt in approx_polygon]

        return polygon_3d

    def publish_polygon_marker(self, polygon_3d, header):
        """
        Publish the polygon points as a Marker message for visualization in RViz.
        :param polygon_3d: A list of 3D points representing the polygon vertices.
        :param header: The header to use for the marker message.
        """
        marker = Marker()
        marker.header = header
        marker.ns = 'table_polygon'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05  # Line width

        # Set the color (blue)
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0  # Fully opaque

        # Add the polygon points to the marker
        for point in polygon_3d:
            p = Point()
            p.x = float(point[0])  # Ensure it's a Python float
            p.y = float(point[1])  # Ensure it's a Python float
            p.z = float(point[2])  # Ensure it's a Python float
            marker.points.append(p)

        # Close the loop by adding the first point to the end
        if len(polygon_3d) > 0:
            p = Point()
            p.x = float(polygon_3d[0][0])
            p.y = float(polygon_3d[0][1])
            p.z = float(polygon_3d[0][2])
            marker.points.append(p)

        # Publish the marker
        self.polygon_pub.publish(marker)

    def find_possible_placements(self, polygon, padding_offset, location_sampling_distance):
        """
        Calculate possible placement locations along the polygon edges, avoiding the end points.
        
        :param polygon: List of 3D vertices
        :param padding_offset: Padding distance from the edges
        :param location_sampling_distance: Distance between sampled points
        :return: List of sampled placement points
        """
        collision_points = [] 

        padded_polygon = self.calculate_padded_lines(polygon, padding_offset)
        padded_polygon.append(padded_polygon[0])  # Close the loop
        
        for i in range(len(padded_polygon) - 1):
            points_on_line = []
            line_start = np.array(padded_polygon[i])
            line_end = np.array(padded_polygon[i + 1])
            line_length = np.linalg.norm(line_end - line_start)
            num_samples = int(line_length / location_sampling_distance)
            
            if num_samples <= 1:
                continue  # Not enough points to sample between the endpoints without including them
            
            for j in range(1, num_samples):
                ratio = j / num_samples
                sampled_point = (1 - ratio) * line_start + ratio * line_end
                points_on_line.append(sampled_point.tolist())
            collision_points.append(points_on_line)
        
        # After calculating the points, publish them
        self.publish_placements(collision_points)
        
        return collision_points

    def calculate_padded_lines(self, polygon, padding_offset):
        """
        Generate padded polygon edges by shifting them by the padding_offset.

        :param polygon: List of polygon vertices
        :param padding_offset: Padding distance from the edges
        :return: List of padded polygon edges
        """
        padded_edges = []
        polygon.append(polygon[0])  # Close the loop

        # Convert polygon to 2D points
        contour = [np.array([point[0], point[1]], dtype=np.float32) for point in polygon]
        contour.pop()  # Remove the last point to avoid repeating the first point in a closed loop

        # Calculate the center of the contour (mean of vertices)
        contour_center = np.mean(contour, axis=0)

        # for i in range(1): #for debug
        for i in range(len(polygon) - 1):
            p1 = np.array([polygon[i][0], polygon[i][1]], dtype=np.float32)
            p2 = np.array([polygon[i + 1][0], polygon[i + 1][1]], dtype=np.float32)

            # Compute direction of the line
            line_vector = p2 - p1
            line_length = np.linalg.norm(line_vector)

            if line_length == 0:
                continue  # Skip degenerate lines

            # Normalize the direction vector
            line_direction = line_vector / line_length

            # Compute the perpendicular direction
            perpendicular_direction = np.array([-line_direction[1], line_direction[0]])

            # Compute padded points by moving points along the perpendicular direction
            padded_point1 = p1 + padding_offset * perpendicular_direction
            padded_point2 = p1 - padding_offset * perpendicular_direction
            padded_point3 = p2 + padding_offset * perpendicular_direction
            padded_point4 = p2 - padding_offset * perpendicular_direction

            # Calculate the midpoints of padded lines
            mid_point1 = (padded_point1 + padded_point3) / 2.0
            mid_point2 = (padded_point2 + padded_point4) / 2.0

            # Calculate distances from the midpoints to the contour center
            distance_mid_point1 = np.linalg.norm(mid_point1 - contour_center)
            distance_mid_point2 = np.linalg.norm(mid_point2 - contour_center)

            # Choose the midpoint with the greater distance from the center
            if distance_mid_point1 > distance_mid_point2:
                padded_edges.append([padded_point1[0], padded_point1[1], polygon[i][2]])
                padded_edges.append([padded_point3[0], padded_point3[1], polygon[i][2]])
            else:
                padded_edges.append([padded_point2[0], padded_point2[1], polygon[i][2]])
                padded_edges.append([padded_point4[0], padded_point4[1], polygon[i + 1][2]])

        return padded_edges

    def filter_for_collision(self, locations, scene_tree, wheelchair_dimensions):
        """
        Filters out locations that would result in collisions with the environment.

        :param locations: List of possible placement locations (clusters of points)
        :param scene_cloud: Point cloud of the scene (Nx3 array).
        :param wheelchair_dimensions: List of dimensions (width, length, depth).
        :return: List of non-colliding locations.
        """
        desirable_locations = []
        wheelchair_width, wheelchair_length, wheelchair_depth = wheelchair_dimensions
        collision_radius = 0.55  # Example small radius

        self.get_logger().info(f"Filtering collision with wheelchair dimensions: {wheelchair_dimensions}")

        for ii, cluster in enumerate(locations):
            # Convert cluster to NumPy array for efficient operations
            cluster_array = np.array(cluster)
            
            # Query points in the cluster against the KD-Tree to find collisions
            non_colliding_points = []
            for point in cluster_array:
                # Query points within the collision radius
                indices = scene_tree.query_ball_point(point[:3], collision_radius)

                if not indices:  # If no points are within the radius
                    non_colliding_points.append(point.tolist())

            self.get_logger().info(f"Cluster {ii}: Found {len(non_colliding_points)} non-colliding points.")
            desirable_locations.append(non_colliding_points)

        total_non_colliding_points = sum(len(cluster) for cluster in desirable_locations)
        self.get_logger().info(f"Total non-colliding points: {total_non_colliding_points}")
        return desirable_locations

    def select_least_collision_risk(self, desirable_locations, scene_cloud, cube_dimensions):
        """
        Selects the location with the lowest summed-up distance cost between its z-axis-aligned centerline
        and points in the scene cloud within a defined cube.

        :param desirable_locations: List of placement clusters (list of lists of points).
        :param scene_cloud: Scene point cloud (Nx3 array).
        :param cube_dimensions: Dimensions of the cube [width, length, height] centered at each location.
        :return: The best point (x, y, z) with the lowest cost and its cost value.
        """
        if not desirable_locations or not any(len(cluster) for cluster in desirable_locations):
            self.get_logger().warn("No desirable locations available.")
            return None, float('inf')  # No locations available

        cube_width, cube_length, cube_height = cube_dimensions
        half_width = cube_width / 2.0
        half_length = cube_length / 2.0
        half_height = cube_height / 2.0

        best_point = None
        best_cost = float('inf')  # Lower costs are better

        table_center_y = self.table_center[1]

        for cluster in desirable_locations:
            if not cluster:
                continue

            for point in cluster:
                point_array = np.array(point)

                # Define the cube bounds
                x_min = point_array[0] - half_width
                x_max = point_array[0] + half_width
                y_min = point_array[1] - half_length
                y_max = point_array[1] + half_length
                z_min = point_array[2] - half_height
                z_max = point_array[2] + half_height

                # Filter scene points inside the cube using NumPy logical indexing
                cube_mask = (
                    (scene_cloud[:, 0] >= x_min) & (scene_cloud[:, 0] <= x_max) &
                    (scene_cloud[:, 1] >= y_min) & (scene_cloud[:, 1] <= y_max) &
                    (scene_cloud[:, 2] >= z_min) & (scene_cloud[:, 2] <= z_max)
                )
                points_in_cube = scene_cloud[cube_mask]

                # Calculate cost as the summed-up distances to the centerline (x, y)
                if points_in_cube.size > 0:
                    # cost = np.sum(np.linalg.norm(points_in_cube[:, :2] - point_array[:2], axis=1)) - point_array[1]
                    cost =  (point_array[1]**2 + point_array[0]**2)**0.5
                    self.get_logger().info(f"-------------- cost: {cost}")
                else:
                    cost =  (point_array[1]**2 + point_array[0]**2)**0.5  # No points in the cube means no collision risk

                # Update the best point if this location has a lower cost
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

    def publish_best_point(self, best_point, padded_line, marker_topic="/best_point_marker"):
        """
        Publishes the best point as a vector with orientation perpendicular to the padded line.

        :param best_point: The selected best point (x, y, z).
        :param padded_line: The list of points defining the padded line.
        :param marker_topic: Topic to publish the marker for visualization.
        """
        start_time = time.time()
        if best_point is None:
            self.get_logger().warn("No best point to publish.")
            return

        if not padded_line or len(padded_line) < 2:
            self.get_logger().warn("Padded line is empty or has insufficient points.")
            return

        # Calculate the center of the padded line
        contour = [np.array([point[0], point[1]], dtype=np.float32) for point in padded_line]
        table_center = np.mean(contour, axis=0)

        # Calculate the direction vector from the table center to the best point
        direction_vector = np.array(table_center) - np.array(best_point[:2])
        direction_vector /= np.linalg.norm(direction_vector)  # Normalize the vector

        # Find the nearest segment of the padded line to determine the perpendicular direction
        closest_distance = float('inf')
        perpendicular_vector = None

        for i in range(len(padded_line) - 1):
            segment_start = np.array(padded_line[i][:2])
            segment_end = np.array(padded_line[i + 1][:2])

            line_vector = segment_end - segment_start
            line_vector /= np.linalg.norm(line_vector)  # Normalize the line vector

            # Calculate the distance from the best point to the segment
            proj_length = np.dot((np.array(best_point[:2]) - segment_start), line_vector)
            proj_point = segment_start + proj_length * line_vector
            distance = np.linalg.norm(np.array(best_point[:2]) - proj_point)

            if distance < closest_distance:
                closest_distance = distance
                # Perpendicular vector (rotate 90 degrees in the 2D plane)
                perpendicular_vector = np.array([-line_vector[1], line_vector[0], 0])

        if perpendicular_vector is None:
            self.get_logger().warn("No valid perpendicular vector found.")
            return

        # Create a quaternion for the orientation of the vector
        yaw_angle = np.arctan2(perpendicular_vector[1], perpendicular_vector[0])
        quaternion = tf_transformations.quaternion_from_euler(0, 0, yaw_angle)

        # Create a PoseStamped message in 'hesai_lidar' frame
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

        # Update the transform
        self.update_transform()

        if self.transform_hesai_to_map is None:
            self.get_logger().warn("Transform not available, cannot publish best point.")
            return

        # Transform the pose using the cached transform
        transformed_pose = do_transform_pose_stamped(pose_stamped, self.transform_hesai_to_map)

        table_center_distance = (self.table_center[0]**2 + self.table_center[1]**2)**0.5

        if transformed_pose:
            # self.get_logger().info(f"self.best_point_previous is {self.best_point_previous}")
            # self.get_logger().info(f"best point is {best_point}")
            # self.get_logger().info(f"table_center_y {table_center_y}")
            if table_center_distance > 2.5:
                self.transformed_pose_previous = transformed_pose
                # continue
            else:
                if self.transformed_pose_previous is None:
                    self.transformed_pose_previous = transformed_pose
                    self.get_logger().info(f"self.transformed_pose_previous is None")
                else:
                    transformed_pose = self.transformed_pose_previous
                    self.get_logger().info(f"best pose unchanged {self.transformed_pose_previous}")
            self.get_logger().info(f"Selected best transformed_pose: {transformed_pose}")
        else:
            self.get_logger().warn("No valid location found with a calculated cost.")

        # Publish the vector as a marker
        marker = Marker()
        marker.header.frame_id = "map"  # Use 'map' as the reference frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "best_point_vector"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose = transformed_pose.pose  # Use the transformed pose
        marker.scale.x = 0.3  # Length of the arrow shaft
        marker.scale.y = 0.06  # Width of the arrow
        marker.scale.z = 0.06  # Height of the arrow
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # Publish the marker
        self.best_point_marker_pub.publish(marker)

        # Publish the best point as a topic (also in 'map' frame)
        best_point_msg = transformed_pose.pose.position
        self.best_point_pub.publish(best_point_msg)
        end_time = time.time()
        duration = end_time - start_time
        self.get_logger().info(f"Published best point in 'map' frame: ({best_point_msg.x}, {best_point_msg.y}, {best_point_msg.z}) with orientation.")

    def create_collision_box(self, point, rotation_matrix, width, length, depth):
        """
        Creates the corners of a 3D collision box around a point, given the rotation and dimensions.

        :param point: 3D point (x, y, z)
        :param rotation_matrix: 3x3 rotation matrix for the box
        :param width: Box width
        :param length: Box length
        :param depth: Box depth
        :return: Corners of the 3D box (8 points)
        """
        # Define the eight corners of the box (relative to the center point)
        half_width = width / 2
        half_length = length / 2
        half_depth = depth / 2

        corners = np.array([
            [half_width, half_length, half_depth],
            [half_width, half_length, -half_depth],
            [half_width, -half_length, half_depth],
            [half_width, -half_length, -half_depth],
            [-half_width, half_length, half_depth],
            [-half_width, half_length, -half_depth],
            [-half_width, -half_length, half_depth],
            [-half_width, -half_length, -half_depth]
        ])

        # Rotate and translate the corners
        rotated_corners = (rotation_matrix @ corners.T).T
        translated_corners = rotated_corners + point

        return translated_corners


    def point_in_collision(self, cube_corners, scene_cloud):
        """
        Check if any point in the scene cloud is inside the convex hull of the cube corners.
        Uses a simple point-in-polygon check with ConvexHull filtering.

        :param cube_corners: Corners of the collision box
        :param scene_cloud: Point cloud of the scene
        :return: True if there's a collision, False otherwise
        """
        hull = ConvexHull(cube_corners)
        scene_points = np.array(scene_cloud)[:, :3]  # Extract (x, y, z) points from the scene cloud

        # Simple check if any points in the scene cloud are inside the convex hull
        for point in scene_points:
            if self.is_point_in_hull(point, hull):
                return True  # Collision detected

        return False  # No collision

    def is_point_in_hull(self, point, hull):
        """
        Check if a point is inside a convex hull.
        
        :param point: 3D point to check
        :param hull: ConvexHull object representing the polygon
        :return: True if the point is inside the hull, False otherwise
        """
        new_hull = ConvexHull(np.vstack((hull.points, point)))
        if np.array_equal(new_hull.vertices, hull.vertices):
            return False  # No new vertices were added, so the point is inside the hull
        return True  # A new vertex was added, so the point is outside the hull

    def publish_locations(self, locations, weights, heading):
        """
        Publish the filtered locations and their weights to the ROS topic.
        
        :param publisher: ROS publisher
        :param locations: List of placement locations (clusters of points)
        :param weights: List of weights for each location
        :param heading: List of orientation (Quaternion) for each location
        """
        msg = LocationArray()
        
        for i, cluster in enumerate(locations):
            for j, loc in enumerate(cluster):
                desired_loc = DesiredLocation()
                desired_loc.location.x = loc[0]
                desired_loc.location.y = loc[1]
                desired_loc.location.z = loc[2]
                desired_loc.location_weight = weights[i][j]
                desired_loc.heading = heading[i][j]  # Assuming heading is in Quaternion format
                msg.desired_locations.append(desired_loc)
        
        self.pub_locations.publish(msg)

    def publish_padded_lines(self, padded_polygon, header):
        """
        Publish the padded polygon lines as markers for visualization in RViz.
        
        :param padded_polygon: List of padded polygon vertices.
        :param header: The header to use for the marker message.
        """
        marker = Marker()
        marker.header = header
        marker.ns = 'padded_polygon'
        marker.id = 0
        marker.type = Marker.LINE_STRIP  # Use LINE_STRIP to represent a connected line
        marker.action = Marker.ADD
        marker.scale.x = 0.05  # Line width

        # Set the color (e.g., blue)
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0  # Fully opaque

        # Add the padded points to the marker
        for point in padded_polygon:
            p = Point()
            p.x = float(point[0])  # Ensure it's a Python float
            p.y = float(point[1])  # Ensure it's a Python float
            p.z = float(point[2])  # Ensure it's a Python float
            marker.points.append(p)

        # Close the loop by adding the first point to the end (if you want to close the polygon)
        if len(padded_polygon) > 0:
            first_point = Point()
            first_point.x = float(padded_polygon[0][0])
            first_point.y = float(padded_polygon[0][1])
            first_point.z = float(padded_polygon[0][2])
            marker.points.append(first_point)

        # Publish the marker to the padded_line topic
        self.padded_line_marker_pub.publish(marker)

    def publish_placements(self, placements):
        """
        Publish the placement points as a MarkerArray to visualize in RViz.
        
        :param placements: List of possible placement points (clusters of points).
        """
        # Clear old markers
        marker_array = MarkerArray()
        marker_id = 0

        for cluster in placements:
            for point in cluster:
                try:
                    # Lookup the latest available transform
                    transform = self.tf_buffer.lookup_transform('map', 'hesai_lidar', rclpy.time.Time())
                    transformed_point = self.transform_point(point, transform)
                except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException) as ex:
                    self.get_logger().warn(f"Transform error: {ex}")
                    continue  # Skip this point if transform is not available

                marker = Marker()
                marker.header.frame_id = "map"  # Assuming map is the reference frame
                marker.ns = 'possible_placements'
                marker.id = marker_id
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.scale.x = 0.05  # Small spheres for placement points
                marker.scale.y = 0.05
                marker.scale.z = 0.05

                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0  # Fully opaque

                marker.pose.position.x = transformed_point[0] # point[0]
                marker.pose.position.y = transformed_point[1] # point[1]
                marker.pose.position.z = transformed_point[2] # point[2]

                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0

                marker_array.markers.append(marker)
                marker_id += 1

        # Delete old markers not used in this update
        for old_marker_id in range(marker_id, self.previous_marker_count):
            delete_marker = Marker()
            delete_marker.header.frame_id = "map"
            delete_marker.ns = 'possible_placements'
            delete_marker.id = old_marker_id
            delete_marker.action = Marker.DELETE
            marker_array.markers.append(delete_marker)

        # Update the previous_marker_count
        self.previous_marker_count = marker_id

        # Publish the MarkerArray
        self.placements_marker_pub.publish(marker_array)

    def desirable_locations_callback(self, msg: LocationArray):
        marker_array = MarkerArray()
        marker_id = 0

        # Create a marker for each desired location
        for i, desired_loc in enumerate(msg.desired_locations):
            # Create a marker for the location (as a sphere, for example)
            marker = Marker()
            marker.header.frame_id = "hesai_lidar"  # or the appropriate frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'desirable_locations'
            marker.id = i
            marker.type = Marker.SPHERE  # Can also be ARROW or CUBE
            marker.action = Marker.ADD

            # Set the position and orientation
            marker.pose.position = desired_loc.location
            marker.pose.orientation = desired_loc.heading

            # Set the scale of the marker (size of the sphere/arrow)
            marker.scale.x = 0.15  # Width of the marker
            marker.scale.y = 0.15  # Height of the marker
            marker.scale.z = 0.15  # Depth of the marker

            # Set the color (green with some transparency)
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.8  # Alpha value for transparency

            # Add the marker to the array
            marker_array.markers.append(marker)
            marker_id += 1
        
        # Delete old markers not used in this update
        for old_marker_id in range(marker_id, self.previous_locations_count):
            delete_marker = Marker()
            delete_marker.header.frame_id = "hesai_lidar"
            delete_marker.ns = 'desirable_locations'
            delete_marker.id = old_marker_id
            delete_marker.action = Marker.DELETE
            marker_array.markers.append(delete_marker)

        # Update the previous_marker_count
        self.previous_locations_count = marker_id

        # Publish the marker array to RViz
        self.desirable_locations_marker_pub.publish(marker_array)


    def downsample_cloud_high(self, cloud_msg):
        # Create Open3D point cloud
        o3d_cloud = o3d.geometry.PointCloud()
        o3d_cloud.points = o3d.utility.Vector3dVector(cloud_msg)

        # Downsample the point cloud
        downsampled_cloud = o3d_cloud.voxel_down_sample(0.1)

        # Convert the downsampled Open3D point cloud back to the same type as points_list slicing
        # downsampled_points = [p for p in np.asarray(downsampled_cloud.points)]

        return np.asarray(downsampled_cloud.points)

    def cloud_to_transformed_cloud(self, cloud, source_frame, target_frame, stamp):
        try:
            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, stamp)
            return self.apply_transform(cloud, transform)
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as ex:
            self.get_logger().warn(f"Transform error: {ex}")
            return cloud

    def apply_transform(self, cloud, transform):
        points = cloud
        transform_matrix = self.transform_to_matrix(transform)
        transformed_points = (transform_matrix[:3, :3] @ points.T).T + transform_matrix[:3, 3]
        return transformed_points

    def transform_point(self, point, transform):
        """
        Transforms a 3D point using a given TransformStamped.

        :param point: A list or array with x, y, z coordinates.
        :param transform: A TransformStamped object.
        :return: Transformed point as a NumPy array [x, y, z].
        """
        # Convert the point to a homogeneous coordinate
        point_hom = np.array([point[0], point[1], point[2], 1.0])

        # Get the transformation matrix
        transform_matrix = self.transform_to_matrix(transform)

        # Apply the transformation
        transformed_point_hom = transform_matrix @ point_hom

        # Return the transformed point (first three components)
        return transformed_point_hom[:3]


    def get_camera_model(self, camera_info_msg):
        # Camera model creation
        camera_matrix = np.array(camera_info_msg.k).reshape((3, 3))
        distortion_coeffs = np.array(camera_info_msg.d)
        frame_id = camera_info_msg.header.frame_id
        return {'camera_matrix': camera_matrix, 'distortion_coeffs': distortion_coeffs, 'frame_id': frame_id}

    def process_points_with_bbox(self, cloud, detection2d_msg, detection_cloud_raw, cam_model):
        """
        Process all points in the cloud and project them to pixel coordinates.
        Check if the projected points are within the bounding box.
        """

        # Extract x, y, z from the cloud points as separate arrays
        x = cloud[:, 0]
        y = cloud[:, 1]
        z = cloud[:, 2]

        valid_indices = z > 0        # Filter points where z > 0 (points in front of the camera)

        x_valid = x[valid_indices]
        y_valid = y[valid_indices]
        z_valid = z[valid_indices]

        # Stack valid points as 3D coordinates in homogenous form
        points_3d = np.vstack((x_valid, y_valid, z_valid)).T

        # Project 3D points to 2D pixel coordinates (vectorized)
        pixel_coords = self.project_3d_to_pixel_matrix(points_3d, cam_model)

        # Check which pixel coordinates are inside the bounding box (vectorized)
        inside_bbox_mask = self.is_point_in_bbox_matrix(pixel_coords, detection2d_msg)

        # Filter out points that are inside the bounding box
        filtered_points = points_3d[inside_bbox_mask]

        # Add the filtered points to detection_cloud_raw
        detection_cloud_raw.extend(filtered_points.tolist())

    def project_3d_to_pixel_matrix(self, points_3d, cam_model):
        """
        Vectorized projection of 3D points to 2D pixel coordinates.
        """
        camera_matrix = cam_model['camera_matrix']

        # Apply the projection using matrix multiplication (3x3 camera matrix)
        pixel_coords_hom = points_3d @ camera_matrix.T

        # Normalize homogeneous coordinates to get pixel coordinates (u, v)
        u = pixel_coords_hom[:, 0] / pixel_coords_hom[:, 2]
        v = pixel_coords_hom[:, 1] / pixel_coords_hom[:, 2]

        return np.vstack((u, v)).T

    def is_point_in_bbox_matrix(self, pixel_coords, detection2d_msg):
        bbox = detection2d_msg.bbox

        u = pixel_coords[:, 0]
        v = pixel_coords[:, 1]

        u_min = bbox.center.position.x - bbox.size_x / 2
        u_max = bbox.center.position.x + bbox.size_x / 2
        v_min = bbox.center.position.y - bbox.size_y / 2
        v_max = bbox.center.position.y + bbox.size_y / 2

        # Check which points fall inside the bounding box
        inside_u = (u >= u_min) & (u <= u_max)
        inside_v = (v >= v_min) & (v <= v_max)

        return inside_u & inside_v

    def euclidean_cluster_extraction(self, cloud, header):
        if len(cloud) == 0:
            self.get_logger().warn("Cloud is empty, skipping clustering.")
            return None

        # Create Open3D point cloud
        o3d_cloud = o3d.geometry.PointCloud()
        o3d_cloud.points = o3d.utility.Vector3dVector(cloud)

        # Perform DBSCAN clustering
        labels = np.array(o3d_cloud.cluster_dbscan(eps=self.get_parameter("cluster_tolerance").get_parameter_value().double_value,
                                                min_points=self.get_parameter("min_cluster_size").get_parameter_value().integer_value,
                                                print_progress=True))

        # Find the cluster with the closest centroid
        min_distance = float('inf')
        closest_cluster_points = []

        for cluster_id in np.unique(labels):
            if cluster_id == -1:
                continue  # Ignore noise points

            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_points = np.asarray(o3d_cloud.select_by_index(cluster_indices).points)

            centroid = np.mean(cluster_points, axis=0)
            distance = np.linalg.norm(centroid)

            if distance < min_distance:
                min_distance = distance
                closest_cluster_points = cluster_points

        # Convert closest cluster points to PointCloud2 message
        if len(closest_cluster_points) > 0:
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]
            cloud_data = [tuple(point) for point in closest_cluster_points]
            closest_cluster_msg = point_cloud2.create_cloud(header, fields, cloud_data)
            return closest_cluster_points
        else:
            self.get_logger().warn("No valid clusters found.")
            return None

    def create_marker_array(self, detection3d_array_msg, duration):
        marker_array_msg = MarkerArray()

        for i, detection in enumerate(detection3d_array_msg.detections):
            marker = Marker()
            marker.header = detection3d_array_msg.header
            marker.ns = 'detection'
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose = detection.bbox.center
            marker.scale.x = detection.bbox.size.x
            marker.scale.y = detection.bbox.size.y
            marker.scale.z = detection.bbox.size.z

            detection_id = None
            if detection.results:
                detection_id = detection.results[0].hypothesis.class_id

            if detection_id == 'dining table':
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
                marker.color.a = 0.5
            else:
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 0.5

            sec = int(duration)
            nanosec = int((duration - sec) * 1e9)
            nanosec = min(max(nanosec, 0), 4294967295)

            marker.lifetime = Duration(sec=sec, nanosec=nanosec)

            marker_array_msg.markers.append(marker)

        return marker_array_msg

def main(args=None):
    rclpy.init(args=args)
    node = PlaneCloudPublisherNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()