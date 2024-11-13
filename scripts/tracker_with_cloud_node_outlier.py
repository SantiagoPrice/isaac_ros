#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, CameraInfo
from vision_msgs.msg import Detection3DArray
from vision_msgs.msg import Detection3D, BoundingBox3D
from geometry_msgs.msg import Pose, Point, Quaternion
from ultralytics_ros.msg import YoloResult
from visualization_msgs.msg import MarkerArray, Marker
from sensor_msgs_py import point_cloud2
import numpy as np
import message_filters
import tf2_ros
import tf_transformations
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import PointCloud2, PointField
import pcl
from builtin_interfaces.msg import Duration
import cv2


class TrackerWithCloudNode(Node):

    def __init__(self):
        super().__init__('tracker_with_cloud_node')

        # Declare and get parameters
        self.declare_parameter('camera_info_topic', 'camera_info')
        self.declare_parameter('lidar_topic', 'points_raw')
        self.declare_parameter('yolo_result_topic', 'yolo_result')
        self.declare_parameter('yolo_3d_result_topic', 'yolo_3d_result')
        self.declare_parameter('cluster_tolerance', 0.5)
        self.declare_parameter('voxel_leaf_size', 0.5)
        self.declare_parameter('min_cluster_size', 100)
        self.declare_parameter('max_cluster_size', 25000)

        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        lidar_topic = self.get_parameter('lidar_topic').get_parameter_value().string_value
        yolo_result_topic = self.get_parameter('yolo_result_topic').get_parameter_value().string_value

        # Message filters to synchronize topics
        self.camera_info_sub = message_filters.Subscriber(self, CameraInfo, camera_info_topic)
        self.lidar_sub = message_filters.Subscriber(self, PointCloud2, lidar_topic)
        self.yolo_result_sub = message_filters.Subscriber(self, YoloResult, yolo_result_topic)

        self.sync = message_filters.ApproximateTimeSynchronizer([self.camera_info_sub, self.lidar_sub, self.yolo_result_sub], 500, 2.0)
        self.sync.registerCallback(self.sync_callback)

        # Publishers
        yolo_3d_result_topic = self.get_parameter('yolo_3d_result_topic').get_parameter_value().string_value
        self.detection3d_pub = self.create_publisher(Detection3DArray, yolo_3d_result_topic, 1)
        self.detection_cloud_pub = self.create_publisher(PointCloud2, 'detection_cloud', 1)
        self.marker_pub = self.create_publisher(MarkerArray, 'detection_marker', 1)
        self.plane_cloud_pub = self.create_publisher(PointCloud2, 'plane_cloud', 1)  # Publisher for detected plane clouds
        self.polygon_pub = self.create_publisher(Marker, 'table_polygon', 1)

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

        downsampled_cloud = self.downsample_cloud(cloud_msg)

        # Use the camera model for projection
        cam_model = self.get_camera_model(camera_info_msg)
        transformed_cloud = self.cloud_to_transformed_cloud(downsampled_cloud, cloud_msg.header.frame_id, cam_model['frame_id'], cloud_msg.header.stamp)
        self.get_logger().info(f"transformed_cloud: {len(transformed_cloud)}")

        # Initialize message objects
        detection3d_array_msg = Detection3DArray()

        # Project points and create bounding boxes
        detection_cloud_msg = self.project_cloud(transformed_cloud, yolo_result_msg, cloud_msg.header, detection3d_array_msg, cam_model)

        marker_array_msg = self.create_marker_array(detection3d_array_msg, callback_interval.nanoseconds / 1e9)

        self.get_logger().info(f"Number of 3D detections: {len(detection3d_array_msg.detections)}")

        # Publish the results
        self.detection3d_pub.publish(detection3d_array_msg)
        if detection_cloud_msg is not None:
            self.detection_cloud_pub.publish(detection_cloud_msg)
        else:
            self.get_logger().warn("detection_cloud_msg is None, skipping publish.")
        self.marker_pub.publish(marker_array_msg)

    def project_cloud(self, cloud, yolo_result_msg, header, detection3d_array_msg, cam_model):
        detection3d_array_msg.header = header
        detection3d_array_msg.header.stamp = yolo_result_msg.header.stamp

        self.get_logger().info(f"Number of detections: {len(yolo_result_msg.detections.detections)}")

        # List to collect all the closest detection clouds
        all_detection_clouds = []
        detected_planes_cloud = []  # List to store plane points for 'dining table'

        for detection in yolo_result_msg.detections.detections:
            detection_cloud_raw = []
            self.process_points_with_bbox(cloud, detection, detection_cloud_raw, cam_model)

            if detection_cloud_raw:
                transformed_cloud = self.cloud_to_transformed_cloud(np.array(detection_cloud_raw), cam_model['frame_id'], header.frame_id, header.stamp)
                closest_detection_cloud = self.euclidean_cluster_extraction(transformed_cloud, header)

                # Extract the detection ID to check if it's 'dining table'
                detection_id = None
                if detection.results:
                    detection_id = detection.results[0].hypothesis.class_id  # Assuming results[0] contains the detection name or id

                # Only process the cloud for plane detection if the detection is for 'dining table'
                if detection_id == 'dining table':
                    self.get_logger().info("Processing plane for 'dining table' detection.")
                    
                    # Detect the plane using RANSAC
                    plane_cloud = self.detect_plane(closest_detection_cloud)

                    if plane_cloud is not None:
                        detected_planes_cloud.extend(plane_cloud)  # Add detected plane to the list

                        # Convert detected plane to minimum area shape (polygon)
                        table_polygon_3d = self.findMinimumAreaShape(np.array(plane_cloud))
                        self.get_logger().info(f"Detected table polygon has {len(table_polygon_3d)} vertices.")

                        # Publish the polygon as a visualization marker in RViz
                        self.publish_polygon_marker(table_polygon_3d, header)

                all_detection_clouds.extend(transformed_cloud)

                # Create bounding box regardless of the detection ID
                self.create_bounding_box(detection3d_array_msg, closest_detection_cloud, detection.results)

        # Combine all the collected closest detection clouds into one point cloud
        if all_detection_clouds:
            combined_cloud_array = np.array(all_detection_clouds, dtype=np.float32)

            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]

            # Create cloud data by converting the combined array to a list of tuples (x, y, z)
            cloud_data = [tuple(point[:3]) for point in combined_cloud_array]

            # Create PointCloud2 message with correct format
            detection_cloud_msg = point_cloud2.create_cloud(header, fields, cloud_data)

            # Publish the detected planes if any and the detection ID matches 'dining table'
            if detected_planes_cloud:
                detected_planes_cloud_array = np.array(detected_planes_cloud, dtype=np.float32)
                plane_cloud_data = [tuple(point[:3]) for point in detected_planes_cloud_array]
                plane_cloud_msg = point_cloud2.create_cloud(header, fields, plane_cloud_data)
                self.plane_cloud_pub.publish(plane_cloud_msg)

            return detection_cloud_msg
        else:
            self.get_logger().warn("No valid points in detection clouds.")
            return None



    def detect_plane(self, cloud):
        """Detect a plane in the given PointCloud2 cloud using RANSAC."""
        # Convert PointCloud2 to a list of points with x, y, z
        points = list(point_cloud2.read_points(cloud, field_names=("x", "y", "z"), skip_nans=True))

        if len(points) == 0:
            self.get_logger().warn("PointCloud2 is empty, skipping plane detection.")
            return None

        # Convert the list of points into a NumPy array (Nx3) for PCL processing
        points_array = np.array([[p[0], p[1], p[2]] for p in points], dtype=np.float32)

        # Convert the numpy array to a PCL PointCloud
        pcl_cloud = pcl.PointCloud(points_array)

        # Set up RANSAC for plane segmentation
        seg = pcl_cloud.make_segmenter()
        seg.set_model_type(pcl.SACMODEL_PLANE)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_distance_threshold(0.01)

        # Perform segmentation to extract inliers (plane points)
        indices, coefficients = seg.segment()

        if len(indices) == 0:
            self.get_logger().info("No plane detected in the point cloud.")
            return None

        self.get_logger().info(f"Plane detected with {len(indices)} inliers.")

        # Extract plane points
        plane_points = pcl_cloud.extract(indices)
        return np.array(plane_points)

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



    def downsample_cloud(self, cloud_msg):
        points_gen = point_cloud2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)

        # Convert the generator into a list of points
        points_list = np.array([[p[0], p[1], p[2]] for p in points_gen], dtype=np.float32)

        # Perform voxel grid downsampling (this is a simple subsample for now)
        voxel_leaf_size = self.get_parameter('voxel_leaf_size').get_parameter_value().double_value

        # Apply downsampling
        downsampled_points = points_list[::int(1/voxel_leaf_size)]  # Simple downsampling by taking every nth point

        return downsampled_points

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

    def transform_to_matrix(self, transform):
        # Convert TransformStamped to a 4x4 transformation matrix
        translation = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
        rotation = [transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]

        # Create a 4x4 transformation matrix from the quaternion
        matrix = tf_transformations.quaternion_matrix(rotation)
        matrix[:3, 3] = translation  # Set the translation part of the matrix

        return matrix

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

        # Filter points where z > 0 (points in front of the camera)
        valid_indices = z > 0
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
        pcl_cloud = pcl.PointCloud(np.array(cloud, dtype=np.float32))

        cluster_tolerance = self.get_parameter("cluster_tolerance").get_parameter_value().double_value
        min_cluster_size = self.get_parameter("min_cluster_size").get_parameter_value().integer_value
        max_cluster_size = self.get_parameter("max_cluster_size").get_parameter_value().integer_value

        tree = pcl_cloud.make_kdtree()

        ec = pcl_cloud.make_EuclideanClusterExtraction()
        ec.set_ClusterTolerance(cluster_tolerance)
        ec.set_MinClusterSize(min_cluster_size)
        ec.set_MaxClusterSize(max_cluster_size)
        ec.set_SearchMethod(tree)
        
        cluster_indices = ec.Extract()

        min_distance = np.finfo(np.float32).max
        closest_cluster_points = []

        for indices in cluster_indices:
            cluster_points = []

            for i in indices:
                cluster_points.append(pcl_cloud[i])

            cluster_points_np = np.array(cluster_points, dtype=np.float32)
            centroid = np.mean(cluster_points_np, axis=0)
            distance = np.linalg.norm(centroid)

            if distance < min_distance:
                min_distance = distance
                closest_cluster_points = cluster_points_np

        closest_cluster_np = np.array(closest_cluster_points)

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud_data = [tuple(p[:3]) for p in closest_cluster_np]

        closest_cluster_msg = point_cloud2.create_cloud(header, fields, cloud_data)

        return closest_cluster_msg

    def create_bounding_box(self, detection3d_array_msg, cloud, results):
        points = list(point_cloud2.read_points(cloud, field_names=("x", "y", "z"), skip_nans=True))

        cloud_array = np.array([[p[0], p[1], p[2]] for p in points], dtype=np.float32)

        cloud = cloud_array

        if cloud.size == 0:
            self.get_logger().warn("Cloud is empty, skipping bounding box creation.")
            return

        centroid = np.mean(cloud, axis=0)
        theta = -np.arctan2(centroid[1], np.sqrt(centroid[0]**2 + centroid[2]**2))

        transform_matrix = tf_transformations.quaternion_matrix(
            tf_transformations.quaternion_about_axis(theta, [0, 0, 1])
        )
        
        transformed_cloud = np.dot(transform_matrix[:3, :3], cloud.T).T + transform_matrix[:3, 3]

        min_pt = np.min(transformed_cloud, axis=0)
        max_pt = np.max(transformed_cloud, axis=0)

        transformed_bbox_center = (min_pt + max_pt) / 2

        bbox_center = np.dot(tf_transformations.inverse_matrix(transform_matrix), np.append(transformed_bbox_center, 1))[:3]

        rotation_quat = tf_transformations.quaternion_from_matrix(tf_transformations.inverse_matrix(transform_matrix))

        detection3d_msg = Detection3D()

        detection3d_msg.bbox = BoundingBox3D()
        detection3d_msg.bbox.center = Pose()
        detection3d_msg.bbox.center.position = Point()
        detection3d_msg.bbox.center.position.x = bbox_center[0]
        detection3d_msg.bbox.center.position.y = bbox_center[1]
        detection3d_msg.bbox.center.position.z = bbox_center[2]

        detection3d_msg.bbox.center.orientation = Quaternion()
        detection3d_msg.bbox.center.orientation.x = rotation_quat[0]
        detection3d_msg.bbox.center.orientation.y = rotation_quat[1]
        detection3d_msg.bbox.center.orientation.z = rotation_quat[2]
        detection3d_msg.bbox.center.orientation.w = rotation_quat[3]

        detection3d_msg.bbox.size.x = max_pt[0] - min_pt[0]
        detection3d_msg.bbox.size.y = max_pt[1] - min_pt[1]
        detection3d_msg.bbox.size.z = max_pt[2] - min_pt[2]

        detection3d_msg.results = results

        detection3d_array_msg.detections.append(detection3d_msg)

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
    node = TrackerWithCloudNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
