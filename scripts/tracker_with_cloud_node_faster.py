#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, CameraInfo, PointField
from vision_msgs.msg import Detection3DArray, Detection3D, BoundingBox3D
from geometry_msgs.msg import Pose, Point, Quaternion
from ultralytics_ros.msg import YoloResult
from visualization_msgs.msg import MarkerArray, Marker
from sensor_msgs_py import point_cloud2
import numpy as np
import message_filters
import tf2_ros
import tf_transformations
from cv_bridge import CvBridge
from builtin_interfaces.msg import Duration
import pcl

class TrackerWithCloudNode(Node):

    def __init__(self):
        super().__init__('tracker_with_cloud_node')

        # Fetch parameters once at initialization to avoid repeated lookups
        self.camera_info_topic = self.declare_parameter('camera_info_topic', 'camera_info').get_parameter_value().string_value
        self.lidar_topic = self.declare_parameter('lidar_topic', 'points_raw').get_parameter_value().string_value
        self.yolo_result_topic = self.declare_parameter('yolo_result_topic', 'yolo_result').get_parameter_value().string_value
        self.yolo_3d_result_topic = self.declare_parameter('yolo_3d_result_topic', 'yolo_3d_result').get_parameter_value().string_value
        self.cluster_tolerance = self.declare_parameter('cluster_tolerance', 0.5).get_parameter_value().double_value
        self.voxel_leaf_size = self.declare_parameter('voxel_leaf_size', 0.5).get_parameter_value().double_value
        self.min_cluster_size = self.declare_parameter('min_cluster_size', 100).get_parameter_value().integer_value
        self.max_cluster_size = self.declare_parameter('max_cluster_size', 25000).get_parameter_value().integer_value

        # Message filters to synchronize topics
        self.camera_info_sub = message_filters.Subscriber(self, CameraInfo, self.camera_info_topic)
        self.lidar_sub = message_filters.Subscriber(self, PointCloud2, self.lidar_topic)
        self.yolo_result_sub = message_filters.Subscriber(self, YoloResult, self.yolo_result_topic)
        self.sync = message_filters.ApproximateTimeSynchronizer([self.camera_info_sub, self.lidar_sub, self.yolo_result_sub], 100, 0.1)
        self.sync.registerCallback(self.sync_callback)

        # Publishers
        self.detection3d_pub = self.create_publisher(Detection3DArray, self.yolo_3d_result_topic, 1)
        self.detection_cloud_pub = self.create_publisher(PointCloud2, 'detection_cloud', 1)
        self.marker_pub = self.create_publisher(MarkerArray, 'detection_marker', 1)

        # TF Buffer and Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.bridge = CvBridge()
        self.last_call_time = self.get_clock().now()

    def sync_callback(self, camera_info_msg, cloud_msg, yolo_result_msg):
        current_call_time = self.get_clock().now()
        callback_interval = current_call_time - self.last_call_time
        self.last_call_time = current_call_time

        downsampled_cloud = self.downsample_cloud(cloud_msg)

        # Use the camera model for projection
        cam_model = self.get_camera_model(camera_info_msg)
        transformed_cloud = self.cloud_to_transformed_cloud(downsampled_cloud, cloud_msg.header.frame_id, cam_model['frame_id'], cloud_msg.header.stamp)

        # Initialize message objects
        detection3d_array_msg = Detection3DArray()

        # Project points and create bounding boxes
        detection_cloud_msg = self.project_cloud(transformed_cloud, yolo_result_msg, cloud_msg.header, detection3d_array_msg, cam_model)

        marker_array_msg = self.create_marker_array(detection3d_array_msg, callback_interval.nanoseconds / 1e9)

        self.get_logger().info(f"Number of 3D detections: {len(detection3d_array_msg.detections)}")

        # Publish the results
        self.detection3d_pub.publish(detection3d_array_msg)
        # self.detection_cloud_pub.publish(detection_cloud_msg)
        if detection_cloud_msg is not None:
            self.detection_cloud_pub.publish(detection_cloud_msg)
        else:
            self.get_logger().warn("detection_cloud_msg is None, skipping publish.")
        self.marker_pub.publish(marker_array_msg)

    def downsample_cloud(self, cloud_msg):
        # Directly convert point cloud into NumPy array and downsample using voxel grid size
        points_gen = point_cloud2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
        # points_list = np.array([p[:3] for p in points_gen], dtype=np.float32)
        points_list = np.array([[p[0], p[1], p[2]] for p in points_gen], dtype=np.float32)

        # Apply downsampling (subsample points with step size according to voxel_leaf_size)
        step = max(1, int(1 / self.voxel_leaf_size))  # Ensure step is at least 1
        return points_list[::step]

    def cloud_to_transformed_cloud(self, cloud, source_frame, target_frame, stamp):
        try:
            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, stamp)
            return self.apply_transform(cloud, transform)
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as ex:
            self.get_logger().warn(f"Transform error: {ex}")
            return cloud

    def apply_transform(self, cloud, transform):
        # Use NumPy for fast matrix multiplication for the cloud transformation
        transform_matrix = self.transform_to_matrix(transform)
        return np.dot(cloud, transform_matrix[:3, :3].T) + transform_matrix[:3, 3]

    def transform_to_matrix(self, transform):
        # Convert TransformStamped to a 4x4 transformation matrix
        translation = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])
        rotation = np.array([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w])
        matrix = tf_transformations.quaternion_matrix(rotation)
        matrix[:3, 3] = translation
        return matrix

    def get_camera_model(self, camera_info_msg):
        # Create the camera model from the camera info message (cached)
        camera_matrix = np.array(camera_info_msg.k).reshape((3, 3))
        distortion_coeffs = np.array(camera_info_msg.d)
        frame_id = camera_info_msg.header.frame_id
        return {'camera_matrix': camera_matrix, 'distortion_coeffs': distortion_coeffs, 'frame_id': frame_id}

    def project_cloud(self, cloud, yolo_result_msg, header, detection3d_array_msg, cam_model):
        detection3d_array_msg.header = header
        detection3d_array_msg.header.stamp = yolo_result_msg.header.stamp
        all_detection_clouds = []

        for detection in yolo_result_msg.detections.detections:
            detection_cloud_raw = []
            self.process_points_with_bbox(cloud, detection, detection_cloud_raw, cam_model)

            if detection_cloud_raw:
                transformed_cloud = self.cloud_to_transformed_cloud(np.array(detection_cloud_raw), "camera_color_left", header.frame_id, header.stamp)
                closest_detection_cloud = self.euclidean_cluster_extraction(transformed_cloud, header)
                all_detection_clouds.extend(transformed_cloud)
                self.create_bounding_box(detection3d_array_msg, closest_detection_cloud, detection.results)

        if all_detection_clouds:
            return self.create_pointcloud2_msg(header, all_detection_clouds)
        else:
            self.get_logger().warn("No valid points in detection clouds.")
            return None

    def create_pointcloud2_msg(self, header, cloud_data):
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        return point_cloud2.create_cloud(header, fields, [tuple(point[:3]) for point in cloud_data])

    def process_points_with_bbox(self, cloud, detection2d_msg, detection_cloud_raw, cam_model):
        bbox = detection2d_msg.bbox
        for point in cloud:
            if point[2] > 0:  # z > 0 check for valid points
                uv = self.project_3d_to_pixel(point[0], point[1], point[2], cam_model)
                if self.is_point_in_bbox(uv, bbox):
                    detection_cloud_raw.append(point)

    def project_3d_to_pixel(self, x, y, z, cam_model):
        point = np.array([x, y, z])
        pixel_coords = cam_model['camera_matrix'] @ point
        return int(pixel_coords[0] / pixel_coords[2]), int(pixel_coords[1] / pixel_coords[2])

    def is_point_in_bbox(self, uv, bbox):
        u, v = uv
        return (bbox.center.position.x - bbox.size_x / 2 <= u <= bbox.center.position.x + bbox.size_x / 2) and \
               (bbox.center.position.y - bbox.size_y / 2 <= v <= bbox.center.position.y + bbox.size_y / 2)

    def euclidean_cluster_extraction(self, cloud, header):
        pcl_cloud = pcl.PointCloud(cloud.astype(np.float32))
        tree = pcl_cloud.make_kdtree()
        ec = pcl_cloud.make_EuclideanClusterExtraction()
        ec.set_ClusterTolerance(self.cluster_tolerance)
        ec.set_MinClusterSize(self.min_cluster_size)
        ec.set_MaxClusterSize(self.max_cluster_size)
        ec.set_SearchMethod(tree)
        cluster_indices = ec.Extract()

        min_distance = np.inf
        closest_cluster_points = []

        for indices in cluster_indices:
            cluster_points = cloud[indices]
            centroid = np.mean(cluster_points, axis=0)
            distance = np.linalg.norm(centroid)
            if distance < min_distance:
                min_distance = distance
                closest_cluster_points = cluster_points

                # Convert closest_cluster_points back to a NumPy array
        closest_cluster_np = np.array(closest_cluster_points)

        # Create PointCloud2 message using sensor_msgs_py.point_cloud2
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud_data = [tuple(p[:3]) for p in closest_cluster_np]  # Ensure (x, y, z) format

        # Create PointCloud2 message
        closest_cluster_msg = point_cloud2.create_cloud(header, fields, cloud_data)

        return closest_cluster_msg
        # return closest_cluster_points if len(closest_cluster_points) > 0 else []

    def create_bounding_box(self, detection3d_array_msg, cloud, results):
        points = list(point_cloud2.read_points(cloud, field_names=("x", "y", "z"), skip_nans=True))
    
        # Convert the points to a NumPy array
        cloud_array = np.array([[p[0], p[1], p[2]] for p in points], dtype=np.float32)

        cloud = cloud_array

        # Check if cloud is empty
        if cloud.size == 0:
            self.get_logger().warn("Cloud is empty, skipping bounding box creation.")
            return

        # Ensure cloud is a 2D array
        if len(cloud.shape) != 2 or cloud.shape[1] != 3:
            self.get_logger().error("Invalid cloud shape. Expected (N, 3), but got shape: {}".format(cloud.shape))
            return

        # Compute the centroid of the point cloud
        centroid = np.mean(cloud, axis=0)  # Equivalent to pcl::compute3DCentroid
        theta = -np.arctan2(centroid[1], np.sqrt(centroid[0]**2 + centroid[2]**2))  # Compute the yaw angle

        # Create the transformation matrix for rotation around the Z axis (yaw)
        transform_matrix = tf_transformations.quaternion_matrix(
            tf_transformations.quaternion_about_axis(theta, [0, 0, 1])
        )
        
        # Transform the point cloud
        transformed_cloud = np.dot(transform_matrix[:3, :3], cloud.T).T + transform_matrix[:3, 3]

        # Compute the minimum and maximum points in the transformed cloud
        min_pt = np.min(transformed_cloud, axis=0)
        max_pt = np.max(transformed_cloud, axis=0)

        # Compute the center of the bounding box in the transformed space
        transformed_bbox_center = (min_pt + max_pt) / 2

        # Transform the center back to the original coordinate space
        bbox_center = np.dot(tf_transformations.inverse_matrix(transform_matrix), np.append(transformed_bbox_center, 1))[:3]

        # Get the inverse quaternion for the orientation
        rotation_quat = tf_transformations.quaternion_from_matrix(tf_transformations.inverse_matrix(transform_matrix))

        # Fill in the Detection3D message
        detection3d_msg = Detection3D()

        # Set the center of the bounding box
        detection3d_msg.bbox = BoundingBox3D()
        detection3d_msg.bbox.center = Pose()
        detection3d_msg.bbox.center.position = Point()
        detection3d_msg.bbox.center.position.x = bbox_center[0]
        detection3d_msg.bbox.center.position.y = bbox_center[1]
        detection3d_msg.bbox.center.position.z = bbox_center[2]

        # Set the orientation (quaternion)
        detection3d_msg.bbox.center.orientation = Quaternion()
        detection3d_msg.bbox.center.orientation.x = rotation_quat[0]
        detection3d_msg.bbox.center.orientation.y = rotation_quat[1]
        detection3d_msg.bbox.center.orientation.z = rotation_quat[2]
        detection3d_msg.bbox.center.orientation.w = rotation_quat[3]

        # Set the size of the bounding box
        detection3d_msg.bbox.size.x = max_pt[0] - min_pt[0]
        detection3d_msg.bbox.size.y = max_pt[1] - min_pt[1]
        detection3d_msg.bbox.size.z = max_pt[2] - min_pt[2]

        # Attach the detection results (classification/confidence)
        detection3d_msg.results = results

        # Add the detection to the Detection3DArray
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
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.5

            # Calculate seconds and nanoseconds, ensuring nanoseconds are within bounds
            sec = int(duration)
            nanosec = int((duration - sec) * 1e9)

            # Ensure nanosec is non-negative and within the valid range for uint32
            nanosec = min(max(nanosec, 0), 4294967295)

            # Assign the lifetime to the marker using Duration
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
