# this node defines docking from bt nodes directives
# also includes parameter for handling bed and table dcking

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
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
from nav2_msgs.srv import GetCostmap
import math
import struct
import random
from std_msgs.msg import Header ,String
from builtin_interfaces.msg import Time
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# Action to stop goal momentarily 
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose

from action_msgs.srv import CancelGoal

# <-- Import your custom service interface
from multi_goal_planner_interfaces.srv import MultiGoalPlan
from std_srvs.srv import Trigger

from rclpy.parameter import Parameter
from rcl_interfaces.srv import GetParameters ,SetParameters
from rclpy.callback_groups import ReentrantCallbackGroup

# Update nav2 parameters via yaml files
import yaml


class PlaneCloudProcessorNode(Node):
    def __init__(self):
        super().__init__('plane_cloud_processor_node')



        # ------------------------------------------------------------
        #Auxuliar variables (declared before callback funcions)
        #-------------------------------------------------------------
        self.coll_cloud_pts = []
        self.coll_pts_calls = 0
        self.debug_collision= False
        self.docking_id=None
        self.mode= "Phase 3"
   
        # ------------------------------------------------------------
        # Service clients to read parameters from ultralytics_ros (important to have this before the callbacks)
        # ------------------------------------------------------------
        self.get_params_cli = self.create_client(GetParameters, "/tracker_with_cloud_node/get_parameters")
        while not self.get_params_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for 'tracker_with_cloud_node' service...")
        self.get_logger().info("'tracker_with_cloud_node' service client ready!")
        
        future = self.get_params_cli.call_async(GetParameters.Request(names=["GOAL"]))
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.docking_id = future.result().values[0].string_value
        else:
            self.get_logger().error("Failed to get docking_id parameter. Shutting down.")
            rclpy.shutdown()
            raise SystemExit("Failed to get docking_id parameter.")


        # Publishers
        self.polygon_pub = self.create_publisher(Marker, 'table_polygon', 1)
        self.padded_line_marker_pub = self.create_publisher(Marker, '/padded_line', 1)
        self.placements_marker_pub = self.create_publisher(MarkerArray, '/possible_placements', 1)
        self.filtered_placements_marker_pub = self.create_publisher(MarkerArray, '/filtered_placements', 1)
        self.best_point_marker_pub = self.create_publisher(Marker, "/best_point_marker", 1)
        self.filtered_placements_fprint_marker_pub = self.create_publisher(MarkerArray, '/filtered_placements_fprints', 1)
        self.coll_pts_pub = self.create_publisher(PointCloud2, '/collision_points', 1)



        # Subscribers
        qos_pc = QoSProfile( reliability=QoSReliabilityPolicy.BEST_EFFORT,history=QoSHistoryPolicy.KEEP_LAST,depth=1) # ensuring the newest message is read
        self.create_subscription(PointCloud2, 'plane_cloud', self.plane_cloud_callback, qos_pc)
        self.create_subscription(PointCloud2, '/lidar_filtered', self.lidar_filtered_callback, 1)
        self.create_subscription(String ,'mode', self.mode_callback, 1)

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
        #self.wheelchair_dimensions = [0.3, 0.8, 0.7] # [0.3, 0.8, 0.55]  # center to back, center to front, width

        # For convenience in marker frames
        self.marker_base = 'map'

        # Point clud accumulator
        self.raw_array_buffer=[]
        self.num_point_t=[]

        # Goal related variable
        self._goal_handle = None
        
        #Nav2 config files
        self.nav2_ph3_yaml='../../../../docker_projects/nav2_ws/nav2_params_humble_phase3.yaml'

        # ------------------------------------------------------------
        # Create a service client to the C++ "MultiGoalPlanner" node
        # ------------------------------------------------------------
        self.plan_cli = self.create_client(MultiGoalPlan, 'multi_goal_plan')
        while not self.plan_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for 'multi_goal_plan' service...")
        self.get_logger().info("'multi_goal_plan' service client ready!")


        # ------------------------------------------------------------
        # Create a service client to the Nav2 costmap service
        # ------------------------------------------------------------
        self.costmap_client = self.create_client(GetCostmap, '/global_costmap/get_costmap')

        if not self.costmap_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("Costmap service not available.")
            return

        self.costmap_future = self.costmap_client.call_async(GetCostmap.Request())
        self.costmap_future.add_done_callback(self.costmap_response_callback)

        while not self.costmap_future.done():
            rclpy.spin_once(self)
        

        def periodic_costmap_request(self):
            future = self.costmap_client.call_async(GetCostmap.Request())
            future.add_done_callback(self.costmap_response_callback)

        self.create_timer(1.0, lambda : periodic_costmap_request(self))    

        
        #------------------------------------------
        # Erase every marker in rviz
        #------------------------------------------
        marker = Marker()
        marker.action = Marker.DELETEALL
        self.polygon_pub.publish(marker)


        #------------------------------------------------------------
        # Docking parameters for each target
        #------------------------------------------------------------

        self.docking_params = {"59": {"min_edge":1.4 ,   #Bed
                                      "near_start": 0.2,
                                      "near_end": 0.2,
                                      "yaw": 145.0 ,
                                      "pad_edge": 0.72,
                                      "wheelc_dim": [0.3, 0.8, 0.8],
                                       "p3_thr_r": 4,
                                       "cr": 2.2, #2.5 , 1.3
                                       "ws_type": "linear"}, 
                                "60": {"min_edge":0.2 ,  #Table
                                       "near_start": 0.1,
                                       "near_end": 0.1,
                                       "yaw": 90.0,
                                       "pad_edge": 0.87,
                                       "wheelc_dim": [0.3, 0.8, 0.9],
                                       "p3_thr_r": 2.5,
                                       "cr": 1.75,
                                       "ws_type": "linear"},
                                "72": {"min_edge":0.2 ,  #Fridge
                                       "yaw": 90.0,
                                       "radius": 0.9,
                                       "wheelc_dim": [0.3, 0.8, 0.9],
                                       "p3_thr_r": 2.5,
                                       "rel_yaw": 3.14,
                                       "cr": 1.75,
                                       "ws_type": "door_approach"},
                                "62": {"min_edge":0.2 ,   #TV 62
                                       "yaw": 90.0,
                                       "radius": 1.75,
                                       "wheelc_dim": [0.3, 0.8, 0.9],
                                       "p3_thr_r": 4.0,
                                       "rel_yaw": 1.57,
                                       "centering": "Mid Front",
                                       "cr": 1.85,
                                       "ws_type": "arch"},
        } #pad edge 0.83





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

        self.raw_array_buffer.append(raw_array)
        self.num_point_t.append(num_points)

        if len(self.raw_array_buffer) < 1:
            self.get_logger().info("Waiting for second lidar message...")
            return

        raw_array=np.hstack(self.raw_array_buffer)
        num_points=sum(np.array(self.num_point_t))

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
        xyz_in_map[:,-1]=0 # Set z to 0 for 2D processing
        # 7) Build/update KD-tree
        if xyz_in_map.shape[0]> 1E3: # 6E3 is a threshold for performance

            self.scene_tree = cKDTree(xyz_in_map)
            self.latest_lidar_cloud = xyz_in_map
            #self.get_logger().info(f"Updated scene_tree with {xyz_in_map.shape[0]} points.")
        else:
            self.scene_tree = None
            self.latest_lidar_cloud = None
            #self.get_logger().warn("No valid points remain after transform.")
        
        self.raw_array_buffer.clear()
        self.num_point_t.clear()


    ###########################################################################
    # Plane callback -> polygon -> padded edges -> sample -> collision check
    ###########################################################################
    def plane_cloud_callback(self, plane_cloud_msg):
        self.get_logger().info("Received plane_cloud")
        if self.mode is None:
            self.get_logger().warn("Mode not set yet. Ignoring plane cloud.")
            return
        plane_cloud = self.convert_cloud_to_array(plane_cloud_msg)

        future = self.get_params_cli.call_async(GetParameters.Request(names=["GOAL"]))
        future.add_done_callback(self.get_params_callback)

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

        # Extract yaw from the transform
        robot_q= base_link_map_tf.transform.rotation
        robot_yaw = tf_transformations.euler_from_quaternion([robot_q.x, robot_q.y, robot_q.z, robot_q.w])[2]
        # froma base_link to center of table
        if self.mode == "Phase 2":

            self.get_logger().info("Phase 2:")
            self.get_logger().info(
            f"Distance={distance:.2f}m > 2.5m. Generating circle of radius=3.0m around table."
            )

            # 3a) Visualize the 3D circle around the table center
            circle_radius = self.docking_params[self.docking_id]["cr"]  # 1.9

            # 3b) Approximate that circle as a polygon (or boundary) for sampling
            #     We'll assume you have a helper like: create_circle_polygon(center, radius, num_segments)
            circle_polygon_3d = self.create_circle_polygon(self.table_center, circle_radius, 36)
            self.publish_padded_lines(circle_polygon_3d, plane_cloud_msg.header)

            # 3c) Generate possible placements around the circle
            location_sampling_distance = 0.2
            possible_poses_circle = self.find_possible_placements_circle(circle_polygon_3d, location_sampling_distance, distance_vector)

            # 3d) Filter them by collision
            if self.scene_tree is not None:
                filtered_poses_circle = self.filter_for_collision(
                    possible_poses_circle, 
                    self.scene_tree, 
                    self.docking_params[self.docking_id]["wheelc_dim"]
                )
            else:
                self.get_logger().warn("No scene data for collision. Using all possible placements.")
                filtered_poses_circle = possible_poses_circle

            # 3e) Publish them for visualization
            self.publish_possible_placements(possible_poses_circle)
            self.publish_filtered_locations(filtered_poses_circle)
            best_pose, best_cost = self.select_least_collision_risk(filtered_poses_circle, circle_polygon_3d)
        
        elif self.mode == "Phase 3":
            self.get_logger().info("Phase 3:")

            # 1) Filter outliers (example: local density)
            filtered_cloud = self.remove_outliers_radius(plane_cloud)

            # 1) Get 2D polygon
            polygon_3d = self.findMinimumAreaShape(filtered_cloud) #plane_cloud
            self.get_logger().info(f"Detected polygon with {len(polygon_3d)} vertices.")

            # 2) Publish polygon
            self.publish_polygon_marker(polygon_3d, plane_cloud_msg.header)

            # 3) Build padded edges (modify this function to handle fridge docking)
            if self.docking_params[self.docking_id]["ws_type"] == "linear":
                padding_offset = self.docking_params[self.docking_id]["pad_edge"]
                padded_edges = self.calculate_padded_edges(polygon_3d, padding_offset)
                # 4) Find possible placements (PoseStamped)
                location_sampling_distance = 0.1
                possible_poses = self.find_possible_placements(padded_edges, location_sampling_distance)
                
            elif self.docking_params[self.docking_id]["ws_type"] == "door_approach":
                target_pose = self.compute_pose_to_open_a_door(robot_yaw, robot_position[0], robot_position[1], polygon_3d)
                target_pose.header.stamp = plane_cloud_msg.header.stamp
                self.publish_best_point(target_pose, ...)
                return
            else:
                radius_arch = self.docking_params[self.docking_id]["radius"]
                center_arch = self.docking_params[self.docking_id]["centering"]
                padded_edges = self.create_arch_polygon(polygon_3d,
                                                        -distance_vector,
                                                        radius_arch,
                                                        center=center_arch)
                # 4) Find possible placements (PoseStamped)
                location_sampling_distance = 0.1
                possible_poses = self.find_possible_placements_circle(padded_edges, location_sampling_distance, distance_vector,self.docking_params[self.docking_id]["rel_yaw"])

            self.publish_padded_lines(padded_edges, plane_cloud_msg.header)

            

            # 5) Collision filtering
            if self.scene_tree is not None:
                filtered_poses = self.filter_for_collision(possible_poses,
                                                             self.scene_tree,
                                                             self.docking_params[self.docking_id]["wheelc_dim"]
                                                            )
            else:
                self.get_logger().warn("No scene data for collision. Using all possible placements.")
                filtered_poses = possible_poses

            # 6) Pick best pose (calls your C++ node via the service)
            best_pose, best_cost = self.select_least_collision_risk(filtered_poses, padded_edges)

            # 7) Publish all possible & filtered
            self.publish_possible_placements(possible_poses)
            self.publish_filtered_locations(filtered_poses)
        
        else:
            self.get_logger().info("Empty iteration.")
        # self.mode = None # reset mode after each iteration

    

    def get_params_callback(self , future):
            if future.result() is not None:
                self.docking_id = future.result().values[0].string_value
                self.get_logger().info(f"Docking ID set to: {self.docking_id}")
            else:
                self.get_logger().warn("Failed to get docking_id parameter.")

    def reload_nav2_conf(self, callbackFunc , client , param_dict):

        request = SetParameters.Request()
        request.parameters = [
            Parameter(name=k, value=v).to_parameter_msg()
            for k, v in param_dict.items()
        ]
        future = client.call_async(request)
        future.add_done_callback(callbackFunc)
        

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
        #hull_points = cv2.convexHull(points_2d)
        best_fitting_box= cv2.minAreaRect(points_2d)

        if best_fitting_box[1][0] < 0.1:
            bw = 0.1
        else:
            bw = best_fitting_box[1][0]
        if best_fitting_box[1][1] < 0.1:
            bh = 0.1
        else:
            bh = best_fitting_box[1][1]
        best_fitting_box= (best_fitting_box[0],(bw,bh),best_fitting_box[2])
        hull_points = cv2.boxPoints(best_fitting_box)
        epsilon = 0#0.05 * cv2.arcLength(hull_points, True)
        approx_polygon = cv2.approxPolyDP(hull_points, epsilon, True)
        
        polygon_3d = [np.array([pt[0][0], pt[0][1], avg_height]) for pt in approx_polygon]
        return polygon_3d

    def create_circle_polygon(self, center, radius, num_segments=36, arange= 6.28, aref=0):
        """
        Create a list of line segments forming a circle
        in the horizontal plane at z = center[2], 
        each segment is [[x1, y1, z1], [x2, y2, z2]].

        :param center: [cx, cy, cz], the center of the circle
        :param radius: float, the desired radius of the circle
        :param num_segments: int, how many segments to approximate the circle
        :param arange: float (0 to 2pi), poly angular range, 2pi is a circle, less ang is an arch
        :param aref: float (0 to 2pi), starting ang to build and arch, this var is irrelevant for building circles
        :return: List of edges, where each edge is [[x1, y1, z1], [x2, y2, z2]]
        """
        import math
        circle_edges = []
        cx, cy, cz = center[0], center[1], center[2]

        # Generate the points around the circle
        angle_step = arange/ (num_segments)
        circle_points = []
        for i in range(num_segments+1):
            theta = i * angle_step
            x = cx + radius * math.cos(theta+aref-arange/2)
            y = cy + radius * math.sin(theta+aref-arange/2)
            z = cz
            circle_points.append([x, y, z])
        

        # Convert the consecutive points into edges
        # e.g. [pt0->pt1, pt1->pt2, ..., pt(n-1)->pt0]
        for i in range(num_segments):
            p1 = circle_points[i]
            p2 = circle_points[(i + 1)]  
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

    ###########################################################################
    # 1) Expanded arch workspace
    ###########################################################################
    def create_arch_polygon(self, polygon, distance_vector, radius, arange = np.pi/6 , center="Mid Front" , num_segments=36):
        """
        Returns a list of edges, each edge is [ [x1, y1, z1], [x2, y2, z2] ].
        """
        # Copy + close
        polygon_copy = np.array(polygon[:])
        #polygon_copy.append(polygon_copy[0])

        # 2D center
        contour_2d = [np.array([pt[0], pt[1]]) for pt in polygon_copy[:-1]]
        contour_center = np.mean(polygon_copy, axis=0)
        local_corners =  polygon_copy - contour_center.reshape(1,-1)

        angle_TR = np.degrees(np.arctan2(distance_vector[1], distance_vector[0]))
        corner_theta =  [(np.degrees(np.arctan2(pt[1], pt[0])) - angle_TR + 360.0) % 360.0 for pt in local_corners]

        sorted_theta = sorted(corner_theta, key=lambda x: x)
        sorted_theta_ind = [corner_theta.index(t) for t in sorted_theta]
        print(sorted_theta_ind)
        sorted_polygon= polygon_copy[sorted_theta_ind]
        if center == "Centroid":
            c = np.mean(polygon_copy, axis=0)
        elif center == "Right Frontal Vertex":
            c =  sorted_polygon[0]
        if center == "Mid Front":
            c = np.mean(sorted_polygon[[0,-1]], axis=0)
        ang_ref = np.radians (angle_TR+ (sorted_theta[0] + sorted_theta[-1])/2-180)
        return self.create_circle_polygon(c, radius , arange = arange , aref = ang_ref,num_segments=6)



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
    def compute_orientation_from_nearest_edge(self, point_3d, padded_edges, yaw_rel = 1.57):
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

        perp = np.array([best_line_dir[0]*np.cos(yaw_rel)-best_line_dir[1]*np.sin(yaw_rel),
                         best_line_dir[0]*np.sin(yaw_rel)+best_line_dir[1]*np.cos(yaw_rel)])
        yaw = np.arctan2(perp[1], perp[0])
        q = tf_transformations.quaternion_from_euler(0, 0, yaw)
        return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])


    ###########################################################################
    # 3) Sample edges -> list-of-lists of PoseStamped
    ###########################################################################
    def find_possible_placements_circle(self, padded_edges, location_sampling_distance, distance_vector , yaw_rel= 1.57):
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

                # judge if the candidate is in the closer quarter circle
                candidate_vector = sample_xyz - self.table_center
                dot_product = np.dot(candidate_vector, distance_vector)
                cos_alfa= dot_product/(np.linalg.norm(candidate_vector)* np.linalg.norm(distance_vector))
                
                print(f"Cosine of angle {cos_alfa:.2f}")

                if True: #(dot_product < 0) and (abs(cos_alfa) > 0.86):  # cos(30 degrees) = 0.86

                    q = self.compute_orientation_from_nearest_edge(sample_xyz, padded_edges,yaw_rel)

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
            #MIN_EDGE_LENGTH = 0.2 #1.4 for bed
            MIN_EDGE_LENGTH = self.docking_params[self.docking_id]["min_edge"] #1.4 for bed


            # Distances (meters) near each edge extreme in which we want to generate poses
            #NEAR_START_DIST = 0.1 # 0.3 for bed
            NEAR_START_DIST = self.docking_params[self.docking_id]["near_start"]

            #NEAR_END_DIST   = 0.1 # 0.6 for bed
            NEAR_END_DIST   = self.docking_params[self.docking_id]["near_end"] 

            # Docking orientation
            #YAW= 90.0  # degrees, bearing to the edge 30.0 for bed
            YAW = self.docking_params[self.docking_id]["yaw"]  # degrees, bearing to the edge 30.0 for bed

            for edge in padded_edges:
                # If the edge isn’t exactly two endpoints, skip
                if len(edge) != 2:
                    all_clusters.append([])
                    continue

                p1 = np.array(edge[0])
                p2 = np.array(edge[1])
                vec = p2 - p1
                length = np.linalg.norm(vec)

                # Only consider edges in [1.5, 2.0]
                if not (MIN_EDGE_LENGTH <= length):
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
                end_ratio   = NEAR_END_DIST / length  # clamp if edge < 0.5m

                if (1-end_ratio) > start_ratio:
                    # Decide how many samples we want
                    valid_len   = (1 - end_ratio - start_ratio) * length
                    num_samples = int(valid_len / location_sampling_distance)
                    num_samples = max(num_samples, 1)  # at least 1 pose

                    for i in range(num_samples):
                        alpha = start_ratio + i * (1 - end_ratio - start_ratio) / (num_samples - 1)
                        sample_xyz = p1 + alpha * vec

                        # Yaw = edge_yaw + 30° bearing
                        if i <num_samples // 2:
                            yaw_30 = edge_yaw + YAW
                        else:
                            yaw_30 = edge_yaw + YAW # (180.0-YAW)

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
                else:
                    self.get_logger().warn(f"Edge condition are wrong: {start_ratio} > {end_ratio}. Skipping edge.")

                all_clusters.append(cluster_list)

            return all_clusters

    def compute_pose_to_open_a_door(self, robot_yaw , robot_x ,robot_y ,polygon_3d):

        # ─────────────────────────────────────────────
        # 2.  Find table center and edges
        # ─────────────────────────────────────────────
        pol_3d_cpy=np.array(polygon_3d)
        edges = np.array ([pol_3d_cpy[i]-pol_3d_cpy[i+1] for i in range(2)])
        cent = np.mean(pol_3d_cpy,axis=0)
        xr_axis = np.array([np.cos(robot_yaw), np.sin(robot_yaw), 0.0])

        XrDotAxpc0 = xr_axis.dot(edges[0])
        XrDotAxpc1 = xr_axis.dot(edges[1])

        if abs(XrDotAxpc0) > abs(XrDotAxpc1):
            xpc_axis = edges[0] * (-1.0 if XrDotAxpc0 > 0 else 1.0)
            ty = np.linalg.norm(edges[1])
        else:
            xpc_axis = edges[1] * (-1.0 if XrDotAxpc1 > 0 else 1.0)
            ty = np.linalg.norm(edges[0])

        xpc_axis/=np.linalg.norm(xpc_axis)
        ypc_axis= np.array([xpc_axis[1],-xpc_axis[0],xpc_axis[2]]) # pointing left
        # ─────────────────────────────────────────────
        # 3.  Define shoulder position and approach length (these are the key parameters for the door approach workspace)
        # ─────────────────────────────────────────────
        sx  = 0.05    # shoulder on x about base link
        sy  = 0.2    # shoulder on y about base link
        al  = 0.75   # 0.54   # approach length

        # ─────────────────────────────────────────────
        # 4.  Compute the pose (position + orientation) for the robot to open the door, based on the geometry and the predefined approach length
        # ─────────────────────────────────────────────
        b     = -ty / (2 * sy)
        a     = (al + sx) / (sy)
        gamma = -np.arcsin(b / np.sqrt(1 + a**2)) + np.arctan(a**-1)
        alpha = np.arctan(sx / sy)

        dx1 = np.cos(gamma) * al
        dx2 = np.sin(gamma + alpha) * np.sqrt(sx**2 + sy**2)
        dx  = dx1 + dx2
        print(f"gamma: {np.rad2deg(gamma):.2f}°")
        print(dx1, dx2, dx)
        print(sy*np.cos(gamma) - (al+sx)*np.sin(gamma)+ty/2)
        pose_xyz = cent + xpc_axis * (ty / 2 + dx) + ypc_axis * 0.2 # moving the pose 20cm to the left to prevent colission during approaching
        pc_yaw   = np.arctan2(-xpc_axis[1], -xpc_axis[0])
        goal_yaw = gamma + pc_yaw + np.pi/9       # orientation around Z (pose_q as a yaw angle) plus correction

        # ─────────────────────────────────────────────
        # 5.  Convert the computed pose into a PoseStamped message for ROS
        # ─────────────────────────────────────────────
        q = tf_transformations.quaternion_from_euler(0, 0, goal_yaw)
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = 'map'
        pose_stamped.pose.position.x = float(pose_xyz[0])
        pose_stamped.pose.position.y = float(pose_xyz[1])
        pose_stamped.pose.position.z = 0.0
        pose_stamped.pose.orientation.x = q[0]
        pose_stamped.pose.orientation.y = q[1]
        pose_stamped.pose.orientation.z = q[2]
        pose_stamped.pose.orientation.w = q[3]
        return pose_stamped

    def quaternion_from_yaw(self, yaw_degrees):
        """
        Helper function to convert a yaw (in degrees) about Z-axis
        into a quaternion in (x, y, z, w) order.
        """
        yaw_radians = math.radians(yaw_degrees)
        # Roll = 0, Pitch = 0, Yaw = yaw_radians
        q = tf_transformations.quaternion_from_euler(0, 0, yaw_radians)
        return q


    ###########################################################################
    # 4) Filter for collision
    ###########################################################################
    def filter_for_collision(self, pose_clusters, scene_tree, wheelchair_dims):
        print("                       ")

        collision_radius = 0.3
        self.get_logger().info(f"Filtering collisions using radius {collision_radius}")
        dside= wheelchair_dims[2] / 2.0
        dfront = wheelchair_dims[1]
        dback = wheelchair_dims[0]
        footprint_loc = np.array([[-dback,-dside,0],[-dback,dside,0],[dfront,dside,0],[dfront,-dside,0]], dtype=np.float32)


        filtered_clusters = []
        
        if self.debug_collision:
            marker_array = MarkerArray()
            delete_marker = Marker()
            delete_marker.action = Marker.DELETEALL
            marker_array.markers.append(delete_marker)
            self.filtered_placements_fprint_marker_pub.publish(marker_array)
            marker_id=0
            cl_pts_msg = PointCloud2()
            self.publish_coll_pts([],reset=True)

        for c_idx, cluster in enumerate(pose_clusters):
            valid_poses = []
            footprints_in_map = []
            for pose_stamped in cluster:
                x = pose_stamped.pose.position.x
                y = pose_stamped.pose.position.y
                z = pose_stamped.pose.position.z

                q = pose_stamped.pose.orientation
                quat = [q.x, q.y, q.z, q.w]

                # Get rotation matrix from quaternion
                transform_matrix = tf_transformations.quaternion_matrix(quat)


                # Transform footprint to map frame
                transform_matrix[0, 3] = x
                transform_matrix[1, 3] = y
                transform_matrix[2, 3] = z
                ones_col = np.ones((footprint_loc.shape[0], 1), dtype=np.float32)
                footprint_hom = np.hstack((footprint_loc, ones_col))  # shape (N,4)
                footprint_map_hom = (transform_matrix @ footprint_hom.T).T
                

                
                

                # Local x-axis (forward direction) in world frame
                forward_vector = transform_matrix[:3, 0]  # x-axis of pose frame
                side_vector = transform_matrix[:3, 1]  # y-axis of pose frame

                # Center of the ball for collision check at the middle front of the footprint
                shift_distance = dfront-collision_radius #0.5
                center_ball_front = [x + forward_vector[0] * shift_distance,
                            y + forward_vector[1] * shift_distance,
                            0.0]

                # shift_distance_2 = -0.5
                # shifted_pos_2 = [x + forward_vector[0] * shift_distance_2,
                #             y + forward_vector[1] * shift_distance_2,
                #             0.0]

                shift_distance = dfront-collision_radius-0.2 # 0.3
                shift_distance_side = dside/2-collision_radius #0.05  #0.15
                center_ball_left = [x + shift_distance * forward_vector[0] + side_vector[0] * shift_distance_side,
                            y + shift_distance * forward_vector[1] + side_vector[1] * shift_distance_side,
                            0.0]    
                center_ball_right = [x + shift_distance * forward_vector[0] - side_vector[0] * shift_distance_side,
                            y + shift_distance * forward_vector[1] - side_vector[1] * shift_distance_side,
                            0.0]

                
                idxs = scene_tree.query_ball_point([x,y,0], collision_radius,workers=-1)
                idxs += scene_tree.query_ball_point(center_ball_front, collision_radius,workers=-1)
                # idxs += scene_tree.query_ball_point(shifted_pos_2, collision_radius,workers=-1)
                idxs += scene_tree.query_ball_point(center_ball_left, collision_radius,workers=-1)
                idxs += scene_tree.query_ball_point(center_ball_right, collision_radius,workers=-1)

                self.get_logger().info(f"Cluster {c_idx}, pose ({x:.2f}, {y:.2f}, {z:.2f}). Shifted ({center_ball_front[0]:.2f}, {center_ball_front[1]:.2f}, {center_ball_front[2]:.2f}). ")
                if  not idxs and not self.is_pose_occupied(pose_stamped,  self.latest_global_costmap, fprint_l=footprint_loc):
                    valid_poses.append(pose_stamped)
                elif idxs and self.debug_collision:
                    cl_pts_msg= self.publish_coll_pts(idxs)
                fp=self.fprint_pot(pose_stamped,  self.latest_global_costmap, fprint_l=footprint_loc)
                #self.get_logger().info(f"Footprint size : {fp}")
                footprints_in_map += [fp]#[footprint_map_hom[:, :3]]

            filtered_clusters.append(valid_poses)

            #publish the footprints for visualization
            if valid_poses and self.debug_collision:
                marker = Marker()
                marker.header.frame_id = 'map'
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD
                marker.scale.x = 0.05
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0

                for i , fprint in enumerate(footprints_in_map):
                    marker.points= []
                    marker.ns = f'footprint {c_idx}{i}'
                    marker.id = marker_id
                    for point in fprint:
                        marker.points.append(Point(x=float(point[0]), y=float(point[1]), z=float(point[2])))
                    #Close the loop
                    marker.points.append(Point(x=float(fprint[0][0]),
                                               y=float(fprint[0][1]),
                                               z=float(fprint[0][2])))
                    marker_array.markers.append(marker)
                    marker_id += 1

                

            self.get_logger().info(f"Cluster {c_idx}: {len(valid_poses)} valid out of {len(cluster)}")
        
        if self.debug_collision:
            self.filtered_placements_fprint_marker_pub.publish(marker_array)
            self.coll_pts_pub.publish(cl_pts_msg)


        return filtered_clusters

    def publish_coll_pts(self, point_idxs, reset=False):
        """
        Converts a list of 3D points into a sensor_msgs/msg/PointCloud2 message for ROS 2.

        Args:
            points (list of [x, y, z]): Points to include.
            add (bool): Whether to add to the previous message or start fresh.

        Returns:
            sensor_msgs.msg.PointCloud2: The resulting point cloud message.
        """

        if reset:
            self.coll_cloud_pts = []
            self.coll_pts_calls = 0
            return

        # Generate deterministic but unique color per call
        random.seed(self.coll_pts_calls)
        r = random.randint(50, 255)
        g = random.randint(50, 255)
        b = random.randint(50, 255)
        a = 255
        rgba = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]


        

        # Add points with color
        for indxs in point_idxs:
            pt = self.scene_tree.data[indxs]
            self.coll_cloud_pts.append([pt[0], pt[1], pt[2], rgba])

        # Create header
        now = rclpy.time.Time().to_msg()
        header = Header()
        header.stamp = now
        header.frame_id = "map"

        # Define fields
        fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgba', offset=12, datatype=PointField.UINT32, count=1),
        ]

        point_step = 16
        data = b''.join([
            struct.pack('fffI', x, y, z, rgba)
            for x, y, z, rgba in self.coll_cloud_pts
        ])

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(self.coll_cloud_pts)
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = point_step
        msg.row_step = point_step * len(self.coll_cloud_pts)
        msg.is_dense = True
        msg.data = data

        self.coll_pts_calls += 1
        return msg

    def is_pose_occupied(self, pose_stamped, costmap, fprint_l=np.array([[0,1],[0,-1],[1,1],[1,-1]]) , lethal_threshold=255):
        info = costmap.metadata
        res = info.resolution
        width = info.size_x
        height = info.size_y
        origin = info.origin.position
        data = np.array(costmap.data).reshape((height, width))




        x = pose_stamped.pose.position.x
        y = pose_stamped.pose.position.y
        mx = int((x - origin.x) / res)
        my = int((y - origin.y) / res)
        
        q=[pose_stamped.pose.orientation.x,pose_stamped.pose.orientation.y,pose_stamped.pose.orientation.z,pose_stamped.pose.orientation.w]
        _ ,_ , yaw = tf_transformations.euler_from_quaternion(q)

        cyaw = np.cos(yaw)
        syaw = np.sin(yaw)

        u_cells_back = int(fprint_l[0,0] / res)
        u_cells_front = int(fprint_l[2,0] / res)
        v_cells_side =  -int(fprint_l[0,1] / res)
        
        dx_du= cyaw
        dy_du= syaw
        dx_dv= -syaw
        dy_dv= cyaw
        
        costs = []
        for du in range(u_cells_back, u_cells_front + 1):
            for dv in range(-v_cells_side, v_cells_side + 1):
                dx = int((dx_du* (du*res) + dx_dv * (dv*res))/res)
                dy = int(( dy_du * (du*res) + dy_dv * (dv*res))/res)

                cx, cy = mx + dx, my + dy
                if 0 <= cx < width and 0 <= cy < height:
                    cost = data[cy, cx]
                    costs.append(cost)
                    if cost >= lethal_threshold:
                        return True
        self.get_logger().info(f"Max cost in footprint: {max(costs)}")
        return False

    def fprint_pot(self, pose_stamped, costmap, fprint_l=np.array([[0,1],[0,-1],[1,1],[1,-1]])):
        info = costmap.metadata
        res = info.resolution
        width = info.size_x
        height = info.size_y
        origin = info.origin.position
        data = np.array(costmap.data).reshape((height, width))




        x = pose_stamped.pose.position.x
        y = pose_stamped.pose.position.y
        mx = (x - origin.x)
        my = (y - origin.y)

        q=[pose_stamped.pose.orientation.x,pose_stamped.pose.orientation.y,pose_stamped.pose.orientation.z,pose_stamped.pose.orientation.w]
        _ ,_ , yaw = tf_transformations.euler_from_quaternion(q)

        cyaw = np.cos(yaw)
        syaw = np.sin(yaw)
        u_cells_back = fprint_l[0,0] 
        u_cells_front = fprint_l[2,0]
        v_cells_side =  -fprint_l[0,1] 
        
        dx_du= cyaw
        dy_du= syaw
        dx_dv= -syaw
        dy_dv= cyaw
        
        costs = []
        fp=[]
        for du in [u_cells_back, u_cells_front ]:
            for dv in [v_cells_side, -v_cells_side ]:
                dx = (dx_du* (du) + dx_dv * (dv))
                dy = ( dy_du * (du) + dy_dv * (dv))
                fp.append([x + dx, y + dy , 0])

        return np.array(fp)

    def costmap_response_callback(self, future):
        try:
            response = future.result()
            self.latest_global_costmap = response.map  # Store for use later
            info=self.latest_global_costmap.metadata
            if info.size_x == 0 or info.size_y == 0:
                self.get_logger().warn("Received empty costmap.")
            else:
                # self.get_logger().info(f"Received costmap with size: {info.size_x}x{info.size_y}")
                # self.get_logger().info(f"Costmap resolution: {info.resolution}m, origin: ({info.origin.position.x}, {info.origin.position.y})")
                pass
        except Exception as e:
            self.get_logger().error(f"Failed to get costmap: {e}")
            self.latest_global_costmap = None


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
        return cost * 15 #8

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

        self.best_point_marker_pub.publish(marker)


    def mode_callback(self, msg):
        self.mode = msg.data
        self.get_logger().info(f"Mode changed to: {self.mode}")


def main(args=None):
    rclpy.init(args=args)
    node = PlaneCloudProcessorNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
