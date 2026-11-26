# this pose generator iteration, handles different nac config for phase 2 and 3
# also includes parameter for handling bed and table dcking

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


from tf2_ros import LookupException, ExtrapolationException, TransformException

import math

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# Action to stop goal momentarily 


from action_msgs.srv import CancelGoal

# <-- Import your custom service interface
from rcl_interfaces.srv import SetParameters
from rclpy.callback_groups import ReentrantCallbackGroup

# Update nav2 parameters via yaml files
import yaml


class PlaneCloudProcessorNode(Node):
    def __init__(self):
        super().__init__('plane_cloud_processor_node')
        
        # Publishers
        self.polygon_pub = self.create_publisher(Marker, '/table_marker_rel', 1)
        self.polygon2_pub = self.create_publisher(Marker, '/robot_marker_rel', 1)




        # Subscribers
        qos_pc = QoSProfile( reliability=QoSReliabilityPolicy.BEST_EFFORT,history=QoSHistoryPolicy.KEEP_LAST,depth=1) # ensuring the newest message is read
        self.create_subscription(PointCloud2, '/lidar_RAM_segm', self.plane_cloud_callback, qos_pc)


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



        self.lmap_par_cli= self.create_client(
            srv_type=SetParameters,  # This was missing
            srv_name='/local_costmap/local_costmap/set_parameters',
            callback_group=ReentrantCallbackGroup()
        )

        self.lmap_pars= {
            "inflation_layer.inflation_radius":0.35,
            #"voxel_layer.z_resolution":0.05,
            #"voxel_layer.z_voxels":16,
            "voxel_layer.scan.clearing":True,
            }

        self.cont_par_cli= self.create_client(
            srv_type=SetParameters,  # This was missing
            srv_name='/controller_server/set_parameters',
            callback_group=ReentrantCallbackGroup()
        )

        self.cont_pars = {
            "FollowPath.vx_max": 0.5,
            "FollowPath.vx_min": -0.5,
            "FollowPath.vx_std": 0.3,
            "FollowPath.wz_max": 0.8,
            "FollowPath.wz_std": 0.6,
            "FollowPath.ConstraintCritic.cost_weight": 5.0,
            "FollowPath.GoalProgressCritic.enabled": False,
            "FollowPath.GoalCritic.cost_weight": 8.0,
            "FollowPath.GoalCritic.threshold_to_consider": 0.3,
            "FollowPath.GoalAngleCritic.cost_weight": 11.0,
            "FollowPath.GoalAngleCritic.cost_power": 1.0,
            "FollowPath.GoalAngleCritic.threshold_to_consider": 0.3,
            "FollowPath.ObstaclesCritic.inflation_radius": 0.2,
            "FollowPath.CostCritic.cost_weight": 3.0,
            "FollowPath.PathAlignCritic.cost_weight": 10.0,
            "FollowPath.PathAlignCritic.offset_from_furthest": 15,
            "FollowPath.PathFollowCritic.cost_weight": 10.0,
            "FollowPath.PathFollowCritic.offset_from_furthest": 7,
            "FollowPath.PathAngleCritic.cost_weight": 1.0,
            "FollowPath.PathAngleCritic.enabled": False,
            "FollowPath.PathAngleCritic.offset_from_furthest": 20,
        }
        #17/10/2025
        # obs inflation radius: 0.35 -> 0.2
        # path allign critic.offset: 20->10  
        # bed pad: 0.65->0.72
        # gcrit off: 0.3->0.1
        # gang off: 0.4->0.1
        #cr bed: 1.3 -> 1.7
        # yaw : 150 -> 145

        # table  goalangle weight 22 , power2 / goal progress true goalcrit threshold: 
        # bed goalangle weight 10 , power1 / goal progress false
        #self.cont_pars = {"FollowPath.vx_max":0.5,"FollowPath.vx_min":-0.5,"FollowPath.vx_std":0.3,"FollowPath.wz_max":1.0,"FollowPath.wz_std":0.8}

        

        

        self._cancel_rviz_goals_client = self.create_client(CancelGoal, 'navigate_to_pose/_action/cancel_goal')

        # ------------------------------------------------------------
        #Auxuliar variables
        #-------------------------------------------------------------

        self.can_trig_init=True
        self.can_assess_init=False
        self.phase3_triggered = False # prevents moving from phase3 to phase 2
        self.phase2_skip = False # make phase3 to read plane cloud once
        self.phase3_skip= False
        self.coll_cloud_pts = []
        self.coll_pts_calls = 0
        self.debug_collision= True
        self.rviz_goal_cancelled = False # dummy flag to trigger servo phase
        self.docking_id="59"

        #------------------------------------------
        # Erase every marker in rviz
        #------------------------------------------
        marker = Marker()
        marker.action = Marker.DELETEALL
        self.polygon_pub.publish(marker)


        #------------------------------------------------------------
        # Docking parameters for each target
        #------------------------------------------------------------

        self.docking_params = {"59": {"min_edge":1.4 ,
                                      "near_start": 0.2,
                                      "near_end": 0.2,
                                      "yaw": 145.0 ,
                                      "pad_edge": 0.72,
                                      "wheelc_dim": [0.3, 0.8, 0.8],
                                       "p3_thr_r": 4,
                                       "cr": 2.2}, #2.5 , 1.3
                                "60": {"min_edge":0.2 , 
                                       "near_start": 0.1,
                                       "near_end": 0.1,
                                       "yaw": 90.0,
                                       "pad_edge": 0.87,
                                       "wheelc_dim": [0.3, 0.8, 0.9],
                                       "p3_thr_r": 2.5,
                                       "cr": 1.75},
        } #pad edge 0.83





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


        # froma base_link to center of table

                

        self.get_logger().info("Phase 3:")
        
        # 1) Filter outliers (example: local density)
        filtered_cloud = self.remove_outliers_radius(plane_cloud)

        # 1) Get 2D polygon
        polygon_3d = self.findMinimumAreaShape(filtered_cloud) #plane_cloud
        self.get_logger().info(f"Detected polygon with {len(polygon_3d)} vertices.")

        # 2) Publish polygon
        self.publish_polygon_marker(polygon_3d, plane_cloud_msg.header)

       



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
        return best_fitting_box

    def publish_polygon_marker(self, polygon_3d, header):

        now = self.get_clock().now().to_msg()

         #---table_Marker---

        q_table = tf_transformations.quaternion_from_euler(0, 0, polygon_3d[2]*3.14/180, axes='sxyz')
        m_table = Marker()
        m_table.header.stamp = header.stamp
        m_table.header.frame_id = 'map'
        m_table.ns = 'table'
        m_table.id = 0
        m_table.type = Marker.CUBE
        m_table.action = Marker.ADD
        # tabel_position
        m_table.pose.position.x = polygon_3d[0][0]
        m_table.pose.position.y = polygon_3d[0][1]
        m_table.pose.position.z = 0.0

        m_table.pose.orientation.x = q_table[0]
        m_table.pose.orientation.y = q_table[1]
        m_table.pose.orientation.z = q_table[2]
        m_table.pose.orientation.w = q_table[3]
        # table_size
        m_table.scale.x = polygon_3d[1][0]
        m_table.scale.y = polygon_3d[1][1]
        m_table.scale.z = 1.0

        m_table.color.r = 0.0
        m_table.color.g = 0.0
        m_table.color.b = 1.0
        m_table.color.a = 1.0
        m_table.lifetime = Duration(sec=0, nanosec=0)

        last_transform_time = None

        # 2) Lookup start pose
        try:
            transform_stamped = self.tf_buffer.lookup_transform(
                'map',         # target frame
                'base_link',   # source frame
                rclpy.time.Time()
            )
        except (LookupException, ExtrapolationException, TransformException) as ex:
            self.get_logger().error(f"Could not lookup transform 'base_link'->'map': {ex}")
            return 
        
        m_robot = Marker()
        m_robot.header.stamp = header.stamp
        m_robot.header.frame_id = 'map'
        m_robot.ns = 'table'
        m_robot.id = 0
        m_robot.type = Marker.CUBE
        m_robot.action = Marker.ADD
        # tabel_position
        m_robot.pose.position.x = transform_stamped.transform.translation.x
        m_robot.pose.position.y = transform_stamped.transform.translation.y
        m_robot.pose.position.z = transform_stamped.transform.translation.z

        m_robot.pose.orientation.x = transform_stamped.transform.rotation.x
        m_robot.pose.orientation.y = transform_stamped.transform.rotation.y
        m_robot.pose.orientation.z = transform_stamped.transform.rotation.z
        m_robot.pose.orientation.w = transform_stamped.transform.rotation.w
        # table_size
        m_robot.scale.x = 1.0
        m_robot.scale.y = 1.0
        m_robot.scale.z = 1.0

        m_robot.color.r = 0.0
        m_robot.color.g = 1.0
        m_robot.color.b = 0.0
        m_robot.color.a = 1.0
        m_robot.lifetime = Duration(sec=0, nanosec=0)


        self.polygon_pub.publish(m_table)
        self.polygon2_pub.publish(m_robot)

   

    def quaternion_from_yaw(self, yaw_degrees):
        """
        Helper function to convert a yaw (in degrees) about Z-axis
        into a quaternion in (x, y, z, w) order.
        """
        yaw_radians = math.radians(yaw_degrees)
        # Roll = 0, Pitch = 0, Yaw = yaw_radians
        q = tf_transformations.quaternion_from_euler(0, 0, yaw_radians)
        return q


   
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



def main(args=None):
    rclpy.init(args=args)
    node = PlaneCloudProcessorNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
