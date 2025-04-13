# request multi goals and recieve a response with multi costs

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

        # 1) Get 2D polygon
        polygon_3d = self.findMinimumAreaShape(plane_cloud)
        self.get_logger().info(f"Detected polygon with {len(polygon_3d)} vertices.")

        # 2) Publish polygon
        self.publish_polygon_marker(polygon_3d, plane_cloud_msg.header)

        # 3) Build padded edges
        padding_offset = 0.9
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
        best_pose, best_cost = self.select_least_collision_risk(filtered_poses)
        if best_pose is not None:
            self.publish_best_point(best_pose, padded_edges)
        else:
            self.get_logger().warn("No valid placement pose found.")

        # 7) Publish all possible & filtered
        self.publish_possible_placements(possible_poses)
        self.publish_filtered_locations(filtered_poses)


    def convert_cloud_to_array(self, cloud_msg):
        points_gen = point_cloud2.read_points(cloud_msg, field_names=("x","y","z"), skip_nans=True)
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
    def find_possible_placements(self, padded_edges, location_sampling_distance):
        skip_distance = 0.3
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

                q = self.compute_orientation_from_nearest_edge(sample_xyz, padded_edges)

                pose_stamped = PoseStamped()
                pose_stamped.header.frame_id = 'map'  # these are in map frame
                pose_stamped.pose.position.x = float(sample_xyz[0])
                pose_stamped.pose.position.y = float(sample_xyz[1])
                pose_stamped.pose.position.z = float(sample_xyz[2])
                pose_stamped.pose.orientation = q

                cluster_list.append(pose_stamped)
            all_clusters.append(cluster_list)

        return all_clusters


    ###########################################################################
    # 4) Filter for collision
    ###########################################################################
    def filter_for_collision(self, pose_clusters, scene_tree, wheelchair_dims):
        collision_radius = 0.8
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
    def select_least_collision_risk(self, filtered_clusters):
        """
        1) Flatten the 'filtered_clusters' into a single list of candidate PoseStamped.
        2) Lookup the current robot pose (base_link -> map) as the 'start' pose.
        3) Call the C++ node's service '/multi_goal_plan' which returns:
        - A list of costs (one cost per goal in the same order).
        - success flag (boolean).
        4) Find the candidate with the lowest cost and return (best_pose, best_cost).
        """

        # 1) Flatten
        candidate_goals = []
        for cluster in filtered_clusters:
            candidate_goals.extend(cluster)  # accumulate them in one list

        if not candidate_goals:
            self.get_logger().warn("No candidate goals available for planning.")
            return None, float('inf')

        # --------------------------------------------
        # 2) Lookup start pose from TF (base_link->map)
        # --------------------------------------------
        try:
            transform_stamped = self.tf_buffer.lookup_transform(
                'map',         # target frame
                'base_link',   # source frame
                rclpy.time.Time()
            )
        except (LookupException, ExtrapolationException, TransformException) as ex:
            self.get_logger().error(f"Could not lookup transform 'base_link'->'map': {ex}")
            return None, float('inf')

        # Convert the transform to a PoseStamped in 'map' frame
        start_pose = PoseStamped()
        start_pose.header.frame_id = 'map'
        start_pose.header.stamp = transform_stamped.header.stamp  # or self.get_clock().now().to_msg()

        start_pose.pose.position.x = transform_stamped.transform.translation.x
        start_pose.pose.position.y = transform_stamped.transform.translation.y
        start_pose.pose.position.z = transform_stamped.transform.translation.z

        q = transform_stamped.transform.rotation
        start_pose.pose.orientation = Quaternion(x=q.x, y=q.y, z=q.z, w=q.w)

        # Print start pose for debugging
        self.get_logger().info(
            f"Start pose from TF (map->base_link): "
            f"({start_pose.pose.position.x:.2f}, "
            f"{start_pose.pose.position.y:.2f}, "
            f"{start_pose.pose.position.z:.2f})"
        )

        # Print candidate goals for debugging
        for i, goal_pose in enumerate(candidate_goals):
            gx = goal_pose.pose.position.x
            gy = goal_pose.pose.position.y
            gz = goal_pose.pose.position.z
            self.get_logger().info(f"Goal {i}: x={gx:.2f}, y={gy:.2f}, z={gz:.2f}")

        # ------------------------------------
        # 3) Build the service request
        # ------------------------------------
        from multi_goal_planner_interfaces.srv import MultiGoalPlan
        request = MultiGoalPlan.Request()
        request.start = start_pose
        request.goals = candidate_goals  # Our candidate poses (in 'map' frame)

        # 4) Call the service (blocking)
        self.get_logger().info(f"Calling /multi_goal_plan with {len(candidate_goals)} goals.")
        self.future = self.plan_cli.call_async(request)
        self.future.add_done_callback(self.plan_response_callback)

        # Check result
        if self.future.result() is None:
            self.get_logger().error("Service call to /multi_goal_plan failed!")
            return None, float('inf')

        response = self.future.result()
        if not response.success:
            self.get_logger().warn("No valid path found for any candidate goal (service returned success=False).")
            return None, float('inf')

        # ----------------------------------------------------------------
        # Now the response is assumed to have parallel arrays: goals, costs
        # or at least a 'costs' array in the same order as candidate_goals.
        # We find the minimum cost among them.
        # ----------------------------------------------------------------
        if len(response.costs) != len(candidate_goals):
            self.get_logger().error(
                f"Mismatch: received {len(response.costs)} costs, but {len(candidate_goals)} candidate goals."
            )
            return None, float('inf')

        min_cost_index = None
        min_cost_value = float('inf')
        for i, cost in enumerate(response.costs):
            if cost < min_cost_value:
                min_cost_value = cost
                min_cost_index = i

        if min_cost_index is None:
            self.get_logger().warn("No valid minimum cost found among candidate goals.")
            return None, float('inf')

        best_pose = candidate_goals[min_cost_index]
        best_cost = min_cost_value

        self.get_logger().info(
            f"/multi_goal_plan => best goal at "
            f"({best_pose.pose.position.x:.2f}, {best_pose.pose.position.y:.2f}), "
            f"cost={best_cost}"
        )

        return best_pose, best_cost

    def plan_response_callback(self, future):
        # This callback is invoked automatically once the service result arrives
        try:
            response = future.result()
            if not response or not response.success:
                self.get_logger().warn("No valid path found (service returned success=False).")
                return
            best_pose = response.best_goal
            best_cost = response.best_cost
            self.get_logger().info(
                f"/multi_goal_plan => best goal at "
                f"({best_pose.pose.position.x}, {best_pose.pose.position.y}), cost={best_cost}"
            )
            # You can now publish or store the best pose
            self.publish_best_point(best_pose, ...)
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

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

        # Also publish /best_point_topic (position only)
        best_point_msg = Point(
            x=transformed_pose.pose.position.x,
            y=transformed_pose.pose.position.y,
            z=transformed_pose.pose.position.z
        )
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
