// replace visual servoing with intermedium docking pose for the 2nd phase
// add bed recognition
#include "tracker_with_cloud_node/tracker_with_cloud_node.h"
#include <pcl/common/transforms.h>
#include <deque>
#include <utility>    // std::pair
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2/LinearMath/Quaternion.h>

TrackerWithCloudNode::TrackerWithCloudNode() 
  : rclcpp::Node("tracker_with_cloud_node"),
    // plane_memory_duration_(1, 0) // store planes for 1 second
    plane_memory_duration_{rclcpp::Duration::from_seconds(0.6)}

{
  // ---------------------------
  // Declare and retrieve params
  // ---------------------------
  this->declare_parameter<std::string>("camera_info_topic", "camera_info");
  this->declare_parameter<std::string>("lidar_topic", "points_raw");
  this->declare_parameter<std::string>("yolo_result_topic", "yolo_result");
  this->declare_parameter<std::string>("yolo_3d_result_topic", "yolo_3d_result");
  this->declare_parameter<float>("cluster_tolerance", 0.5f);
  this->declare_parameter<float>("voxel_leaf_size", 0.05f);
  this->declare_parameter<int>("min_cluster_size", 100);
  this->declare_parameter<int>("max_cluster_size", 25000);

  this->get_parameter("camera_info_topic", camera_info_topic_);
  this->get_parameter("lidar_topic", lidar_topic_);
  this->get_parameter("yolo_result_topic", yolo_result_topic_);
  this->get_parameter("yolo_3d_result_topic", yolo_3d_result_topic_);

  // ---------------------------
  // Subscribers
  // ---------------------------
  camera_info_sub_.subscribe(this, camera_info_topic_);
  lidar_sub_.subscribe(this, lidar_topic_);
  yolo_result_sub_.subscribe(this, yolo_result_topic_);

  // Approximate sync
  sync_ = std::make_shared<message_filters::Synchronizer<ApproximateSyncPolicy>>(
            ApproximateSyncPolicy(500));
  sync_->connectInput(camera_info_sub_, lidar_sub_, yolo_result_sub_);
  sync_->registerCallback(std::bind(&TrackerWithCloudNode::syncCallback,
                                    this,
                                    std::placeholders::_1,
                                    std::placeholders::_2,
                                    std::placeholders::_3));

  // ---------------------------
  // Publishers
  // ---------------------------
  detection3d_pub_ = this->create_publisher<vision_msgs::msg::Detection3DArray>(
                        yolo_3d_result_topic_, 1);
  detection_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
                            "detection_cloud", 1);
  marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
                  "detection_marker", 1);
  plane_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("plane_cloud", 1);
  goal_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
      "/rviz_goal",   // or your chosen topic name
      10             // queue size
  );
  visual_servo_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(
    "/cmd_vel", 
    10
  );

  last_call_time_ = this->now();

  // TF Buffer
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
}

void TrackerWithCloudNode::syncCallback(
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info_msg,
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg,
  const ultralytics_ros::msg::YoloResult::ConstSharedPtr& yolo_result_msg)
{
  rclcpp::Time current_call_time = this->now();
  rclcpp::Duration callback_interval = current_call_time - last_call_time_;
  last_call_time_ = current_call_time;

  // 1) Downsample
  pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  downsampled_cloud = downsampleCloudMsg(cloud_msg);

  // 2) Camera model
  cam_model_.fromCameraInfo(camera_info_msg);

  // 3) Transform cloud -> camera frame
  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  transformed_cloud = cloud2TransformedCloud(
                        downsampled_cloud,
                        cloud_msg->header.frame_id,
                        cam_model_.tfFrame(),
                        cloud_msg->header.stamp);

  // 4) Build detection3d array + detection cloud
  vision_msgs::msg::Detection3DArray detection3d_array_msg;
  sensor_msgs::msg::PointCloud2 detection_cloud_msg;

  projectCloud(transformed_cloud,
               yolo_result_msg,
               cloud_msg->header,
               detection3d_array_msg,
               detection_cloud_msg);

  // 5) Publish results
  visualization_msgs::msg::MarkerArray marker_array_msg =
      createMarkerArray(detection3d_array_msg, callback_interval.seconds());

  detection3d_pub_->publish(detection3d_array_msg);
  detection_cloud_pub_->publish(detection_cloud_msg);
  marker_pub_->publish(marker_array_msg);

  RCLCPP_INFO(this->get_logger(), "syncCallback");
}

void TrackerWithCloudNode::projectCloud(
  const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
  const ultralytics_ros::msg::YoloResult::ConstSharedPtr& yolo_result_msg,
  const std_msgs::msg::Header& header,
  vision_msgs::msg::Detection3DArray& detection3d_array_msg,
  sensor_msgs::msg::PointCloud2& combine_detection_cloud_msg)
{
  detection3d_array_msg.header = header;
  detection3d_array_msg.header.stamp = yolo_result_msg->header.stamp;

  RCLCPP_INFO(this->get_logger(),
              "Number of detections: %zu",
              yolo_result_msg->detections.detections.size());

  bool have_single_mask = (!yolo_result_msg->masks.empty());  // only 1 combined mask if not empty

  for (size_t i = 0; i < yolo_result_msg->detections.detections.size(); i++)
  {
    // Filter for class_id == "60"
    bool is_class_60 = false;
    for (const auto &r : yolo_result_msg->detections.detections[i].results)
    {
      if (r.hypothesis.class_id == "59") {
        is_class_60 = true;
        break;
      }
    }
    if (!is_class_60) {
      // skip if not class 60
      continue;
    }


    // Gather points
    pcl::PointCloud<pcl::PointXYZ>::Ptr detection_cloud_raw(new pcl::PointCloud<pcl::PointXYZ>());

    // Single combined mask scenario
    if (have_single_mask)
    {
      RCLCPP_INFO(this->get_logger(), "Using single combined mask for detection idx=%zu", i);
      // We pass 'i' so we know to look for pixel == i+1
      processPointsWithMask(
        cloud,
        yolo_result_msg->masks[0],  // the single combined mask
        i,                          // detection index
        detection_cloud_raw
      );
    }
    else
    {
      // fallback to bounding box
      RCLCPP_INFO(this->get_logger(), "No mask found, using bounding box for detection idx=%zu", i);
      processPointsWithBbox(cloud,
                            yolo_result_msg->detections.detections[i],
                            detection_cloud_raw);
    }

    if (!detection_cloud_raw->empty())
    {
      // Transform detection cloud back -> original frame
      pcl::PointCloud<pcl::PointXYZ>::Ptr detection_cloud(new pcl::PointCloud<pcl::PointXYZ>());
      detection_cloud = cloud2TransformedCloud(
                          detection_cloud_raw,
                          cam_model_.tfFrame(),
                          header.frame_id,
                          header.stamp);

      // cluster extraction (optional)
      pcl::PointCloud<pcl::PointXYZ>::Ptr closest_cloud = euclideanClusterExtraction(detection_cloud);

      // optional plane fitting
      pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
      pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
      pcl::SACSegmentation<pcl::PointXYZ> seg;
      seg.setOptimizeCoefficients(true);
      seg.setModelType(pcl::SACMODEL_PLANE);
      seg.setMethodType(pcl::SAC_RANSAC);
      seg.setDistanceThreshold(0.02);
      seg.setInputCloud(closest_cloud);
      seg.segment(*inliers, *coefficients);

      if (!inliers->indices.empty())
      {
        // example plane inliers
        RCLCPP_INFO(this->get_logger(),
                    "Plane with %zu inliers for detection idx=%zu (class=60)",
                    inliers->indices.size(), i);

        pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(closest_cloud);
        extract.setIndices(inliers);
        extract.filter(*plane_cloud);

        // 5) Compute centroid of the plane in "header.frame_id"
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*plane_cloud, centroid);

        // 6) Transform the centroid -> map frame (or whichever global frame you want)
        geometry_msgs::msg::Point plane_center_local;
        plane_center_local.x = centroid[0];
        plane_center_local.y = centroid[1];
        plane_center_local.z = centroid[2];

        last_plane_center_map_ = plane_center_local;
        // Mark that we do have a plane center
        geometry_msgs::msg::Point center_in_base;
        bool tf_ok = transformPointFrameToFrame(last_plane_center_map_, center_in_base, "hesai_lidar", "base_link");
        float dx = center_in_base.x;
        float dy = center_in_base.y;
        float dz = center_in_base.z;
        float dist_plane = std::sqrt(dx*dx + dy*dy);
        RCLCPP_INFO(this->get_logger(),
                        "plane distance %.2f", dist_plane);

        have_plane_center_ = true;
        // updateVisualServo(yolo_result_msg->detections.detections[i]);

        float image_width  = static_cast<float>(cam_model_.fullResolution().width);
        float image_height = static_cast<float>(cam_model_.fullResolution().height);
        float edge_threshold_px = 20.0f;  // example threshold

        const auto & detection2d_msg = yolo_result_msg->detections.detections[i];
        bool near_edge = isBboxNearEdge(detection2d_msg, image_width, image_height, edge_threshold_px);

        // ------------------------------------------------
        // (B) Decide whether to store the plane
        // ------------------------------------------------
        // Condition = near edge OR dist < 4
        bool condition = (dist_plane > 3.0f);
        bool condition_2 = (near_edge || dist_plane < 2.0f);

        // 3) Get the robot pose (base_link) in map frame
        if (condition)
        {
          geometry_msgs::msg::TransformStamped base_link_map_tf;
          try
          {
            // We want the transform FROM base_link TO map (or the other way around),
            // so we can figure out the robot's position in map frame. Typically, we do:
            //    lookupTransform("map", "base_link", ... )
            // that means: "Give me a transform that transforms points from base_link -> map."
            base_link_map_tf = tf_buffer_->lookupTransform("map", "base_link",
                                                          tf2::TimePointZero);
          }
          catch (tf2::TransformException &ex)
          {
            RCLCPP_WARN(this->get_logger(), "Could not get base_link->map transform: %s", ex.what());
            return;
          }

          // Extract robot position (rx, ry, rz) in map frame
          double rx = base_link_map_tf.transform.translation.x;
          double ry = base_link_map_tf.transform.translation.y;
          double rz = base_link_map_tf.transform.translation.z;

          // 4) Distance and direction from robot -> plane center
          geometry_msgs::msg::Point plane_center_map;
          bool tf_ok = transformPointFrameToFrame(plane_center_local, plane_center_map, "hesai_lidar", "map");
          double px = plane_center_map.x;
          double py = plane_center_map.y;
          double pz = plane_center_map.z;

          double dx = px - rx;
          double dy = py - ry;
          double dz = pz - rz;
          double dist_plane = std::sqrt(dx*dx + dy*dy);

          RCLCPP_INFO(this->get_logger(), "Distance to plane center (map frame): %.2f m", dist_plane);
          RCLCPP_INFO(this->get_logger(), "px py pz %.2f,  %.2f, %.2f",plane_center_map.x, plane_center_map.y, plane_center_map.z);
          RCLCPP_INFO(this->get_logger(), "rx ry rz %.2f,  %.2f, %.2f",rx, ry, rz);

          if (dist_plane < 1e-3) {
            RCLCPP_WARN(this->get_logger(),
                        "Plane center is essentially at the robot position; skipping goal calc.");
            return;
          }

          // 5) Compute the goal pose 2.5 m away from plane center (map frame)
          double offset = 2.5; // 2.5 m away from the plane
          // Unit direction from robot to plane
          double ux = dx / dist_plane;
          double uy = dy / dist_plane;
          double uz = dz / dist_plane;

          geometry_msgs::msg::PoseStamped goal_pose;
          goal_pose.header.stamp = this->now();
          goal_pose.header.frame_id = "map";

          // Position: plane center minus offset in the direction from robot->plane
          goal_pose.pose.position.x = px - offset * ux;
          goal_pose.pose.position.y = py - offset * uy;
          goal_pose.pose.position.z = pz - offset * uz;

          // Orientation: face the plane in XY-plane (simple yaw for ground robot)
          double yaw = std::atan2((py - goal_pose.pose.position.y),
                                  (px - goal_pose.pose.position.x));
          tf2::Quaternion q;
          q.setRPY(0.0, 0.0, yaw);
          goal_pose.pose.orientation = tf2::toMsg(q);

          // 6) Publish the goal
          goal_pose_pub_->publish(goal_pose);

          RCLCPP_INFO(this->get_logger(),
                      "Published goal in map frame, 2.5 m from plane center.");
        }
        else
        {
          if (!condition_2)
          {// servo_done_ == true => we no longer do servo, but we keep storing plane
          rclcpp::Time now_t = this->now();
          plane_buffer_.push_back({now_t, plane_cloud});
          while (!plane_buffer_.empty() &&
                (now_t - plane_buffer_.front().first) > plane_memory_duration_) {
            plane_buffer_.pop_front();
          }
          publishCombinedPlane(now_t, header);

          RCLCPP_INFO(this->get_logger(),
                      "servo_done_=true => ignoring servo, always storing plane now");
          }
          else
          {
            RCLCPP_INFO(this->get_logger(),
                      "get too close to the table, skip storing plane.");
          }
        }
      }

      // create bounding box in 3D
      createBoundingBox(
        detection3d_array_msg,
        closest_cloud,
        yolo_result_msg->detections.detections[i].results
      );
    }
  }

  // empty detection cloud
  pcl::PointCloud<pcl::PointXYZ> empty_cloud;
  pcl::toROSMsg(empty_cloud, combine_detection_cloud_msg);
  combine_detection_cloud_msg.header = header;
}

void TrackerWithCloudNode::updateVisualServo(const vision_msgs::msg::Detection2D &detection2d_msg)
{
  // 1) Retrieve bounding box info + image size
  float image_width  = static_cast<float>(cam_model_.fullResolution().width);
  float image_height = static_cast<float>(cam_model_.fullResolution().height);
  if (image_width <= 0.0f || image_height <= 0.0f) {
    RCLCPP_WARN(this->get_logger(), "No valid camera info => cannot do servo");
    return;
  }

  float cx = detection2d_msg.bbox.center.position.x;
  float cy = detection2d_msg.bbox.center.position.y;
  float w  = detection2d_msg.bbox.size_x;
  float h  = detection2d_msg.bbox.size_y;

  float left   = cx - w * 0.5f;
  float right  = cx + w * 0.5f;
  float top    = cy - h * 0.5f;
  float bottom = cy + h * 0.5f;

  float dist_left  = left;                   // distance from left image border
  float dist_right = image_width  - right;   // distance from right border
  float dist_top   = top;                    // distance from top border
  float dist_down  = image_height - bottom;  // distance from bottom border

  float edge_threshold = 20.0f;  // px
  bool near_left  = (dist_left  < edge_threshold);
  bool near_right = (dist_right < edge_threshold);
  bool near_top   = (dist_top   < edge_threshold);
  bool near_down  = (dist_down  < edge_threshold);

  // 2) Prepare a Twist
  geometry_msgs::msg::Twist cmd;
  cmd.linear.x  = 0.0f;
  cmd.angular.z = 0.0f;

  // Gains and speeds
  float rot_speed     = 0.17f;   // rad/s for simple in-place rotation
  float linear_speed  = 0.1f;   // m/s for in-place forward/back

  // 3) Priority: edges
  if (near_left)
  {
    // Turn left in place
    cmd.angular.z = +rot_speed;
  }
  else if (near_right)
  {
    // Turn right in place
    cmd.angular.z = -rot_speed;
  }
  else if (near_top)
  {
    // If near top => move backward
    cmd.linear.x = -linear_speed;
  }
  else if (near_down)
  {
    // If near bottom => move forward
    cmd.linear.x = +linear_speed;
  }
  else
  {
    // 4) No edges near => check distance to plane center
    if (!have_plane_center_) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "No plane center => cannot approach table");
    }
    else {
      geometry_msgs::msg::Point center_in_base;
      bool tf_ok = transformPointFrameToFrame(last_plane_center_map_, center_in_base, "hesai_lidar", "base_link");
      if (!tf_ok) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                             "TF map->base_link not available => skip forward");
      }
      else {
        float dx = center_in_base.x;
        float dy = center_in_base.y;
        float dz = center_in_base.z;
        float dist_plane = std::sqrt(dx*dx + dy*dy + dz*dz);

        float desired_dist = 2.0f;  // we want to stop if within 4 m
        if (dist_plane > desired_dist)
        {
          // (A) Combined rotation + forward:
          //  We'll rotate toward (dx,dy) and move forward to reduce distance.
          //  heading_err = atan2(dy, dx) => 0 means pointing directly at table.
          float heading_err = std::atan2(dy, dx);

          // A small P-control on heading:
          float heading_gain = 0.5f;  // tune
          float ang_cmd = heading_gain * heading_err;
          // saturate
          float max_rot = 0.5f; 
          if (ang_cmd >  max_rot)  ang_cmd =  max_rot;
          if (ang_cmd < -max_rot)  ang_cmd = -max_rot;

          // Forward speed
          float forward_err = dist_plane - desired_dist;
          float forward_gain = 0.2f;
          float fwd_cmd = forward_gain * forward_err;
          // saturate
          float max_fwd = 0.3f;
          if (fwd_cmd >  max_fwd) fwd_cmd =  max_fwd;
          if (fwd_cmd < -max_fwd) fwd_cmd = -max_fwd;

          cmd.linear.x  = fwd_cmd;
          cmd.angular.z = ang_cmd;
        }
        else
        {
          // (B) Already within 4 m => stop
          cmd.linear.x  = 0.0f;
          cmd.angular.z = 0.0f;
          RCLCPP_INFO(this->get_logger(), "Within 4 m => stopping");
        }
      }
    }
  }

  // 5) Publish final cmd
  visual_servo_pub_->publish(cmd);

  RCLCPP_INFO(this->get_logger(),
    "[updateVisualServo] near_edge=(L:%d,R:%d,T:%d,D:%d) => cmd( lin=%.2f, ang=%.2f )",
    (near_left?1:0),(near_right?1:0),(near_top?1:0),(near_down?1:0),
    cmd.linear.x, cmd.angular.z
  );
}

bool TrackerWithCloudNode::transformPointFrameToFrame(
    const geometry_msgs::msg::Point &pt_in,
    geometry_msgs::msg::Point &pt_out,
    const std::string &from_frame,
    const std::string &to_frame)
{
  // Example: transform at the current time
  rclcpp::Time now_t = this->now();
  // Wait up to 2s for the transform
  rclcpp::Duration timeout(2, 0);

  bool tf_ok = waitForTransform(to_frame, from_frame, now_t, timeout);
  if (!tf_ok)
  {
    RCLCPP_WARN(this->get_logger(),
                "transformPointFrameToFrame: No TF from %s->%s after %.1f s",
                from_frame.c_str(), to_frame.c_str(), timeout.seconds());
    return false;
  }

  try
  {
    // Lookup the transform
    geometry_msgs::msg::TransformStamped tf_stamped =
        tf_buffer_->lookupTransform(to_frame, from_frame, now_t);

    // Convert the geometry_msgs to TF2 types
    tf2::Vector3 p(pt_in.x, pt_in.y, pt_in.z);
    tf2::Quaternion q(tf_stamped.transform.rotation.x,
                      tf_stamped.transform.rotation.y,
                      tf_stamped.transform.rotation.z,
                      tf_stamped.transform.rotation.w);
    tf2::Vector3 t(tf_stamped.transform.translation.x,
                   tf_stamped.transform.translation.y,
                   tf_stamped.transform.translation.z);

    tf2::Transform transform(q, t);

    // Apply the transform
    tf2::Vector3 p_new = transform * p;
    pt_out.x = p_new.x();
    pt_out.y = p_new.y();
    pt_out.z = p_new.z();

    return true;
  }
  catch (tf2::TransformException &ex)
  {
    RCLCPP_WARN(this->get_logger(),
                "transformPointFrameToFrame failed: %s", ex.what());
    return false;
  }
}

// bool TrackerWithCloudNode::transformPointFrameToFrame(
//     const geometry_msgs::msg::Point &pt_in_map,
//     geometry_msgs::msg::Point &pt_in_base)
// {
//   std::string from_frame = "hesai_lidar";
//   std::string to_frame   = "base_link";
//   rclcpp::Time now_t = this->now();

//   // Wait up to 2s for transform
//   rclcpp::Duration timeout(2,0);
//   bool tf_ok = waitForTransform(to_frame, from_frame, now_t, timeout);
//   if (!tf_ok) {
//     RCLCPP_WARN(this->get_logger(), 
//                 "transformPointFrameToFrame: No TF from %s->%s after %f s",
//                 from_frame.c_str(), to_frame.c_str(), timeout.seconds());
//     return false;
//   }

//   try
//   {
//     geometry_msgs::msg::TransformStamped tf_stamped =
//       tf_buffer_->lookupTransform(to_frame, from_frame, now_t);

//     // transform the point
//     tf2::Vector3 p(pt_in_map.x, pt_in_map.y, pt_in_map.z);
//     tf2::Quaternion q(
//       tf_stamped.transform.rotation.x,
//       tf_stamped.transform.rotation.y,
//       tf_stamped.transform.rotation.z,
//       tf_stamped.transform.rotation.w
//     );
//     tf2::Vector3 t(
//       tf_stamped.transform.translation.x,
//       tf_stamped.transform.translation.y,
//       tf_stamped.transform.translation.z
//     );
//     tf2::Transform tf(q, t);

//     tf2::Vector3 p_new = tf * p;
//     pt_in_base.x = p_new.x();
//     pt_in_base.y = p_new.y();
//     pt_in_base.z = p_new.z();
//     return true;
//   }
//   catch (tf2::TransformException &ex)
//   {
//     RCLCPP_WARN(this->get_logger(),
//                 "transformPointFrameToFrame failed: %s", ex.what());
//     return false;
//   }
// }

bool TrackerWithCloudNode::isBboxNearEdge(
  const vision_msgs::msg::Detection2D &detection2d_msg,
  float image_width,
  float image_height,
  float edge_threshold)
{
  // Extract bounding box info
  float cx = detection2d_msg.bbox.center.position.x;
  float cy = detection2d_msg.bbox.center.position.y;
  float w  = detection2d_msg.bbox.size_x;
  float h  = detection2d_msg.bbox.size_y;

  // Edges of the bbox
  float left   = cx - w * 0.5f;
  float right  = cx + w * 0.5f;
  float top    = cy - h * 0.5f;
  float bottom = cy + h * 0.5f;

  // Distances from each image border
  float dist_left  = left;
  float dist_right = image_width - right;
  float dist_top   = top;
  float dist_down  = image_height - bottom;

  // If any is below edge_threshold => near edge
  if (dist_left < edge_threshold || dist_right < edge_threshold ||
      dist_top  < edge_threshold || dist_down  < edge_threshold)
  {
    RCLCPP_INFO(this->get_logger(),
                "Bbox is near edge: (left=%.1f, right=%.1f, top=%.1f, down=%.1f)",
                dist_left, dist_right, dist_top, dist_down);
    return true;
  }
  return false;
}

// --------------------------------------------------------------------
// Keep the original function name, but add detection_index
// so we know which ID => i+1 in the mask
// --------------------------------------------------------------------
void TrackerWithCloudNode::processPointsWithMask(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const sensor_msgs::msg::Image& mask_image_msg,
    size_t detection_index,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& detection_cloud_raw)
{
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(mask_image_msg, sensor_msgs::image_encodings::MONO8);
  }
  catch (cv_bridge::Exception &e)
  {
    RCLCPP_ERROR(this->get_logger(), "processPointsWithMask: cv_bridge exception: %s", e.what());
    return;
  }

  // Single mask => pixel value == (detection_index + 1)
  uint8_t mask_val = static_cast<uint8_t>(detection_index + 1);

  double mask_width  = static_cast<double>(cv_ptr->image.cols);
  double mask_height = static_cast<double>(cv_ptr->image.rows);

  double cam_width  = static_cast<double>(cam_model_.fullResolution().width);
  double cam_height = static_cast<double>(cam_model_.fullResolution().height);

  double scale_x = mask_width  / cam_width;
  double scale_y = mask_height / cam_height;

  int total_points    = 0;
  int points_in_front = 0;
  int points_matched  = 0;

  for (const auto& p : cloud->points)
  {
    total_points++;
    if (p.z <= 0) {
      continue;  // skip behind camera
    }
    cv::Point3d pt_cv(p.x, p.y, p.z);
    cv::Point2d uv = cam_model_.project3dToPixel(pt_cv);

    if (uv.x >= 0 && uv.x < cam_width &&
        uv.y >= 0 && uv.y < cam_height)
    {
      points_in_front++;

      double mask_u = uv.x * scale_x;
      double mask_v = uv.y * scale_y;

      if (mask_u >= 0.0 && mask_u < mask_width &&
          mask_v >= 0.0 && mask_v < mask_height)
      {
        uint8_t pixel_val = cv_ptr->image.at<uchar>(
                              static_cast<int>(mask_v),
                              static_cast<int>(mask_u));
        // check if pixel == (detection_index + 1)
        if (pixel_val == mask_val)
        {
          detection_cloud_raw->points.push_back(p);
          points_matched++;
        }
      }
    }
  }

  RCLCPP_INFO(
    this->get_logger(),
    "[processPointsWithMask] detection=%zu => mask_val=%u, total=%d, front=%d, matched=%d => raw=%zu",
    detection_index, mask_val, total_points, points_in_front, points_matched,
    detection_cloud_raw->points.size()
  );
}

// Keep the rest of your functions: processPointsWithBbox, createBoundingBox, 
// downsampleCloudMsg, transformPointCloud, publishCombinedPlane, etc.
// unchanged from the original code, other than removing references to yolo_result_msg->masks[i].

void TrackerWithCloudNode::processPointsWithBbox(
  const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
  const vision_msgs::msg::Detection2D& detection2d_msg,
  pcl::PointCloud<pcl::PointXYZ>::Ptr& detection_cloud_raw)
{
  // For each point, project to (u,v) and check if it falls within the 2D bounding box
  for (const auto& point : cloud->points)
  {
    cv::Point3d pt_cv(point.x, point.y, point.z);
    cv::Point2d uv = cam_model_.project3dToPixel(pt_cv);

    float half_w = detection2d_msg.bbox.size_x * 0.5f;
    float half_h = detection2d_msg.bbox.size_y * 0.5f;
    float x_center = detection2d_msg.bbox.center.position.x;
    float y_center = detection2d_msg.bbox.center.position.y;

    if (point.z > 0 &&
        uv.x >= (x_center - half_w) && uv.x <= (x_center + half_w) &&
        uv.y >= (y_center - half_h) && uv.y <= (y_center + half_h))
    {
      detection_cloud_raw->points.push_back(point);
    }
  }
}

void TrackerWithCloudNode::createBoundingBox(
  vision_msgs::msg::Detection3DArray& detection3d_array_msg,
  const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
  const std::vector<vision_msgs::msg::ObjectHypothesisWithPose>& detections_results_msg)
{
  RCLCPP_INFO(this->get_logger(), "createBoundingBox");
  if (cloud->empty()) {
    // No points => skip
    return;
  }

  // Compute centroid
  Eigen::Vector4f centroid;
  pcl::compute3DCentroid(*cloud, centroid);

  // Compute an orientation (rotating around Z to flatten bounding box)
  double theta = -atan2(centroid[1], sqrt(pow(centroid[0], 2) + pow(centroid[2], 2)));
  Eigen::Affine3f transform = Eigen::Affine3f::Identity();
  transform.rotate(Eigen::AngleAxisf(theta, Eigen::Vector3f::UnitZ()));

  // Transform the cloud to get min/max
  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  transformPointCloud(cloud, transformed_cloud, transform);

  pcl::PointXYZ min_pt, max_pt;
  pcl::getMinMax3D(*transformed_cloud, min_pt, max_pt);

  Eigen::Vector4f transformed_bbox_center =
      Eigen::Vector4f((min_pt.x + max_pt.x) * 0.5f,
                      (min_pt.y + max_pt.y) * 0.5f,
                      (min_pt.z + max_pt.z) * 0.5f,
                      1.0f);
  Eigen::Vector4f bbox_center = transform.inverse() * transformed_bbox_center;
  Eigen::Quaternionf q(transform.inverse().rotation());

  // Fill a Detection3D
  vision_msgs::msg::Detection3D detection3d_msg;
  detection3d_msg.bbox.center.position.x = bbox_center[0];
  detection3d_msg.bbox.center.position.y = bbox_center[1];
  detection3d_msg.bbox.center.position.z = bbox_center[2];
  detection3d_msg.bbox.center.orientation.x = q.x();
  detection3d_msg.bbox.center.orientation.y = q.y();
  detection3d_msg.bbox.center.orientation.z = q.z();
  detection3d_msg.bbox.center.orientation.w = q.w();
  detection3d_msg.bbox.size.x = (max_pt.x - min_pt.x);
  detection3d_msg.bbox.size.y = (max_pt.y - min_pt.y);
  detection3d_msg.bbox.size.z = (max_pt.z - min_pt.z);

  // Attach detection results (class, score, etc.)
  detection3d_msg.results = detections_results_msg;

  detection3d_array_msg.detections.push_back(detection3d_msg);
}

pcl::PointCloud<pcl::PointXYZ>::Ptr
TrackerWithCloudNode::downsampleCloudMsg(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::fromROSMsg(*cloud_msg, *cloud);

  this->get_parameter("voxel_leaf_size", voxel_leaf_size_);
  pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::VoxelGrid<pcl::PointXYZ>::Ptr voxel_grid(new pcl::VoxelGrid<pcl::PointXYZ>());
  voxel_grid->setInputCloud(cloud);
  voxel_grid->setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
  voxel_grid->filter(*downsampled_cloud);

  return downsampled_cloud;
}

bool TrackerWithCloudNode::waitForTransform(
  const std::string &target_frame,
  const std::string &source_frame,
  const rclcpp::Time &time,
  const rclcpp::Duration &timeout)
{
  auto start_time = this->now();

  // We'll loop until the TF is available or until we exceed 'timeout'
  while (rclcpp::ok() && (this->now() - start_time) < timeout) {
    // Check if TF is available
    bool can = tf_buffer_->canTransform(target_frame, source_frame, time, rclcpp::Duration(0,0));
    if (can) {
      return true;  // We found the transform
    }
    // Sleep a bit, then try again
    rclcpp::sleep_for(std::chrono::milliseconds(100));
  }

  // If we get here, it's not available within 'timeout'
  return false;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr
TrackerWithCloudNode::cloud2TransformedCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const std::string& source_frame,
    const std::string& target_frame,
    const rclcpp::Time& stamp)
{
  // Wait for transform to become available (e.g., 2 seconds)
  rclcpp::Duration timeout(2, 0); // 2s
  bool tf_ok = waitForTransform(target_frame, source_frame, stamp, timeout);
  if (!tf_ok) {
    // Could not get the transform in time. Handle it gracefully.
    RCLCPP_WARN(this->get_logger(),
                "Timed out waiting for transform %s -> %s after %.2f seconds",
                source_frame.c_str(), target_frame.c_str(),
                timeout.seconds());
    // Return the original cloud or an empty cloud, as you prefer
    return cloud;
  }

  try
  {
    geometry_msgs::msg::TransformStamped tf_stamped =
        tf_buffer_->lookupTransform(target_frame, source_frame, stamp);
    Eigen::Affine3f eigen_transform =
        tf2::transformToEigen(tf_stamped.transform).cast<float>();

    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    transformPointCloud(cloud, transformed_cloud, eigen_transform);

    return transformed_cloud;
  }
  catch (tf2::TransformException& e)
  {
    RCLCPP_WARN(this->get_logger(), "%s", e.what());
    // If we can't transform, just return the original cloud
    return cloud;
  }
}

void TrackerWithCloudNode::transformPointCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_out,
    const Eigen::Affine3f& transform)
{
  int cloud_size = cloud_in->size();
  cloud_out->resize(cloud_size);

  for (int i = 0; i < cloud_size; i++)
  {
    const auto& point = cloud_in->points[i];
    cloud_out->points[i].x =
        transform(0, 0) * point.x +
        transform(0, 1) * point.y +
        transform(0, 2) * point.z +
        transform(0, 3);
    cloud_out->points[i].y =
        transform(1, 0) * point.x +
        transform(1, 1) * point.y +
        transform(1, 2) * point.z +
        transform(1, 3);
    cloud_out->points[i].z =
        transform(2, 0) * point.x +
        transform(2, 1) * point.y +
        transform(2, 2) * point.z +
        transform(2, 3);
  }
}

pcl::PointCloud<pcl::PointXYZ>::Ptr
TrackerWithCloudNode::euclideanClusterExtraction(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
  this->get_parameter("cluster_tolerance", cluster_tolerance_);
  this->get_parameter("min_cluster_size", min_cluster_size_);
  this->get_parameter("max_cluster_size", max_cluster_size_);

  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
  tree->setInputCloud(cloud);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance(cluster_tolerance_);
  ec.setMinClusterSize(min_cluster_size_);
  ec.setMaxClusterSize(max_cluster_size_);
  ec.setSearchMethod(tree);
  ec.setInputCloud(cloud);
  ec.extract(cluster_indices);

  // We'll pick the cluster whose centroid is closest to the origin
  float min_distance = std::numeric_limits<float>::max();
  pcl::PointCloud<pcl::PointXYZ>::Ptr closest_cluster(new pcl::PointCloud<pcl::PointXYZ>());

  for (const auto& cluster : cluster_indices)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>());
    for (const auto& idx : cluster.indices)
    {
      cloud_cluster->push_back((*cloud)[idx]);
    }

    // Compute centroid
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud_cluster, centroid);
    float distance = centroid.norm();

    // Keep the cluster that is closest
    if (distance < min_distance)
    {
      min_distance = distance;
      *closest_cluster = *cloud_cluster;
    }
  }

  return closest_cluster;
}


void TrackerWithCloudNode::publishCombinedPlane(const rclcpp::Time &stamp,
                                               const std_msgs::msg::Header &header)
{
  // 0) If there's nothing in the buffer, just return.
  if (plane_buffer_.empty()) {
    RCLCPP_INFO(this->get_logger(), "No planes to combine in the last 1s");
    return;
  }

  // 1) Prepare a container for the combined planes in the map frame
  pcl::PointCloud<pcl::PointXYZ> combined_in_map;
  combined_in_map.clear();

  // For each plane we stored: (timestamp, plane_cloud_ptr)
  for (auto &pair : plane_buffer_)
  {
    rclcpp::Time plane_time = pair.first;
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_ptr = pair.second;

    // 2) Wait for transform from 'header.frame_id' to 'map', using the plane's own timestamp.
    rclcpp::Duration tf_timeout(2, 0); // 2 seconds
    if (!waitForTransform("map", header.frame_id, plane_time, tf_timeout)) {
      RCLCPP_WARN(this->get_logger(),
                  "[publishCombinedPlane] Could not get TF from %s to map (time=%.2f). "
                  "Skipping this plane.",
                  header.frame_id.c_str(),
                  plane_time.seconds());
      continue;
    }

    // 3) Lookup the transform
    pcl::PointCloud<pcl::PointXYZ> plane_in_map;
    plane_in_map.clear();

    try
    {
      geometry_msgs::msg::TransformStamped tf_stamped =
          tf_buffer_->lookupTransform("map", header.frame_id, plane_time);

      // Convert geometry_msgs transform -> Eigen
      Eigen::Affine3f eigen_transform =
          tf2::transformToEigen(tf_stamped.transform).cast<float>();

      // 4) Transform the plane into 'map'
      //    Make sure you have #include <pcl/common/transforms.h>
      pcl::transformPointCloud(*plane_ptr, plane_in_map, eigen_transform);
    }
    catch (tf2::TransformException &ex)
    {
      RCLCPP_ERROR(this->get_logger(),
                   "[publishCombinedPlane] Failed to transform plane to 'map': %s",
                   ex.what());
      // Skip this plane if we cannot transform
      continue;
    }

    // 5) Merge the transformed plane with our growing "combined_in_map" cloud
    combined_in_map += plane_in_map;
  }

  // 6) If we have at least some points, publish them
  if (combined_in_map.empty()) {
    RCLCPP_INFO(this->get_logger(),
                "[publishCombinedPlane] After transforming, no planes to publish.");
    return;
  }

  // 7) Convert to ROS message and publish with frame_id = "map"
  sensor_msgs::msg::PointCloud2 combined_msg;
  pcl::toROSMsg(combined_in_map, combined_msg);
  combined_msg.header.stamp = stamp;   // or use now() if appropriate
  combined_msg.header.frame_id = "map";

  plane_pub_->publish(combined_msg);

  RCLCPP_INFO(this->get_logger(),
              "[publishCombinedPlane] Published combined plane in 'map' frame with %zu points",
              combined_in_map.size());
}

visualization_msgs::msg::MarkerArray
TrackerWithCloudNode::createMarkerArray(
    const vision_msgs::msg::Detection3DArray& detection3d_array_msg,
    const double& duration)
{
  visualization_msgs::msg::MarkerArray marker_array_msg;

  for (size_t i = 0; i < detection3d_array_msg.detections.size(); i++)
  {
    const auto &bbox = detection3d_array_msg.detections[i].bbox;
    if (std::isfinite(bbox.size.x) &&
        std::isfinite(bbox.size.y) &&
        std::isfinite(bbox.size.z))
    {
      visualization_msgs::msg::Marker marker_msg;
      marker_msg.header = detection3d_array_msg.header;
      marker_msg.ns = "detection";
      marker_msg.id = i;
      marker_msg.type = visualization_msgs::msg::Marker::CUBE;
      marker_msg.action = visualization_msgs::msg::Marker::ADD;
      marker_msg.pose = bbox.center;
      marker_msg.scale.x = bbox.size.x;
      marker_msg.scale.y = bbox.size.y;
      marker_msg.scale.z = bbox.size.z;

      // Green, partially transparent
      marker_msg.color.r = 0.0f;
      marker_msg.color.g = 1.0f;
      marker_msg.color.b = 0.0f;
      marker_msg.color.a = 0.5f;

      marker_msg.lifetime = rclcpp::Duration(std::chrono::duration<double>(duration));
      marker_array_msg.markers.push_back(marker_msg);
    }
  }

  return marker_array_msg;
}

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<TrackerWithCloudNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

