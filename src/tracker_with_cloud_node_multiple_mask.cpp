/**
 * ultralytics_ros
 * Copyright (C) 2023-2024  Alpaca-zip
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "tracker_with_cloud_node/tracker_with_cloud_node.h"
#include <pcl/common/transforms.h>

// --------------------
// Additional includes
// --------------------
#include <deque>
#include <utility>    // for std::pair

TrackerWithCloudNode::TrackerWithCloudNode() 
  : rclcpp::Node("tracker_with_cloud_node"),
    plane_memory_duration_(1, 0) // store planes for 1 second
{
  // Declare and retrieve parameters
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

  // Subscribers
  camera_info_sub_.subscribe(this, camera_info_topic_);
  lidar_sub_.subscribe(this, lidar_topic_);
  yolo_result_sub_.subscribe(this, yolo_result_topic_);

  // Synchronize all three inputs (Approximate)
  sync_ = std::make_shared<message_filters::Synchronizer<ApproximateSyncPolicy>>(ApproximateSyncPolicy(500));
  sync_->connectInput(camera_info_sub_, lidar_sub_, yolo_result_sub_);
  sync_->registerCallback(std::bind(&TrackerWithCloudNode::syncCallback, this,
                                    std::placeholders::_1,
                                    std::placeholders::_2,
                                    std::placeholders::_3));

  // Publishers
  this->get_parameter("yolo_3d_result_topic", yolo_3d_result_topic_);
  detection3d_pub_ = this->create_publisher<vision_msgs::msg::Detection3DArray>(yolo_3d_result_topic_, 1);
  detection_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("detection_cloud", 1);
  marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("detection_marker", 1);

  // We'll reuse plane_pub_ for both the immediate plane and the combined plane
  plane_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("plane_cloud", 1);

  last_call_time_ = this->now();

  // TF buffer/listener
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

  // 1) Downsample the incoming cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  downsampled_cloud = downsampleCloudMsg(cloud_msg);

  // 2) Get camera model from camera_info
  cam_model_.fromCameraInfo(camera_info_msg);

  // 3) Transform the cloud into camera's frame for projection
  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  transformed_cloud = cloud2TransformedCloud(
                        downsampled_cloud,
                        cloud_msg->header.frame_id,
                        cam_model_.tfFrame(),
                        cloud_msg->header.stamp);

  // 4) Build the detection3d_array and an optional combined detection cloud
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
  // Prepare the 3D array message
  detection3d_array_msg.header = header;
  detection3d_array_msg.header.stamp = yolo_result_msg->header.stamp;

  RCLCPP_INFO(this->get_logger(), "Number of detections: %zu",
              yolo_result_msg->detections.detections.size());

  int bounding_box_count = 0;

  // Iterate through each detection
  for (size_t i = 0; i < yolo_result_msg->detections.detections.size(); i++)
  {
    // Only process detection if it has class_id == "60"
    bool process_detection = false;
    for (const auto &result : yolo_result_msg->detections.detections[i].results)
    {
      // If YOLO published "60" as a string for class 60
      if (result.hypothesis.class_id == "60")
      {
        process_detection = true;
        break;
      }
    }

    if (!process_detection)
    {
      // Skip if not class 60
      continue;
    }

    // Gather points for this detection
    pcl::PointCloud<pcl::PointXYZ>::Ptr detection_cloud_raw(new pcl::PointCloud<pcl::PointXYZ>());

    // Check for a corresponding mask
    // (We assume 1:1 relationship: masks.size() == detections.size())
    if (!yolo_result_msg->masks.empty() && i < yolo_result_msg->masks.size())
    {
      RCLCPP_INFO(this->get_logger(), "Using mask for detection index i=%zu", i);
      processPointsWithMask(cloud, yolo_result_msg->masks[i], detection_cloud_raw);
    }
    else
    {
      // Fallback to bounding box if no mask is available
      RCLCPP_INFO(this->get_logger(), "No mask found for i=%zu, using bounding box", i);
      processPointsWithBbox(cloud, yolo_result_msg->detections.detections[i], detection_cloud_raw);
    }

    // If we found any points
    if (!detection_cloud_raw->points.empty())
    {
      // Transform detection points from camera frame -> original frame
      pcl::PointCloud<pcl::PointXYZ>::Ptr detection_cloud(new pcl::PointCloud<pcl::PointXYZ>());
      detection_cloud = cloud2TransformedCloud(
                          detection_cloud_raw,
                          cam_model_.tfFrame(),
                          header.frame_id,
                          header.stamp);

      // Optionally cluster to find the closest cluster
      pcl::PointCloud<pcl::PointXYZ>::Ptr closest_detection_cloud(new pcl::PointCloud<pcl::PointXYZ>());
      closest_detection_cloud = euclideanClusterExtraction(detection_cloud);

      // Example plane fitting (RANSAC)
      pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
      pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
      pcl::SACSegmentation<pcl::PointXYZ> seg;
      seg.setOptimizeCoefficients(true);
      seg.setModelType(pcl::SACMODEL_PLANE);
      seg.setMethodType(pcl::SAC_RANSAC);
      seg.setDistanceThreshold(0.02);
      seg.setInputCloud(closest_detection_cloud);
      seg.segment(*inliers, *coefficients);

      if (!inliers->indices.empty())
      {
        RCLCPP_INFO(this->get_logger(),
                    "Plane detected with %zu inliers for detection ID 60",
                    inliers->indices.size());

        // Extract plane points
        pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(closest_detection_cloud);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*plane_cloud);

        // Publish immediate plane
        // sensor_msgs::msg::PointCloud2 plane_cloud_msg;
        // pcl::toROSMsg(*plane_cloud, plane_cloud_msg);
        // plane_cloud_msg.header = header;
        // plane_pub_->publish(plane_cloud_msg);

        // ---------------------------------------------------
        // Now store this plane in our 1-second plane_buffer_
        // ---------------------------------------------------
        rclcpp::Time now_t = this->now();
        plane_buffer_.push_back({now_t, plane_cloud});

        // Remove older than 1 second from the front
        while (!plane_buffer_.empty() && (now_t - plane_buffer_.front().first) > plane_memory_duration_) {
          plane_buffer_.pop_front();
        }

        // Re-publish the combined plane from the last second
        publishCombinedPlane(now_t, header);
      }

      // Create bounding box for the detection
      createBoundingBox(detection3d_array_msg,
                        closest_detection_cloud,
                        yolo_result_msg->detections.detections[i].results);

      bounding_box_count++;
    }
  }

  RCLCPP_INFO(this->get_logger(), "Number of BoundingBoxes created: %zu",
              detection3d_array_msg.detections.size());

  // No combined detection cloud => publish an empty
  pcl::PointCloud<pcl::PointXYZ> empty_cloud;
  pcl::toROSMsg(empty_cloud, combine_detection_cloud_msg);
  combine_detection_cloud_msg.header = header;
}

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

void TrackerWithCloudNode::processPointsWithMask(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const sensor_msgs::msg::Image& mask_image_msg,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& detection_cloud_raw)
{
  cv_bridge::CvImagePtr cv_ptr;

  try
  {
    // Convert the incoming mask to an OpenCV mono8 image
    cv_ptr = cv_bridge::toCvCopy(mask_image_msg, sensor_msgs::image_encodings::MONO8);
  }
  catch (cv_bridge::Exception &e)
  {
    RCLCPP_ERROR(this->get_logger(), "Error in processPointsWithMask: %s", e.what());
    return;
  }

  double mask_width  = static_cast<double>(cv_ptr->image.cols);
  double mask_height = static_cast<double>(cv_ptr->image.rows);

  double cam_width  = static_cast<double>(cam_model_.fullResolution().width);
  double cam_height = static_cast<double>(cam_model_.fullResolution().height);

  double scale_x = mask_width  / cam_width;
  double scale_y = mask_height / cam_height;

  int total_points    = 0;
  int points_in_front = 0;
  int points_in_mask  = 0;

  for (const auto& point : cloud->points)
  {
    total_points++;

    cv::Point3d pt_cv(point.x, point.y, point.z);
    cv::Point2d uv = cam_model_.project3dToPixel(pt_cv);

    if (point.z > 0 &&
        uv.x >= 0 && uv.x < cam_width &&
        uv.y >= 0 && uv.y < cam_height)
    {
      points_in_front++;

      double mask_u = uv.x * scale_x;
      double mask_v = uv.y * scale_y;

      if (mask_u >= 0.0 && mask_u < mask_width &&
          mask_v >= 0.0 && mask_v < mask_height)
      {
        if (cv_ptr->image.at<uchar>(
              static_cast<int>(mask_v),
              static_cast<int>(mask_u)) > 0)
        {
          detection_cloud_raw->points.push_back(point);
          points_in_mask++;
        }
      }
    }
  }

  RCLCPP_INFO(
    this->get_logger(),
    "processPointsWithMask: total=%d, in_front=%d, in_mask=%d => detection cloud size=%zu",
    total_points,
    points_in_front,
    points_in_mask,
    detection_cloud_raw->points.size()
  );
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

// -------------------------------------------------------
// Additional method: publishCombinedPlane
// merges all plane clouds in plane_buffer_ over last 1s
// and re-publishes on plane_pub_
// -------------------------------------------------------

// void TrackerWithCloudNode::publishCombinedPlane(const rclcpp::Time &stamp, const std_msgs::msg::Header &header)
// {
//   // Merge all plane clouds from plane_buffer_ into a single point cloud
//   pcl::PointCloud<pcl::PointXYZ> combined;

//   for (auto &pair : plane_buffer_) {
//     auto &plane_ptr = pair.second;  // plane_ptr is a pcl::PointCloud<pcl::PointXYZ>::Ptr
//     combined += *plane_ptr;         // operator+= merges points
//   }

//   if (combined.empty()) {
//     RCLCPP_INFO(this->get_logger(), "No planes to combine in the last 1s");
//     return;
//   }

//   sensor_msgs::msg::PointCloud2 combined_msg;
//   pcl::toROSMsg(combined, combined_msg);

//   // Use the same header for continuity
//   combined_msg.header = header;

//   // Publish on the same plane_pub_ topic
//   plane_pub_->publish(combined_msg);

//   RCLCPP_INFO(this->get_logger(),
//               "Published combined plane from last 1s with %zu points",
//               combined.size());
// }

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
