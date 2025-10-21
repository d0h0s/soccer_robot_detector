//
// Created by xianghong on 10/20/25.
//

#pragma once

#include <memory>
#include <vector>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/PointStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_geometry/pinhole_camera_model.h>

class SoccerRobotDetector
{
public:
  explicit SoccerRobotDetector(ros::NodeHandle& nh);

  void syncCallback(const sensor_msgs::ImageConstPtr& image_msg,
                    const sensor_msgs::ImageConstPtr& depth_msg);
  void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& info_msg);

private:
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::SubscriberFilter image_sub_;
  image_transport::SubscriberFilter depth_sub_;
  using ApproxSyncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image>;
  std::unique_ptr<message_filters::Synchronizer<ApproxSyncPolicy>> sync_;
  image_transport::Publisher debug_pub_;
  ros::Subscriber camera_info_sub_;
  ros::Publisher center_pub_;

  image_geometry::PinholeCameraModel camera_model_;
  bool has_camera_info_ = false;

  double dp_, minDist_, param1_, param2_;
  int minRadius_, maxRadius_;
  bool use_harris_;

  // Hough line parameters
  double hough_rho_;
  double hough_theta_;
  int hough_threshold_;
  double minLineLength_;
  double maxLineGap_;

  // Corner detection parameters
  int maxCorners_;
  double qualityLevel_;
  double minCornerDistance_;
  cv::Point2f last_center_;
  int last_radius_ = 0;
  bool has_last_circle_ = false;

  double extractMedianDepth(const cv::Mat& depth_roi,
                            const cv::Point& roi_center,
                            int radius) const;
};

