//
// Created by xianghong on 10/20/25.
//

#pragma once
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>

class SoccerRobotDetector
{
public:
  SoccerRobotDetector(ros::NodeHandle& nh);
  void imageCallback(const sensor_msgs::ImageConstPtr& msg);

private:
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher debug_pub_;

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

};

