//
// Created by xianghong on 10/20/25.
//

#include "soccer_robot_detector/detector.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <iomanip>
#include <boost/bind.hpp>
#include <sensor_msgs/image_encodings.h>

namespace
{
constexpr double kDefaultDp = 1.2;
constexpr double kDefaultMinDist = 50.0;
constexpr double kDefaultParam1 = 100.0;
constexpr double kDefaultParam2 = 30.0;
constexpr int kDefaultMinRadius = 30;
constexpr int kDefaultMaxRadius = 200;
}  // namespace

SoccerRobotDetector::SoccerRobotDetector(ros::NodeHandle& nh)
    : nh_(nh),
      it_(nh_),
      image_sub_(),
      depth_sub_()
{
  std::string image_topic;
  std::string depth_topic;
  std::string camera_info_topic;
  std::string debug_topic;
  std::string center_topic;

  nh_.param("image_topic", image_topic, std::string("/camera/infra1/image_rect_raw"));
  nh_.param("depth_topic", depth_topic, std::string("/camera/depth/image_rect_raw"));
  nh_.param("camera_info_topic", camera_info_topic, std::string("/camera/infra1/camera_info"));
  nh_.param("debug_topic", debug_topic, std::string("/soccer_robot_detector/debug"));
  nh_.param("center_topic", center_topic, std::string("/soccer_robot_detector/center"));

  nh_.param("dp", dp_, kDefaultDp);
  nh_.param("minDist", minDist_, kDefaultMinDist);
  nh_.param("param1", param1_, kDefaultParam1);
  nh_.param("param2", param2_, kDefaultParam2);
  nh_.param("minRadius", minRadius_, kDefaultMinRadius);
  nh_.param("maxRadius", maxRadius_, kDefaultMaxRadius);
  nh_.param("use_harris", use_harris_, true);
  nh_.param("hough_rho", hough_rho_, 1.0);
  nh_.param("hough_theta", hough_theta_, 1.0);
  nh_.param("hough_threshold", hough_threshold_, 50);
  nh_.param("minLineLength", minLineLength_, 30.0);
  nh_.param("maxLineGap", maxLineGap_, 10.0);
  nh_.param("maxCorners", maxCorners_, 50);
  nh_.param("qualityLevel", qualityLevel_, 0.02);
  nh_.param("minCornerDistance", minCornerDistance_, 10.0);

  image_sub_.subscribe(it_, image_topic, 1);
  depth_sub_.subscribe(it_, depth_topic, 1);

  sync_ = std::make_unique<message_filters::Synchronizer<ApproxSyncPolicy>>(ApproxSyncPolicy(10), image_sub_, depth_sub_);
  sync_->registerCallback(boost::bind(&SoccerRobotDetector::syncCallback, this, _1, _2));

  camera_info_sub_ = nh_.subscribe(camera_info_topic, 1, &SoccerRobotDetector::cameraInfoCallback, this);
  debug_pub_ = it_.advertise(debug_topic, 1);
  center_pub_ = nh_.advertise<geometry_msgs::PointStamped>(center_topic, 1);
}

void SoccerRobotDetector::cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& info_msg)
{
  camera_model_.fromCameraInfo(info_msg);
  has_camera_info_ = true;
}

void SoccerRobotDetector::syncCallback(const sensor_msgs::ImageConstPtr& image_msg,
                                       const sensor_msgs::ImageConstPtr& depth_msg)
{
  cv::Mat gray;
  try {
    gray = cv_bridge::toCvShare(image_msg, sensor_msgs::image_encodings::MONO8)->image.clone();
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR_STREAM("cv_bridge exception while converting intensity image: " << e.what());
    return;
  }

  cv::Mat depth_float;
  try {
    if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
      cv::Mat depth_16 = cv_bridge::toCvShare(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1)->image;
      depth_16.convertTo(depth_float, CV_32F, 0.001);
    } else {
      depth_float = cv_bridge::toCvShare(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1)->image.clone();
    }
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR_STREAM("cv_bridge exception while converting depth image: " << e.what());
    return;
  }

  if (depth_float.empty()) {
    ROS_WARN_THROTTLE(5.0, "Depth image is empty, skipping frame.");
    return;
  }

  cv::Mat blur_img;
  cv::medianBlur(gray, blur_img, 5);

  std::vector<cv::Vec3f> circles;
  cv::HoughCircles(blur_img, circles, cv::HOUGH_GRADIENT, dp_, minDist_, param1_, param2_, minRadius_, maxRadius_);

  cv::Mat debug_img;
  cv::cvtColor(gray, debug_img, cv::COLOR_GRAY2BGR);

  if (circles.empty()) {
    has_last_circle_ = false;
    sensor_msgs::ImagePtr debug_msg = cv_bridge::CvImage(image_msg->header, sensor_msgs::image_encodings::BGR8, debug_img).toImageMsg();
    debug_pub_.publish(debug_msg);
    return;
  }

  size_t best_index = 0;
  if (circles.size() > 1) {
    if (has_last_circle_) {
      double best_dist = std::numeric_limits<double>::max();
      for (size_t i = 0; i < circles.size(); ++i) {
        cv::Point2f c(circles[i][0], circles[i][1]);
        double dist = cv::norm(c - last_center_);
        if (dist < best_dist) {
          best_dist = dist;
          best_index = i;
        }
      }
    } else {
      float largest_radius = 0.f;
      for (size_t i = 0; i < circles.size(); ++i) {
        if (circles[i][2] > largest_radius) {
          largest_radius = circles[i][2];
          best_index = i;
        }
      }
    }
  }

  cv::Vec3f c = circles[best_index];
  cv::Point2f center_f(c[0], c[1]);
  int radius = cvRound(c[2]);

  if (has_last_circle_) {
    center_f.x = 0.7f * last_center_.x + 0.3f * center_f.x;
    center_f.y = 0.7f * last_center_.y + 0.3f * center_f.y;
    radius = static_cast<int>(0.7f * static_cast<float>(last_radius_) + 0.3f * static_cast<float>(radius));
  }

  cv::Point center(cvRound(center_f.x), cvRound(center_f.y));
  last_center_ = center_f;
  last_radius_ = radius;
  has_last_circle_ = true;

  cv::circle(debug_img, center, radius, cv::Scalar(0, 255, 0), 2);
  cv::circle(debug_img, center, 3, cv::Scalar(0, 0, 255), -1);

  int x1 = std::max(center.x - radius, 0);
  int y1 = std::max(center.y - radius, 0);
  int x2 = std::min(center.x + radius, gray.cols - 1);
  int y2 = std::min(center.y + radius, gray.rows - 1);
  cv::Rect roi(cv::Point(x1, y1), cv::Point(x2 + 1, y2 + 1));

  if (roi.width <= 0 || roi.height <= 0) {
    sensor_msgs::ImagePtr debug_msg = cv_bridge::CvImage(image_msg->header, sensor_msgs::image_encodings::BGR8, debug_img).toImageMsg();
    debug_pub_.publish(debug_msg);
    return;
  }

  cv::Point roi_center(center.x - roi.x, center.y - roi.y);
  cv::Mat circle_roi = gray(roi).clone();
  cv::Mat depth_roi = depth_float(roi);

  cv::Mat grad_x, grad_y, grad_mag;
  cv::Sobel(circle_roi, grad_x, CV_32F, 1, 0, 3);
  cv::Sobel(circle_roi, grad_y, CV_32F, 0, 1, 3);
  cv::magnitude(grad_x, grad_y, grad_mag);

  double mean_intensity = cv::mean(circle_roi)[0];
  double canny_low = std::max(50.0, 120.0 - mean_intensity / 2.0);
  double canny_high = canny_low * 2.5;

  cv::Mat edges;
  cv::Canny(circle_roi, edges, canny_low, canny_high);
  cv::dilate(edges, edges, cv::Mat(), cv::Point(-1, -1), 1);

  std::vector<cv::Vec4i> raw_lines;
  cv::HoughLinesP(edges, raw_lines, hough_rho_, CV_PI / 180.0 * hough_theta_,
                  hough_threshold_ + 30, minLineLength_ + 30, maxLineGap_ / 2.0);

  std::vector<cv::Vec4i> filtered_lines;
  filtered_lines.reserve(raw_lines.size());
  for (const auto& l : raw_lines) {
    cv::Point p1(l[0], l[1]);
    cv::Point p2(l[2], l[3]);
    double len = cv::norm(p1 - p2);
    if (len < 30.0 || len > radius * 1.3) {
      continue;
    }

    cv::Point mid((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);
    if (cv::norm(mid - roi_center) > radius * 0.85) {
      continue;
    }

    double angle = std::fabs(std::atan2(static_cast<double>(p2.y - p1.y),
                                        static_cast<double>(p2.x - p1.x)) * 180.0 / CV_PI);
    if (angle < 15.0 || angle > 165.0) {
      continue;
    }

    cv::LineIterator it(circle_roi, p1, p2);
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    for (int i = 0; i < it.count; ++i, ++it) {
      float val = static_cast<float>(circle_roi.at<uchar>(it.pos()));
      min_val = std::min(min_val, val);
      max_val = std::max(max_val, val);
    }
    if (max_val - min_val < 30.0f) {
      continue;
    }

    filtered_lines.push_back(l);
  }

  for (const auto& l : filtered_lines) {
    cv::line(debug_img(roi), cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255, 0, 0), 1);
  }

  if (use_harris_) {
    cv::Mat corners, dst_norm;
    cv::cornerHarris(circle_roi, corners, 2, 3, 0.04);
    cv::normalize(corners, dst_norm, 0, 255, cv::NORM_MINMAX);

    for (int y = 0; y < dst_norm.rows; ++y) {
      for (int x = 0; x < dst_norm.cols; ++x) {
        if (static_cast<int>(dst_norm.at<float>(y, x)) > 180) {
          cv::circle(debug_img(roi), cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
        }
      }
    }
  } else {
    std::vector<cv::Point2f> shi_tomasi_corners;
    cv::goodFeaturesToTrack(circle_roi,
                            shi_tomasi_corners,
                            maxCorners_,
                            qualityLevel_,
                            minCornerDistance_);

    for (const auto& pt : shi_tomasi_corners) {
      cv::circle(debug_img(roi), pt, 2, cv::Scalar(0, 0, 255), -1);
    }
  }

  if (has_camera_info_) {
    double median_depth = extractMedianDepth(depth_roi, roi_center, radius);
    if (std::isfinite(median_depth) && median_depth > 0.0) {
      cv::Point2d pixel(static_cast<double>(center.x), static_cast<double>(center.y));
      cv::Point3d ray = camera_model_.projectPixelTo3dRay(pixel);
      cv::Point3d position = ray * median_depth;

      geometry_msgs::PointStamped center_msg;
      center_msg.header = image_msg->header;
      center_msg.point.x = position.x;
      center_msg.point.y = position.y;
      center_msg.point.z = position.z;
      center_pub_.publish(center_msg);

      std::ostringstream text_stream;
      text_stream << std::fixed << std::setprecision(2)
                  << "(x:" << position.x << ", y:" << position.y
                  << ", z:" << position.z << ")";
      cv::putText(debug_img, text_stream.str(),
                  center + cv::Point(-radius, -radius - 10),
                  cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
    } else {
      ROS_WARN_THROTTLE(2.0, "Unable to estimate robot depth inside detected circle.");
    }
  } else {
    ROS_WARN_THROTTLE(5.0, "Waiting for camera info to estimate 3D position.");
  }

  sensor_msgs::ImagePtr debug_msg = cv_bridge::CvImage(image_msg->header, sensor_msgs::image_encodings::BGR8, debug_img).toImageMsg();
  debug_pub_.publish(debug_msg);
}

double SoccerRobotDetector::extractMedianDepth(const cv::Mat& depth_roi,
                                               const cv::Point& roi_center,
                                               int radius) const
{
  std::vector<float> depths;
  depths.reserve(static_cast<size_t>(radius * radius));

  for (int y = 0; y < depth_roi.rows; ++y) {
    const float* depth_ptr = depth_roi.ptr<float>(y);
    for (int x = 0; x < depth_roi.cols; ++x) {
      if (cv::norm(cv::Point(x, y) - roi_center) > radius) {
        continue;
      }
      float depth = depth_ptr[x];
      if (!std::isfinite(depth) || depth <= 0.0f) {
        continue;
      }
      depths.push_back(depth);
    }
  }

  if (depths.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  size_t mid_index = depths.size() / 2;
  std::nth_element(depths.begin(), depths.begin() + mid_index, depths.end());
  double median = depths[mid_index];

  if (depths.size() % 2 == 0) {
    float lower = *std::max_element(depths.begin(), depths.begin() + mid_index);
    median = 0.5 * (median + static_cast<double>(lower));
  }

  return median;
}

