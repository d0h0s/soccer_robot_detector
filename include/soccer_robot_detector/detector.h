//
// Created by xianghong on 10/20/25.
//

#pragma once
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <string>
#include <vector>

class SoccerRobotDetector
{
public:
  SoccerRobotDetector(ros::NodeHandle& nh);
  void imageCallback(const sensor_msgs::ImageConstPtr& msg);

private:
  struct ArcSegment
  {
    cv::Point2f center;
    float radius;
    std::vector<cv::Point2f> points;
    cv::Rect bounding_box;
    double coverage_deg = 0.0;
    double mean_error = 0.0;
  };

  enum class LineClassification
  {
    ShortThin,
    LongThick,
    Rejected
  };

  struct LineFeature
  {
    cv::Vec4i line;
    LineClassification classification;
    double length = 0.0;
    double thickness = 0.0;
  };

  bool detectArcSegment(const cv::Mat& gray, ArcSegment& best_arc) const;
  static double computeAngularCoverage(const std::vector<cv::Point2f>& points,
                                       const cv::Point2f& center);
  double estimateLineThickness(const cv::Mat& edge_img, const cv::Point& p1,
                               const cv::Point& p2) const;
  double sampleThicknessAtPoint(const cv::Mat& edge_img, const cv::Point2f& point,
                                const cv::Point2f& normal) const;
  LineClassification classifyLine(double length_ratio, double thickness) const;
  std::string formatLineLabel(const LineFeature& feature) const;

  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher debug_pub_;

  double dp_, minDist_, param1_, param2_;
  int minRadius_, maxRadius_;
  bool use_harris_;
  bool use_hough_circle_ = true;

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

  // Arc detection parameters
  int min_arc_points_;
  double arc_max_fit_error_;
  double arc_min_coverage_deg_;
  double roi_padding_scale_;

  // Soccer pattern prior parameters
  double short_length_ratio_min_;
  double short_length_ratio_max_;
  double long_length_ratio_min_;
  double long_length_ratio_max_;
  double thin_thickness_min_;
  double thin_thickness_max_;
  double thick_thickness_min_;
  double thick_thickness_max_;
  int thickness_search_radius_;
};

