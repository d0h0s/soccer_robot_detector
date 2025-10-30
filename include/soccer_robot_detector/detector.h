//
// Created by xianghong on 10/20/25.
//

#pragma once
#include <cv_bridge/cv_bridge.h>
#include <dynamic_reconfigure/server.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Header.h>
#include <string>
#include <vector>

#include <soccer_robot_detector/SoccerRobotDetectorConfig.h>

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

  struct ArcDebugImages
  {
    cv::Mat blurred;
    cv::Mat edges;
    cv::Mat dilated_edges;
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

  bool detectArcSegment(const cv::Mat& gray, ArcSegment& best_arc,
                        ArcDebugImages* debug) const;
  static double computeAngularCoverage(const std::vector<cv::Point2f>& points,
                                       const cv::Point2f& center);
  double estimateLineThickness(const cv::Mat& edge_img, const cv::Point& p1,
                               const cv::Point& p2) const;
  double sampleThicknessAtPoint(const cv::Mat& edge_img, const cv::Point2f& point,
                                const cv::Point2f& normal) const;
  LineClassification classifyLine(double length_ratio, double thickness) const;
  std::string formatLineLabel(const LineFeature& feature) const;
  void publishArcDebugImages(const ArcDebugImages& images, const std_msgs::Header& header,
                             const cv::Mat& gray, const ArcSegment* arc,
                             bool arc_found);
  void enforceParameterConstraints(soccer_robot_detector::SoccerRobotDetectorConfig& config);
  void reconfigureCallback(soccer_robot_detector::SoccerRobotDetectorConfig& config,
                           uint32_t level);

  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher debug_pub_;
  image_transport::Publisher arc_blur_pub_;
  image_transport::Publisher arc_edges_pub_;
  image_transport::Publisher arc_dilated_pub_;
  image_transport::Publisher arc_contours_pub_;

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
  int arc_blur_kernel_size_;
  double arc_canny_low_;
  double arc_canny_high_;
  int arc_dilate_iterations_;
  double arc_approx_epsilon_;
  double arc_score_error_weight_;
  bool arc_visualize_blur_ = false;
  bool arc_visualize_edges_ = false;
  bool arc_visualize_dilated_edges_ = false;
  bool arc_visualize_contours_ = false;

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

  int median_blur_kernel_size_;
  double roi_canny_low_min_;
  double roi_canny_low_base_;
  double roi_canny_low_mean_scale_;
  double roi_canny_high_ratio_;
  int roi_edge_dilation_iterations_;
  double line_hough_threshold_extra_;
  double line_min_length_extra_;
  double line_max_gap_scale_;
  double line_length_min_px_;
  double line_length_max_radius_ratio_;
  double line_midpoint_max_radius_ratio_;
  double line_angle_min_deg_;
  double line_angle_max_deg_;
  double line_intensity_diff_min_;

  dynamic_reconfigure::Server<soccer_robot_detector::SoccerRobotDetectorConfig> reconfigure_server_;
};

