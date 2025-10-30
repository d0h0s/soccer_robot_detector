//
// Created by xianghong on 10/20/25.
//

#include "soccer_robot_detector/detector.h"

#include <algorithm>
#include <boost/bind/bind.hpp>
#include <boost/bind.hpp>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>

namespace
{
cv::Rect clampRectToImage(const cv::Rect& rect, const cv::Size& image_size)
{
  int x = std::max(rect.x, 0);
  int y = std::max(rect.y, 0);
  int width = std::min(rect.x + rect.width, image_size.width) - x;
  int height = std::min(rect.y + rect.height, image_size.height) - y;
  if (width <= 0 || height <= 0)
    return cv::Rect();
  return cv::Rect(x, y, width, height);
}

int ensureOddKernel(int value, int min_value)
{
  if (value < min_value)
    value = min_value;
  if (value % 2 == 0)
    ++value;
  return value;
}

bool fitCircleLeastSquares(const std::vector<cv::Point2f>& points, cv::Point2f& center,
                           float& radius)
{
  if (points.size() < 3)
    return false;

  cv::Point2f mean(0.0f, 0.0f);
  for (const auto& pt : points)
  {
    mean.x += pt.x;
    mean.y += pt.y;
  }
  mean.x /= static_cast<float>(points.size());
  mean.y /= static_cast<float>(points.size());

  cv::Mat A(points.size(), 3, CV_64F);
  cv::Mat b(points.size(), 1, CV_64F);
  for (size_t i = 0; i < points.size(); ++i)
  {
    double x = static_cast<double>(points[i].x - mean.x);
    double y = static_cast<double>(points[i].y - mean.y);
    A.at<double>(i, 0) = 2.0 * x;
    A.at<double>(i, 1) = 2.0 * y;
    A.at<double>(i, 2) = 1.0;
    b.at<double>(i, 0) = x * x + y * y;
  }

  cv::Mat sol;
  if (!cv::solve(A, b, sol, cv::DECOMP_SVD))
    return false;

  double cx = sol.at<double>(0);
  double cy = sol.at<double>(1);
  double d = sol.at<double>(2);
  double r_sq = cx * cx + cy * cy + d;
  if (r_sq <= 0.0)
    return false;

  cx += static_cast<double>(mean.x);
  cy += static_cast<double>(mean.y);

  center = cv::Point2f(static_cast<float>(cx), static_cast<float>(cy));
  radius = static_cast<float>(std::sqrt(r_sq));

  return std::isfinite(center.x) && std::isfinite(center.y) && std::isfinite(radius) &&
         radius > 0.0f;
}
}  // namespace

SoccerRobotDetector::SoccerRobotDetector(ros::NodeHandle& nh)
    : nh_(nh), it_(nh), reconfigure_server_(nh)
{
  nh_.param("dp", dp_, 1.2);
  nh_.param("minDist", minDist_, 50.0);
  nh_.param("param1", param1_, 100.0);
  nh_.param("param2", param2_, 30.0);
  nh_.param("minRadius", minRadius_, 30);
  nh_.param("maxRadius", maxRadius_, 200);
  nh_.param("use_hough_circle", use_hough_circle_, true);
  nh_.param("use_harris", use_harris_, true);
  nh_.param("roi_padding_scale", roi_padding_scale_, 0.15);
  nh_.param("median_blur_kernel_size", median_blur_kernel_size_, 5);

  nh_.param("hough_rho", hough_rho_, 1.0);
  nh_.param("hough_theta", hough_theta_, 1.0);
  nh_.param("hough_threshold", hough_threshold_, 50);
  nh_.param("minLineLength", minLineLength_, 30.0);
  nh_.param("maxLineGap", maxLineGap_, 10.0);
  nh_.param("line_hough_threshold_extra", line_hough_threshold_extra_, 30.0);
  nh_.param("line_min_length_extra", line_min_length_extra_, 30.0);
  nh_.param("line_max_gap_scale", line_max_gap_scale_, 0.5);
  nh_.param("line_length_min_px", line_length_min_px_, 30.0);
  nh_.param("line_length_max_radius_ratio", line_length_max_radius_ratio_, 1.3);
  nh_.param("line_midpoint_max_radius_ratio", line_midpoint_max_radius_ratio_, 0.85);
  nh_.param("line_angle_min_deg", line_angle_min_deg_, 15.0);
  nh_.param("line_angle_max_deg", line_angle_max_deg_, 165.0);
  nh_.param("line_intensity_diff_min", line_intensity_diff_min_, 30.0);

  nh_.param("maxCorners", maxCorners_, 50);
  nh_.param("qualityLevel", qualityLevel_, 0.02);
  nh_.param("minCornerDistance", minCornerDistance_, 10.0);

  nh_.param("roi_canny_low_min", roi_canny_low_min_, 50.0);
  nh_.param("roi_canny_low_base", roi_canny_low_base_, 120.0);
  nh_.param("roi_canny_low_mean_scale", roi_canny_low_mean_scale_, 0.5);
  nh_.param("roi_canny_high_ratio", roi_canny_high_ratio_, 2.5);
  nh_.param("roi_edge_dilation_iterations", roi_edge_dilation_iterations_, 1);

  nh_.param("min_arc_points", min_arc_points_, 80);
  nh_.param("arc_max_fit_error", arc_max_fit_error_, 8.0);
  nh_.param("arc_min_coverage_deg", arc_min_coverage_deg_, 80.0);
  nh_.param("arc_blur_kernel_size", arc_blur_kernel_size_, 5);
  nh_.param("arc_canny_low", arc_canny_low_, 60.0);
  nh_.param("arc_canny_high", arc_canny_high_, 150.0);
  nh_.param("arc_dilate_iterations", arc_dilate_iterations_, 1);
  nh_.param("arc_approx_poly_epsilon", arc_approx_epsilon_, 2.0);
  nh_.param("arc_score_error_weight", arc_score_error_weight_, 5.0);
  nh_.param("arc_visualize_blur", arc_visualize_blur_, false);
  nh_.param("arc_visualize_edges", arc_visualize_edges_, false);
  nh_.param("arc_visualize_dilated_edges", arc_visualize_dilated_edges_, false);
  nh_.param("arc_visualize_contours", arc_visualize_contours_, false);

  nh_.param("short_length_ratio_min", short_length_ratio_min_, 0.25);
  nh_.param("short_length_ratio_max", short_length_ratio_max_, 0.55);
  nh_.param("long_length_ratio_min", long_length_ratio_min_, 0.55);
  nh_.param("long_length_ratio_max", long_length_ratio_max_, 0.95);
  nh_.param("thin_thickness_min", thin_thickness_min_, 1.0);
  nh_.param("thin_thickness_max", thin_thickness_max_, 3.0);
  nh_.param("thick_thickness_min", thick_thickness_min_, 3.0);
  nh_.param("thick_thickness_max", thick_thickness_max_, 6.0);
  nh_.param("thickness_search_radius", thickness_search_radius_, 8);

  median_blur_kernel_size_ = ensureOddKernel(median_blur_kernel_size_, 3);
  arc_blur_kernel_size_ = ensureOddKernel(arc_blur_kernel_size_, 3);
  if (arc_canny_high_ < arc_canny_low_)
    arc_canny_high_ = arc_canny_low_;
  if (line_angle_min_deg_ > line_angle_max_deg_)
    line_angle_min_deg_ = line_angle_max_deg_;
  roi_edge_dilation_iterations_ = std::max(0, roi_edge_dilation_iterations_);
  arc_dilate_iterations_ = std::max(0, arc_dilate_iterations_);
  thickness_search_radius_ = std::max(1, thickness_search_radius_);

  image_sub_ = it_.subscribe("/camera/infra1/image_rect_raw", 1,
                             &SoccerRobotDetector::imageCallback, this);
  debug_pub_ = it_.advertise("/soccer_robot_detector/debug", 1);
  arc_blur_pub_ = it_.advertise("/soccer_robot_detector/arc/blurred", 1);
  arc_edges_pub_ = it_.advertise("/soccer_robot_detector/arc/edges", 1);
  arc_dilated_pub_ = it_.advertise("/soccer_robot_detector/arc/dilated_edges", 1);
  arc_contours_pub_ = it_.advertise("/soccer_robot_detector/arc/contours", 1);

  has_last_circle_ = false;

  soccer_robot_detector::SoccerRobotDetectorConfig initial_config;
  initial_config.use_hough_circle = use_hough_circle_;
  initial_config.dp = dp_;
  initial_config.minDist = minDist_;
  initial_config.param1 = param1_;
  initial_config.param2 = param2_;
  initial_config.minRadius = minRadius_;
  initial_config.maxRadius = maxRadius_;
  initial_config.roi_padding_scale = roi_padding_scale_;
  initial_config.median_blur_kernel_size = median_blur_kernel_size_;
  initial_config.use_harris = use_harris_;
  initial_config.maxCorners = maxCorners_;
  initial_config.qualityLevel = qualityLevel_;
  initial_config.minCornerDistance = minCornerDistance_;
  initial_config.hough_rho = hough_rho_;
  initial_config.hough_theta = hough_theta_;
  initial_config.hough_threshold = hough_threshold_;
  initial_config.minLineLength = minLineLength_;
  initial_config.maxLineGap = maxLineGap_;
  initial_config.line_hough_threshold_extra = line_hough_threshold_extra_;
  initial_config.line_min_length_extra = line_min_length_extra_;
  initial_config.line_max_gap_scale = line_max_gap_scale_;
  initial_config.line_length_min_px = line_length_min_px_;
  initial_config.line_length_max_radius_ratio = line_length_max_radius_ratio_;
  initial_config.line_midpoint_max_radius_ratio = line_midpoint_max_radius_ratio_;
  initial_config.line_angle_min_deg = line_angle_min_deg_;
  initial_config.line_angle_max_deg = line_angle_max_deg_;
  initial_config.line_intensity_diff_min = line_intensity_diff_min_;
  initial_config.roi_canny_low_min = roi_canny_low_min_;
  initial_config.roi_canny_low_base = roi_canny_low_base_;
  initial_config.roi_canny_low_mean_scale = roi_canny_low_mean_scale_;
  initial_config.roi_canny_high_ratio = roi_canny_high_ratio_;
  initial_config.roi_edge_dilation_iterations = roi_edge_dilation_iterations_;
  initial_config.min_arc_points = min_arc_points_;
  initial_config.arc_max_fit_error = arc_max_fit_error_;
  initial_config.arc_min_coverage_deg = arc_min_coverage_deg_;
  initial_config.arc_blur_kernel_size = arc_blur_kernel_size_;
  initial_config.arc_canny_low = arc_canny_low_;
  initial_config.arc_canny_high = arc_canny_high_;
  initial_config.arc_dilate_iterations = arc_dilate_iterations_;
  initial_config.arc_approx_poly_epsilon = arc_approx_epsilon_;
  initial_config.arc_score_error_weight = arc_score_error_weight_;
  initial_config.short_length_ratio_min = short_length_ratio_min_;
  initial_config.short_length_ratio_max = short_length_ratio_max_;
  initial_config.long_length_ratio_min = long_length_ratio_min_;
  initial_config.long_length_ratio_max = long_length_ratio_max_;
  initial_config.thin_thickness_min = thin_thickness_min_;
  initial_config.thin_thickness_max = thin_thickness_max_;
  initial_config.thick_thickness_min = thick_thickness_min_;
  initial_config.thick_thickness_max = thick_thickness_max_;
  initial_config.thickness_search_radius = thickness_search_radius_;
  initial_config.arc_visualize_blur = arc_visualize_blur_;
  initial_config.arc_visualize_edges = arc_visualize_edges_;
  initial_config.arc_visualize_dilated_edges = arc_visualize_dilated_edges_;
  initial_config.arc_visualize_contours = arc_visualize_contours_;

  enforceParameterConstraints(initial_config);
  reconfigureCallback(initial_config, 0);

  dynamic_reconfigure::Server<soccer_robot_detector::SoccerRobotDetectorConfig>::CallbackType cb;
  cb = boost::bind(&SoccerRobotDetector::reconfigureCallback, this, boost::placeholders::_1,
                   boost::placeholders::_2);
  reconfigure_server_.setCallback(cb);
  reconfigure_server_.updateConfig(initial_config);
}

void SoccerRobotDetector::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  cv::Mat gray;
  try
  {
    gray = cv_bridge::toCvShare(msg, "mono8")->image.clone();
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  cv::Mat blur_img;
  cv::medianBlur(gray, blur_img, median_blur_kernel_size_);

  cv::Mat debug_img;
  cv::cvtColor(gray, debug_img, cv::COLOR_GRAY2BGR);

  std::vector<cv::Vec3f> circles;
  if (use_hough_circle_)
  {
    cv::HoughCircles(blur_img, circles, cv::HOUGH_GRADIENT, dp_, minDist_, param1_, param2_,
                     minRadius_, maxRadius_);
  }

  bool detection_success = false;
  cv::Rect roi;
  cv::Point center_pixel;
  int radius_px = 0;
  std::vector<cv::Point> arc_polyline;

  if (!circles.empty())
  {
    cv::Vec3f c = circles.front();
    cv::Point2f center_f(c[0], c[1]);
    float radius_f = c[2];

    if (has_last_circle_)
    {
      center_f.x = 0.7f * last_center_.x + 0.3f * center_f.x;
      center_f.y = 0.7f * last_center_.y + 0.3f * center_f.y;
      radius_f = 0.7f * static_cast<float>(last_radius_) + 0.3f * radius_f;
    }

    center_pixel = cv::Point(cvRound(center_f.x), cvRound(center_f.y));
    radius_px = static_cast<int>(std::round(radius_f));

    last_center_ = center_f;
    last_radius_ = radius_px;
    has_last_circle_ = true;

    int padding = static_cast<int>(roi_padding_scale_ * radius_px);
    int x1 = center_pixel.x - radius_px - padding;
    int y1 = center_pixel.y - radius_px - padding;
    int x2 = center_pixel.x + radius_px + padding;
    int y2 = center_pixel.y + radius_px + padding;
    roi = clampRectToImage(cv::Rect(x1, y1, x2 - x1, y2 - y1), gray.size());
    detection_success = roi.area() > 0;

    cv::circle(debug_img, center_pixel, radius_px, cv::Scalar(0, 255, 0), 2);
    cv::circle(debug_img, center_pixel, 3, cv::Scalar(0, 0, 255), -1);
  }

  if (!detection_success)
  {
    ArcSegment arc_candidate;
    bool need_arc_debug = arc_visualize_blur_ || arc_visualize_edges_ ||
                          arc_visualize_dilated_edges_ || arc_visualize_contours_;
    ArcDebugImages arc_debug_images;
    ArcDebugImages* debug_ptr = need_arc_debug ? &arc_debug_images : nullptr;

    bool arc_found = detectArcSegment(blur_img, arc_candidate, debug_ptr);
    if (need_arc_debug)
      publishArcDebugImages(arc_debug_images, msg->header, gray,
                            arc_found ? &arc_candidate : nullptr, arc_found);

    if (arc_found)
    {
      cv::Point2f center_f = arc_candidate.center;
      float radius_f = arc_candidate.radius;
      if (has_last_circle_)
      {
        center_f.x = 0.7f * last_center_.x + 0.3f * center_f.x;
        center_f.y = 0.7f * last_center_.y + 0.3f * center_f.y;
        radius_f = 0.7f * static_cast<float>(last_radius_) + 0.3f * radius_f;
      }

      center_pixel = cv::Point(cvRound(center_f.x), cvRound(center_f.y));
      radius_px = static_cast<int>(std::round(radius_f));

      last_center_ = center_f;
      last_radius_ = radius_px;
      has_last_circle_ = true;

      int padding = static_cast<int>(roi_padding_scale_ * radius_px);
      int x1 = center_pixel.x - radius_px - padding;
      int y1 = center_pixel.y - radius_px - padding;
      int x2 = center_pixel.x + radius_px + padding;
      int y2 = center_pixel.y + radius_px + padding;
      roi = clampRectToImage(cv::Rect(x1, y1, x2 - x1, y2 - y1), gray.size());
      detection_success = roi.area() > 0;

      arc_polyline.reserve(arc_candidate.points.size());
      for (const auto& pt : arc_candidate.points)
        arc_polyline.emplace_back(cv::Point(cvRound(pt.x), cvRound(pt.y)));

      cv::circle(debug_img, center_pixel, radius_px, cv::Scalar(0, 255, 255), 2);
      if (!arc_polyline.empty())
      {
        cv::polylines(debug_img, arc_polyline, false, cv::Scalar(0, 165, 255), 2, cv::LINE_AA);
        cv::circle(debug_img, arc_polyline.front(), 3, cv::Scalar(0, 200, 255), -1);
        cv::circle(debug_img, arc_polyline.back(), 3, cv::Scalar(0, 200, 255), -1);
      }
      cv::rectangle(debug_img, arc_candidate.bounding_box, cv::Scalar(0, 140, 255), 1);
      cv::circle(debug_img, cv::Point(cvRound(arc_candidate.center.x),
                                      cvRound(arc_candidate.center.y)),
                 3, cv::Scalar(0, 69, 255), -1);
    }
  }

  if (!detection_success)
  {
    has_last_circle_ = false;
    sensor_msgs::ImagePtr debug_msg =
        cv_bridge::CvImage(std_msgs::Header(), "bgr8", debug_img).toImageMsg();
    debug_pub_.publish(debug_msg);
    return;
  }

  cv::Mat circle_roi = gray(roi).clone();
  cv::Point roi_center(center_pixel.x - roi.x, center_pixel.y - roi.y);

  cv::Mat grad_x, grad_y, grad_mag;
  cv::Sobel(circle_roi, grad_x, CV_32F, 1, 0, 3);
  cv::Sobel(circle_roi, grad_y, CV_32F, 0, 1, 3);
  cv::magnitude(grad_x, grad_y, grad_mag);

  double mean_intensity = cv::mean(circle_roi)[0];
  double adaptive_low = roi_canny_low_base_ - mean_intensity * roi_canny_low_mean_scale_;
  double canny_low = std::max(roi_canny_low_min_, adaptive_low);
  double canny_high = canny_low * roi_canny_high_ratio_;

  cv::Mat edges;
  cv::Canny(circle_roi, edges, canny_low, canny_high);
  cv::Mat edges_dilated;
  if (roi_edge_dilation_iterations_ > 0)
    cv::dilate(edges, edges_dilated, cv::Mat(), cv::Point(-1, -1), roi_edge_dilation_iterations_);
  else
    edges.copyTo(edges_dilated);

  std::vector<cv::Vec4i> raw_lines;
  double theta_resolution = CV_PI / 180.0 * hough_theta_;
  double hough_threshold = hough_threshold_ + line_hough_threshold_extra_;
  double min_line_length = minLineLength_ + line_min_length_extra_;
  double max_line_gap = maxLineGap_ * line_max_gap_scale_;
  cv::HoughLinesP(edges_dilated, raw_lines, hough_rho_, theta_resolution, hough_threshold,
                  min_line_length, max_line_gap);

  std::vector<LineFeature> filtered_lines;
  for (const auto& l : raw_lines)
  {
    cv::Point p1(l[0], l[1]);
    cv::Point p2(l[2], l[3]);
    double len = cv::norm(p1 - p2);
    if (len < line_length_min_px_ || len > radius_px * line_length_max_radius_ratio_)
      continue;

    cv::Point mid((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);
    if (cv::norm(mid - roi_center) > radius_px * line_midpoint_max_radius_ratio_)
      continue;

    double angle = std::fabs(std::atan2(p2.y - p1.y, p2.x - p1.x) * 180.0 / CV_PI);
    if (angle < line_angle_min_deg_ || angle > line_angle_max_deg_)
      continue;

    cv::LineIterator it(circle_roi, p1, p2);
    std::vector<float> profile;
    profile.reserve(it.count);
    for (int i = 0; i < it.count; ++i, ++it)
      profile.push_back(static_cast<float>(circle_roi.at<uchar>(it.pos())));

    auto minmax = std::minmax_element(profile.begin(), profile.end());
    double diff = std::fabs(*minmax.second - *minmax.first);
    if (diff < line_intensity_diff_min_)
      continue;

    double length_ratio = len / static_cast<double>(radius_px);
    double thickness = estimateLineThickness(edges, p1, p2);

    LineClassification classification = classifyLine(length_ratio, thickness);
    if (classification == LineClassification::Rejected)
      continue;

    filtered_lines.push_back({l, classification, len, thickness});
  }

  for (const auto& lf : filtered_lines)
  {
    const auto& l = lf.line;
    cv::Scalar color = lf.classification == LineClassification::ShortThin
                           ? cv::Scalar(255, 200, 0)
                           : cv::Scalar(255, 0, 0);
    int thickness_px = lf.classification == LineClassification::ShortThin ? 1 : 2;
    cv::line(debug_img(roi), cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), color,
             thickness_px, cv::LINE_AA);
    cv::Point mid((l[0] + l[2]) / 2, (l[1] + l[3]) / 2);
    cv::putText(debug_img(roi), formatLineLabel(lf), mid, cv::FONT_HERSHEY_SIMPLEX, 0.3,
                color, 1, cv::LINE_AA);
  }

  if (use_harris_)
  {
    cv::Mat corners, dst_norm;
    cv::cornerHarris(circle_roi, corners, 2, 3, 0.04);
    cv::normalize(corners, dst_norm, 0, 255, cv::NORM_MINMAX);

    for (int y = 0; y < dst_norm.rows; ++y)
    {
      for (int x = 0; x < dst_norm.cols; ++x)
      {
        if (static_cast<int>(dst_norm.at<float>(y, x)) > 180)
          cv::circle(debug_img(roi), cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
      }
    }
  }
  else
  {
    std::vector<cv::Point2f> shi_tomasi_corners;
    cv::goodFeaturesToTrack(circle_roi, shi_tomasi_corners, maxCorners_, qualityLevel_,
                            minCornerDistance_);

    for (const auto& pt : shi_tomasi_corners)
      cv::circle(debug_img(roi), pt, 2, cv::Scalar(0, 0, 255), -1);
  }

  if (!roi.empty())
    cv::rectangle(debug_img, roi, cv::Scalar(0, 255, 255), 1);

  sensor_msgs::ImagePtr debug_msg =
      cv_bridge::CvImage(std_msgs::Header(), "bgr8", debug_img).toImageMsg();
  debug_pub_.publish(debug_msg);
}

double SoccerRobotDetector::estimateLineThickness(const cv::Mat& edge_img, const cv::Point& p1,
                                                  const cv::Point& p2) const
{
  cv::Point2f dir(static_cast<float>(p2.x - p1.x), static_cast<float>(p2.y - p1.y));
  double length = cv::norm(dir);
  if (length < 1e-3)
    return 0.0;

  dir *= static_cast<float>(1.0 / length);
  cv::Point2f normal(-dir.y, dir.x);

  int samples = std::max(5, static_cast<int>(length / 10.0));
  double total_width = 0.0;
  int valid_samples = 0;
  for (int i = 1; i <= samples; ++i)
  {
    double t = static_cast<double>(i) / (samples + 1);
    cv::Point2f pt = cv::Point2f(static_cast<float>(p1.x), static_cast<float>(p1.y)) +
                     dir * static_cast<float>(length * t);
    double width = sampleThicknessAtPoint(edge_img, pt, normal);
    if (width > 0.0)
    {
      total_width += width;
      ++valid_samples;
    }
  }

  if (valid_samples == 0)
    return 0.0;

  return total_width / static_cast<double>(valid_samples);
}

namespace
{
bool isInside(const cv::Mat& img, int x, int y)
{
  return x >= 0 && y >= 0 && x < img.cols && y < img.rows;
}
}  // namespace

double SoccerRobotDetector::sampleThicknessAtPoint(const cv::Mat& edge_img,
                                                   const cv::Point2f& point,
                                                   const cv::Point2f& normal) const
{
  if (edge_img.empty())
    return 0.0;

  auto scan = [&](int direction) {
    int last_on = -1;
    bool seen_on = false;
    for (int step = 0; step <= thickness_search_radius_; ++step)
    {
      float fx = point.x + normal.x * static_cast<float>(direction * step);
      float fy = point.y + normal.y * static_cast<float>(direction * step);
      int x = cvRound(fx);
      int y = cvRound(fy);
      if (!isInside(edge_img, x, y))
        break;
      if (edge_img.at<uchar>(y, x) > 0)
      {
        seen_on = true;
        last_on = step;
      }
      else if (seen_on)
      {
        break;
      }
    }
    return last_on < 0 ? 0 : last_on;
  };

  int pos = scan(1);
  int neg = scan(-1);
  if (pos == 0 && neg == 0)
    return 0.0;

  return static_cast<double>(pos + neg + 1);
}

SoccerRobotDetector::LineClassification
SoccerRobotDetector::classifyLine(double length_ratio, double thickness) const
{
  if (length_ratio >= short_length_ratio_min_ && length_ratio <= short_length_ratio_max_ &&
      thickness >= thin_thickness_min_ && thickness <= thin_thickness_max_)
    return LineClassification::ShortThin;

  if (length_ratio >= long_length_ratio_min_ && length_ratio <= long_length_ratio_max_ &&
      thickness >= thick_thickness_min_ && thickness <= thick_thickness_max_)
    return LineClassification::LongThick;

  return LineClassification::Rejected;
}

std::string SoccerRobotDetector::formatLineLabel(const LineFeature& feature) const
{
  std::ostringstream oss;
  oss << (feature.classification == LineClassification::ShortThin ? "S" : "L");
  oss << " " << std::fixed << std::setprecision(0) << feature.length;
  oss << "px /" << std::setprecision(1) << feature.thickness;
  return oss.str();
}

void SoccerRobotDetector::publishArcDebugImages(const ArcDebugImages& images,
                                                const std_msgs::Header& header,
                                                const cv::Mat& gray, const ArcSegment* arc,
                                                bool arc_found)
{
  if (arc_visualize_blur_ && !images.blurred.empty())
  {
    cv_bridge::CvImage cv_image(header, "mono8", images.blurred);
    arc_blur_pub_.publish(cv_image.toImageMsg());
  }

  if (arc_visualize_edges_ && !images.edges.empty())
  {
    cv_bridge::CvImage cv_image(header, "mono8", images.edges);
    arc_edges_pub_.publish(cv_image.toImageMsg());
  }

  if (arc_visualize_dilated_edges_ && !images.dilated_edges.empty())
  {
    cv_bridge::CvImage cv_image(header, "mono8", images.dilated_edges);
    arc_dilated_pub_.publish(cv_image.toImageMsg());
  }

  if (arc_visualize_contours_ && !gray.empty())
  {
    cv::Mat overlay;
    cv::cvtColor(gray, overlay, cv::COLOR_GRAY2BGR);
    if (arc_found && arc != nullptr)
    {
      std::vector<cv::Point> pts;
      pts.reserve(arc->points.size());
      for (const auto& pt : arc->points)
        pts.emplace_back(cvRound(pt.x), cvRound(pt.y));
      if (!pts.empty())
        cv::polylines(overlay, pts, false, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

      cv::circle(overlay, cv::Point(cvRound(arc->center.x), cvRound(arc->center.y)),
                 cvRound(arc->radius), cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
      cv::rectangle(overlay, arc->bounding_box, cv::Scalar(0, 140, 255), 1);
    }

    cv_bridge::CvImage cv_image(header, "bgr8", overlay);
    arc_contours_pub_.publish(cv_image.toImageMsg());
  }
}

bool SoccerRobotDetector::detectArcSegment(const cv::Mat& gray, ArcSegment& best_arc,
                                           ArcDebugImages* debug) const
{
  cv::Mat blurred;
  cv::GaussianBlur(gray, blurred, cv::Size(arc_blur_kernel_size_, arc_blur_kernel_size_), 0);
  if (debug && arc_visualize_blur_)
    blurred.copyTo(debug->blurred);

  cv::Mat edges;
  cv::Canny(blurred, edges, arc_canny_low_, arc_canny_high_);
  if (debug && arc_visualize_edges_)
    edges.copyTo(debug->edges);

  cv::Mat dilated_edges;
  if (arc_dilate_iterations_ > 0)
    cv::dilate(edges, dilated_edges, cv::Mat(), cv::Point(-1, -1), arc_dilate_iterations_);
  else
    dilated_edges = edges.clone();

  if (debug && arc_visualize_dilated_edges_)
    dilated_edges.copyTo(debug->dilated_edges);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(dilated_edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

  double best_score = -1.0;
  for (const auto& contour : contours)
  {
    if (static_cast<int>(contour.size()) < min_arc_points_)
      continue;

    std::vector<cv::Point> simplified;
    cv::approxPolyDP(contour, simplified, arc_approx_epsilon_, false);
    const auto& candidate = simplified.empty() ? contour : simplified;

    if (candidate.size() < 5)
      continue;

    std::vector<cv::Point2f> pts;
    pts.reserve(candidate.size());
    for (const auto& p : candidate)
      pts.emplace_back(static_cast<float>(p.x), static_cast<float>(p.y));

    cv::Point2f center;
    float radius;
    if (!fitCircleLeastSquares(pts, center, radius))
      continue;

    if (radius < static_cast<float>(minRadius_) || radius > static_cast<float>(maxRadius_))
      continue;

    double mean_error = 0.0;
    double max_error = 0.0;
    for (const auto& pt : pts)
    {
      double dist = std::fabs(cv::norm(pt - center) - radius);
      mean_error += dist;
      max_error = std::max(max_error, dist);
    }
    mean_error /= static_cast<double>(pts.size());
    if (max_error > arc_max_fit_error_)
      continue;

    double coverage_deg = computeAngularCoverage(pts, center);
    if (coverage_deg < arc_min_coverage_deg_)
      continue;

    double score = coverage_deg - mean_error * arc_score_error_weight_;
    if (score > best_score)
    {
      best_score = score;
      best_arc.center = center;
      best_arc.radius = radius;
      best_arc.points = std::move(pts);
      best_arc.bounding_box = cv::boundingRect(contour);
      best_arc.coverage_deg = coverage_deg;
      best_arc.mean_error = mean_error;
    }
  }

  return best_score >= 0.0;
}

void SoccerRobotDetector::enforceParameterConstraints(
    soccer_robot_detector::SoccerRobotDetectorConfig& config)
{
  config.median_blur_kernel_size = ensureOddKernel(config.median_blur_kernel_size, 3);
  config.arc_blur_kernel_size = ensureOddKernel(config.arc_blur_kernel_size, 3);

  if (config.maxRadius < config.minRadius)
    config.maxRadius = config.minRadius;

  if (config.arc_canny_high < config.arc_canny_low)
    config.arc_canny_high = config.arc_canny_low;

  if (config.line_angle_min_deg > config.line_angle_max_deg)
    config.line_angle_min_deg = config.line_angle_max_deg;

  if (config.short_length_ratio_max < config.short_length_ratio_min)
    config.short_length_ratio_max = config.short_length_ratio_min;

  if (config.long_length_ratio_max < config.long_length_ratio_min)
    config.long_length_ratio_max = config.long_length_ratio_min;

  if (config.thin_thickness_max < config.thin_thickness_min)
    config.thin_thickness_max = config.thin_thickness_min;

  if (config.thick_thickness_max < config.thick_thickness_min)
    config.thick_thickness_max = config.thick_thickness_min;

  if (config.line_length_min_px < 0.0)
    config.line_length_min_px = 0.0;

  if (config.line_length_max_radius_ratio < 0.0)
    config.line_length_max_radius_ratio = 0.0;

  if (config.line_midpoint_max_radius_ratio < 0.0)
    config.line_midpoint_max_radius_ratio = 0.0;

  if (config.line_intensity_diff_min < 0.0)
    config.line_intensity_diff_min = 0.0;

  if (config.roi_canny_high_ratio < 1.0)
    config.roi_canny_high_ratio = 1.0;

  if (config.line_max_gap_scale < 0.0)
    config.line_max_gap_scale = 0.0;

  config.roi_edge_dilation_iterations = std::max(0, config.roi_edge_dilation_iterations);
  config.arc_dilate_iterations = std::max(0, config.arc_dilate_iterations);
  config.thickness_search_radius = std::max(1, config.thickness_search_radius);
}

void SoccerRobotDetector::reconfigureCallback(
    soccer_robot_detector::SoccerRobotDetectorConfig& config, uint32_t)
{
  enforceParameterConstraints(config);

  use_hough_circle_ = config.use_hough_circle;
  dp_ = config.dp;
  minDist_ = config.minDist;
  param1_ = config.param1;
  param2_ = config.param2;
  minRadius_ = config.minRadius;
  maxRadius_ = config.maxRadius;
  roi_padding_scale_ = config.roi_padding_scale;
  median_blur_kernel_size_ = config.median_blur_kernel_size;
  use_harris_ = config.use_harris;
  maxCorners_ = config.maxCorners;
  qualityLevel_ = config.qualityLevel;
  minCornerDistance_ = config.minCornerDistance;

  hough_rho_ = config.hough_rho;
  hough_theta_ = config.hough_theta;
  hough_threshold_ = config.hough_threshold;
  minLineLength_ = config.minLineLength;
  maxLineGap_ = config.maxLineGap;
  line_hough_threshold_extra_ = config.line_hough_threshold_extra;
  line_min_length_extra_ = config.line_min_length_extra;
  line_max_gap_scale_ = config.line_max_gap_scale;
  line_length_min_px_ = config.line_length_min_px;
  line_length_max_radius_ratio_ = config.line_length_max_radius_ratio;
  line_midpoint_max_radius_ratio_ = config.line_midpoint_max_radius_ratio;
  line_angle_min_deg_ = config.line_angle_min_deg;
  line_angle_max_deg_ = config.line_angle_max_deg;
  line_intensity_diff_min_ = config.line_intensity_diff_min;

  roi_canny_low_min_ = config.roi_canny_low_min;
  roi_canny_low_base_ = config.roi_canny_low_base;
  roi_canny_low_mean_scale_ = config.roi_canny_low_mean_scale;
  roi_canny_high_ratio_ = config.roi_canny_high_ratio;
  roi_edge_dilation_iterations_ = config.roi_edge_dilation_iterations;

  min_arc_points_ = config.min_arc_points;
  arc_max_fit_error_ = config.arc_max_fit_error;
  arc_min_coverage_deg_ = config.arc_min_coverage_deg;
  arc_blur_kernel_size_ = config.arc_blur_kernel_size;
  arc_canny_low_ = config.arc_canny_low;
  arc_canny_high_ = config.arc_canny_high;
  arc_dilate_iterations_ = config.arc_dilate_iterations;
  arc_approx_epsilon_ = config.arc_approx_poly_epsilon;
  arc_score_error_weight_ = config.arc_score_error_weight;

  short_length_ratio_min_ = config.short_length_ratio_min;
  short_length_ratio_max_ = config.short_length_ratio_max;
  long_length_ratio_min_ = config.long_length_ratio_min;
  long_length_ratio_max_ = config.long_length_ratio_max;
  thin_thickness_min_ = config.thin_thickness_min;
  thin_thickness_max_ = config.thin_thickness_max;
  thick_thickness_min_ = config.thick_thickness_min;
  thick_thickness_max_ = config.thick_thickness_max;
  thickness_search_radius_ = config.thickness_search_radius;

  arc_visualize_blur_ = config.arc_visualize_blur;
  arc_visualize_edges_ = config.arc_visualize_edges;
  arc_visualize_dilated_edges_ = config.arc_visualize_dilated_edges;
  arc_visualize_contours_ = config.arc_visualize_contours;
}

double SoccerRobotDetector::computeAngularCoverage(const std::vector<cv::Point2f>& points,
                                                   const cv::Point2f& center)
{
  if (points.size() < 2)
    return 0.0;

  std::vector<double> angles;
  angles.reserve(points.size());
  for (const auto& pt : points)
    angles.push_back(std::atan2(pt.y - center.y, pt.x - center.x));

  std::sort(angles.begin(), angles.end());
  double max_gap = 0.0;
  for (size_t i = 1; i < angles.size(); ++i)
  {
    double gap = angles[i] - angles[i - 1];
    if (gap > max_gap)
      max_gap = gap;
  }

  double wrap_gap = (angles.front() + 2.0 * CV_PI) - angles.back();
  if (wrap_gap > max_gap)
    max_gap = wrap_gap;

  double coverage = 2.0 * CV_PI - max_gap;
  return coverage * 180.0 / CV_PI;
}

