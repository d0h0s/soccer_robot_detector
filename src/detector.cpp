//
// Created by xianghong on 10/20/25.
//

#include "soccer_robot_detector/detector.h"

#include <algorithm>
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
    : nh_(nh), it_(nh)
{
  nh_.param("dp", dp_, 1.2);
  nh_.param("minDist", minDist_, 50.0);
  nh_.param("param1", param1_, 100.0);
  nh_.param("param2", param2_, 30.0);
  nh_.param("minRadius", minRadius_, 30);
  nh_.param("maxRadius", maxRadius_, 200);
  nh_.param("use_hough_circle", use_hough_circle_, true);
  nh_.param("use_harris", use_harris_, true);
  nh_.param("hough_rho", hough_rho_, 1.0);
  nh_.param("hough_theta", hough_theta_, 1.0);
  nh_.param("hough_threshold", hough_threshold_, 50);
  nh_.param("minLineLength", minLineLength_, 30.0);
  nh_.param("maxLineGap", maxLineGap_, 10.0);

  nh_.param("maxCorners", maxCorners_, 50);
  nh_.param("qualityLevel", qualityLevel_, 0.02);
  nh_.param("minCornerDistance", minCornerDistance_, 10.0);

  nh_.param("min_arc_points", min_arc_points_, 80);
  nh_.param("arc_max_fit_error", arc_max_fit_error_, 8.0);
  nh_.param("arc_min_coverage_deg", arc_min_coverage_deg_, 80.0);
  nh_.param("roi_padding_scale", roi_padding_scale_, 0.15);

  nh_.param("short_length_ratio_min", short_length_ratio_min_, 0.25);
  nh_.param("short_length_ratio_max", short_length_ratio_max_, 0.55);
  nh_.param("long_length_ratio_min", long_length_ratio_min_, 0.55);
  nh_.param("long_length_ratio_max", long_length_ratio_max_, 0.95);
  nh_.param("thin_thickness_min", thin_thickness_min_, 1.0);
  nh_.param("thin_thickness_max", thin_thickness_max_, 3.0);
  nh_.param("thick_thickness_min", thick_thickness_min_, 3.0);
  nh_.param("thick_thickness_max", thick_thickness_max_, 6.0);
  nh_.param("thickness_search_radius", thickness_search_radius_, 8);

  image_sub_ = it_.subscribe("/camera/infra1/image_rect_raw", 1,
                             &SoccerRobotDetector::imageCallback, this);
  debug_pub_ = it_.advertise("/soccer_robot_detector/debug", 1);

  has_last_circle_ = false;
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
  cv::medianBlur(gray, blur_img, 5);

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
    if (detectArcSegment(blur_img, arc_candidate))
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
      cv::Rect padded = arc_candidate.bounding_box;
      padded.x -= padding;
      padded.y -= padding;
      padded.width += padding * 2;
      padded.height += padding * 2;
      roi = clampRectToImage(padded, gray.size());
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
  double canny_low = std::max(50.0, 120.0 - mean_intensity / 2.0);
  double canny_high = canny_low * 2.5;

  cv::Mat edges;
  cv::Canny(circle_roi, edges, canny_low, canny_high);
  cv::Mat edges_dilated;
  cv::dilate(edges, edges_dilated, cv::Mat(), cv::Point(-1, -1), 1);

  std::vector<cv::Vec4i> raw_lines;
  cv::HoughLinesP(edges_dilated, raw_lines, hough_rho_, CV_PI / 180.0 * hough_theta_,
                  hough_threshold_ + 30, minLineLength_ + 30, maxLineGap_ / 2.0);

  std::vector<LineFeature> filtered_lines;
  for (const auto& l : raw_lines)
  {
    cv::Point p1(l[0], l[1]);
    cv::Point p2(l[2], l[3]);
    double len = cv::norm(p1 - p2);
    if (len < 30 || len > radius_px * 1.3)
      continue;

    cv::Point mid((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);
    if (cv::norm(mid - roi_center) > radius_px * 0.85)
      continue;

    double angle = std::fabs(std::atan2(p2.y - p1.y, p2.x - p1.x) * 180.0 / CV_PI);
    if (angle < 15 || angle > 165)
      continue;

    cv::LineIterator it(circle_roi, p1, p2);
    std::vector<float> profile;
    profile.reserve(it.count);
    for (int i = 0; i < it.count; ++i, ++it)
      profile.push_back(static_cast<float>(circle_roi.at<uchar>(it.pos())));

    auto minmax = std::minmax_element(profile.begin(), profile.end());
    double diff = std::fabs(*minmax.second - *minmax.first);
    if (diff < 30)
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

bool SoccerRobotDetector::detectArcSegment(const cv::Mat& gray, ArcSegment& best_arc) const
{
  cv::Mat blurred;
  cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

  double canny_low = 60.0;
  double canny_high = 150.0;
  cv::Mat edges;
  cv::Canny(blurred, edges, canny_low, canny_high);
  cv::dilate(edges, edges, cv::Mat(), cv::Point(-1, -1), 1);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

  double best_score = -1.0;
  for (const auto& contour : contours)
  {
    if (static_cast<int>(contour.size()) < min_arc_points_)
      continue;

    std::vector<cv::Point> simplified;
    cv::approxPolyDP(contour, simplified, 2.0, false);
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

    double score = coverage_deg - mean_error * 5.0;
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

