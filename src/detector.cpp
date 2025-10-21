//
// Created by xianghong on 10/20/25.
//
#include "soccer_robot_detector/detector.h"

SoccerRobotDetector::SoccerRobotDetector(ros::NodeHandle& nh)
    : nh_(nh), it_(nh)
{
  // ��������
  nh_.param("dp", dp_, 1.2);
  nh_.param("minDist", minDist_, 50.0);
  nh_.param("param1", param1_, 100.0);
  nh_.param("param2", param2_, 30.0);
  nh_.param("minRadius", minRadius_, 30);
  nh_.param("maxRadius", maxRadius_, 200);
  nh_.param("use_harris", use_harris_, true);
  nh_.param("hough_rho", hough_rho_, 1.0);
  nh_.param("hough_theta", hough_theta_, 1.0);  // ��λΪ�ȣ��Ժ�ת��Ϊ����
  nh_.param("hough_threshold", hough_threshold_, 50);
  nh_.param("minLineLength", minLineLength_, 30.0);
  nh_.param("maxLineGap", maxLineGap_, 10.0);

  nh_.param("maxCorners", maxCorners_, 50);
  nh_.param("qualityLevel", qualityLevel_, 0.02);
  nh_.param("minCornerDistance", minCornerDistance_, 10.0);

  image_sub_ = it_.subscribe("/camera/infra1/image_rect_raw", 1,
                             &SoccerRobotDetector::imageCallback, this);
  debug_pub_ = it_.advertise("/soccer_robot_detector/debug", 1);

  has_last_circle_ = false;
}

void SoccerRobotDetector::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  cv::Mat gray;
  try {
    gray = cv_bridge::toCvShare(msg, "mono8")->image.clone();
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  cv::Mat blur_img;
  cv::medianBlur(gray, blur_img, 5);

  // ����Բ���
  std::vector<cv::Vec3f> circles;
  cv::HoughCircles(blur_img, circles, cv::HOUGH_GRADIENT, dp_, minDist_, param1_, param2_, minRadius_, maxRadius_);

  cv::Mat debug_img;
  cv::cvtColor(gray, debug_img, cv::COLOR_GRAY2BGR);

  for (size_t i = 0; i < circles.size(); i++)
  {
    cv::Vec3f c = circles[i];
    cv::Point center(cvRound(c[0]), cvRound(c[1]));
    int radius = cvRound(c[2]);

    // ---------- ƽ��Բ�ĺͰ뾶 ----------
    if (has_last_circle_) {
      center.x = 0.7 * last_center_.x + 0.3 * center.x;
      center.y = 0.7 * last_center_.y + 0.3 * center.y;
      radius   = static_cast<int>(0.7 * last_radius_ + 0.3 * radius);
    }
    last_center_ = center;
    last_radius_ = radius;
    has_last_circle_ = true;
    // ---------------------------------

    // ����Բ
    cv::circle(debug_img, center, radius, cv::Scalar(0,255,0), 2);
    cv::circle(debug_img, center, 3, cv::Scalar(0,0,255), -1);

    // ---------- ��ȡROI ----------
    int x1 = std::max(center.x - radius, 0);
    int y1 = std::max(center.y - radius, 0);
    int x2 = std::min(center.x + radius, gray.cols);
    int y2 = std::min(center.y + radius, gray.rows);
    cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
    cv::Mat circle_roi = gray(roi).clone();

    // ---------- �Ż��� ����ֱ�߼�� + ���� ----------

    // �����ݶ�ǿ��ͼ���������θ߹�����
    cv::Mat grad_x, grad_y, grad_mag;
    cv::Sobel(circle_roi, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(circle_roi, grad_y, CV_32F, 0, 1, 3);
    cv::magnitude(grad_x, grad_y, grad_mag);

    // �������Ⱦ�ֵ��̬���� Canny ��ֵ
    double mean_intensity = cv::mean(circle_roi)[0];
    double canny_low = std::max(50.0, 120.0 - mean_intensity / 2);
    double canny_high = canny_low * 2.5;

    // ��Ե��� + ����
    cv::Mat edges;
    cv::Canny(circle_roi, edges, canny_low, canny_high);
    cv::dilate(edges, edges, cv::Mat(), cv::Point(-1,-1), 1);

    // ������
    std::vector<cv::Vec4i> raw_lines;
    cv::HoughLinesP(edges, raw_lines, hough_rho_, CV_PI/180.0 * hough_theta_,
                    hough_threshold_ + 30, minLineLength_ + 30, maxLineGap_ / 2.0);

    // ���ˣ����� + ǿ��
    std::vector<cv::Vec4i> filtered_lines;
    for (auto &l : raw_lines) {
      cv::Point p1(l[0], l[1]), p2(l[2], l[3]);
      double len = cv::norm(p1 - p2);
      if (len < 30 || len > radius * 1.3) continue;

      // Բ���ų�
      cv::Point mid((p1.x+p2.x)/2, (p1.y+p2.y)/2);
      if (cv::norm(mid - cv::Point(radius, radius)) > radius * 0.85) continue;

      // ȥ����ˮƽ��
      double angle = std::fabs(std::atan2(p2.y - p1.y, p2.x - p1.x) * 180.0 / CV_PI);
      if (angle < 15 || angle > 165) continue;

      // ���������ҶȲ�ж��Ƿ�Ϊ�����ƣ�
      cv::LineIterator it(circle_roi, p1, p2);
      std::vector<float> profile;
      for(int i=0; i<it.count; i++, ++it)
        profile.push_back((float)circle_roi.at<uchar>(it.pos()));
      double diff = std::fabs(*std::max_element(profile.begin(), profile.end()) -
                              *std::min_element(profile.begin(), profile.end()));
      if (diff < 30) continue;  // �ڰײ���ԣ�����

      filtered_lines.push_back(l);
    }

    // ���ƽ��
    for (auto &l : filtered_lines) {
      cv::line(debug_img(roi),
               cv::Point(l[0], l[1]),
               cv::Point(l[2], l[3]),
               cv::Scalar(255,0,0), 1);
    }

    // ---------- �ǵ��� ----------
    if (use_harris_) {
      cv::Mat corners, dst_norm;
      cv::cornerHarris(circle_roi, corners, 2, 3, 0.04);
      cv::normalize(corners, dst_norm, 0, 255, cv::NORM_MINMAX);

      for (int y = 0; y < dst_norm.rows; y++) {
        for (int x = 0; x < dst_norm.cols; x++) {
          if ((int)dst_norm.at<float>(y, x) > 180)
            cv::circle(debug_img(roi), cv::Point(x,y), 2, cv::Scalar(0,0,255), -1);
        }
      }
    } else {
      std::vector<cv::Point2f> shi_tomasi_corners;
      cv::goodFeaturesToTrack(circle_roi,
                              shi_tomasi_corners,
                              maxCorners_,
                              qualityLevel_,
                              minCornerDistance_);

      for (auto &pt : shi_tomasi_corners)
        cv::circle(debug_img(roi), pt, 2, cv::Scalar(0,0,255), -1);
    }
  }

  // ---------- ����debugͼ�� ----------
  sensor_msgs::ImagePtr debug_msg =
      cv_bridge::CvImage(std_msgs::Header(), "bgr8", debug_img).toImageMsg();
  debug_pub_.publish(debug_msg);
}