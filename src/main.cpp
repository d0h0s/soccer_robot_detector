//
// Created by xianghong on 10/20/25.
//
#include "ros/ros.h"
#include "soccer_robot_detector/detector.h"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "soccer_robot_detector_node");
  ros::NodeHandle nh("~");
  SoccerRobotDetector detector(nh);
  ros::spin();
  return 0;
}
