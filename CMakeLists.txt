cmake_minimum_required(VERSION 3.0.2)
project(soccer_robot_detector)

## 编译选项
add_compile_options(-std=c++14 -O2 -Wall)

## 查找依赖包
find_package(catkin REQUIRED COMPONENTS
        roscpp
        sensor_msgs
        cv_bridge
        image_transport
)

## OpenCV
find_package(OpenCV REQUIRED)

## 包声明
catkin_package(
        CATKIN_DEPENDS roscpp sensor_msgs cv_bridge image_transport
)

## 头文件目录
include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)

## 可执行文件
add_executable(${PROJECT_NAME}_node
        src/main.cpp
        src/detector.cpp
)

## 依赖与链接
target_link_libraries(${PROJECT_NAME}_node
        ${catkin_LIBRARIES}
        ${OpenCV_LIBRARIES}
)

## 安装规则（可选）
install(TARGETS ${PROJECT_NAME}_node
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch config
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
