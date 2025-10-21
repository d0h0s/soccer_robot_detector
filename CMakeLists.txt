cmake_minimum_required(VERSION 3.0.2)
project(soccer_robot_detector)

## ѡ
add_compile_options(-std=c++14 -O2 -Wall)

## 
find_package(catkin REQUIRED COMPONENTS
        roscpp
        sensor_msgs
        cv_bridge
        image_transport
        message_filters
        geometry_msgs
        image_geometry
)

## OpenCV
find_package(OpenCV REQUIRED)

## 
catkin_package(
        CATKIN_DEPENDS roscpp sensor_msgs cv_bridge image_transport message_filters geometry_msgs image_geometry
)

## ͷļĿ¼
include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)

## ִļ
add_executable(${PROJECT_NAME}_node
        src/main.cpp
        src/detector.cpp
)

## 
target_link_libraries(${PROJECT_NAME}_node
        ${catkin_LIBRARIES}
        ${OpenCV_LIBRARIES}
)

## װ򣨿ѡ
install(TARGETS ${PROJECT_NAME}_node
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch config
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
