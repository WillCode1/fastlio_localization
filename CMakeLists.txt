cmake_minimum_required(VERSION 3.5)
project(fastlio_localization)

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "-std=c++17 -O3")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fexceptions -Wno-unused-local-typedefs -Wno-format-truncation -Wno-unused-result -Wno-pedantic -Wno-unused-parameter -Wno-unused-variable")

# echo "/home/will/coredump/core-%e-%p-%s-%t" > /proc/sys/kernel/core_pattern
# SET(CMAKE_BUILD_TYPE "Debug")
# SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
# SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)|(aarch32)|(AARCH32)|(aarch64)|(AARCH64)")
  include(ProcessorCount)
  ProcessorCount(N)
  message("Processer number:  ${N}")
  if(N GREATER 6)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=4)
    add_definitions(-DBNB_PROC_NUM=6)
    message("core for MP: 4")
  elseif(N GREATER 4)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=4)
    add_definitions(-DBNB_PROC_NUM=4)
    message("core for MP: 4")
  elseif(N GREATER 3)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=3)
    add_definitions(-DBNB_PROC_NUM=3)
    message("core for MP: 3")
  else()
    add_definitions(-DMP_PROC_NUM=1)
    add_definitions(-DBNB_PROC_NUM=2)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
  add_definitions(-DBNB_PROC_NUM=2)
endif()

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

if(ROS_EDITION STREQUAL "ROS1")
  find_package(catkin REQUIRED COMPONENTS
    geometry_msgs
    nav_msgs
    sensor_msgs
    roscpp
    rospy
    std_msgs
    pcl_ros
    tf
    livox_ros_driver
    eigen_conversions
    slam_interfaces
  )

  find_package(Eigen3 REQUIRED)
  find_package(PCL 1.8 REQUIRED)
  find_package(OpenCV REQUIRED)

  include_directories(
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    include)

  catkin_package(
    CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs
    DEPENDS EIGEN3 PCL
    INCLUDE_DIRS
  )

  add_library(${PROJECT_NAME} SHARED
    include/ikd-Tree/ikd_Tree.cpp
    include/global_localization/scancontext/Scancontext.cpp
    include/system/frontend/ImuProcessor.cpp
    include/system/frontend/use-ikfom.cpp
    include/system/System.hpp)
  target_link_libraries(${PROJECT_NAME} stdc++fs ${PCL_LIBRARIES})

  add_executable(fastlio_localization_ros1 src/ros1_main.cpp)
  target_link_libraries(fastlio_localization_ros1 ${PROJECT_NAME} ${catkin_LIBRARIES} ${PCL_LIBRARIES})

  add_executable(fastlio_localization_test_bag_lcalization test/read_rosbag_test_lcalization.cpp)
  target_link_libraries(fastlio_localization_test_bag_lcalization ${PROJECT_NAME} ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

  add_executable(fastlio_localization_test_bag_relcalization test/read_rosbag_test_relcalization.cpp)
  target_link_libraries(fastlio_localization_test_bag_relcalization ${PROJECT_NAME} ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

  install(TARGETS ${PROJECT_NAME} fastlio_localization_ros1
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )

  install(DIRECTORY launch config rviz_cfg Log
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  )

else(ROS_EDITION STREQUAL "ROS2")
  if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
  endif()

  # find dependencies
  find_package(ament_cmake REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(rclpy REQUIRED)
  find_package(nav_msgs REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(pcl_conversions REQUIRED)
  find_package(tf2_ros REQUIRED)
  find_package(livox_ros_driver2 REQUIRED)
  find_package(slam_interfaces REQUIRED)

  find_package(Eigen3 REQUIRED)
  find_package(PCL 1.8 REQUIRED)
  find_package(OpenCV REQUIRED)

  include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${pcl_conversions_INCLUDE_DIRS}
    include)

  if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    # the following line skips the linter which checks for copyrights
    # uncomment the line when a copyright and license is not present in all source files
    #set(ament_cmake_copyright_FOUND TRUE)
    # the following line skips cpplint (only works in a git repo)
    # uncomment the line when this package is not in a git repo
    #set(ament_cmake_cpplint_FOUND TRUE)
    ament_lint_auto_find_test_dependencies()
  endif()

  ament_package()

  add_executable(fastlio_localization_ros2 src/ros2_main.cpp
    include/ikd-Tree/ikd_Tree.cpp
    include/global_localization/scancontext/Scancontext.cpp
    include/system/frontend/ImuProcessor.cpp
    include/system/frontend/use-ikfom.cpp
    include/system/System.hpp)
  target_link_libraries(fastlio_localization_ros2 stdc++fs ${PCL_LIBRARIES})
  ament_target_dependencies(fastlio_localization_ros2 rclcpp sensor_msgs nav_msgs tf2_ros livox_ros_driver2)

  # add_executable(fastlio_sam_test_bag test/read_rosbag_test.cpp)
  # target_link_libraries(fastlio_sam_test_bag ${PROJECT_NAME} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

  install(TARGETS fastlio_localization_ros2
    DESTINATION lib/${PROJECT_NAME}
  )
  install(DIRECTORY launch config rviz_cfg
    DESTINATION share/${PROJECT_NAME})
endif()
