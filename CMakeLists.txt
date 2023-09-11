cmake_minimum_required(VERSION 3.0.2)
project(fastlio_localization)

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "-std=c++17 -O3")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fexceptions -Wno-unused-local-typedefs")

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" )
  include(ProcessorCount)
  ProcessorCount(N)
  message("Processer number:  ${N}")
  if(N GREATER 4)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=3)
    message("core for MP: 3")
  elseif(N GREATER 3)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=2)
    message("core for MP: 2")
  else()
    add_definitions(-DMP_PROC_NUM=1)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

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
  ant_robot_msgs
)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(yaml-cpp REQUIRED)

include_directories(
	${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
    include)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs
  DEPENDS EIGEN3 PCL
  INCLUDE_DIRS
)

add_library(${PROJECT_NAME} SHARED
  include/ikd-Tree/ikd_Tree.cpp
  include/global_localization/scancontext/Scancontext.cpp
  include/system/ImuProcessor.cpp
  include/system/System.hpp)
target_link_libraries(${PROJECT_NAME} stdc++fs ${PCL_LIBRARIES})

add_executable(fastlio_localization_ros1 src/ros1_main.cpp)
target_link_libraries(fastlio_localization_ros1 ${PROJECT_NAME} ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${YAML_CPP_LIBRARIES})

add_executable(fastlio_localization_test_bag_lcalization test/read_rosbag_test_lcalization.cpp)
target_link_libraries(fastlio_localization_test_bag_lcalization ${PROJECT_NAME} ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${YAML_CPP_LIBRARIES})

add_executable(fastlio_localization_test_bag_relcalization test/read_rosbag_test_relcalization.cpp)
target_link_libraries(fastlio_localization_test_bag_relcalization ${PROJECT_NAME} ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${YAML_CPP_LIBRARIES})
