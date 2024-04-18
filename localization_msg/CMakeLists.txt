cmake_minimum_required(VERSION 2.8.3)
project(localization_msg)


find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
)

add_message_files(
  DIRECTORY msg
  FILES
  vehicle_pose.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)


catkin_package(
  CATKIN_DEPENDS 
  message_runtime 
  message_generation
  std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)



