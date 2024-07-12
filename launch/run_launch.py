import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

def generate_launch_description():
    config = os.path.join(get_package_share_directory('fastlio_localization'), 'config', 'ros2_param.yaml')
    rviz_config = os.path.join(get_package_share_directory('fastlio_localization'), 'rviz_cfg', 'localization_ros2.rviz')

    fastlio_localization = Node(package="fastlio_localization", executable="fastlio_localization_ros2", prefix=['stdbuf -o L'], output='screen', parameters=[config])
    # fastlio_localization = Node(package="fastlio_localization", executable="fastlio_localization_ros2", prefix=['gdb -ex run --args'], output='screen', parameters=[config])
    rviz2 = Node(package='rviz2', executable='rviz2', arguments=['-d', rviz_config])
    map_publisher = Node(package='pcl_ros', executable='pointcloud_to_pcd', output='screen',
                         parameters=[{'file_name': '/path/to/your/pcd/file.pcd', 'frame_id': 'map'}],
                         remappings=[('/cloud_pcd', '/your/pointcloud_topic')])
    return LaunchDescription([fastlio_localization, rviz2])
