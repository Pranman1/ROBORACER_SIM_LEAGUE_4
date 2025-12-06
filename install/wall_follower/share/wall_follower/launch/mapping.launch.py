"""
Mapping Launch - Driver + SLAM + RViz (all together!)
"""
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_share = get_package_share_directory('wall_follower')
    slam_params = os.path.join(pkg_share, 'config', 'slam_params.yaml')
    rviz_config = os.path.join(pkg_share, 'rviz', 'racing.rviz')
    
    return LaunchDescription([
        # SLAM Toolbox
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            parameters=[slam_params],
        ),
        # Map driver
        Node(
            package='wall_follower',
            executable='map_driver',
            name='map_driver',
            output='screen',
        ),
        # RViz to see the map
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            output='screen',
        ),
    ])
