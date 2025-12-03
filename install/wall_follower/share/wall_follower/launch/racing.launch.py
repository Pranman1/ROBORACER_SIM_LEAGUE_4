"""
Racing Launch - Minimal 3-node stack
1. local_planner - Gap follower + waypoint recorder (Lap 1)
2. global_planner - Smooths path, computes speeds
3. mpc_controller - DBM MPC path following (Lap 2+)
"""
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package='wall_follower', executable='local_planner', output='screen'),
        Node(package='wall_follower', executable='global_planner', output='screen'),
        Node(package='wall_follower', executable='mpc_controller', output='screen'),
    ])
