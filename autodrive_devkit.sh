#!/bin/bash

# ============================================
# AutoDRIVE RoboRacer Competition Entry Point
# ============================================

# Source ROS2
source /opt/ros/humble/setup.bash
source /home/autodrive_devkit/install/setup.bash

# Set working directory
cd /home/autodrive_devkit

# Launch AutoDRIVE Bridge (headless - no GUI)
ros2 launch autodrive_roboracer bringup_headless.launch.py &
sleep 2

# Launch Racing Algorithm
# Change 'comp_driver' to 'slam_driver' for qualification
ros2 run wall_follower comp_driver &

# Keep container alive
wait

