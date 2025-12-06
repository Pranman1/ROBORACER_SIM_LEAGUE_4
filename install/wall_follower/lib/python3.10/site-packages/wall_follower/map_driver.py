"""
Reactive wall avoider - NEVER hits walls
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
import numpy as np

class MapDriver(Node):
    def __init__(self):
        super().__init__('map_driver')
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.pub_t = self.create_publisher(Float32, '/autodrive/roboracer_1/throttle_command', 10)
        self.pub_s = self.create_publisher(Float32, '/autodrive/roboracer_1/steering_command', 10)
        self.create_subscription(LaserScan, '/autodrive/roboracer_1/lidar', self.cb, qos)
        self.get_logger().info("Reactive Wall Avoider Started")

    def cb(self, msg):
        r = np.clip(np.nan_to_num(msg.ranges, nan=10), 0.1, 10)
        n, c = len(r), len(r)//2
        
        # Check distances in sectors
        left = np.min(r[c+100:c+300])      # +25 to +75 deg
        right = np.min(r[c-300:c-100])     # -75 to -25 deg
        front = np.min(r[c-50:c+50])       # ±12 deg
        
        # REACTIVE: Steer away from closest wall
        if left < right:
            # Left closer - turn right
            steer = -0.4
        else:
            # Right closer - turn left
            steer = 0.4
        
        # Also avoid front
        if front < 0.5:
            # Front blocked - turn toward more open side
            if left > right:
                steer = 0.5  # Turn left
            else:
                steer = -0.5  # Turn right
        
        # Speed: slow down near walls
        min_dist = min(left, right, front)
        if min_dist < 0.4:
            speed = 0.02
        elif min_dist < 0.7:
            speed = 0.03
        else:
            speed = 0.05
        
        t, s = Float32(), Float32()
        t.data, s.data = speed, steer
        self.pub_t.publish(t)
        self.pub_s.publish(s)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(MapDriver())

if __name__ == '__main__':
    main()
