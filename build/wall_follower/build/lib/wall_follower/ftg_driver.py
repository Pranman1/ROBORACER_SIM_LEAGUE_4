"""
Follow The Gap (FTG) Driver - F1Tenth Standard Algorithm
https://f1tenth.org/learn.html

1. Find closest obstacle
2. Create safety bubble around it
3. Find longest consecutive gap
4. Drive toward gap center
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
import numpy as np

class FTGDriver(Node):
    def __init__(self):
        super().__init__('ftg_driver')
        
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        
        self.pub_throttle = self.create_publisher(Float32, '/autodrive/roboracer_1/throttle_command', 10)
        self.pub_steering = self.create_publisher(Float32, '/autodrive/roboracer_1/steering_command', 10)
        self.sub_lidar = self.create_subscription(LaserScan, '/autodrive/roboracer_1/lidar', self.lidar_callback, qos)
        
        # Params
        self.BUBBLE_RADIUS = 0.3  # meters - safety bubble around closest point
        self.MAX_STEER = 0.5236
        self.SPEED_FAST = 0.18
        self.SPEED_SLOW = 0.10
        
        self.prev_steer = 0.0
        self.get_logger().info("FTG Driver Started - F1Tenth Algorithm")

    def preprocess(self, ranges, angle_min, angle_inc):
        """Clean lidar data and limit to forward 180 degrees"""
        proc = np.array(ranges)
        proc = np.where(np.isfinite(proc), proc, 10.0)
        proc = np.clip(proc, 0.0, 10.0)
        
        # Only use forward-facing 180 degrees
        n = len(proc)
        center = n // 2
        fov_points = int(np.pi / angle_inc)  # 180 degrees worth
        start = max(0, center - fov_points // 2)
        end = min(n, center + fov_points // 2)
        
        return proc[start:end], start

    def find_closest(self, ranges):
        """Find index of closest point"""
        return np.argmin(ranges)

    def create_bubble(self, ranges, closest_idx, angle_inc):
        """Zero out points around closest obstacle"""
        proc = ranges.copy()
        closest_dist = ranges[closest_idx]
        
        if closest_dist < 0.01:
            closest_dist = 0.01
        
        # How many indices does BUBBLE_RADIUS span at this distance?
        bubble_angle = np.arctan(self.BUBBLE_RADIUS / closest_dist)
        bubble_indices = int(bubble_angle / angle_inc)
        
        start = max(0, closest_idx - bubble_indices)
        end = min(len(ranges), closest_idx + bubble_indices + 1)
        proc[start:end] = 0.0
        
        return proc

    def find_best_gap(self, ranges):
        """Find the longest consecutive sequence of non-zero readings"""
        # Find all gaps (consecutive non-zero regions)
        nonzero = ranges > 0.1
        
        # Find start and end of each gap
        gaps = []
        in_gap = False
        gap_start = 0
        
        for i in range(len(nonzero)):
            if nonzero[i] and not in_gap:
                gap_start = i
                in_gap = True
            elif not nonzero[i] and in_gap:
                gaps.append((gap_start, i - 1))
                in_gap = False
        
        if in_gap:
            gaps.append((gap_start, len(nonzero) - 1))
        
        if not gaps:
            return len(ranges) // 2  # Default to center
        
        # Find longest gap
        longest_gap = max(gaps, key=lambda g: g[1] - g[0])
        
        # Return center of longest gap, weighted by distance
        start, end = longest_gap
        gap_ranges = ranges[start:end+1]
        
        if len(gap_ranges) == 0:
            return (start + end) // 2
        
        # Weight toward the deepest part of the gap
        best_in_gap = np.argmax(gap_ranges)
        return start + best_in_gap

    def lidar_callback(self, msg):
        # 1. Preprocess
        ranges, offset = self.preprocess(msg.ranges, msg.angle_min, msg.angle_increment)
        
        if len(ranges) == 0:
            return
        
        # 2. Find closest point
        closest_idx = self.find_closest(ranges)
        closest_dist = ranges[closest_idx]
        
        # 3. Create bubble around closest point
        bubble_ranges = self.create_bubble(ranges, closest_idx, msg.angle_increment)
        
        # 4. Find best gap
        best_idx = self.find_best_gap(bubble_ranges)
        
        # 5. Calculate steering angle
        # Convert index to angle
        center_idx = len(ranges) // 2
        angle_from_center = (best_idx - center_idx) * msg.angle_increment
        
        # Steering proportional to angle
        steer = angle_from_center * 1.0
        
        # Smooth steering
        steer = 0.7 * steer + 0.3 * self.prev_steer
        steer = np.clip(steer, -self.MAX_STEER, self.MAX_STEER)
        self.prev_steer = steer
        
        # 6. Speed control
        if closest_dist < 0.5:
            speed = 0.05
        elif closest_dist < 1.0 or abs(steer) > 0.3:
            speed = self.SPEED_SLOW
        else:
            speed = self.SPEED_FAST
        
        # Publish
        t_msg, s_msg = Float32(), Float32()
        t_msg.data = float(speed)
        s_msg.data = float(steer)
        self.pub_throttle.publish(t_msg)
        self.pub_steering.publish(s_msg)


def main(args=None):
    rclpy.init(args=args)
    node = FTGDriver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

