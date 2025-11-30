import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
import numpy as np
import math

class MPCDriver(Node):
    def __init__(self):
        super().__init__('mpc_driver')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # PUBLISHERS
        self.pub_throttle = self.create_publisher(Float32, '/autodrive/roboracer_1/throttle_command', 10)
        self.pub_steering = self.create_publisher(Float32, '/autodrive/roboracer_1/steering_command', 10)

        # SUBSCRIBERS
        self.sub_lidar = self.create_subscription(LaserScan, '/autodrive/roboracer_1/lidar', self.lidar_callback, qos_profile)

        # TUNING -- ULTRA CONSERVATIVE NO-CRASH MODE
        self.BASE_SPEED = 0.12        # Slow throughout
        self.TURN_SPEED = 0.08        # Very slow in corners
        self.CRAWL_SPEED = 0.05       # Crawl in tight spots
        self.MAX_STEER = 0.45         # Max steering
        
        # Smoothing
        self.prev_steer = 0.0         # For smooth steering changes

        self.get_logger().info("Ultra Conservative - Zero Crashes!")

    def lidar_callback(self, scan_data):
        # Clean lidar data
        ranges = np.array(scan_data.ranges, dtype=np.float32)
        ranges = np.where(np.isinf(ranges), 10.0, ranges)
        ranges = np.where(np.isnan(ranges), 10.0, ranges)
        ranges = np.clip(ranges, 0.1, 10.0)
        
        num_ranges = len(ranges)
        angle_min = scan_data.angle_min
        angle_inc = scan_data.angle_increment
        center_idx = num_ranges // 2
        
        # === STEP 1: FIND SAFEST DIRECTION ===
        # Look in forward 70 degrees - not too wide
        search_width = int(num_ranges * 0.19)  # ~35 degrees each side
        search_start = max(0, center_idx - search_width)
        search_end = min(num_ranges, center_idx + search_width)
        
        # Find direction with most clearance - larger window for smoothness
        window_size = 9
        best_idx = center_idx
        best_score = 0.0
        
        for i in range(search_start, search_end - window_size):
            window_avg = np.mean(ranges[i:i+window_size])
            # Strongly prefer staying near center
            distance_from_center = abs(i - center_idx)
            center_penalty = (distance_from_center / num_ranges) * 1.5
            score = window_avg * (1.0 - center_penalty)
            
            if score > best_score:
                best_score = score
                best_idx = i + window_size // 2
        
        # Convert to angle
        target_angle = angle_min + (best_idx * angle_inc)
        
        # === STEP 2: SMOOTH STEERING ===
        # Gentle steering with smoothing to prevent zigzag
        steer_cmd = target_angle * 0.7  # Lower gain for smoothness
        
        # Smooth steering changes (low-pass filter)
        steer_cmd = 0.6 * steer_cmd + 0.4 * self.prev_steer
        steer_cmd = np.clip(steer_cmd, -self.MAX_STEER, self.MAX_STEER)
        self.prev_steer = steer_cmd
        
        # === STEP 3: CHECK CLEARANCES ===
        forward_width = int(num_ranges * 0.08)
        forward_start = max(0, center_idx - forward_width)
        forward_end = min(num_ranges, center_idx + forward_width)
        forward_min = np.min(ranges[forward_start:forward_end])
        
        # Check side walls
        left_side = int(center_idx + num_ranges * 0.2)
        right_side = int(center_idx - num_ranges * 0.2)
        left_dist = np.mean(ranges[left_side:left_side+8]) if left_side < len(ranges)-8 else 10.0
        right_dist = np.mean(ranges[right_side-8:right_side]) if right_side >= 8 else 10.0
        min_side_dist = min(left_dist, right_dist)
        
        # === STEP 4: ULTRA CONSERVATIVE SPEED CONTROL ===
        # Default to slow
        throttle_cmd = self.TURN_SPEED
        
        # Crawl if anything remotely tight
        if forward_min < 1.2 or min_side_dist < 0.8 or abs(steer_cmd) > 0.2:
            throttle_cmd = self.CRAWL_SPEED
        
        # Only go base speed if everything is perfect
        if forward_min > 2.5 and min_side_dist > 1.5 and abs(steer_cmd) < 0.1:
            throttle_cmd = self.BASE_SPEED
        
        # Publish
        self.publish_command(throttle_cmd, steer_cmd)

    def publish_command(self, throttle, steer):
        t_msg = Float32()
        t_msg.data = float(throttle)
        self.pub_throttle.publish(t_msg)
        
        s_msg = Float32()
        s_msg.data = float(steer)
        self.pub_steering.publish(s_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MPCDriver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()