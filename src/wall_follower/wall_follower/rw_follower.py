"""
CONVEX WALL FOLLOWER - FINAL VERSION
- Positive convexity check (prevents switching into concave gaps)
- Diagonal clearance check (prevents switching into corners)
- Proper speed control for safety
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
import numpy as np

class ConvexWallFollower(Node):
    def __init__(self):
        super().__init__('rw_follower')
        
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, 
            depth=10
        )
        
        self.pub_throttle = self.create_publisher(Float32, '/autodrive/roboracer_1/throttle_command', 10)
        self.pub_steering = self.create_publisher(Float32, '/autodrive/roboracer_1/steering_command', 10)
        self.sub_lidar = self.create_subscription(LaserScan, '/autodrive/roboracer_1/lidar', self.lidar_callback, qos)
        
        # ===== SPEED SETTINGS =====
        self.SPEED_NORMAL = 0.06    # Normal cruising
        self.SPEED_TURN = 0.045     # When turning
        self.SPEED_TIGHT = 0.035    # Tight spots
        
        self.MAX_STEER = 0.5236
        self.TARGET_DIST = 0.65
        self.Kp = 1.0
        self.Kd = 0.3
        
        self.SWITCH_THRESHOLD = 0.5
        self.CORNER_CLEARANCE_TO_LEFT = 1.5   # Strict: don't dive into gap
        self.CORNER_CLEARANCE_TO_RIGHT = 0.8  # Lenient: easy to go back to right
        
        self.prev_error = 0.0
        self.count = 0
        self.following_right = True 

        self.get_logger().info("=" * 50)
        self.get_logger().info("CONVEX FOLLOWER - FINAL VERSION")
        self.get_logger().info("Gap protection + Safe speed control")
        self.get_logger().info("=" * 50)

    def analyze_wall_convexity(self, ranges, side='right'):
        n = len(ranges)
        center = n // 2
        
        # TWO FOV ZONES - weighted average
        # FRONT zone: -90° to -30° (side to front) - sees upcoming bends
        # REAR zone:  -120° to -60° (rear to side) - sees hairpin type features
        
        if side == 'right':
            # Front zone
            front_start = max(0, center - 360)   # -90°
            front_end = max(0, center - 120)     # -30°
            # Rear zone
            rear_start = max(0, center - 480)    # -120°
            rear_end = max(0, center - 240)      # -60°
        else:
            # Front zone
            front_start = min(n, center + 120)   # +30°
            front_end = min(n, center + 360)     # +90°
            # Rear zone
            rear_start = min(n, center + 240)    # +60°
            rear_end = min(n, center + 480)      # +120°
        
        # Calculate convexity for FRONT zone
        front_ranges = ranges[front_start:front_end]
        front_valid = front_ranges[(front_ranges > 0.1) & (front_ranges < 6.0)]
        if len(front_valid) >= 10:
            mid = len(front_valid) // 2
            front_convex = np.mean(front_valid[mid:]) - np.mean(front_valid[:mid])
            front_dist = float(np.percentile(front_valid, 25))
        else:
            front_convex = -1.0
            front_dist = 3.0
        
        # Calculate convexity for REAR zone
        rear_ranges = ranges[rear_start:rear_end]
        rear_valid = rear_ranges[(rear_ranges > 0.1) & (rear_ranges < 6.0)]
        if len(rear_valid) >= 10:
            mid = len(rear_valid) // 2
            rear_convex = np.mean(rear_valid[mid:]) - np.mean(rear_valid[:mid])
            rear_dist = float(np.percentile(rear_valid, 25))
        else:
            rear_convex = -1.0
            rear_dist = 3.0
        
        # WEIGHTED AVERAGE: 60% front, 40% rear
        # Front is more important for seeing upcoming walls
        convexity = 0.6 * front_convex + 0.4 * rear_convex
        distance = min(front_dist, rear_dist)  # Use closer distance for safety
        
        return distance, float(convexity)

    def get_front_distance(self, ranges):
        n = len(ranges)
        center = n // 2
        front_ranges = ranges[max(0, center-80):min(n, center+80)]
        valid = front_ranges[front_ranges > 0.05]
        return float(np.min(valid)) if len(valid) > 0 else 10.0

    def lidar_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isfinite(ranges), ranges, 10.0)
        ranges = np.clip(ranges, 0.0, 10.0)
        
        right_dist, right_convex = self.analyze_wall_convexity(ranges, 'right')
        left_dist, left_convex = self.analyze_wall_convexity(ranges, 'left')
        front_dist = self.get_front_distance(ranges)
        
        # === DIAGONAL CLEARANCE ===
        center_idx = len(ranges) // 2
        fr_idx = max(10, center_idx - 135)
        fl_idx = min(len(ranges) - 10, center_idx + 135)
        
        fr_clearance = float(np.mean(ranges[fr_idx-10 : fr_idx+10]))
        fl_clearance = float(np.mean(ranges[fl_idx-10 : fl_idx+10]))
        
        # === SWITCHING LOGIC WITH GAP PROTECTION ===
        if self.following_right:
            # Switch to LEFT only if:
            # 1. Left is actually CONVEX (> 0.05)
            # 2. Left is significantly more convex than right
            # 3. Front-left is clear (STRICT - don't dive into gap)
            if (left_convex > 0.05 and 
                left_convex > right_convex + self.SWITCH_THRESHOLD and 
                fl_clearance > self.CORNER_CLEARANCE_TO_LEFT):
                
                self.following_right = False
                self.get_logger().info(f">>> SWITCH LEFT (convex:{left_convex:.2f} clear:{fl_clearance:.1f})")
        else:
            # Switch back to RIGHT - more lenient!
            # Just need right to be convex and have some clearance
            if (right_convex > 0.0 and 
                right_convex >= left_convex - 0.2 and
                fr_clearance > self.CORNER_CLEARANCE_TO_RIGHT):
                
                self.following_right = True
                self.get_logger().info(f">>> SWITCH RIGHT (convex:{right_convex:.2f} clear:{fr_clearance:.1f})")
        
        # Logging
        self.count += 1
        if self.count % 15 == 0:
            side = "RIGHT" if self.following_right else "LEFT"
            self.get_logger().info(
                f"[{side}] R={right_dist:.2f}({right_convex:+.2f}) L={left_dist:.2f}({left_convex:+.2f}) F={front_dist:.2f}"
            )
        
        # ========== WALL FOLLOWING PID ==========
        if self.following_right:
            error = right_dist - self.TARGET_DIST
            derivative = error - self.prev_error
            self.prev_error = error
            steer = -(self.Kp * error + self.Kd * derivative)
        else:
            error = left_dist - self.TARGET_DIST
            derivative = error - self.prev_error
            self.prev_error = error
            steer = self.Kp * error + self.Kd * derivative
        
        steer = np.clip(steer, -self.MAX_STEER, self.MAX_STEER)
        
        # ========== SMART SPEED CONTROL ==========
        min_clearance = min(fr_clearance, fl_clearance, front_dist)
        
        if min_clearance < 0.5:
            # Very tight - crawl
            speed = self.SPEED_TIGHT
        elif min_clearance < 1.0 or abs(steer) > 0.35:
            # Tight or turning hard
            speed = self.SPEED_TURN
        elif abs(steer) > 0.2:
            # Moderate turn
            speed = self.SPEED_TURN
        else:
            # Clear and straight
            speed = self.SPEED_NORMAL
        
        t_msg = Float32()
        s_msg = Float32()
        t_msg.data = float(speed)
        s_msg.data = float(steer)
        self.pub_throttle.publish(t_msg)
        self.pub_steering.publish(s_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ConvexWallFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
