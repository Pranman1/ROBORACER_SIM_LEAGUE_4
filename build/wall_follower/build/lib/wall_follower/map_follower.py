import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
import numpy as np
import math
from scipy.spatial import distance
from collections import deque

class MapFollower(Node):
    def __init__(self):
        super().__init__('map_follower')

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
        self.sub_odom = self.create_subscription(Odometry, '/autodrive/roboracer_1/ips', self.odom_callback, qos_profile)

        # State
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        self.velocity = 0.0
        
        # Mapping phase
        self.mapping_mode = True
        self.centerline_points = []
        self.recorded_positions = deque(maxlen=5000)
        self.start_position = None
        self.lap_complete = False
        
        # Control parameters
        self.LOOKAHEAD_DISTANCE = 1.5  # meters
        self.BASE_SPEED = 0.15
        self.TURN_SPEED = 0.10
        self.MAX_STEER = 0.45
        
        # Simple gap follower for mapping lap
        self.prev_steer = 0.0

        self.get_logger().info("Map-Based Controller: Starting mapping lap...")

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.velocity = msg.twist.twist.linear.x
        
        # Extract heading from quaternion
        qw = msg.pose.pose.orientation.w
        qz = msg.pose.pose.orientation.z
        self.heading = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)
        
        current_pos = np.array([self.x, self.y])
        
        # Record positions during mapping
        if self.mapping_mode and not self.lap_complete:
            self.recorded_positions.append(current_pos.copy())
            
            # Check if we've completed a lap
            if self.start_position is None and len(self.recorded_positions) > 10:
                self.start_position = current_pos.copy()
            elif self.start_position is not None and len(self.recorded_positions) > 100:
                dist_to_start = np.linalg.norm(current_pos - self.start_position)
                if dist_to_start < 1.5:  # Close to start
                    self.complete_mapping()

    def complete_mapping(self):
        """Process recorded positions into centerline"""
        if len(self.recorded_positions) < 50:
            return
            
        self.lap_complete = True
        
        # Downsample recorded positions
        positions = np.array(list(self.recorded_positions))
        step = max(1, len(positions) // 200)
        self.centerline_points = positions[::step]
        
        # Smooth the centerline
        self.centerline_points = self.smooth_centerline(self.centerline_points)
        
        self.mapping_mode = False
        self.get_logger().info(f"Mapping complete! Centerline has {len(self.centerline_points)} points. Switching to following mode!")

    def smooth_centerline(self, points, window=5):
        """Smooth centerline using moving average"""
        if len(points) < window:
            return points
        
        smoothed = []
        for i in range(len(points)):
            start = max(0, i - window//2)
            end = min(len(points), i + window//2 + 1)
            smoothed.append(np.mean(points[start:end], axis=0))
        
        return np.array(smoothed)

    def get_lookahead_point(self):
        """Find lookahead point on centerline"""
        if len(self.centerline_points) == 0:
            return None
        
        current_pos = np.array([self.x, self.y])
        
        # Find closest point on centerline
        distances = np.linalg.norm(self.centerline_points - current_pos, axis=1)
        closest_idx = np.argmin(distances)
        
        # Find lookahead point
        lookahead_dist = 0.0
        lookahead_idx = closest_idx
        
        for i in range(closest_idx, closest_idx + len(self.centerline_points)):
            idx = i % len(self.centerline_points)
            next_idx = (idx + 1) % len(self.centerline_points)
            
            segment_length = np.linalg.norm(self.centerline_points[next_idx] - self.centerline_points[idx])
            lookahead_dist += segment_length
            
            if lookahead_dist >= self.LOOKAHEAD_DISTANCE:
                lookahead_idx = next_idx
                break
        
        return self.centerline_points[lookahead_idx]

    def pure_pursuit_control(self):
        """Calculate steering using pure pursuit"""
        lookahead_point = self.get_lookahead_point()
        
        if lookahead_point is None:
            return 0.0
        
        # Transform lookahead point to car frame
        dx = lookahead_point[0] - self.x
        dy = lookahead_point[1] - self.y
        
        # Rotate to car frame
        cos_h = math.cos(-self.heading)
        sin_h = math.sin(-self.heading)
        local_x = dx * cos_h - dy * sin_h
        local_y = dx * sin_h + dy * cos_h
        
        # Pure pursuit steering
        ld = np.linalg.norm([local_x, local_y])
        if ld < 0.1:
            return 0.0
        
        curvature = 2.0 * local_y / (ld ** 2)
        steer = np.arctan(curvature * 0.33)  # 0.33 is wheelbase
        
        return np.clip(steer, -self.MAX_STEER, self.MAX_STEER)

    def lidar_callback(self, scan_data):
        ranges = np.array(scan_data.ranges, dtype=np.float32)
        ranges = np.where(np.isinf(ranges), 10.0, ranges)
        ranges = np.where(np.isnan(ranges), 10.0, ranges)
        ranges = np.clip(ranges, 0.1, 10.0)
        
        num_ranges = len(ranges)
        angle_min = scan_data.angle_min
        angle_inc = scan_data.angle_increment
        center_idx = num_ranges // 2
        
        if self.mapping_mode:
            # Use safe gap follower during mapping
            steer_cmd, throttle_cmd = self.mapping_lap_control(ranges, num_ranges, center_idx, angle_min, angle_inc)
        else:
            # Use pure pursuit for following
            steer_cmd = self.pure_pursuit_control()
            
            # Speed control based on curvature
            forward_width = int(num_ranges * 0.08)
            forward_min = np.min(ranges[center_idx - forward_width:center_idx + forward_width])
            
            if abs(steer_cmd) > 0.25 or forward_min < 1.2:
                throttle_cmd = self.TURN_SPEED
            else:
                throttle_cmd = self.BASE_SPEED
        
        self.publish_command(throttle_cmd, steer_cmd)

    def mapping_lap_control(self, ranges, num_ranges, center_idx, angle_min, angle_inc):
        """Very conservative control for mapping lap"""
        # Find safest direction
        search_width = int(num_ranges * 0.19)
        search_start = max(0, center_idx - search_width)
        search_end = min(num_ranges, center_idx + search_width)
        
        window_size = 9
        best_idx = center_idx
        best_score = 0.0
        
        for i in range(search_start, search_end - window_size):
            window_avg = np.mean(ranges[i:i+window_size])
            distance_from_center = abs(i - center_idx)
            center_penalty = (distance_from_center / num_ranges) * 1.5
            score = window_avg * (1.0 - center_penalty)
            
            if score > best_score:
                best_score = score
                best_idx = i + window_size // 2
        
        target_angle = angle_min + (best_idx * angle_inc)
        steer_cmd = target_angle * 0.7
        steer_cmd = 0.6 * steer_cmd + 0.4 * self.prev_steer
        steer_cmd = np.clip(steer_cmd, -self.MAX_STEER, self.MAX_STEER)
        self.prev_steer = steer_cmd
        
        # Very slow during mapping
        forward_width = int(num_ranges * 0.08)
        forward_min = np.min(ranges[center_idx - forward_width:center_idx + forward_width])
        
        if abs(steer_cmd) > 0.2 or forward_min < 1.2:
            throttle_cmd = 0.08
        else:
            throttle_cmd = 0.12
        
        return steer_cmd, throttle_cmd

    def publish_command(self, throttle, steer):
        t_msg = Float32()
        t_msg.data = float(throttle)
        self.pub_throttle.publish(t_msg)
        
        s_msg = Float32()
        s_msg.data = float(steer)
        self.pub_steering.publish(s_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MapFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

