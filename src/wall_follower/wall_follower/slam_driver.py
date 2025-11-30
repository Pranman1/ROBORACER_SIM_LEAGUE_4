import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
import numpy as np
import math

class SlamDriver(Node):
    def __init__(self):
        super().__init__('slam_driver')
        
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        
        # Publishers
        self.pub_throttle = self.create_publisher(Float32, '/autodrive/roboracer_1/throttle_command', 10)
        self.pub_steering = self.create_publisher(Float32, '/autodrive/roboracer_1/steering_command', 10)
        
        # Subscribers
        self.sub_lidar = self.create_subscription(LaserScan, '/autodrive/roboracer_1/lidar', self.lidar_callback, qos)
        self.sub_odom = self.create_subscription(Odometry, '/autodrive/roboracer_1/ips', self.odom_callback, qos)
        
        # State
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        
        # DEBUG COUNTERS
        self.lidar_count = 0
        self.odom_count = 0
        
        # PHASE CONTROL
        self.phase = "MAPPING"
        
        # Mapping data
        self.path_points = []
        self.start_pos = None
        self.min_points_before_lap_check = 100  # Reduced - check earlier
        self.lap_close_distance = 3.5  # Increased - easier to detect
        
        # Following data
        self.centerline = None
        
        # Speeds
        self.MAP_SPEED = 0.10
        self.FOLLOW_SPEED = 0.12
        self.MAX_STEER = 0.40
        
        self.prev_steer = 0.0
        
        self.get_logger().info("="*50)
        self.get_logger().info("SLAM DRIVER STARTED")
        self.get_logger().info("Waiting for LIDAR and ODOM data...")
        self.get_logger().info("="*50)

    def odom_callback(self, msg):
        self.odom_count += 1
        
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        
        qw = msg.pose.pose.orientation.w
        qz = msg.pose.pose.orientation.z
        self.heading = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)
        
        # DEBUG: Log first few odom messages
        if self.odom_count <= 3:
            self.get_logger().info(f"ODOM RECEIVED #{self.odom_count}: x={self.x:.2f} y={self.y:.2f}")
        
        if self.phase == "MAPPING":
            self.record_position()

    def record_position(self):
        pos = np.array([self.x, self.y])
        
        if self.start_pos is None:
            self.start_pos = pos.copy()
            self.get_logger().info(f">>> START POSITION SET: ({self.x:.2f}, {self.y:.2f})")
        
        if len(self.path_points) == 0:
            self.path_points.append(pos)
        else:
            last = self.path_points[-1]
            if np.linalg.norm(pos - last) > 0.1:
                self.path_points.append(pos)
                
                dist_to_start = np.linalg.norm(pos - self.start_pos)
                
                # Log every 10 points with distance to start
                if len(self.path_points) % 10 == 0:
                    self.get_logger().info(f"MAP: {len(self.path_points)} pts | Pos:({self.x:.1f},{self.y:.1f}) | TO_START:{dist_to_start:.1f}m")
                
                # Extra log when getting close to start
                if len(self.path_points) > self.min_points_before_lap_check and dist_to_start < 6.0:
                    self.get_logger().warn(f">>> APPROACHING START! Dist={dist_to_start:.1f}m (need <{self.lap_close_distance}m)")
        
        # Check for lap completion
        if len(self.path_points) > self.min_points_before_lap_check:
            dist_to_start = np.linalg.norm(pos - self.start_pos)
            if dist_to_start < self.lap_close_distance:
                self.get_logger().info(f">>> LAP DETECTED! Dist={dist_to_start:.1f}m")
                self.complete_mapping()

    def complete_mapping(self):
        self.phase = "STOPPED"
        self.get_logger().info("="*50)
        self.get_logger().info(">>> LAP COMPLETE! STOPPING CAR...")
        self.get_logger().info(f">>> Recorded {len(self.path_points)} points")
        self.get_logger().info("="*50)
        
        self.publish_drive(0.0, 0.0)
        self.process_centerline()
        
        self.phase = "FOLLOWING"
        self.get_logger().info("="*50)
        self.get_logger().info(">>> PHASE 2: FOLLOWING - Using recorded path")
        self.get_logger().info("="*50)

    def process_centerline(self):
        points = np.array(self.path_points)
        step = max(1, len(points) // 100)
        self.centerline = points[::step]
        
        smoothed = []
        window = 3
        for i in range(len(self.centerline)):
            start = max(0, i - window)
            end = min(len(self.centerline), i + window + 1)
            smoothed.append(np.mean(self.centerline[start:end], axis=0))
        
        self.centerline = np.array(smoothed)
        self.get_logger().info(f">>> Centerline ready: {len(self.centerline)} waypoints")

    def lidar_callback(self, data):
        self.lidar_count += 1
        
        # DEBUG: Log first few lidar messages
        if self.lidar_count <= 3:
            self.get_logger().info(f"LIDAR RECEIVED #{self.lidar_count}: {len(data.ranges)} points")
        
        ranges = np.array(data.ranges, dtype=np.float32)
        ranges = np.where(np.isinf(ranges), 10.0, ranges)
        ranges = np.where(np.isnan(ranges), 10.0, ranges)
        ranges = np.clip(ranges, 0.1, 10.0)
        
        num_points = len(ranges)
        center = num_points // 2
        angle_min = data.angle_min
        angle_inc = data.angle_increment
        
        if self.phase == "MAPPING":
            steer, speed = self.mapping_control(ranges, num_points, center, angle_min, angle_inc)
            self.publish_drive(speed, steer)
            
        elif self.phase == "STOPPED":
            self.publish_drive(0.0, 0.0)
            
        elif self.phase == "FOLLOWING":
            steer = self.pure_pursuit()
            
            forward_arc = int(num_points * 0.1)
            forward_min = np.min(ranges[center - forward_arc:center + forward_arc])
            
            speed = self.FOLLOW_SPEED
            if abs(steer) > 0.2 or forward_min < 1.0:
                speed = 0.10
            if forward_min < 0.6:
                speed = 0.06
                
            # Log every 10th message
            if self.lidar_count % 10 == 0:
                self.get_logger().info(f"FOLLOW: St={steer:.2f} Sp={speed} Fwd={forward_min:.1f}")
            
            self.publish_drive(speed, steer)

    def mapping_control(self, ranges, num_points, center, angle_min, angle_inc):
        forward_arc = int(num_points * 0.17)
        start = center - forward_arc
        end = center + forward_arc
        forward_ranges = ranges[start:end]
        forward_min = np.min(forward_ranges)
        
        window = 7
        smoothed = np.convolve(forward_ranges, np.ones(window)/window, mode='same')
        best_idx = start + np.argmax(smoothed)
        target_angle = angle_min + best_idx * angle_inc
        
        steer = target_angle * 0.5
        steer = 0.6 * steer + 0.4 * self.prev_steer
        steer = np.clip(steer, -self.MAX_STEER, self.MAX_STEER)
        self.prev_steer = steer
        
        left_idx = center + int(num_points * 0.15)
        right_idx = center - int(num_points * 0.15)
        left_dist = np.mean(ranges[left_idx:left_idx+10]) if left_idx < num_points-10 else 10.0
        right_dist = np.mean(ranges[max(0,right_idx-10):right_idx]) if right_idx > 10 else 10.0
        
        if left_dist < 0.75:
            steer = -self.MAX_STEER
        elif right_dist < 0.75:
            steer = self.MAX_STEER
        
        speed = self.MAP_SPEED
        if abs(steer) > 0.2 or forward_min < 1.0:
            speed = 0.08
            
        return steer, speed

    def pure_pursuit(self):
        if self.centerline is None or len(self.centerline) == 0:
            return 0.0
        
        pos = np.array([self.x, self.y])
        dists = np.linalg.norm(self.centerline - pos, axis=1)
        closest_idx = np.argmin(dists)
        
        lookahead_idx = (closest_idx + 8) % len(self.centerline)
        target = self.centerline[lookahead_idx]
        
        dx = target[0] - self.x
        dy = target[1] - self.y
        
        cos_h = math.cos(-self.heading)
        sin_h = math.sin(-self.heading)
        local_x = dx * cos_h - dy * sin_h
        local_y = dx * sin_h + dy * cos_h
        
        ld = math.sqrt(local_x**2 + local_y**2)
        if ld < 0.1:
            return 0.0
        
        curvature = 2.0 * local_y / (ld ** 2)
        steer = math.atan(curvature * 0.33)
        
        return np.clip(steer, -self.MAX_STEER, self.MAX_STEER)

    def publish_drive(self, speed, steer):
        t_msg = Float32()
        t_msg.data = float(speed)
        self.pub_throttle.publish(t_msg)
        
        s_msg = Float32()
        s_msg.data = float(steer)
        self.pub_steering.publish(s_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SlamDriver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
