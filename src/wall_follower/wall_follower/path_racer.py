"""
PHASE 2: PATH RACER
- Loads the saved path from track_mapper
- Uses pure pursuit to follow it
- Ignores LIDAR traps!

RUN: ros2 run wall_follower path_racer
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from geometry_msgs.msg import Point
import numpy as np
import json
import os

class PathRacer(Node):
    def __init__(self):
        super().__init__('path_racer')
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        
        # Publishers
        self.pub_throttle = self.create_publisher(Float32, '/autodrive/roboracer_1/throttle_command', 10)
        self.pub_steering = self.create_publisher(Float32, '/autodrive/roboracer_1/steering_command', 10)
        
        # Subscribers
        self.sub_ips = self.create_subscription(Point, '/autodrive/roboracer_1/ips', self.ips_cb, 10)
        self.sub_lidar = self.create_subscription(LaserScan, '/autodrive/roboracer_1/lidar', self.lidar_cb, qos)
        
        # State
        self.x, self.y = 0.0, 0.0
        self.yaw = 0.0
        self.prev_x, self.prev_y = 0.0, 0.0
        self.waypoints = []
        self.current_idx = 0
        
        # Tuning
        self.LOOKAHEAD = 0.8  # How far ahead to look (meters)
        self.SPEED = 0.15    # Base speed
        self.WHEELBASE = 0.324  # Vehicle wheelbase
        
        # Load path
        self.load_path()
        
        if len(self.waypoints) > 0:
            self.get_logger().info(f"=== PATH RACER STARTED ===")
            self.get_logger().info(f"Loaded {len(self.waypoints)} waypoints")
            self.get_logger().info("Following the recorded path!")
        else:
            self.get_logger().error("NO PATH FOUND! Run track_mapper first!")

    def load_path(self):
        """Load waypoints from JSON file"""
        path_file = os.path.expanduser('~/roboracer_ws/src/wall_follower/track_path.json')
        try:
            with open(path_file, 'r') as f:
                data = json.load(f)
                self.waypoints = np.array(data['waypoints'])
                self.get_logger().info(f"Loaded path from {path_file}")
        except FileNotFoundError:
            self.get_logger().error(f"Path file not found: {path_file}")
            self.waypoints = np.array([])
        except Exception as e:
            self.get_logger().error(f"Error loading path: {e}")
            self.waypoints = np.array([])

    def ips_cb(self, msg):
        """Update position and estimate heading"""
        # Estimate heading from movement
        if abs(msg.x - self.prev_x) > 0.01 or abs(msg.y - self.prev_y) > 0.01:
            self.yaw = np.arctan2(msg.y - self.prev_y, msg.x - self.prev_x)
            self.prev_x, self.prev_y = self.x, self.y
        
        self.x, self.y = msg.x, msg.y

    def find_lookahead_point(self):
        """Find the target point on the path to steer toward"""
        if len(self.waypoints) == 0:
            return None
        
        # Find closest waypoint
        pos = np.array([self.x, self.y])
        dists = np.linalg.norm(self.waypoints - pos, axis=1)
        closest_idx = np.argmin(dists)
        
        # Look ahead from closest point
        n = len(self.waypoints)
        best_idx = closest_idx
        best_dist = 0
        
        for i in range(20):  # Look up to 20 waypoints ahead
            idx = (closest_idx + i) % n
            d = np.linalg.norm(self.waypoints[idx] - pos)
            if d >= self.LOOKAHEAD:
                best_idx = idx
                break
            if d > best_dist:
                best_dist = d
                best_idx = idx
        
        self.current_idx = best_idx
        return self.waypoints[best_idx]

    def pure_pursuit_steering(self, target):
        """Calculate steering angle using pure pursuit"""
        if target is None:
            return 0.0
        
        # Vector to target
        dx = target[0] - self.x
        dy = target[1] - self.y
        
        # Distance to target
        L = np.sqrt(dx*dx + dy*dy)
        if L < 0.1:
            return 0.0
        
        # Angle to target in world frame
        target_angle = np.arctan2(dy, dx)
        
        # Angle relative to car heading
        alpha = target_angle - self.yaw
        
        # Normalize to [-pi, pi]
        while alpha > np.pi:
            alpha -= 2*np.pi
        while alpha < -np.pi:
            alpha += 2*np.pi
        
        # Pure pursuit: steering = atan(2 * L_wheelbase * sin(alpha) / L_lookahead)
        steer = np.arctan2(2.0 * self.WHEELBASE * np.sin(alpha), self.LOOKAHEAD)
        
        # Limit steering
        steer = np.clip(steer, -0.5, 0.5)
        
        return steer

    def lidar_cb(self, msg):
        """Control loop - follows path with emergency wall avoidance"""
        if len(self.waypoints) == 0:
            self.publish(0.0, 0.0)
            return
        
        # Get target point and pure pursuit steering
        target = self.find_lookahead_point()
        steer = self.pure_pursuit_steering(target)
        speed = self.SPEED
        
        # === MINIMAL SAFETY: Only emergency wall avoidance ===
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.where(np.isfinite(ranges), ranges, 10.0)
        ranges = np.clip(ranges, 0.1, 10.0)
        
        n = len(ranges)
        center = n // 2
        
        # Check front distance
        front_ranges = ranges[center-50:center+50]
        front_min = np.min(front_ranges)
        
        # Only intervene if about to crash
        if front_min < 0.35:
            speed = 0.05
            # Check which side is more open
            left = np.mean(ranges[center+100:center+200])
            right = np.mean(ranges[center-200:center-100])
            if left > right:
                steer = max(steer, 0.3)
            else:
                steer = min(steer, -0.3)
        elif front_min < 0.5:
            speed = min(speed, 0.10)
        
        self.publish(speed, steer)
        
        # Log occasionally
        if hasattr(self, '_count'):
            self._count += 1
        else:
            self._count = 0
        if self._count % 30 == 0:
            self.get_logger().info(f"Pos:({self.x:.1f},{self.y:.1f}) WP:{self.current_idx}/{len(self.waypoints)} St:{steer:.2f}")

    def publish(self, speed, steer):
        t_msg, s_msg = Float32(), Float32()
        t_msg.data = float(speed)
        s_msg.data = float(steer)
        self.pub_throttle.publish(t_msg)
        self.pub_steering.publish(s_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PathRacer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

