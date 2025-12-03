"""
Local Planner - Wall follower with UNCONSTRAINED steering
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Bool, Int32
from geometry_msgs.msg import Point
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import numpy as np

MAX_STEER = 0.5236  # Physical limit only
TARGET_WALL_DIST = 0.7

class LocalPlanner(Node):
    def __init__(self):
        super().__init__('local_planner')
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        
        self.pub_throttle = self.create_publisher(Float32, '/autodrive/roboracer_1/throttle_command', 10)
        self.pub_steering = self.create_publisher(Float32, '/autodrive/roboracer_1/steering_command', 10)
        self.pub_path = self.create_publisher(Path, '/recorded_path', 10)
        self.pub_ready = self.create_publisher(Bool, '/map_ready', 10)
        
        self.sub_lidar = self.create_subscription(LaserScan, '/autodrive/roboracer_1/lidar', self.lidar_cb, qos)
        self.sub_ips = self.create_subscription(Point, '/autodrive/roboracer_1/ips', self.ips_cb, 10)
        self.sub_lap = self.create_subscription(Int32, '/autodrive/roboracer_1/lap_count', self.lap_cb, 10)
        self.sub_mpc = self.create_subscription(Bool, '/mpc_active', self.mpc_cb, 10)
        
        self.x, self.y = 0.0, 0.0
        self.waypoints = []
        self.lap = 0
        self.mpc_active = False
        
        self.get_logger().info("LOCAL PLANNER: Unconstrained steering")

    def ips_cb(self, msg):
        self.x, self.y = msg.x, msg.y
        if self.lap == 0:
            pos = np.array([self.x, self.y])
            if len(self.waypoints) == 0 or np.linalg.norm(pos - self.waypoints[-1]) > 0.2:
                self.waypoints.append(pos)

    def lap_cb(self, msg):
        if msg.data > self.lap and self.lap == 0:
            self.get_logger().info(f"LAP 1 DONE! {len(self.waypoints)} waypoints")
            self.publish_path()
        self.lap = msg.data

    def mpc_cb(self, msg):
        self.mpc_active = msg.data

    def publish_path(self):
        msg = Path()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        for pt in self.waypoints:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x, pose.pose.position.y = float(pt[0]), float(pt[1])
            msg.poses.append(pose)
        self.pub_path.publish(msg)
        ready = Bool()
        ready.data = True
        self.pub_ready.publish(ready)

    def lidar_cb(self, msg):
        if self.mpc_active:
            return
        
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isfinite(ranges), ranges, 10.0)
        ranges = np.clip(ranges, 0.1, 10.0)
        
        n = len(ranges)
        angle_min = msg.angle_min
        angle_inc = msg.angle_increment
        
        def angle_to_idx(angle_deg):
            angle_rad = np.radians(angle_deg)
            idx = int((angle_rad - angle_min) / angle_inc)
            return np.clip(idx, 0, n-1)
        
        def get_dist(idx):
            start = max(0, idx - 15)
            end = min(n, idx + 15)
            return np.min(ranges[start:end])
        
        # Key distances
        d_right = get_dist(angle_to_idx(-90))   # Right side
        d_right_45 = get_dist(angle_to_idx(-45))
        d_front = get_dist(angle_to_idx(0))     # Front
        d_left_45 = get_dist(angle_to_idx(45))
        d_left = get_dist(angle_to_idx(90))     # Left side
        
        # === STEERING LOGIC ===
        # Positive steer = turn LEFT
        # Negative steer = turn RIGHT
        
        steer = 0.0
        
        # 1. If too close to RIGHT wall -> steer LEFT (positive)
        if d_right < 0.4:
            steer = MAX_STEER  # Hard left
        elif d_right < 0.6:
            steer = MAX_STEER * 0.7
        elif d_right < 0.8:
            steer = MAX_STEER * 0.4
            
        # 2. If too close to LEFT wall -> steer RIGHT (negative)
        if d_left < 0.4:
            steer = -MAX_STEER  # Hard right
        elif d_left < 0.6:
            steer = -MAX_STEER * 0.7
        elif d_left < 0.8:
            steer = -MAX_STEER * 0.4
        
        # 3. If front blocked -> turn toward more open side
        if d_front < 0.5:
            if d_left > d_right:
                steer = MAX_STEER  # Turn left
            else:
                steer = -MAX_STEER  # Turn right
        elif d_front < 1.0:
            if d_left > d_right:
                steer = MAX_STEER * 0.6
            else:
                steer = -MAX_STEER * 0.6
        
        # 4. If all clear, center between walls
        if d_front > 1.5 and d_left > 0.8 and d_right > 0.8:
            # Steer toward center
            steer = (d_right - d_left) * 0.3
        
        # NO CLAMPING except physical limit
        steer = np.clip(steer, -MAX_STEER, MAX_STEER)
        
        # Speed
        min_dist = min(d_front, d_left, d_right)
        if min_dist < 0.4:
            speed = 0.0
        elif min_dist < 0.6:
            speed = 0.05
        elif d_front < 0.8:
            speed = 0.08
        elif d_front < 1.2:
            speed = 0.10
        else:
            speed = 0.15
        
        # Publish
        t, s = Float32(), Float32()
        t.data, s.data = float(speed), float(steer)
        self.pub_throttle.publish(t)
        self.pub_steering.publish(s)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(LocalPlanner())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
