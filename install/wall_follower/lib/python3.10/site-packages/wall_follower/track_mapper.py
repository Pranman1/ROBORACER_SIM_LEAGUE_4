"""
PURE RIGHT WALL FOLLOWER - Simplest possible
- Just stay 0.6m from right wall
- Follow it around the track
- No gap detection, no convexity
- Just PID wall following

If this doesn't work, we try MPC with a recorded path.

RUN: ros2 run wall_follower track_mapper
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Int32
from geometry_msgs.msg import Point, TransformStamped
from tf2_ros import TransformBroadcaster
import numpy as np
import subprocess
import time

class TrackMapper(Node):
    def __init__(self):
        super().__init__('track_mapper')
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        
        self.pub_throttle = self.create_publisher(Float32, '/autodrive/roboracer_1/throttle_command', 10)
        self.pub_steering = self.create_publisher(Float32, '/autodrive/roboracer_1/steering_command', 10)
        self.pub_scan = self.create_publisher(LaserScan, '/scan', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        self.sub_lidar = self.create_subscription(LaserScan, '/autodrive/roboracer_1/lidar', self.lidar_cb, qos)
        self.sub_ips = self.create_subscription(Point, '/autodrive/roboracer_1/ips', self.ips_cb, 10)
        self.sub_lap = self.create_subscription(Int32, '/autodrive/roboracer_1/lap_count', self.lap_cb, 10)
        
        self.x, self.y, self.yaw = 0.0, 0.0, 0.0
        self.prev_x, self.prev_y = 0.0, 0.0
        self.lap = 0
        self.done = False
        
        # Simple PID wall following
        self.TARGET_DIST = 0.8  # Stay 0.6m from right wall
        self.Kp = 2.0  # Even stronger!
        self.Kd = 0.6
        self.prev_error = 0.0
        
        self.SPEED = 0.04
        self.MAX_STEER = 0.7  # FULL physical limit!
        
        self.count = 0
        self.start_time = time.time()
        self.STARTUP_DELAY = 3.5
        
        # Simple reversing
        self.reversing = False
        self.reverse_start = 0
        self.position_history = []
        self.heading_history = []  # Track heading to prevent 180 turns
        
        self.create_timer(0.02, self.publish_tf)
        self.create_timer(0.3, self.check_stuck)
        self.launch_slam()
        
        self.get_logger().info("="*50)
        self.get_logger().info("PURE RIGHT WALL FOLLOWER")
        self.get_logger().info("Just follow the right wall. Simple.")
        self.get_logger().info("="*50)

    def launch_slam(self):
        config = """
slam_toolbox:
  ros__parameters:
    odom_frame: odom
    map_frame: map
    base_frame: base_link
    scan_topic: /scan
    mode: mapping
    resolution: 0.05
    max_laser_range: 10.0
    map_update_interval: 1.0
    use_scan_matching: true
    minimum_travel_distance: 0.08
    do_loop_closing: true
"""
        rviz = """
Visualization Manager:
  Displays:
    - Class: rviz_default_plugins/Grid
      Cell Size: 1
      Plane Cell Count: 100
    - Class: rviz_default_plugins/Map
      Topic:
        Value: /map
      Value: true
    - Class: rviz_default_plugins/LaserScan
      Topic:
        Value: /scan
      Size (m): 0.05
      Value: true
  Global Options:
    Fixed Frame: map
"""
        try:
            with open('/tmp/slam_p.yaml', 'w') as f: f.write(config)
            with open('/tmp/slam_r.rviz', 'w') as f: f.write(rviz)
            subprocess.Popen(['ros2', 'run', 'slam_toolbox', 'async_slam_toolbox_node',
                            '--ros-args', '--params-file', '/tmp/slam_p.yaml'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.Popen(['rviz2', '-d', '/tmp/slam_r.rviz'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            pass

    def publish_tf(self):
        now = self.get_clock().now().to_msg()
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.rotation.z = np.sin(self.yaw / 2)
        t.transform.rotation.w = np.cos(self.yaw / 2)
        self.tf_broadcaster.sendTransform(t)
        
        t2 = TransformStamped()
        t2.header.stamp = now
        t2.header.frame_id = 'base_link'
        t2.child_frame_id = 'laser'
        t2.transform.translation.x = 0.1
        t2.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t2)

    def check_stuck(self):
        if self.reversing or self.done:
            return
        elapsed = time.time() - self.start_time
        if elapsed < self.STARTUP_DELAY + 2:
            return
        
        self.position_history.append((self.x, self.y, time.time()))
        if len(self.position_history) > 20:
            self.position_history.pop(0)
        
        if len(self.position_history) >= 15:
            old = self.position_history[0]
            new = self.position_history[-1]
            dist = np.sqrt((new[0]-old[0])**2 + (new[1]-old[1])**2)
            dt = new[2] - old[2]
            if dist < 0.04 and dt > 4.0:
                self.get_logger().warn("STUCK!")
                self.reversing = True
                self.reverse_start = time.time()
                self.position_history.clear()

    def ips_cb(self, msg):
        dx, dy = msg.x - self.prev_x, msg.y - self.prev_y
        if dx*dx + dy*dy > 0.0001:
            self.yaw = 0.85 * self.yaw + 0.15 * np.arctan2(dy, dx)
            self.heading_history.append(self.yaw)
            if len(self.heading_history) > 40:
                self.heading_history.pop(0)
        self.prev_x, self.prev_y = self.x, self.y
        self.x, self.y = msg.x, msg.y

    def lap_cb(self, msg):
        if msg.data > self.lap and not self.done:
            self.done = True
            self.get_logger().info("LAP COMPLETE!")
        self.lap = msg.data

    def get_right_wall_dist(self, ranges, n, center):
        """Get distance to right wall at -90 degrees"""
        # Right is at about 1/4 of the scan (assuming 270 deg FOV)
        right_idx = center - n // 4  # -67.5 degrees
        
        # Take average of a window
        window = 30
        start = max(0, right_idx - window)
        end = min(n, right_idx + window)
        
        right_ranges = ranges[start:end]
        valid = right_ranges[right_ranges < 5.0]
        
        if len(valid) > 0:
            return float(np.percentile(valid, 25))  # Use 25th percentile (closer readings)
        return 3.0

    def get_front_dist(self, ranges, n, center):
        """Get distance ahead"""
        front = ranges[center-40:center+40]
        return float(np.min(front))

    def lidar_cb(self, msg):
        # Republish for SLAM
        scan = LaserScan()
        scan.header = msg.header
        scan.header.frame_id = 'laser'
        scan.angle_min = msg.angle_min
        scan.angle_max = msg.angle_max
        scan.angle_increment = msg.angle_increment
        scan.time_increment = msg.time_increment
        scan.scan_time = msg.scan_time
        scan.range_min = msg.range_min
        scan.range_max = msg.range_max
        scan.ranges = msg.ranges
        scan.intensities = msg.intensities
        self.pub_scan.publish(scan)
        
        elapsed = time.time() - self.start_time
        if elapsed < self.STARTUP_DELAY:
            self.publish(0.0, 0.0)
            return
        
        if self.done:
            self.publish(0.0, 0.0)
            return
        
        self.count += 1
        
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.where(np.isfinite(ranges), ranges, 10.0)
        ranges = np.clip(ranges, 0.1, 10.0)
        
        n = len(ranges)
        center = n // 2
        
        # Get distances
        d_right = self.get_right_wall_dist(ranges, n, center)
        d_front = self.get_front_dist(ranges, n, center)
        d_back = min(float(np.min(ranges[:n//8])), float(np.min(ranges[-n//8:])))
        
        # === REVERSING - STRAIGHT BACK! ===
        if self.reversing:
            elapsed_rev = time.time() - self.reverse_start
            if elapsed_rev < 0.8 and d_back > 0.3:
                # Reverse STRAIGHT - no turning! PID will fix it going forward
                self.publish(-0.018, 0.0)  # Zero steering!
                self.get_logger().info(f"REV STRAIGHT {0.8-elapsed_rev:.1f}s")
                return
            else:
                self.reversing = False
                self.prev_error = 0.0  # Reset PID
        
        # === CHECK FOR 180 TURN (going backwards!) ===
        if len(self.heading_history) > 30:
            old_heading = self.heading_history[0]
            new_heading = self.heading_history[-1]
            diff = abs(new_heading - old_heading)
            while diff > np.pi:
                diff = abs(diff - 2*np.pi)
            if diff > 2.2:  # ~125 degrees = going back on ourselves!
                self.get_logger().warn("TURNING AROUND! Force forward!")
                self.heading_history.clear()
                # Don't reverse, just keep going and let PID correct
        
        # === FRONT BLOCKED ===
        if d_front < 0.35:
            self.get_logger().warn("FRONT BLOCKED!")
            self.reversing = True
            self.reverse_start = time.time()
            return
        
        # === PURE RIGHT WALL FOLLOWING PID ===
        error = d_right - self.TARGET_DIST
        derivative = error - self.prev_error
        self.prev_error = error
        
        # Negative because: too far from wall (error+) -> turn right (steer-)
        steer = -(self.Kp * error + self.Kd * derivative)
        steer = np.clip(steer, -self.MAX_STEER, self.MAX_STEER)
        
        # === FRONT OBSTACLE - EMERGENCY SHARP TURN! ===
        if d_front < 0.7:
            steer = max(steer, 0.4)  # Start turning
        if d_front < 0.55:
            steer = max(steer, self.MAX_STEER)  # MAXIMUM TURN!
        if d_front < 0.45:
            steer = self.MAX_STEER  # FORCE maximum!
        
        # === SPEED ===
        if d_front < 0.5:
            speed = 0.02
        elif d_front < 0.8 or abs(steer) > 0.3:
            speed = 0.03
        else:
            speed = self.SPEED
        
        self.publish(speed, steer)
        
        if self.count % 30 == 0:
            self.get_logger().info(f"R:{d_right:.2f} F:{d_front:.2f} err:{error:.2f} st:{steer:.2f}")

    def publish(self, speed, steer):
        t, s = Float32(), Float32()
        t.data, s.data = float(speed), float(steer)
        self.pub_throttle.publish(t)
        self.pub_steering.publish(s)


def main(args=None):
    rclpy.init(args=args)
    node = TrackMapper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
