"""
PURE RIGHT WALL FOLLOWER - CONTINUOUS
- Right wall following (your tuned params!)
- No mapping (removed)
- Runs continuously for multiple laps
- Test endurance!

RUN: ros2 run wall_follower track_mapper
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Int32
from geometry_msgs.msg import Point
import numpy as np
import time

class TrackMapper(Node):
    def __init__(self):
        super().__init__('track_mapper')
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        
        self.pub_throttle = self.create_publisher(Float32, '/autodrive/roboracer_1/throttle_command', 10)
        self.pub_steering = self.create_publisher(Float32, '/autodrive/roboracer_1/steering_command', 10)
        
        self.sub_lidar = self.create_subscription(LaserScan, '/autodrive/roboracer_1/lidar', self.lidar_cb, qos)
        self.sub_ips = self.create_subscription(Point, '/autodrive/roboracer_1/ips', self.ips_cb, 10)
        self.sub_lap = self.create_subscription(Int32, '/autodrive/roboracer_1/lap_count', self.lap_cb, 10)
        
        self.x, self.y, self.yaw = 0.0, 0.0, 0.0
        self.prev_x, self.prev_y = 0.0, 0.0
        self.lap = 0
        self.max_lap = 10  # Run for 10 laps!
        
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
        
        self.create_timer(0.3, self.check_stuck)
        
        self.get_logger().info("="*50)
        self.get_logger().info("ENDURANCE TEST - 10 LAPS")
        self.get_logger().info("="*50)

    def OLD_launch_rviz(self):
        """Launch RViz to visualize the map"""
        rviz = """
Visualization Manager:
  Displays:
    - Class: rviz_default_plugins/Grid
      Cell Size: 1
      Plane Cell Count: 50
    - Class: rviz_default_plugins/Map
      Name: Track Map
      Topic:
        Value: /map
      Color Scheme: map
      Value: true
  Global Options:
    Fixed Frame: map
    Frame Rate: 30
"""
        try:
            with open('/tmp/mapper_rviz.rviz', 'w') as f: f.write(rviz)
            subprocess.Popen(['rviz2', '-d', '/tmp/mapper_rviz.rviz'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.get_logger().info("RViz launched!")
        except:
            pass

    def OLD_world_to_grid(self, wx, wy):
        """Convert world coordinates to grid indices"""
        gx = int((wx - self.grid_origin_x) / self.RESOLUTION)
        gy = int((wy - self.grid_origin_y) / self.RESOLUTION)
        return gx, gy

    def OLD_bresenham(self, x0, y0, x1, y1):
        """Bresenham line algorithm for ray tracing"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            if 0 <= x0 < self.grid_size and 0 <= y0 < self.grid_size:
                points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return points

    def OLD_update_map(self, ranges, angle_min, angle_inc):
        """Update occupancy grid with confidence-based filtering"""
        robot_gx, robot_gy = self.world_to_grid(self.x, self.y)
        
        if not (0 <= robot_gx < self.grid_size and 0 <= robot_gy < self.grid_size):
            return
        
        # Only use every 5th ray (reduce noise)
        for i in range(0, len(ranges), 5):
            r = ranges[i]
            if r < 0.3 or r > 7.0:  # Tighter range for reliability
                continue
            
            ray_angle = angle_min + i * angle_inc + self.yaw
            
            # Hit point
            hit_x = self.x + r * np.cos(ray_angle)
            hit_y = self.y + r * np.sin(ray_angle)
            hit_gx, hit_gy = self.world_to_grid(hit_x, hit_y)
            
            # Ray trace
            line = self.bresenham(robot_gx, robot_gy, hit_gx, hit_gy)
            
            # Mark free space (misses)
            for gx, gy in line[:-1]:
                if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                    self.miss_count[gy, gx] += 1
                    # Only mark as free after enough misses
                    if self.miss_count[gy, gx] >= self.MISS_THRESHOLD:
                        if self.hit_count[gy, gx] < self.HIT_THRESHOLD:  # Not confidently an obstacle
                            self.grid[gy, gx] = 0
            
            # Mark obstacle (hit)
            if 0 <= hit_gx < self.grid_size and 0 <= hit_gy < self.grid_size:
                self.hit_count[hit_gy, hit_gx] += 1
                # Only mark as obstacle after enough hits
                if self.hit_count[hit_gy, hit_gx] >= self.HIT_THRESHOLD:
                    self.grid[hit_gy, hit_gx] = 100

    def OLD_publish_map(self):
        """Publish occupancy grid to RViz"""
        msg = OccupancyGrid()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        
        msg.info.resolution = self.RESOLUTION
        msg.info.width = self.grid_size
        msg.info.height = self.grid_size
        msg.info.origin.position.x = self.grid_origin_x
        msg.info.origin.position.y = self.grid_origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        
        msg.data = self.grid.flatten().tolist()
        self.pub_map.publish(msg)


    def check_stuck(self):
        if self.reversing:
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
        if msg.data > self.lap:
            self.lap = msg.data
            self.get_logger().info(f"*** LAP {self.lap} COMPLETE! ***")
            if self.lap >= self.max_lap:
                self.get_logger().info(f"=== {self.max_lap} LAPS DONE! SUCCESS! ===")
                # Could stop here, but let's keep going to see how far it gets!

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
        elapsed = time.time() - self.start_time
        if elapsed < self.STARTUP_DELAY:
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
