"""
PURE RIGHT WALL FOLLOWER + MAPPING
- Right wall following (your tuned params!)
- Maps track during lap 1
- STOPS after lap 1 to save map

RUN: ros2 run wall_follower track_mapper
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Int32
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
import numpy as np
import time
import json
import os
import subprocess

class TrackMapper(Node):
    def __init__(self):
        super().__init__('track_mapper')
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        
        self.pub_throttle = self.create_publisher(Float32, '/autodrive/roboracer_1/throttle_command', 10)
        self.pub_steering = self.create_publisher(Float32, '/autodrive/roboracer_1/steering_command', 10)
        self.pub_map = self.create_publisher(OccupancyGrid, '/map', 10)
        self.pub_path = self.create_publisher(Path, '/planned_path', 10)
        
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
        
        # ===== MAPPING ONLY (doesn't touch controls!) =====
        self.mapping_active = True
        self.stopped = False  # Flag to keep robot stopped after lap 1
        self.current_steer = 0.0
        self.planned_waypoints = None  # Store waypoints for continuous publishing
        self.start_x, self.start_y = None, None  # Track start position for loop closure
        self.loop_closed = False
        self.RESOLUTION = 0.05
        self.MAP_SIZE = 30.0
        self.grid_size = int(self.MAP_SIZE / self.RESOLUTION)
        self.grid_origin_x = -self.MAP_SIZE / 2
        self.grid_origin_y = -self.MAP_SIZE / 2
        self.grid = np.full((self.grid_size, self.grid_size), -1, dtype=np.int8)
        self.hit_count = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.miss_count = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.HIT_THRESHOLD = 3
        self.MISS_THRESHOLD = 5
        
        self.create_timer(0.3, self.check_stuck)
        self.create_timer(1.0, self.publish_map)
        self.create_timer(0.5, self.publish_path_timer)  # Publish path continuously
        
        self.launch_rviz()
        
        self.get_logger().info("="*50)
        self.get_logger().info("TRACK MAPPER - Loop Closure + STOP")
        self.get_logger().info("="*50)

    def launch_rviz(self):
        """Launch RViz to visualize the map"""
        rviz = """
Visualization Manager:
  Displays:
    - Class: rviz_default_plugins/Grid
      Cell Size: 1
      Plane Cell Count: 50
      Value: true
    - Class: rviz_default_plugins/Map
      Name: Track Map
      Topic:
        Value: /map
      Color Scheme: map
      Alpha: 0.7
      Value: true
    - Class: rviz_default_plugins/Path
      Name: Planned Path
      Topic:
        Value: /planned_path
      Color: 0; 255; 0
      Line Width: 0.1
      Value: true
  Global Options:
    Fixed Frame: map
    Frame Rate: 30
"""
        try:
            with open('/tmp/mapper_rviz.rviz', 'w') as f:
                f.write(rviz)
            subprocess.Popen(['rviz2', '-d', '/tmp/mapper_rviz.rviz'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.get_logger().info("RViz launched!")
        except:
            pass

    def world_to_grid(self, wx, wy):
        """Convert world coordinates to grid indices"""
        gx = int((wx - self.grid_origin_x) / self.RESOLUTION)
        gy = int((wy - self.grid_origin_y) / self.RESOLUTION)
        return gx, gy

    def bresenham(self, x0, y0, x1, y1):
        """Bresenham line algorithm"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        while True:
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                points.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return points

    def update_map(self, ranges, angle_min, angle_inc):
        """Update map - skip during turns/reversing!"""
        if not self.mapping_active:
            return
        
        # Skip during startup to avoid weird cluster
        if time.time() - self.start_time < self.STARTUP_DELAY + 2.0:
            return
        
        # Skip during reversing or sharp turns (KEY FIX!)
        if self.reversing or abs(self.current_steer) > 0.35:
            return
            
        robot_gx, robot_gy = self.world_to_grid(self.x, self.y)
        
        if not (0 <= robot_gx < self.grid_size and 0 <= robot_gy < self.grid_size):
            return
        
        # Every 3rd ray - good balance
        for i in range(0, len(ranges), 3):
            r = ranges[i]
            if r < 0.2 or r > 8.0 or not np.isfinite(r):
                continue
            
            ray_angle = angle_min + i * angle_inc + self.yaw
            
            hit_x = self.x + r * np.cos(ray_angle)
            hit_y = self.y + r * np.sin(ray_angle)
            hit_gx, hit_gy = self.world_to_grid(hit_x, hit_y)
            
            line = self.bresenham(robot_gx, robot_gy, hit_gx, hit_gy)
            
            if len(line) < 2:
                continue
            
            # Mark free
            for gx, gy in line[:-1]:
                if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                    self.miss_count[gy, gx] += 1
                    if self.miss_count[gy, gx] >= self.MISS_THRESHOLD:
                        if self.hit_count[gy, gx] < self.HIT_THRESHOLD:
                            self.grid[gy, gx] = 0
            
            # Mark obstacle
            if 0 <= hit_gx < self.grid_size and 0 <= hit_gy < self.grid_size:
                self.hit_count[hit_gy, hit_gx] += 1
                if self.hit_count[hit_gy, hit_gx] >= self.HIT_THRESHOLD:
                    self.grid[hit_gy, hit_gx] = 100

    def publish_map(self):
        """Publish map to RViz"""
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
    
    def save_map(self):
        """Save map to file"""
        try:
            # Use current directory to avoid path issues
            map_dir = os.getcwd()
            map_path = os.path.join(map_dir, 'track_map.npy')
            meta_path = os.path.join(map_dir, 'track_map_meta.json')
            
            np.save(map_path, self.grid)
            
            meta = {
                'resolution': self.RESOLUTION,
                'width': self.grid_size,
                'height': self.grid_size,
                'origin_x': self.grid_origin_x,
                'origin_y': self.grid_origin_y
            }
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
            
            occupied = np.sum(self.grid == 100)
            free = np.sum(self.grid == 0)
            total = self.grid_size * self.grid_size
            
            self.get_logger().info(f"MAP SAVED: {map_path}")
            self.get_logger().info(f"  Walls: {100*occupied/total:.1f}% | Free: {100*free/total:.1f}%")
        except Exception as e:
            self.get_logger().error(f"Save failed: {e}")
    
    def compute_centerline(self):
        """Compute racing line from map"""
        try:
            from scipy.ndimage import distance_transform_edt, binary_opening, binary_closing
            from scipy.ndimage import gaussian_filter
            
            self.get_logger().info("Computing racing line from map...")
            
            # Heavy cleaning - remove ALL small artifacts
            obstacle_map = (self.grid == 100).astype(bool)
            
            # Opening: removes small objects (like starting artifacts)
            kernel_large = np.ones((15, 15), dtype=bool)
            cleaned = binary_opening(obstacle_map, kernel_large)
            
            # Closing: fills small holes
            cleaned = binary_closing(cleaned, kernel_large)
            
            self.get_logger().info("Map cleaned, computing distance transform...")
            
            # Free space
            free_space = ~cleaned
            
            # Distance transform - find centerline
            dist_map = distance_transform_edt(free_space)
            
            # Find ridge of distance map (centerline)
            # Use points that are local maxima in distance
            smoothed_dist = gaussian_filter(dist_map.astype(float), sigma=2)
            
            # Get points with high distance from walls
            threshold = np.percentile(smoothed_dist[smoothed_dist > 0], 80)
            centerline_mask = smoothed_dist > threshold
            
            cy, cx = np.where(centerline_mask)
            
            if len(cx) < 20:
                self.get_logger().error(f"Only {len(cx)} centerline points!")
                return None
            
            self.get_logger().info(f"Found {len(cx)} centerline points")
            
            # Convert to world coordinates
            wx = self.grid_origin_x + cx * self.RESOLUTION
            wy = self.grid_origin_y + cy * self.RESOLUTION
            
            # Sort by angle from start to create loop
            angles = np.arctan2(wy - self.start_y, wx - self.start_x)
            sorted_idx = np.argsort(angles)
            wx = wx[sorted_idx]
            wy = wy[sorted_idx]
            
            # Smooth the path with moving average
            window = 5
            wx_smooth = np.convolve(wx, np.ones(window)/window, mode='valid')
            wy_smooth = np.convolve(wy, np.ones(window)/window, mode='valid')
            
            # Subsample to ~80 waypoints
            num_waypoints = 80
            step = max(1, len(wx_smooth) // num_waypoints)
            waypoints = [[float(wx_smooth[i]), float(wy_smooth[i])] 
                        for i in range(0, len(wx_smooth), step)]
            
            # Close the loop - add first point at end
            if len(waypoints) > 0:
                waypoints.append(waypoints[0])
            
            self.get_logger().info(f"✅ Computed {len(waypoints)} smooth waypoints")
            
            # Save to file
            map_dir = os.getcwd()
            path_file = os.path.join(map_dir, 'centerline.json')
            data = {
                'waypoints': waypoints,
                'num_waypoints': len(waypoints)
            }
            with open(path_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.get_logger().info(f"💾 Saved to: {path_file}")
            
            return waypoints
            
        except ImportError as e:
            self.get_logger().error(f"Missing library: {e}")
            self.get_logger().error("Run: pip install scipy")
            return None
        except Exception as e:
            self.get_logger().error(f"Centerline failed: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return None
    
    def publish_path_timer(self):
        """Timer callback to continuously publish path"""
        if self.planned_waypoints is None or len(self.planned_waypoints) == 0:
            return
        
        msg = Path()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        
        for wp in self.planned_waypoints:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(wp[0])
            pose.pose.position.y = float(wp[1])
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        
        self.pub_path.publish(msg)


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
        
        # Record start position after startup
        if self.start_x is None and time.time() - self.start_time > self.STARTUP_DELAY + 3.0:
            self.start_x, self.start_y = self.x, self.y
            self.get_logger().info(f"Start position: ({self.start_x:.2f}, {self.start_y:.2f})")

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
        # Keep stopped after lap 1
        if self.stopped:
            self.publish(0.0, 0.0)
            return
        
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
        
        # Update map (doesn't affect controls!)
        self.update_map(ranges, msg.angle_min, msg.angle_increment)
        
        # Check for loop closure
        if self.mapping_active and self.start_x is not None and not self.loop_closed:
            dist_from_start = np.sqrt((self.x - self.start_x)**2 + (self.y - self.start_y)**2)
            # If we've moved far and come back near start
            if elapsed > self.STARTUP_DELAY + 20.0 and dist_from_start < 0.6:
                self.loop_closed = True
                self.mapping_active = False
                self.stopped = True
                
                self.get_logger().info("🔄 LOOP CLOSED! Stopping...")
                self.publish(0.0, 0.0)
                
                self.get_logger().info("💾 Saving map...")
                self.save_map()
                
                self.get_logger().info("🧮 Computing racing line...")
                waypoints = self.compute_centerline()
                
                if waypoints:
                    self.planned_waypoints = waypoints
                    self.get_logger().info("🎯 Racing line computed!")
                
                self.get_logger().info("✅ DONE! Check RViz for green racing line")
                return
        
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
            map_str = ""
            if self.mapping_active:
                occupied = np.sum(self.grid == 100)
                skip = ""
                if self.reversing:
                    skip = " [SKIP: rev]"
                elif abs(steer) > 0.35:
                    skip = " [SKIP: turn]"
                map_str = f" [MAP: {occupied} walls{skip}]"
            self.get_logger().info(f"R:{d_right:.2f} F:{d_front:.2f} err:{error:.2f} st:{steer:.2f}{map_str}")

    def publish(self, speed, steer):
        t, s = Float32(), Float32()
        t.data, s.data = float(speed), float(steer)
        self.pub_throttle.publish(t)
        self.pub_steering.publish(s)
        self.current_steer = float(steer)  # Track for mapping


def main(args=None):
    rclpy.init(args=args)
    node = TrackMapper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
