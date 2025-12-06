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
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
import numpy as np
import time
import json
import os
import subprocess
from scipy.ndimage import distance_transform_edt, uniform_filter1d
from skimage import measure
from skimage.graph import route_through_array

# Try to import matplotlib, but don't fail if it's broken
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except:
    MATPLOTLIB_AVAILABLE = False

class TrackMapper(Node):
    def __init__(self):
        super().__init__('track_mapper')
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        
        self.pub_throttle = self.create_publisher(Float32, '/autodrive/roboracer_1/throttle_command', 10)
        self.pub_steering = self.create_publisher(Float32, '/autodrive/roboracer_1/steering_command', 10)
        self.pub_map = self.create_publisher(OccupancyGrid, '/map', 10)
        
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
        self.start_x, self.start_y = None, None  # Track start position for loop closure
        self.loop_closed = False
        
        # ===== PURE PURSUIT =====
        self.raceline = None  # Will store waypoints after lap 1
        self.pursuing = False  # Switch from wall follow to pure pursuit
        self.LOOKAHEAD_DIST = 0.5  # Look 0.5m ahead
        self.RESOLUTION = 0.05
        self.MAP_SIZE = 30.0
        self.grid_size = int(self.MAP_SIZE / self.RESOLUTION)
        self.grid_origin_x = -self.MAP_SIZE / 2
        self.grid_origin_y = -self.MAP_SIZE / 2
        self.grid = np.full((self.grid_size, self.grid_size), -1, dtype=np.int8)
        self.hit_count = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.miss_count = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.HIT_THRESHOLD = 6  # Need 6 hits - very confident!
        self.MISS_THRESHOLD = 10  # Need 10 misses - very confident!
        self.current_speed = 0.0
        
        self.create_timer(0.3, self.check_stuck)
        self.create_timer(1.0, self.publish_map)
        
        self.launch_rviz()
        
        self.get_logger().info("="*50)
        self.get_logger().info("TRACK MAPPER - IMPROVED MAPPING")
        self.get_logger().info("Maps ONLY when driving straight at good speed")
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
        """Update map - ONLY when driving straight!"""
        if not self.mapping_active:
            return
        
        # Skip during startup to avoid weird cluster
        if time.time() - self.start_time < self.STARTUP_DELAY + 2.0:
            return
        
        # ONLY map when going relatively straight and at good speed
        if self.reversing:
            return
        
        # Lower threshold - skip even moderate turns
        if abs(self.current_steer) > 0.25:
            return
        
        # Skip if going too slow (might be stuck/maneuvering)
        if self.current_speed < 0.025:
            return
            
        robot_gx, robot_gy = self.world_to_grid(self.x, self.y)
        
        if not (0 <= robot_gx < self.grid_size and 0 <= robot_gy < self.grid_size):
            return
        
        # Every 5th ray - more selective, less noise
        for i in range(0, len(ranges), 5):
            r = ranges[i]
            # Stricter filtering
            if r < 0.25 or r > 7.5 or not np.isfinite(r):
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
            
            # Mark obstacle - THICKER walls!
            if 0 <= hit_gx < self.grid_size and 0 <= hit_gy < self.grid_size:
                self.hit_count[hit_gy, hit_gx] += 1
                if self.hit_count[hit_gy, hit_gx] >= self.HIT_THRESHOLD:
                    # Mark the cell and neighbors to make wall thicker
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            ny, nx = hit_gy + dy, hit_gx + dx
                            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                                self.grid[ny, nx] = 100

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
    
    def cleanup_map(self):
        """Post-process map: fill wall gaps, thicken walls, fill holes - NO matplotlib!"""
        try:
            from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes, label
            
            self.get_logger().info("Cleaning up map...")
            
            # Step 1: Fill gaps in walls (make walls solid)
            obstacle_map = (self.grid == 100)
            kernel = np.ones((9, 9), dtype=bool)
            obstacles_filled = binary_closing(obstacle_map, kernel)
            # Make walls thicker
            obstacles_thick = binary_dilation(obstacles_filled, np.ones((4, 4)))
            self.get_logger().info("✓ Filled wall gaps and made thick")
            
            # Step 2: Expand free space into nearby unknown
            free_space = (self.grid == 0)
            free_expanded = binary_dilation(free_space, np.ones((11, 11)))
            free_expanded = free_expanded & ~obstacles_thick
            self.get_logger().info("✓ Expanded free space")
            
            # Step 3: Keep only LARGEST connected component
            labeled, num_features = label(free_expanded)
            if num_features > 0:
                sizes = np.bincount(labeled.ravel())
                sizes[0] = 0
                largest_component = sizes.argmax()
                free_expanded = (labeled == largest_component)
                self.get_logger().info(f"✓ Kept largest track region (removed {num_features-1} isolated areas)")
            
            # Step 4: Fill ALL holes inside free space
            free_filled = binary_fill_holes(free_expanded)
            self.get_logger().info("✓ Filled ALL holes surrounded by white")
            
            # Step 5: Remove overlap with walls
            free_final = free_filled & ~obstacles_thick
            
            # Step 6: Add THIN perimeter around all white areas
            from scipy.ndimage import binary_erosion
            free_inner = binary_erosion(free_final, np.ones((2, 2)))
            perimeter = free_final & ~free_inner
            
            # Combine walls + perimeter
            walls_final = obstacles_thick | perimeter
            self.get_logger().info("✓ Added thin perimeter walls")
            
            # Final free space (away from walls)
            free_final = free_final & ~walls_final
            
            # Apply to grid
            self.grid[:] = -1  # Reset to unknown
            self.grid[free_final] = 0  # Free space
            self.grid[walls_final] = 100  # Walls
            
            free_count = np.sum(self.grid == 0)
            wall_count = np.sum(self.grid == 100)
            
            self.get_logger().info(f"✅ Cleanup done! Free: {free_count}, Walls: {wall_count}")
            
        except Exception as e:
            self.get_logger().error(f"Cleanup failed: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
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
    
    def generate_raceline(self):
        """Generate racing line from cleaned map using quadrant checkpoints"""
        try:
            self.get_logger().info("🏁 Generating raceline...")
            
            # Get cost map and checkpoints
            binary_track = (self.grid == 0)
            dist_map = distance_transform_edt(binary_track)
            
            max_dist = np.max(dist_map)
            cost_map = max_dist - dist_map
            cost_map[self.grid != 0] = np.inf
            
            # Find 4 quadrant checkpoints
            valid_y, valid_x = np.where(dist_map > 0)
            center_y = np.mean(valid_y)
            center_x = np.mean(valid_x)
            
            h, w = dist_map.shape
            y_grid, x_grid = np.ogrid[:h, :w]
            angles = np.arctan2(y_grid - center_y, x_grid - center_x)
            
            sectors = [(-np.pi, -np.pi/2), (-np.pi/2, 0), (0, np.pi/2), (np.pi/2, np.pi)]
            checkpoints = []
            
            for start_ang, end_ang in sectors:
                angle_mask = (angles >= start_ang) & (angles < end_ang)
                valid_mask = angle_mask & (dist_map > 0)
                if np.sum(valid_mask) == 0: continue
                sector_dist = dist_map.copy()
                sector_dist[~valid_mask] = -1
                best_idx = np.argmax(sector_dist)
                checkpoints.append(np.unravel_index(best_idx, dist_map.shape))
            
            # Route through checkpoints
            full_path = []
            num_points = len(checkpoints)
            for i in range(num_points):
                start = checkpoints[i]
                end = checkpoints[(i + 1) % num_points]
                indices, _ = route_through_array(cost_map, start, end, fully_connected=True, geometric=True)
                segment = np.array(indices)
                if i > 0:
                    full_path.append(segment[1:])
                else:
                    full_path.append(segment)
            
            path_pixels = np.vstack(full_path)
            
            # Convert to world coordinates
            py = path_pixels[:, 0]
            px = path_pixels[:, 1]
            wx = (px * self.RESOLUTION) + self.grid_origin_x
            wy = (py * self.RESOLUTION) + self.grid_origin_y
            
            # Smooth
            window = 20
            wx = uniform_filter1d(wx, size=window, mode='wrap')
            wy = uniform_filter1d(wy, size=window, mode='wrap')
            
            # Resample to 200 points
            path_world = np.column_stack((wx, wy))
            self.raceline = self.resample_path(path_world, 200)
            
            self.get_logger().info(f"✅ Raceline generated: {len(self.raceline)} waypoints")
            
            # Visualize and save
            self.visualize_raceline(path_pixels, checkpoints)
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Raceline generation failed: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return False
    
    def resample_path(self, path, num_points=200):
        """Interpolate path to have exactly num_points evenly spaced"""
        dists = np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))
        cumulative_dist = np.r_[0, np.cumsum(dists)]
        total_dist = cumulative_dist[-1]
        
        new_dists = np.linspace(0, total_dist, num_points)
        
        x_new = np.interp(new_dists, cumulative_dist, path[:,0])
        y_new = np.interp(new_dists, cumulative_dist, path[:,1])
        
        return np.column_stack((x_new, y_new))
    
    def visualize_raceline(self, path_pixels, checkpoints):
        """Visualize and save raceline on map"""
        if not MATPLOTLIB_AVAILABLE:
            self.get_logger().info("⚠️  Matplotlib not available, skipping visualization")
            self.get_logger().info("   Run centerline_skeleton.py separately to visualize")
            return
        
        try:
            self.get_logger().info("📊 Saving raceline visualization...")
            
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            
            # Show map
            ax.imshow(self.grid, cmap='gray', origin='lower')
            
            # Plot path
            py = path_pixels[:, 0]
            px = path_pixels[:, 1]
            ax.plot(px, py, 'g-', linewidth=3, label='Racing Line', alpha=0.8)
            
            # Plot checkpoints
            for i, (cy, cx) in enumerate(checkpoints):
                colors = ['red', 'cyan', 'yellow', 'orange']
                ax.scatter(cx, cy, c=colors[i % 4], s=200, marker='*', 
                          edgecolors='black', linewidths=2, zorder=10,
                          label=f'Checkpoint {i+1}')
            
            # Plot start point
            ax.scatter(px[0], py[0], c='lime', s=300, marker='o', 
                      edgecolors='darkgreen', linewidths=3, zorder=11,
                      label='Start')
            
            ax.set_title("Generated Racing Line", fontsize=16, fontweight='bold')
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Save
            save_path = os.path.join(os.getcwd(), 'raceline_map.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.get_logger().info(f"💾 Saved visualization: {save_path}")
            
            # Open automatically
            try:
                subprocess.Popen(['xdg-open', save_path])
                self.get_logger().info("🖼️  Opening visualization automatically...")
            except:
                self.get_logger().info("   Open this file to see your racing line!")
            
        except Exception as e:
            self.get_logger().warn(f"Visualization failed: {e}")
    
    def pure_pursuit(self, ranges, n, center):
        """Pure pursuit controller - follow the raceline"""
        if self.raceline is None or len(self.raceline) == 0:
            self.get_logger().warn("No raceline!")
            return 0.0, 0.0
        
        # Find closest point on raceline
        current_pos = np.array([self.x, self.y])
        distances = np.linalg.norm(self.raceline - current_pos, axis=1)
        closest_idx = np.argmin(distances)
        
        # Look ahead
        lookahead_idx = (closest_idx + 10) % len(self.raceline)  # Look 10 points ahead
        target = self.raceline[lookahead_idx]
        
        # Calculate steering angle
        dx = target[0] - self.x
        dy = target[1] - self.y
        target_angle = np.arctan2(dy, dx)
        
        angle_diff = target_angle - self.yaw
        # Normalize to [-pi, pi]
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        
        steer = np.clip(angle_diff * 2.0, -self.MAX_STEER, self.MAX_STEER)
        
        # Speed based on steering
        speed = self.SPEED * (1.0 - abs(steer) / self.MAX_STEER * 0.5)
        
        return speed, steer
    
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
        
        # PURE PURSUIT MODE - Follow raceline!
        if self.pursuing:
            speed, steer = self.pure_pursuit(ranges, n, center)
            self.publish(speed, steer)
            return
        
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
                
                self.get_logger().info("🧹 Cleaning up map...")
                self.cleanup_map()
                
                self.get_logger().info("💾 Saving map...")
                self.save_map()
                
                self.get_logger().info("🏁 Generating raceline...")
                if self.generate_raceline():
                    self.pursuing = True
                    self.stopped = False
                    self.get_logger().info("✅ Switching to PURE PURSUIT mode!")
                else:
                    self.get_logger().error("❌ Failed to generate raceline, staying stopped")
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
                free_count = np.sum(self.grid == 0)
                skip = ""
                if self.reversing:
                    skip = " [SKIP: rev]"
                elif abs(steer) > 0.25:
                    skip = " [SKIP: turn]"
                elif speed < 0.025:
                    skip = " [SKIP: slow]"
                else:
                    skip = " [MAPPING]"
                map_str = f" [W:{occupied} F:{free_count}{skip}]"
            self.get_logger().info(f"R:{d_right:.2f} F:{d_front:.2f} err:{error:.2f} st:{steer:.2f}{map_str}")

    def publish(self, speed, steer):
        t, s = Float32(), Float32()
        t.data, s.data = float(speed), float(steer)
        self.pub_throttle.publish(t)
        self.pub_steering.publish(s)
        self.current_steer = float(steer)  # Track for mapping
        self.current_speed = float(speed)  # Track for mapping


def main(args=None):
    rclpy.init(args=args)
    node = TrackMapper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
