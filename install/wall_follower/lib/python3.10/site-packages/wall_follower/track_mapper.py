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
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import time
import json
import os
import subprocess
from scipy.ndimage import distance_transform_edt, uniform_filter1d, maximum_filter, gaussian_filter
from skimage import measure
from skimage.graph import route_through_array

class TrackMapper(Node):
    def __init__(self):
        super().__init__('track_mapper')
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        
        self.pub_throttle = self.create_publisher(Float32, '/autodrive/roboracer_1/throttle_command', 10)
        self.pub_steering = self.create_publisher(Float32, '/autodrive/roboracer_1/steering_command', 10)
        self.pub_map = self.create_publisher(OccupancyGrid, '/map', 10)
        self.pub_raceline = self.create_publisher(Path, '/raceline_path', 10)
        self.pub_waypoints = self.create_publisher(MarkerArray, '/waypoint_markers', 10)
        self.pub_target = self.create_publisher(Marker, '/target_waypoint', 10)
        
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
        self.total_distance = 0.0  # Track distance to ensure full lap
        self.passed_start_once = False  # Did we complete one lap?
        self.dist_after_pass = 0.0  # Extra distance after passing start
        
        # ===== PURE PURSUIT =====
        self.raceline = None  # Will store waypoints after lap 1
        self.curvatures = None  # Curvature at each waypoint
        self.pursuing = False  # Switch from wall follow to pure pursuit
        self.last_wp = 0  # Track waypoint progression
        
        # Speed limits - AGGRESSIVE!
        self.PP_MAX_SPEED = 0.15  # 15 cm/s on straights (VERY FAST!)
        self.PP_MIN_SPEED = 0.05  # 5 cm/s in tight turns (still moving!)
        
        # Steering damping
        self.last_steer = 0.0
        
        # CRASH FALLBACK: If we crash, use wall follower forever
        self.pp_crashed = False
        
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
        self.create_timer(0.5, self.publish_raceline_path)  # Publish raceline for RViz
        
        self.launch_rviz()
        
        self.get_logger().info("="*50)
        self.get_logger().info("TRACK MAPPER - IMPROVED MAPPING")
        self.get_logger().info("Maps ONLY when driving straight at good speed")
        self.get_logger().info("="*50)

    def launch_rviz(self):
        """Launch RViz to visualize the map and raceline"""
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
      Name: Racing Line
      Topic:
        Value: /raceline_path
      Color: 0; 255; 0
      Line Width: 0.08
      Alpha: 1.0
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

    def publish_raceline_path(self):
        """Publish raceline as Path for RViz visualization"""
        if self.raceline is None or len(self.raceline) == 0:
            return
        
        msg = Path()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        
        for wp in self.raceline:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(wp[0])
            pose.pose.position.y = float(wp[1])
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        
        self.pub_raceline.publish(msg)
    
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
        """Simple 3-step cleanup: Melt → Fill → Re-Wall (matches manual_cleanup.py)"""
        try:
            from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes, binary_erosion, label
            
            self.get_logger().info("Cleaning: Melt → Fill → Re-Wall...")
            
            # STEP 1: MELT small black noise (keep only BIG walls)
            walls = (self.grid == 100)
            labeled_walls, num_walls = label(walls)
            
            min_wall_size = 100  # Keep walls bigger than 100 pixels
            for i in range(1, num_walls + 1):
                if np.sum(labeled_walls == i) < min_wall_size:
                    self.grid[labeled_walls == i] = 0
            
            removed = num_walls - np.sum([np.sum(labeled_walls == i) >= min_wall_size for i in range(1, num_walls + 1)])
            self.get_logger().info(f"✓ Melted {removed} small black spots")
            
            # STEP 2: FILL white areas (expand + fill holes)
            free = (self.grid == 0)
            
            # Expand white to fill gaps
            free_expanded = binary_dilation(free, structure=np.ones((15, 15)))
            self.get_logger().info("✓ Expanded white")
            
            # Keep only largest white blob (main track)
            labeled_free, num_free = label(free_expanded)
            if num_free > 0:
                sizes = np.bincount(labeled_free.ravel())
                sizes[0] = 0
                largest = sizes.argmax()
                free_main = (labeled_free == largest)
                self.get_logger().info(f"✓ Kept largest track (removed {num_free-1} blobs)")
            else:
                free_main = free_expanded
            
            # Fill all holes inside track
            free_filled = binary_fill_holes(free_main)
            self.get_logger().info("✓ Filled holes")
            
            # STEP 3: RE-ADD big black walls
            # Find original BIG walls (not small noise)
            big_walls = np.zeros_like(self.grid, dtype=bool)
            for i in range(1, num_walls + 1):
                if np.sum(labeled_walls == i) >= min_wall_size:
                    big_walls |= (labeled_walls == i)
            
            # Make walls thick and solid
            walls_thick = binary_closing(big_walls, structure=np.ones((7, 7)))
            walls_thick = binary_dilation(walls_thick, structure=np.ones((3, 3)))
            self.get_logger().info("✓ Thickened walls")
            
            # Add perimeter around white
            free_inner = binary_erosion(free_filled, structure=np.ones((2, 2)))
            perimeter = free_filled & ~free_inner
            walls_final = walls_thick | perimeter
            self.get_logger().info("✓ Added perimeter")
            
            # Final track = white minus walls
            track_final = free_filled & ~walls_final
            
            # Build final grid
            self.grid[:] = -1  # Unknown
            self.grid[track_final] = 0  # Free (white)
            self.grid[walls_final] = 100  # Walls (black)
            
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
        """Generate racing line from cleaned map using arc-length spaced checkpoints"""
        try:
            self.get_logger().info("🏁 Generating raceline...")
            
            # Get cost map
            binary_track = (self.grid == 0)
            n_free = np.sum(binary_track)
            self.get_logger().info(f"   Free space: {n_free} cells")
            
            if n_free < 500:
                self.get_logger().error(f"❌ Too little free space ({n_free} cells)!")
                return False
            
            dist_map = distance_transform_edt(binary_track)
            max_dist = np.max(dist_map)
            cost_map = max_dist - dist_map
            cost_map[self.grid != 0] = np.inf
            
            # Find ridge/centerline checkpoints spaced by arc length
            checkpoints = self.get_arc_length_checkpoints(dist_map)
            if len(checkpoints) < 4:
                self.get_logger().error(f"❌ Need 4 checkpoints, got {len(checkpoints)}")
                return False
            
            # Route through checkpoints: TR -> BR -> BL -> TL -> TR
            ordered_checkpoints = [checkpoints[0], checkpoints[3], checkpoints[2], checkpoints[1]]
            self.get_logger().info("   Route: TR -> BR -> BL -> TL -> TR")
            
            full_path = []
            for i in range(len(ordered_checkpoints)):
                start = ordered_checkpoints[i]
                end = ordered_checkpoints[(i + 1) % len(ordered_checkpoints)]
                try:
                    indices, _ = route_through_array(cost_map, start, end, fully_connected=True, geometric=True)
                    segment = np.array(indices)
                    if i > 0:
                        full_path.append(segment[1:])
                    else:
                        full_path.append(segment)
                    self.get_logger().info(f"   ✓ Segment {i}: {len(segment)} points")
                except Exception as e:
                    self.get_logger().error(f"   ❌ Failed segment {i}: {e}")
                    return False
            
            path_pixels = np.vstack(full_path)
            self.get_logger().info(f"   Total path: {len(path_pixels)} pixels")
            
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
            self.raceline = self.resample_path(path_world, 300)  # 300 points for smooth + fast
            
            self.get_logger().info(f"✅ Raceline generated: {len(self.raceline)} waypoints")
            
            # Calculate curvatures for adaptive speed/lookahead
            self.calculate_curvatures()
            
            # Publish waypoint markers to RViz (highlight WP 480-520 in RED)
            self.publish_waypoint_markers(highlight_range=(480, 520))
            
            # Visualize and save
            self.visualize_raceline(path_pixels, checkpoints)
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Raceline generation failed: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return False
    
    def get_arc_length_checkpoints(self, dist_map):
        """Find 4 checkpoints: ONE in EACH quadrant (X/Y split)"""
        from scipy.ndimage import maximum_filter, gaussian_filter
        
        # Find centerline/ridge points
        threshold = np.percentile(dist_map[dist_map > 0], 50)
        dist_smooth = gaussian_filter(dist_map, sigma=2)
        local_max = maximum_filter(dist_smooth, size=7)
        ridge = (dist_smooth == local_max) & (dist_map > threshold)
        
        ry, rx = np.where(ridge)
        
        if len(rx) < 10:
            self.get_logger().error("❌ Too few ridge points!")
            return []
        
        self.get_logger().info(f"   Found {len(rx)} ridge points")
        
        # Get distances for all ridge points
        dists = np.array([dist_map[ry[i], rx[i]] for i in range(len(rx))])
        
        # Find center
        center_y = np.mean(ry)
        center_x = np.mean(rx)
        
        # Split into 4 quadrants - ONE checkpoint per quadrant!
        quadrants = []
        quad_names = ["TOP-RIGHT", "TOP-LEFT", "BOTTOM-LEFT", "BOTTOM-RIGHT"]
        
        for qname, (y_cond, x_cond) in zip(quad_names, 
                                            [(lambda y: y >= center_y, lambda x: x >= center_x),
                                             (lambda y: y >= center_y, lambda x: x < center_x),
                                             (lambda y: y < center_y, lambda x: x < center_x),
                                             (lambda y: y < center_y, lambda x: x >= center_x)]):
            mask = y_cond(ry) & x_cond(rx)
            if np.any(mask):
                q_idx = np.where(mask)[0]
                best = q_idx[np.argmax(dists[q_idx])]
                cp = (ry[best], rx[best])
                quadrants.append(cp)
                self.get_logger().info(f"   {qname}: ({cp[0]},{cp[1]}) w={dists[best]:.1f}")
        
        self.get_logger().info(f"   Selected {len(quadrants)} checkpoints")
        return quadrants
    
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
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
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
            self.get_logger().info("⚠️  Matplotlib not available, skipping visualization")
            self.get_logger().info(f"   (Error: {e})")
            self.get_logger().info("   Run centerline_skeleton.py separately to visualize")
    
    def calculate_curvatures(self):
        """Calculate curvature at each waypoint for adaptive control"""
        if self.raceline is None or len(self.raceline) < 10:
            return
        
        n = len(self.raceline)
        curvatures = np.zeros(n)
        
        # Use 3-point curvature formula
        step = 3  # Points to skip for 300 waypoints
        for i in range(n):
            p1 = self.raceline[(i - step) % n]
            p2 = self.raceline[i]
            p3 = self.raceline[(i + step) % n]
            
            # Vectors
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Angle change
            angle1 = np.arctan2(v1[1], v1[0])
            angle2 = np.arctan2(v2[1], v2[0])
            dangle = angle2 - angle1
            
            # Normalize
            if dangle > np.pi: dangle -= 2*np.pi
            if dangle < -np.pi: dangle += 2*np.pi
            
            # Distance
            dist = np.linalg.norm(v1) + np.linalg.norm(v2)
            
            # Curvature = angle change / distance
            if dist > 0.01:
                curvatures[i] = abs(dangle) / dist
            else:
                curvatures[i] = 0
        
        # Smooth curvatures
        self.curvatures = uniform_filter1d(curvatures, size=20, mode='wrap')  # Smooth for 300 waypoints
        
        max_k = np.max(self.curvatures)
        avg_k = np.mean(self.curvatures)
        self.get_logger().info(f"📊 Curvature calculated: max={max_k:.3f}, avg={avg_k:.3f}")
        
        # Find problematic waypoints (high curvature OR direction changes)
        problem_wps = []
        for i in range(len(self.curvatures)):
            if self.curvatures[i] > 0.25:  # High curvature
                problem_wps.append(i)
        self.get_logger().info(f"⚠️ High curvature waypoints: {len(problem_wps)}")
    
    def publish_waypoint_markers(self, highlight_range=None):
        """Publish waypoints to RViz for debugging"""
        if self.raceline is None:
            return
            
        marker_array = MarkerArray()
        
        # All waypoints as small spheres
        for i, wp in enumerate(self.raceline):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "waypoints"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            marker.pose.position.x = float(wp[0])
            marker.pose.position.y = float(wp[1])
            marker.pose.position.z = 0.05
            marker.pose.orientation.w = 1.0
            
            # Size: bigger for every 100th waypoint
            if i % 100 == 0:
                marker.scale.x = 0.15
                marker.scale.y = 0.15
                marker.scale.z = 0.15
            else:
                marker.scale.x = 0.04
                marker.scale.y = 0.04
                marker.scale.z = 0.04
            
            # Color: RED for problem region, GREEN otherwise
            if highlight_range and highlight_range[0] <= i <= highlight_range[1]:
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0
                # Make problem ones bigger
                marker.scale.x = 0.1
                marker.scale.y = 0.1
                marker.scale.z = 0.1
            elif self.curvatures is not None and self.curvatures[i] > 0.25:
                # High curvature = YELLOW
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0
                marker.scale.x = 0.08
                marker.scale.y = 0.08
                marker.scale.z = 0.08
            else:
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 0.7
            
            marker_array.markers.append(marker)
        
        # Add text labels for every 100th waypoint
        for i in range(0, len(self.raceline), 100):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "wp_labels"
            marker.id = i + 10000
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            
            marker.pose.position.x = float(self.raceline[i][0])
            marker.pose.position.y = float(self.raceline[i][1])
            marker.pose.position.z = 0.3
            marker.pose.orientation.w = 1.0
            
            marker.scale.z = 0.2
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 1.0
            marker.text = str(i)
            
            marker_array.markers.append(marker)
        
        self.pub_waypoints.publish(marker_array)
        self.get_logger().info(f"📍 Published {len(self.raceline)} waypoint markers to /waypoint_markers")
    
    def pure_pursuit(self, ranges, n, center):
        """ULTIMATE Pure Pursuit: Path + Wall Avoidance + Damping + Safety!"""
        if self.raceline is None or len(self.raceline) == 0:
            return None, None  # Signal to use wall follower
        
        # ===== 1. LIDAR: Get wall distances =====
        left_min = float(np.min(ranges[center+40:center+80]))   # Left
        right_min = float(np.min(ranges[center-80:center-40]))  # Right
        front = float(np.min(ranges[center-25:center+25]))      # Front
        
        # ===== 2. CRASH DETECTION =====
        if front < 0.15:
            self.pp_crashed = True
            self.get_logger().warn("💥 CRASH! Switching to wall follower permanently!")
            return None, None
        
        # ===== 3. FIND CLOSEST WAYPOINT =====
        pos = np.array([self.x, self.y])
        num_wps = len(self.raceline)
        
        # Find ACTUAL closest waypoint (search all)
        dists = np.linalg.norm(self.raceline - pos, axis=1)
        closest = np.argmin(dists)
        xte = dists[closest]
        
        # Update tracker
        self.last_wp = closest
        
        # ===== 4. ADAPTIVE LOOKAHEAD =====
        # Straights: far ahead (smooth), Turns: closer (precise!)
        if self.curvatures is not None:
            curv = self.curvatures[closest]
            max_curv = 0.25
            curv_factor = min(curv / max_curv, 1.0)
            # Straights: 18, Tight turns: 8
            lookahead = int(18 - 10 * curv_factor)  # 18 → 8
            lookahead = max(lookahead, 6)
        else:
            lookahead = 12
            curv = 0
        
        # MINIMUM DISTANCE: Target must be at least 0.4m away for stability
        MIN_TARGET_DIST = 0.4
        target_idx = (closest + lookahead) % num_wps
        target = self.raceline[target_idx]
        target_dist = np.linalg.norm(target - pos)
        
        # If target is too close, look further ahead
        while target_dist < MIN_TARGET_DIST and lookahead < 40:
            lookahead += 3
            target_idx = (closest + lookahead) % num_wps
            target = self.raceline[target_idx]
            target_dist = np.linalg.norm(target - pos)
        
        # Check if target is in front (angle < 90°)
        dx_check = target[0] - self.x
        dy_check = target[1] - self.y
        angle_to_target = np.arctan2(dy_check, dx_check)
        angle_diff = angle_to_target - self.yaw
        if angle_diff > np.pi: angle_diff -= 2*np.pi
        if angle_diff < -np.pi: angle_diff += 2*np.pi
        
        # If target is behind, find one in front
        if abs(angle_diff) > np.pi * 0.7:  # > 126 degrees
            for try_la in range(lookahead - 1, 3, -1):
                try_idx = (closest + try_la) % num_wps
                try_target = self.raceline[try_idx]
                dx_t = try_target[0] - self.x
                dy_t = try_target[1] - self.y
                ang_t = np.arctan2(dy_t, dx_t) - self.yaw
                if ang_t > np.pi: ang_t -= 2*np.pi
                if ang_t < -np.pi: ang_t += 2*np.pi
                if abs(ang_t) < np.pi * 0.7:
                    target_idx = try_idx
                    target = try_target
                    lookahead = try_la
                    break
        
        # ===== 5. PATH FOLLOWING STEERING =====
        dx = target[0] - self.x
        dy = target[1] - self.y
        target_angle = np.arctan2(dy, dx)
        
        err = target_angle - self.yaw
        if err > np.pi: err -= 2*np.pi
        if err < -np.pi: err += 2*np.pi
        
        # STEERING GAIN: Based on situation
        if curv > 0.15:
            gain = 1.3  # Aggressive in curves
            path_steer = err * gain
        else:
            # STRAIGHTS: Speed-aware + anti-oscillation
            # Bigger dead zone, gentler corrections
            dead_zone = np.radians(10)  # 10 degree dead zone
            
            if abs(err) < dead_zone:
                path_steer = 0.0
            else:
                # Very gentle gain on straights
                gain = 0.3
                path_steer = err * gain
                
                # ANTI-OSCILLATION: If about to change direction, reduce!
                if self.last_steer * path_steer < 0:  # Opposite signs = direction change
                    path_steer *= 0.3  # Reduce by 70%!
        
        # TIGHT CORNER LOGIC: If front wall is close, turn HARD!
        # Proportional: closer wall = sharper turn!
        if front < 1.4 and curv > 0.1:
            force_steer = 0.4 + (1.4 - front) * 0.25  # 0.4 at 1.4m, 0.75 at 0m
            if err > 0:
                path_steer = max(path_steer, force_steer)  # Force left
            else:
                path_steer = min(path_steer, -force_steer)  # Force right
        
        # SIDE WALL in curve: steer away!
        if curv > 0.1 and min(left_min, right_min) < 0.4:
            if left_min < right_min:
                path_steer = min(path_steer, -0.4)  # Force right
            else:
                path_steer = max(path_steer, 0.4)  # Force left
        
        # ===== 6. WALL AVOIDANCE STEERING =====
        wall_steer = 0.0
        wall_thresh = 0.5  # Start avoiding at 50cm
        
        if left_min < wall_thresh:
            wall_steer -= (wall_thresh - left_min) * 2.5  # Steer right
        if right_min < wall_thresh:
            wall_steer += (wall_thresh - right_min) * 2.5  # Steer left
        
        # Front wall - AGGRESSIVE turn to avoid!
        if front < 0.6:
            turn_force = (0.6 - front) * 1.5  # Stronger as wall gets closer
            if left_min < right_min:
                wall_steer -= turn_force  # Turn right (away from left wall)
            else:
                wall_steer += turn_force  # Turn left (away from right wall)
        
        # ===== 7. BLEND PATH + WALL AVOIDANCE =====
        min_side = min(left_min, right_min)
        wall_weight = 0.0
        if min_side < wall_thresh:
            wall_weight = min((wall_thresh - min_side) / wall_thresh, 0.6)
        
        raw_steer = (1.0 - wall_weight) * path_steer + wall_weight * wall_steer
        
        # ===== 8. STEERING: Full range allowed, but damped to prevent oscillation =====
        steer_change = raw_steer - self.last_steer
        
        # DAMPING: Minimal in curves, VERY HEAVY on straights
        if curv > 0.15:  # In a curve - need to turn!
            damping = 0.1
        else:  # Straight - VERY HEAVY damping to kill oscillation
            damping = 0.75
        
        damped_steer = raw_steer - steer_change * damping
        
        # FULL STEERING ALLOWED everywhere!
        steer = np.clip(damped_steer, -self.MAX_STEER, self.MAX_STEER)
        
        self.last_steer = steer
        
        # ===== 9. ADAPTIVE SPEED =====
        if self.curvatures is not None:
            # Don't look too far ahead - slow down LATE, not early!
            look_ahead_curv = max(int(lookahead), 8)
            indices = [(closest + i) % num_wps for i in range(look_ahead_curv)]
            upcoming_curv = np.max(self.curvatures[indices])
            curv_factor = min(upcoming_curv / max_curv, 1.0)
            
            # Speed: FAST always, only slow in TIGHT turns
            # Use curv_factor^2 so we only slow for really tight turns
            speed = self.PP_MAX_SPEED - (self.PP_MAX_SPEED - self.PP_MIN_SPEED) * (curv_factor ** 1.5)
        else:
            speed = 0.10
        
        # Reduce speed when steering hard (prevents slipping!)
        steer_factor = abs(steer) / self.MAX_STEER
        speed *= (1.0 - steer_factor * 0.5)  # Up to 50% reduction when max steer
        
        # ===== 10. SAFETY LIMITS (less aggressive!) =====
        if front < 0.25:
            speed = min(speed, 0.03)
        elif front < 0.4:
            speed = min(speed, 0.05)
        
        if min_side < 0.2:
            speed = min(speed, 0.04)
        
        if xte > 0.4:
            speed = min(speed, 0.05)
        if xte > 0.6:
            speed = min(speed, 0.03)
        
        # ===== DETAILED DEBUG =====
        angle_deg = np.degrees(err)
        target_angle_deg = np.degrees(target_angle)
        yaw_deg = np.degrees(self.yaw)
        
        # Print every 20 frames or if there's a problem
        is_problem = xte > 0.4 or abs(steer) > 0.65
        
        if self.count % 20 == 0 or is_problem:
            problem_flag = "⚠️" if is_problem else ""
            self.get_logger().info(
                f"🏎️ WP:{closest} k={curv:.2f} | XTE:{xte:.2f}m | Spd:{speed:.3f} Str:{steer:.2f} {problem_flag}"
            )
        
        # Publish TARGET WAYPOINT as bright purple marker
        target_marker = Marker()
        target_marker.header.frame_id = "world"
        target_marker.header.stamp = self.get_clock().now().to_msg()
        target_marker.ns = "target"
        target_marker.id = 0
        target_marker.type = Marker.SPHERE
        target_marker.action = Marker.ADD
        target_marker.pose.position.x = float(target[0])
        target_marker.pose.position.y = float(target[1])
        target_marker.pose.position.z = 0.3  # Higher so it's visible
        target_marker.pose.orientation.w = 1.0
        target_marker.scale.x = 0.25
        target_marker.scale.y = 0.25
        target_marker.scale.z = 0.25
        target_marker.color.r = 1.0
        target_marker.color.g = 0.0
        target_marker.color.b = 1.0  # PURPLE (magenta)
        target_marker.color.a = 1.0
        self.pub_target.publish(target_marker)
        
        if self.count % 15 == 0:
            curv_str = f"k={curv:.2f}" if self.curvatures is not None else "-"
            self.get_logger().info(
                f"🏎️ WP:{closest} Tgt:{target_idx} LA:{lookahead} XTE:{xte:.2f} Ang:{angle_deg:+.0f}° {curv_str} | "
                f"L:{left_min:.2f} R:{right_min:.2f} F:{front:.2f} | "
                f"Spd:{speed:.3f} Str:{steer:.2f}"
            )
        
        
        return speed, steer
    
    def check_stuck(self):
        # Don't check stuck during pure pursuit!
        if self.pursuing or self.reversing:
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
        dist_moved = np.sqrt(dx*dx + dy*dy)
        
        if dist_moved > 0.005:  # Only update yaw if actually moving (was 0.001)
            new_yaw = np.arctan2(dy, dx)
            
            # LIMIT YAW CHANGE RATE - max 10° per frame to prevent spinning detection
            yaw_diff = new_yaw - self.yaw
            if yaw_diff > np.pi: yaw_diff -= 2*np.pi
            if yaw_diff < -np.pi: yaw_diff += 2*np.pi
            
            max_yaw_change = np.radians(15)  # Max 15° per frame
            yaw_diff = np.clip(yaw_diff, -max_yaw_change, max_yaw_change)
            
            self.yaw = self.yaw + yaw_diff * 0.3  # Slower smoothing (was 0.15 direct)
            
            # Keep yaw in [-pi, pi]
            if self.yaw > np.pi: self.yaw -= 2*np.pi
            if self.yaw < -np.pi: self.yaw += 2*np.pi
            
            self.heading_history.append(self.yaw)
            if len(self.heading_history) > 40:
                self.heading_history.pop(0)
            
            # Track total distance traveled (after start recorded)
            if self.start_x is not None:
                self.total_distance += dist_moved
        
        self.prev_x, self.prev_y = self.x, self.y
        self.x, self.y = msg.x, msg.y
        
        # Record start position after startup
        if self.start_x is None and time.time() - self.start_time > self.STARTUP_DELAY + 3.0:
            self.start_x, self.start_y = self.x, self.y
            self.total_distance = 0.0
            self.get_logger().info(f"📍 Start: ({self.start_x:.2f}, {self.start_y:.2f})")

    def lap_cb(self, msg):
        if msg.data > self.lap:
            self.lap = msg.data
            self.get_logger().info(f"*** LAP {self.lap} COMPLETE! ***")
            
            # Lap 1 done - DON'T stop yet! Drive 5m more to fill gaps!
            if self.lap == 1 and not self.passed_start_once:
                self.passed_start_once = True
                self.dist_after_pass = self.total_distance
                self.get_logger().info(f"📍 LAP 1 @ {self.total_distance:.1f}m! Driving 5m more to fill gaps...")
            
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
        
        # PURE PURSUIT with CRASH FALLBACK
        if self.pursuing and not self.pp_crashed:
            result = self.pure_pursuit(ranges, n, center)
            if result[0] is not None:
                speed, steer = result
                self.publish(speed, steer)
                return
            # If None returned, fall through to wall follower
        
        # If crashed or None, use wall follower (it never crashes!)
        if self.pursuing and self.pp_crashed:
            if self.count % 50 == 0:
                self.get_logger().info("🛡️ Using wall follower (crash fallback)")
        
        # Update map (doesn't affect controls!)
        self.update_map(ranges, msg.angle_min, msg.angle_increment)
        
        # Check for loop closure - drive a bit PAST start before stopping!
        if self.mapping_active and self.start_x is not None and not self.loop_closed:
            dist_from_start = np.sqrt((self.x - self.start_x)**2 + (self.y - self.start_y)**2)
            
            # First: detect when we complete a lap (went far, came back close)
            if not self.passed_start_once:
                if elapsed > self.STARTUP_DELAY + 25.0 and self.total_distance > 15.0 and dist_from_start < 0.8:
                    self.passed_start_once = True
                    self.dist_after_pass = self.total_distance  # Record distance at this moment
                    self.get_logger().info(f"📍 Lap complete @ {self.total_distance:.1f}m! Driving 5m more...")
            
            # Close loop after driving 5m past the start point
            if self.passed_start_once and (self.total_distance - self.dist_after_pass) > 5.0:
                self.loop_closed = True
                self.mapping_active = False
                self.stopped = True
                
                self.get_logger().info(f"🔄 LOOP CLOSED! Time:{elapsed:.1f}s Dist:{self.total_distance:.1f}m FromStart:{dist_from_start:.2f}m")
                self.get_logger().info("🛑 Stopping...")
                self.publish(0.0, 0.0)
                
                self.get_logger().info("🧹 Cleaning up map...")
                self.cleanup_map()
                
                self.get_logger().info("💾 Saving map...")
                self.save_map()
                
                self.get_logger().info("🏁 Generating raceline...")
                if self.generate_raceline():
                    self.pursuing = True
                    self.stopped = False
                    self.get_logger().info("✅ ADAPTIVE Pure Pursuit activated! 🏎️")
                    self.get_logger().info(f"   Current car pos: ({self.x:.3f}, {self.y:.3f})")
                    # Find closest waypoint to current position
                    dists = np.linalg.norm(self.raceline - np.array([self.x, self.y]), axis=1)
                    closest = np.argmin(dists)
                    self.get_logger().info(f"   Closest WP: {closest} @({self.raceline[closest][0]:.3f}, {self.raceline[closest][1]:.3f}) Dist:{dists[closest]:.3f}m")
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
