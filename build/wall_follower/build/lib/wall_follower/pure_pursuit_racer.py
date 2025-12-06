#!/usr/bin/env python3
"""
Pure Pursuit Racer - Follows pre-computed racing line from map
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32, Int32
from geometry_msgs.msg import Point
import numpy as np
import json
import os

# Import the centerline generation functions
import sys
from scipy.ndimage import distance_transform_edt, uniform_filter1d
from skimage import measure
from skimage.graph import route_through_array

class PurePursuitRacer(Node):
    def __init__(self):
        super().__init__('pure_pursuit_racer')
        
        # QoS
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        
        # Publishers
        self.pub_throttle = self.create_publisher(Float32, '/autodrive/roboracer_1/throttle_command', 10)
        self.pub_steering = self.create_publisher(Float32, '/autodrive/roboracer_1/steering_command', 10)
        
        # Subscribers
        self.create_subscription(Point, '/autodrive/roboracer_1/ips', self.ips_cb, qos)
        self.create_subscription(Int32, '/autodrive/roboracer_1/lap_count', self.lap_cb, 10)
        
        # Vehicle state
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.current_lap = 0
        
        # Pure pursuit params
        self.LOOKAHEAD_DIST = 0.5  # meters
        self.BASE_SPEED = 2.0
        self.MAX_SPEED = 3.0
        self.Kp_steering = 1.5
        
        # Waypoints
        self.waypoints = None
        self.current_idx = 0
        
        # Load map and generate raceline
        self.get_logger().info("🏁 Loading map and generating racing line...")
        self.load_and_generate_raceline()
        
        if self.waypoints is None:
            self.get_logger().error("❌ Failed to generate raceline!")
            return
        
        self.get_logger().info(f"✅ Racing line ready: {len(self.waypoints)} waypoints")
        self.get_logger().info("🚗 Pure Pursuit Racer started!")
    
    def load_and_generate_raceline(self):
        """Load map, clean it, and generate optimal racing line"""
        try:
            # Load map
            map_dir = os.getcwd()
            
            if os.path.exists(os.path.join(map_dir, 'track_map_cleaned.npy')):
                grid = np.load(os.path.join(map_dir, 'track_map_cleaned.npy'))
                with open(os.path.join(map_dir, 'track_map_cleaned_meta.json'), 'r') as f:
                    meta = json.load(f)
            else:
                grid = np.load(os.path.join(map_dir, 'track_map.npy'))
                with open(os.path.join(map_dir, 'track_map_meta.json'), 'r') as f:
                    meta = json.load(f)
            
            # Clean map
            clean_grid = self.clean_map(grid)
            
            # Generate raceline
            self.waypoints = self.generate_raceline(clean_grid, meta)
            
        except Exception as e:
            self.get_logger().error(f"Failed to load/generate raceline: {e}")
            self.waypoints = None
    
    def clean_map(self, grid):
        """Remove artifacts, keep only main track"""
        binary_track = (grid == 0).astype(int)
        labeled_array, num_features = measure.label(binary_track, return_num=True, connectivity=2)
        
        if num_features < 2:
            return grid
        
        # Find largest component
        max_label = 0
        max_size = 0
        for i in range(1, num_features + 1):
            size = np.sum(labeled_array == i)
            if size > max_size:
                max_size = size
                max_label = i
        
        new_grid = np.ones_like(grid) * 100
        new_grid[labeled_array == max_label] = 0
        return new_grid
    
    def generate_raceline(self, grid, meta):
        """Generate optimal racing line using quadrant checkpoints"""
        
        # Create cost map
        binary_track = (grid == 0)
        dist_map = distance_transform_edt(binary_track)
        max_dist = np.max(dist_map)
        cost_map = max_dist - dist_map
        cost_map[grid != 0] = np.inf
        
        # Get 4 quadrant checkpoints
        checkpoints = self.get_quadrant_checkpoints(dist_map)
        
        # Route between checkpoints
        path_pixels = self.route_segments(cost_map, checkpoints)
        
        if path_pixels is None:
            return None
        
        # Convert to world coordinates
        res = meta['resolution']
        ox = meta['origin_x']
        oy = meta['origin_y']
        
        py = path_pixels[:, 0]
        px = path_pixels[:, 1]
        wx = (px * res) + ox
        wy = (py * res) + oy
        
        # Smooth
        window = 20
        wx = uniform_filter1d(wx, size=window, mode='wrap')
        wy = uniform_filter1d(wy, size=window, mode='wrap')
        
        # Resample to 200 points
        waypoints = self.resample_path(np.column_stack((wx, wy)), num_points=200)
        
        return waypoints
    
    def get_quadrant_checkpoints(self, dist_map):
        """Find best point in each quadrant"""
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
            if np.sum(valid_mask) == 0:
                continue
            sector_dist = dist_map.copy()
            sector_dist[~valid_mask] = -1
            best_idx = np.argmax(sector_dist)
            checkpoints.append(np.unravel_index(best_idx, dist_map.shape))
        
        return checkpoints
    
    def route_segments(self, cost_map, checkpoints):
        """Route between checkpoints to form closed loop"""
        full_path = []
        num_points = len(checkpoints)
        
        for i in range(num_points):
            start = checkpoints[i]
            end = checkpoints[(i + 1) % num_points]
            try:
                indices, weight = route_through_array(cost_map, start, end, fully_connected=True, geometric=True)
                segment = np.array(indices)
                if i > 0:
                    full_path.append(segment[1:])
                else:
                    full_path.append(segment)
            except:
                return None
        
        return np.vstack(full_path)
    
    def resample_path(self, path, num_points=200):
        """Resample path to have exactly num_points evenly spaced"""
        dists = np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))
        cumulative_dist = np.r_[0, np.cumsum(dists)]
        total_dist = cumulative_dist[-1]
        
        new_dists = np.linspace(0, total_dist, num_points)
        
        x_new = np.interp(new_dists, cumulative_dist, path[:,0])
        y_new = np.interp(new_dists, cumulative_dist, path[:,1])
        
        return np.column_stack((x_new, y_new))
    
    def find_closest_waypoint(self):
        """Find closest waypoint to current position"""
        if self.waypoints is None:
            return 0
        
        distances = np.sqrt((self.waypoints[:, 0] - self.x)**2 + 
                           (self.waypoints[:, 1] - self.y)**2)
        return np.argmin(distances)
    
    def pure_pursuit(self):
        """Pure pursuit control"""
        if self.waypoints is None:
            return 0.0, 0.0
        
        # Find closest waypoint
        closest_idx = self.find_closest_waypoint()
        
        # Look ahead for target point
        lookahead_points = int(self.LOOKAHEAD_DIST / 0.1)  # Assuming ~0.1m between points
        target_idx = (closest_idx + lookahead_points) % len(self.waypoints)
        
        # Target waypoint
        target_x = self.waypoints[target_idx, 0]
        target_y = self.waypoints[target_idx, 1]
        
        # Compute heading error
        dx = target_x - self.x
        dy = target_y - self.y
        target_heading = np.arctan2(dy, dx)
        
        # Heading error (normalized to [-pi, pi])
        heading_error = target_heading - self.yaw
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        
        # Steering (proportional to heading error)
        steering = self.Kp_steering * heading_error
        steering = np.clip(steering, -1.0, 1.0)
        
        # Speed (slow down on sharp turns)
        speed = self.BASE_SPEED if abs(steering) < 0.3 else self.BASE_SPEED * 0.7
        speed = np.clip(speed, 0.5, self.MAX_SPEED)
        
        return speed, steering
    
    def ips_cb(self, msg):
        """IPS callback - updates position and yaw"""
        self.x = msg.x
        self.y = msg.y
        self.yaw = msg.z  # IPS uses z for yaw
        
        # Compute control
        speed, steering = self.pure_pursuit()
        
        # Publish
        self.publish(speed, steering)
    
    def lap_cb(self, msg):
        """Lap counter callback"""
        self.current_lap = msg.data
        
        if self.current_lap >= 10:
            self.get_logger().info("🏁 Completed 10 laps! Stopping...")
            self.publish(0.0, 0.0)
    
    def publish(self, speed, steering):
        """Publish speed and steering commands"""
        throttle_msg = Float32()
        throttle_msg.data = float(speed)
        self.pub_throttle.publish(throttle_msg)
        
        steer_msg = Float32()
        steer_msg.data = float(steering)
        self.pub_steering.publish(steer_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitRacer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

