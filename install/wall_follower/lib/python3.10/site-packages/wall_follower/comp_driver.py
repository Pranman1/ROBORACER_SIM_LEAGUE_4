"""
Competition Driver - All-in-One Node for Docker Submission

Combines Local Planner + Global Planner + MPC into a single node.
This is the recommended driver for competition submission.

Phase 1 (Lap 1): Safe gap-following while building occupancy map
Phase 2 (Laps 2-10): MPC follows optimized racing line

Usage: ros2 run wall_follower comp_driver
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan, Imu
from geometry_msgs.msg import Point
from std_msgs.msg import Float32
import numpy as np
import math
from collections import deque
from scipy import ndimage
from scipy.interpolate import splprep, splev

# Try CasADi, fall back to pure pursuit if not available
try:
    import casadi as ca
    CASADI_AVAILABLE = True
except ImportError:
    CASADI_AVAILABLE = False


class CompDriver(Node):
    """All-in-one competition driver with MPC"""
    
    def __init__(self):
        super().__init__('comp_driver')
        
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # ============ VEHICLE PARAMETERS ============
        self.L = 0.3240  # Wheelbase [m]
        self.MAX_STEER = 0.5236  # Max steering [rad]
        self.MAX_STEER_RATE = 3.2  # Max steering rate [rad/s]
        
        # ============ TUNABLE PARAMETERS ============
        # Mapping phase (conservative)
        self.MAPPING_SPEED = 0.28
        self.CORNER_SPEED = 0.18
        
        # Racing phase (aggressive)
        self.MAX_RACING_SPEED = 4.0  # Start conservative! Increase after testing
        self.MIN_RACING_SPEED = 1.5
        
        # MPC parameters
        self.N = 15  # Prediction horizon
        self.dt = 0.05  # Time step
        
        # MPC weights
        self.Q_pos = 12.0
        self.Q_heading = 6.0
        self.Q_vel = 3.0
        self.R_steer = 40.0
        self.R_accel = 8.0
        self.R_steer_rate = 80.0
        
        # ============ MAPPING PARAMETERS ============
        self.MAP_RESOLUTION = 0.05
        self.MAP_SIZE = 600
        self.MAP_ORIGIN = -15.0
        
        # ============ STATE ============
        self.phase = "MAPPING"  # MAPPING -> PLANNING -> RACING
        self.position = np.array([0.0, 0.0])
        self.yaw = 0.0
        self.velocity = 0.0
        self.prev_steer = 0.0
        self.prev_position = None
        
        # Lap detection
        self.start_position = None
        self.has_moved_away = False
        self.trajectory = deque(maxlen=10000)
        
        # Mapping
        self.occupancy_grid = np.zeros((self.MAP_SIZE, self.MAP_SIZE), dtype=np.int8)
        self.hit_counts = np.zeros((self.MAP_SIZE, self.MAP_SIZE), dtype=np.int32)
        self.miss_counts = np.zeros((self.MAP_SIZE, self.MAP_SIZE), dtype=np.int32)
        
        # Racing line
        self.racing_line = None
        self.racing_velocities = None
        self.racing_headings = None
        
        # MPC solver
        self.solver = None
        self.mpc_ready = False
        
        # ============ PUBLISHERS ============
        self.pub_throttle = self.create_publisher(
            Float32, '/autodrive/roboracer_1/throttle_command', 10)
        self.pub_steering = self.create_publisher(
            Float32, '/autodrive/roboracer_1/steering_command', 10)
        
        # ============ SUBSCRIBERS ============
        self.sub_lidar = self.create_subscription(
            LaserScan, '/autodrive/roboracer_1/lidar', self.lidar_callback, qos)
        self.sub_ips = self.create_subscription(
            Point, '/autodrive/roboracer_1/ips', self.ips_callback, qos)
        self.sub_imu = self.create_subscription(
            Imu, '/autodrive/roboracer_1/imu', self.imu_callback, qos)
        
        # Initialize MPC
        if CASADI_AVAILABLE:
            self.setup_mpc()
            self.get_logger().info("CasADi MPC initialized")
        else:
            self.get_logger().warn("CasADi not available - using Pure Pursuit")
        
        self.get_logger().info("=" * 50)
        self.get_logger().info("COMPETITION DRIVER STARTED")
        self.get_logger().info("Phase: MAPPING (Lap 1)")
        self.get_logger().info("=" * 50)

    # =========================================================
    # SENSOR CALLBACKS
    # =========================================================
    
    def ips_callback(self, msg):
        """Update position from IPS"""
        self.position = np.array([msg.x, msg.y])
        
        # Initialize start position
        if self.start_position is None:
            self.start_position = self.position.copy()
            self.get_logger().info(f"Start: ({msg.x:.2f}, {msg.y:.2f})")
        
        self.trajectory.append(self.position.copy())
        
        # Velocity estimation
        if self.prev_position is not None:
            dx = self.position[0] - self.prev_position[0]
            dy = self.position[1] - self.prev_position[1]
            self.velocity = 0.7 * self.velocity + 0.3 * (np.sqrt(dx**2 + dy**2) / 0.05)
        self.prev_position = self.position.copy()
        
        # Lap detection (mapping phase only)
        if self.phase == "MAPPING":
            self.check_lap_complete()

    def imu_callback(self, msg):
        """Update yaw from IMU"""
        q = msg.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    def lidar_callback(self, msg):
        """Main control loop - triggered by LIDAR"""
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.where(np.isinf(ranges), 10.0, ranges)
        ranges = np.where(np.isnan(ranges), 10.0, ranges)
        ranges = np.clip(ranges, 0.06, 10.0)
        
        if self.phase == "MAPPING":
            self.update_occupancy_grid(msg, ranges)
            self.gap_follower_control(msg, ranges)
        elif self.phase == "PLANNING":
            self.compute_racing_line()
        elif self.phase == "RACING":
            self.racing_control()

    # =========================================================
    # PHASE 1: MAPPING
    # =========================================================
    
    def check_lap_complete(self):
        """Detect when first lap is complete"""
        if self.start_position is None:
            return
        
        dist = np.linalg.norm(self.position - self.start_position)
        
        if dist > 3.0:
            self.has_moved_away = True
        
        if self.has_moved_away and dist < 0.8 and len(self.trajectory) > 200:
            self.get_logger().info("=" * 50)
            self.get_logger().info("LAP 1 COMPLETE!")
            self.get_logger().info("=" * 50)
            self.phase = "PLANNING"

    def update_occupancy_grid(self, msg, ranges):
        """Update occupancy grid from LIDAR"""
        angle = msg.angle_min
        for r in ranges:
            if r < 9.9:
                hit_x = self.position[0] + r * math.cos(self.yaw + angle)
                hit_y = self.position[1] + r * math.sin(self.yaw + angle)
                
                gx = int((hit_x - self.MAP_ORIGIN) / self.MAP_RESOLUTION)
                gy = int((hit_y - self.MAP_ORIGIN) / self.MAP_RESOLUTION)
                
                if 0 <= gx < self.MAP_SIZE and 0 <= gy < self.MAP_SIZE:
                    self.hit_counts[gy, gx] += 1
                
                # Mark free space
                for frac in [0.3, 0.6]:
                    fx = self.position[0] + r * frac * math.cos(self.yaw + angle)
                    fy = self.position[1] + r * frac * math.sin(self.yaw + angle)
                    gfx = int((fx - self.MAP_ORIGIN) / self.MAP_RESOLUTION)
                    gfy = int((fy - self.MAP_ORIGIN) / self.MAP_RESOLUTION)
                    if 0 <= gfx < self.MAP_SIZE and 0 <= gfy < self.MAP_SIZE:
                        self.miss_counts[gfy, gfx] += 1
            
            angle += msg.angle_increment

    def gap_follower_control(self, msg, ranges):
        """Safe gap following for mapping phase"""
        num_points = len(ranges)
        center = num_points // 2
        
        forward_arc = int(num_points * 0.17)
        start = center - forward_arc
        end = center + forward_arc
        forward_ranges = ranges[start:end]
        forward_min = np.min(forward_ranges)
        
        # Find best gap
        window = 7
        smoothed = np.convolve(forward_ranges, np.ones(window)/window, mode='same')
        best_idx = start + np.argmax(smoothed)
        target_angle = msg.angle_min + best_idx * msg.angle_increment
        
        # Steering
        steer = target_angle * 0.55
        steer = 0.65 * steer + 0.35 * self.prev_steer
        steer = np.clip(steer, -self.MAX_STEER, self.MAX_STEER)
        self.prev_steer = steer
        
        # Wall avoidance
        left_idx = center + int(num_points * 0.15)
        right_idx = center - int(num_points * 0.15)
        left_dist = np.mean(ranges[left_idx:min(left_idx+15, num_points)])
        right_dist = np.mean(ranges[max(0, right_idx-15):right_idx])
        
        if left_dist < 0.55:
            steer = max(steer - 0.2, -self.MAX_STEER)
        if right_dist < 0.55:
            steer = min(steer + 0.2, self.MAX_STEER)
        
        # Speed
        speed = self.MAPPING_SPEED
        if abs(steer) > 0.25 or forward_min < 1.0:
            speed = self.CORNER_SPEED
        
        self.publish_control(steer, speed)

    # =========================================================
    # PHASE 2: PLANNING
    # =========================================================
    
    def compute_racing_line(self):
        """Compute optimal racing line from map"""
        self.get_logger().info("Computing racing line...")
        
        try:
            # Build occupancy grid
            total = self.hit_counts + self.miss_counts
            with np.errstate(divide='ignore', invalid='ignore'):
                prob = np.where(total > 0, self.hit_counts / total, 0.5)
            
            self.occupancy_grid = np.where(
                total < 3, -1,
                np.where(prob > 0.65, 100, np.where(prob < 0.35, 0, -1))
            ).astype(np.int8)
            
            # Extract centerline
            centerline = self.extract_centerline()
            if centerline is None or len(centerline) < 10:
                self.get_logger().error("Failed to extract centerline!")
                # Fallback: use trajectory
                centerline = np.array(list(self.trajectory))
            
            # Smooth
            centerline = self.smooth_path(centerline)
            
            # Optimize (simple smoothing for now)
            self.racing_line = self.optimize_line(centerline)
            
            # Velocity profile
            self.compute_velocity_profile()
            
            # Headings
            self.compute_headings()
            
            self.get_logger().info(f"Racing line: {len(self.racing_line)} points")
            self.get_logger().info("=" * 50)
            self.get_logger().info("RACING PHASE - GO FAST!")
            self.get_logger().info("=" * 50)
            
            self.phase = "RACING"
            
        except Exception as e:
            self.get_logger().error(f"Planning failed: {e}")
            # Emergency fallback: use recorded trajectory
            self.racing_line = np.array(list(self.trajectory))
            if len(self.racing_line) > 10:
                self.racing_line = self.smooth_path(self.racing_line)
                self.compute_velocity_profile()
                self.compute_headings()
                self.phase = "RACING"

    def extract_centerline(self):
        """Extract centerline using distance transform"""
        free_space = (self.occupancy_grid == 0).astype(np.uint8)
        dist = ndimage.distance_transform_edt(free_space)
        
        from scipy.ndimage import maximum_filter
        max_filt = maximum_filter(dist, size=5)
        skeleton = (dist == max_filt) & (dist > 2)
        
        points = np.argwhere(skeleton)
        if len(points) < 20:
            threshold = np.percentile(dist[dist > 0], 75) if np.any(dist > 0) else 0
            points = np.argwhere(dist > threshold)
        
        if len(points) < 10:
            return None
        
        # Convert to world coords
        world = np.zeros((len(points), 2))
        world[:, 0] = points[:, 1] * self.MAP_RESOLUTION + self.MAP_ORIGIN
        world[:, 1] = points[:, 0] * self.MAP_RESOLUTION + self.MAP_ORIGIN
        
        return self.order_points(world)

    def order_points(self, points):
        """Order points by nearest neighbor"""
        if len(points) < 2:
            return points
        
        dists = np.linalg.norm(points - self.position, axis=1)
        start_idx = np.argmin(dists)
        
        ordered = [points[start_idx]]
        remaining = list(range(len(points)))
        remaining.remove(start_idx)
        
        while remaining:
            last = ordered[-1]
            dists = [np.linalg.norm(points[i] - last) for i in remaining]
            if min(dists) > 1.0:
                break
            nearest = remaining[np.argmin(dists)]
            ordered.append(points[nearest])
            remaining.remove(nearest)
        
        return np.array(ordered)

    def smooth_path(self, points, s=0.5):
        """Smooth path with spline"""
        if len(points) < 4:
            return points
        
        try:
            if np.linalg.norm(points[0] - points[-1]) < 2.0:
                points = np.vstack([points, points[0]])
            
            tck, u = splprep([points[:, 0], points[:, 1]], s=s, per=True)
            u_new = np.linspace(0, 1, max(100, len(points) * 2))
            return np.array(splev(u_new, tck)).T
        except:
            return points

    def optimize_line(self, centerline):
        """Simple curvature-based smoothing"""
        line = centerline.copy()
        for _ in range(5):
            new_line = line.copy()
            for i in range(1, len(line) - 1):
                v1 = line[i] - line[i-1]
                v2 = line[i+1] - line[i]
                angle = np.arctan2(np.cross(v1, v2), np.dot(v1, v2))
                alpha = min(0.3, abs(angle) * 2)
                new_line[i] = (1 - alpha) * line[i] + alpha * 0.5 * (line[i-1] + line[i+1])
            line = new_line
        return line

    def compute_velocity_profile(self):
        """Compute velocity based on curvature"""
        n = len(self.racing_line)
        curvatures = np.zeros(n)
        
        for i in range(n):
            p1 = self.racing_line[(i-1) % n]
            p2 = self.racing_line[i]
            p3 = self.racing_line[(i+1) % n]
            
            area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
            d1, d2, d3 = np.linalg.norm(p2-p1), np.linalg.norm(p3-p2), np.linalg.norm(p3-p1)
            
            if d1 * d2 * d3 > 1e-9:
                curvatures[i] = 4 * area / (d1 * d2 * d3)
        
        curvatures = np.convolve(curvatures, np.ones(5)/5, mode='same')
        
        self.racing_velocities = np.zeros(n)
        for i in range(n):
            if curvatures[i] > 0.01:
                v_max = math.sqrt(8.0 / curvatures[i])  # a_lat = 8 m/s^2
            else:
                v_max = self.MAX_RACING_SPEED
            self.racing_velocities[i] = np.clip(v_max, self.MIN_RACING_SPEED, self.MAX_RACING_SPEED)
        
        self.racing_velocities = np.convolve(self.racing_velocities, np.ones(5)/5, mode='same')

    def compute_headings(self):
        """Compute heading at each point"""
        n = len(self.racing_line)
        self.racing_headings = np.zeros(n)
        for i in range(n):
            dx = self.racing_line[(i+1) % n, 0] - self.racing_line[i, 0]
            dy = self.racing_line[(i+1) % n, 1] - self.racing_line[i, 1]
            self.racing_headings[i] = math.atan2(dy, dx)

    # =========================================================
    # PHASE 3: RACING (MPC or Pure Pursuit)
    # =========================================================
    
    def setup_mpc(self):
        """Setup CasADi MPC solver"""
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        psi = ca.SX.sym('psi')
        v = ca.SX.sym('v')
        state = ca.vertcat(x, y, psi, v)
        
        delta = ca.SX.sym('delta')
        a = ca.SX.sym('a')
        control = ca.vertcat(delta, a)
        
        # Bicycle model
        x_dot = v * ca.cos(psi)
        y_dot = v * ca.sin(psi)
        psi_dot = v / self.L * ca.tan(delta)
        v_dot = a
        
        f = ca.Function('f', [state, control], [ca.vertcat(x_dot, y_dot, psi_dot, v_dot)])
        
        # RK4 integration
        k1 = f(state, control)
        k2 = f(state + self.dt/2*k1, control)
        k3 = f(state + self.dt/2*k2, control)
        k4 = f(state + self.dt*k3, control)
        state_next = state + self.dt/6*(k1 + 2*k2 + 2*k3 + k4)
        
        F = ca.Function('F', [state, control], [state_next])
        
        # NLP
        w, w0, lbw, ubw = [], [], [], []
        g, lbg, ubg = [], [], []
        
        n_params = 4 + self.N * 4 + 1
        P = ca.SX.sym('P', n_params)
        J = 0
        
        Xk = ca.SX.sym('X0', 4)
        w.append(Xk)
        lbw.extend([-ca.inf]*4)
        ubw.extend([ca.inf]*4)
        w0.extend([0, 0, 0, 1])
        
        g.append(Xk - P[:4])
        lbg.extend([0]*4)
        ubg.extend([0]*4)
        
        prev_delta = P[-1]
        
        for k in range(self.N):
            Uk = ca.SX.sym(f'U{k}', 2)
            w.append(Uk)
            lbw.extend([-self.MAX_STEER, -5.0])
            ubw.extend([self.MAX_STEER, 3.0])
            w0.extend([0, 0])
            
            ref_idx = 4 + k*4
            dx = Xk[0] - P[ref_idx]
            dy = Xk[1] - P[ref_idx+1]
            dpsi = ca.atan2(ca.sin(Xk[2] - P[ref_idx+2]), ca.cos(Xk[2] - P[ref_idx+2]))
            dv = Xk[3] - P[ref_idx+3]
            
            J += self.Q_pos * (dx**2 + dy**2)
            J += self.Q_heading * dpsi**2
            J += self.Q_vel * dv**2
            J += self.R_steer * Uk[0]**2
            J += self.R_accel * Uk[1]**2
            
            if k == 0:
                d_rate = (Uk[0] - prev_delta) / self.dt
            else:
                d_rate = (Uk[0] - w[-3][0]) / self.dt
            J += self.R_steer_rate * d_rate**2
            
            g.append(d_rate)
            lbg.append(-self.MAX_STEER_RATE)
            ubg.append(self.MAX_STEER_RATE)
            
            Xk_next = F(Xk, Uk)
            
            Xk = ca.SX.sym(f'X{k+1}', 4)
            w.append(Xk)
            lbw.extend([-ca.inf, -ca.inf, -ca.inf, 0])
            ubw.extend([ca.inf, ca.inf, ca.inf, self.MAX_RACING_SPEED])
            w0.extend([0, 0, 0, 1])
            
            g.append(Xk - Xk_next)
            lbg.extend([0]*4)
            ubg.extend([0]*4)
        
        nlp = {'x': ca.vertcat(*w), 'f': J, 'g': ca.vertcat(*g), 'p': P}
        opts = {
            'ipopt.print_level': 0,
            'ipopt.max_iter': 50,
            'ipopt.warm_start_init_point': 'yes',
            'print_time': 0
        }
        
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        self.w0 = w0
        self.lbw = lbw
        self.ubw = ubw
        self.lbg = lbg
        self.ubg = ubg
        self.mpc_ready = True

    def racing_control(self):
        """MPC or Pure Pursuit control"""
        if CASADI_AVAILABLE and self.mpc_ready:
            self.mpc_control()
        else:
            self.pure_pursuit_control()

    def find_closest_waypoint(self):
        """Find closest point on racing line"""
        dists = np.sqrt(
            (self.racing_line[:, 0] - self.position[0])**2 +
            (self.racing_line[:, 1] - self.position[1])**2
        )
        return np.argmin(dists)

    def get_reference_window(self, start_idx):
        """Get N reference points with look-ahead"""
        n = len(self.racing_line)
        refs = np.zeros((self.N, 4))
        look_ahead = max(3, int(self.velocity * 0.5))
        
        for i in range(self.N):
            idx = (start_idx + look_ahead + i) % n
            refs[i, 0] = self.racing_line[idx, 0]
            refs[i, 1] = self.racing_line[idx, 1]
            refs[i, 2] = self.racing_headings[idx]
            refs[i, 3] = self.racing_velocities[idx]
        
        return refs

    def mpc_control(self):
        """MPC trajectory tracking"""
        state = np.array([
            self.position[0],
            self.position[1],
            self.yaw,
            max(0.1, self.velocity)
        ])
        
        closest = self.find_closest_waypoint()
        refs = self.get_reference_window(closest)
        
        params = np.concatenate([state, refs.flatten(), [self.prev_steer]])
        
        try:
            sol = self.solver(
                x0=self.w0, lbx=self.lbw, ubx=self.ubw,
                lbg=self.lbg, ubg=self.ubg, p=params
            )
            
            w_opt = sol['x'].full().flatten()
            delta = float(w_opt[4])  # First control
            a = float(w_opt[5])
            
            steer = np.clip(delta, -self.MAX_STEER, self.MAX_STEER)
            
            # Convert to throttle
            target_v = refs[0, 3]
            throttle = 0.25 * target_v + 0.08 * a
            throttle = np.clip(throttle, 0.0, 1.0)
            
            self.publish_control(steer, throttle)
            self.prev_steer = steer
            self.w0 = list(w_opt)
            
        except Exception as e:
            self.get_logger().warn(f"MPC failed: {e}")
            self.pure_pursuit_control()

    def pure_pursuit_control(self):
        """Fallback pure pursuit"""
        closest = self.find_closest_waypoint()
        look_ahead = max(0.5, self.velocity * 0.6)
        
        n = len(self.racing_line)
        target_idx = closest
        cumul = 0.0
        
        for i in range(50):
            idx = (closest + i) % n
            next_idx = (idx + 1) % n
            cumul += np.linalg.norm(self.racing_line[next_idx] - self.racing_line[idx])
            if cumul >= look_ahead:
                target_idx = next_idx
                break
        
        target = self.racing_line[target_idx]
        dx = target[0] - self.position[0]
        dy = target[1] - self.position[1]
        
        local_x = dx * np.cos(-self.yaw) - dy * np.sin(-self.yaw)
        local_y = dx * np.sin(-self.yaw) + dy * np.cos(-self.yaw)
        
        if local_x > 0.1:
            curvature = 2 * local_y / (local_x**2 + local_y**2)
            steer = np.arctan(curvature * self.L)
        else:
            steer = 0.0
        
        steer = np.clip(steer, -self.MAX_STEER, self.MAX_STEER)
        steer = 0.7 * steer + 0.3 * self.prev_steer
        self.prev_steer = steer
        
        target_v = self.racing_velocities[target_idx]
        throttle = np.clip(0.25 * target_v, 0.15, 1.0)
        
        if abs(steer) > 0.3:
            throttle *= 0.7
        
        self.publish_control(steer, throttle)

    def publish_control(self, steer, throttle):
        """Publish control commands"""
        s_msg = Float32()
        s_msg.data = float(steer)
        self.pub_steering.publish(s_msg)
        
        t_msg = Float32()
        t_msg.data = float(throttle)
        self.pub_throttle.publish(t_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CompDriver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

