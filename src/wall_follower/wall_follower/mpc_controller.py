"""
MPC Controller - Dynamic Bicycle Model with scipy.optimize
Simple, debuggable, robust
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Bool, Float32MultiArray
from geometry_msgs.msg import Point
from nav_msgs.msg import Path
import numpy as np
from scipy.optimize import minimize

# Vehicle params (RoboRacer)
L_F = 0.162  # CG to front axle
L_R = 0.162  # CG to rear axle
L = L_F + L_R

# MPC params
DT = 0.15
N = 5  # Horizon

# Limits
STEER_MAX = 100.0
STEER_RATE_MAX = 100.0
ACCEL_MIN, ACCEL_MAX = -1.0, 50.0

# Costs
Q_POS = 50.0
Q_HEAD = 100.0
Q_VEL = 20.0
R_CTRL = 1.0


class MPCController(Node):
    def __init__(self):
        super().__init__('mpc_controller')
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        
        # Pubs
        self.pub_throttle = self.create_publisher(Float32, '/autodrive/roboracer_1/throttle_command', 10)
        self.pub_steering = self.create_publisher(Float32, '/autodrive/roboracer_1/steering_command', 10)
        self.pub_active = self.create_publisher(Bool, '/mpc_active', 10)
        
        # Subs
        self.sub_path = self.create_subscription(Path, '/global_path', self.path_cb, 10)
        self.sub_speeds = self.create_subscription(Float32MultiArray, '/path_speeds', self.speeds_cb, 10)
        self.sub_ips = self.create_subscription(Point, '/autodrive/roboracer_1/ips', self.ips_cb, 10)
        self.sub_lidar = self.create_subscription(LaserScan, '/autodrive/roboracer_1/lidar', self.lidar_cb, qos)
        
        # State: [x, y, theta, v, phi]
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.prev_pos = None
        self.path = None
        self.speeds = None
        self.fwd_min = 10.0
        
        self.create_timer(0.1, self.control_loop)
        self.get_logger().info("MPC CONTROLLER: Waiting for global path...")

    def ips_cb(self, msg):
        if self.prev_pos is not None:
            dx, dy = msg.x - self.prev_pos[0], msg.y - self.prev_pos[1]
            if abs(dx) > 0.005 or abs(dy) > 0.005:
                self.state[2] = np.arctan2(dy, dx)  # theta
                self.state[3] = np.sqrt(dx**2 + dy**2) / 0.05  # v estimate
        self.state[0], self.state[1] = msg.x, msg.y
        self.prev_pos = (msg.x, msg.y)

    def path_cb(self, msg):
        if msg.poses:
            self.path = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses])
            self.get_logger().info(f"MPC: Got path with {len(self.path)} points")

    def speeds_cb(self, msg):
        self.speeds = np.array(msg.data)

    def lidar_cb(self, msg):
        r = np.array(msg.ranges)
        r = np.where(np.isfinite(r), r, 10.0)
        c = len(r) // 2
        self.fwd_min = float(np.min(r[c-60:c+60]))

    def dbm_step(self, state, u):
        """Dynamic Bicycle Model - one step"""
        x, y, theta, v, phi = state
        a, phi_dot = u
        
        # Slip angle
        beta = np.arctan(L_R / L * np.tan(phi))
        
        # Dynamics
        x_new = x + DT * v * np.cos(theta + beta)
        y_new = y + DT * v * np.sin(theta + beta)
        theta_new = theta + DT * v / L_R * np.sin(beta)
        v_new = v + DT * a
        phi_new = np.clip(phi + DT * phi_dot, -STEER_MAX, STEER_MAX)
        
        return np.array([x_new, y_new, theta_new, max(0, v_new), phi_new])

    def get_ref(self, idx):
        """Get reference state at path index"""
        n = len(self.path)
        idx = idx % n
        pos = self.path[idx]
        next_pos = self.path[(idx + 1) % n]
        heading = np.arctan2(next_pos[1] - pos[1], next_pos[0] - pos[0])
        v_ref = self.speeds[idx] if self.speeds is not None else 0.2
        return np.array([pos[0], pos[1], heading, v_ref])

    def mpc_cost(self, u_flat, state, start_idx):
        """Cost function for MPC"""
        u_seq = u_flat.reshape(N, 2)
        cost = 0.0
        s = state.copy()
        
        for k in range(N):
            s = self.dbm_step(s, u_seq[k])
            ref = self.get_ref(start_idx + k * 2)
            
            # Position error
            cost += Q_POS * ((s[0] - ref[0])**2 + (s[1] - ref[1])**2)
            
            # Heading error (wrap)
            h_err = s[2] - ref[2]
            h_err = np.arctan2(np.sin(h_err), np.cos(h_err))
            cost += Q_HEAD * h_err**2
            
            # Speed error
            cost += Q_VEL * (s[3] - ref[3])**2
            
            # Control effort
            cost += R_CTRL * (u_seq[k, 0]**2 + u_seq[k, 1]**2)
        
        return cost

    def solve_mpc(self):
        """Solve MPC optimization"""
        # Find closest path point
        dists = np.linalg.norm(self.path - self.state[:2], axis=1)
        idx = np.argmin(dists)
        
        # Bounds
        bounds = []
        for _ in range(N):
            bounds.append((ACCEL_MIN, ACCEL_MAX))
            bounds.append((-STEER_RATE_MAX, STEER_RATE_MAX))
        
        # Solve
        u_init = np.zeros(N * 2)
        result = minimize(
            lambda u: self.mpc_cost(u, self.state, idx),
            u_init, bounds=bounds, method='SLSQP',
            options={'maxiter': 50, 'ftol': 1e-3}
        )
        
        if result.success:
            u_opt = result.x.reshape(N, 2)
            return u_opt[0]
        return np.array([0.0, 0.0])

    def control_loop(self):
        active = Bool()
        active.data = self.path is not None
        self.pub_active.publish(active)
        
        if self.path is None:
            return
        
        # Safety check
        if self.fwd_min < 0.3:
            self.publish(0.05, 0.0)
            self.get_logger().warn(f"MPC: EMERGENCY fwd={self.fwd_min:.2f}m")
            return
        
        # Solve MPC
        u = self.solve_mpc()
        
        # Apply steering rate to get new steering
        new_phi = self.state[4] + DT * u[1]
        new_phi = np.clip(new_phi, -STEER_MAX, STEER_MAX)
        self.state[4] = new_phi
        
        # Get target speed from path
        dists = np.linalg.norm(self.path - self.state[:2], axis=1)
        idx = np.argmin(dists)
        target_speed = self.speeds[idx] if self.speeds is not None else 0.2
        
        # Safety speed limit
        if self.fwd_min < 0.8:
            target_speed = min(target_speed, 0.15)
        
        self.publish(target_speed, new_phi)

    def publish(self, speed, steer):
        t, s = Float32(), Float32()
        t.data, s.data = float(np.clip(speed, 0, 0.4)), float(np.clip(steer, -STEER_MAX, STEER_MAX))
        self.pub_throttle.publish(t)
        self.pub_steering.publish(s)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(MPCController())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
