#!/usr/bin/env python3
"""
MPC Controller for Racing
Uses generated raceline as reference path
"""
import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass

@dataclass
class MPCConfig:
    """MPC Configuration"""
    N: int = 10  # Prediction horizon
    dt: float = 0.1  # Time step (s)
    
    # Vehicle parameters (from AutoDRIVE)
    L: float = 0.324  # Wheelbase (m)
    max_steer: float = 0.5236  # Max steering (rad) = 30deg
    max_speed: float = 0.10  # Max speed (m/s) - start conservative!
    min_speed: float = 0.02  # Min speed
    
    # Cost weights
    Q_pos: float = 100.0  # Position tracking weight
    Q_heading: float = 10.0  # Heading tracking weight
    Q_speed: float = 1.0  # Speed tracking weight
    R_steer: float = 10.0  # Steering effort
    R_accel: float = 1.0  # Acceleration effort
    R_steer_rate: float = 100.0  # Steering rate (smooth)

class MPCController:
    """Model Predictive Controller for racing"""
    
    def __init__(self, config=None):
        self.config = config or MPCConfig()
        self.prev_steer = 0.0
        self.prev_speed = 0.02
        
    def find_closest_point(self, x, y, raceline):
        """Find closest waypoint on raceline"""
        dists = np.linalg.norm(raceline - np.array([x, y]), axis=1)
        return np.argmin(dists)
    
    def get_reference_trajectory(self, x, y, yaw, raceline):
        """Get N-step reference trajectory ahead on raceline"""
        closest_idx = self.find_closest_point(x, y, raceline)
        
        # Get N waypoints ahead
        ref_traj = []
        for i in range(self.config.N):
            idx = (closest_idx + i * 2) % len(raceline)  # Every 2nd point
            wp = raceline[idx]
            
            # Calculate reference heading (to next waypoint)
            next_idx = (idx + 1) % len(raceline)
            next_wp = raceline[next_idx]
            ref_yaw = np.arctan2(next_wp[1] - wp[1], next_wp[0] - wp[0])
            
            ref_traj.append([wp[0], wp[1], ref_yaw])
        
        return np.array(ref_traj)
    
    def bicycle_model(self, state, u):
        """Kinematic bicycle model"""
        x, y, psi, v = state
        delta, a = u
        
        # Clip inputs
        delta = np.clip(delta, -self.config.max_steer, self.config.max_steer)
        
        # Dynamics
        x_dot = v * np.cos(psi)
        y_dot = v * np.sin(psi)
        psi_dot = v * np.tan(delta) / self.config.L
        v_dot = a
        
        # Integrate (Euler)
        x_new = x + x_dot * self.config.dt
        y_new = y + y_dot * self.config.dt
        psi_new = psi + psi_dot * self.config.dt
        v_new = v + v_dot * self.config.dt
        v_new = np.clip(v_new, self.config.min_speed, self.config.max_speed)
        
        return np.array([x_new, y_new, psi_new, v_new])
    
    def compute_control(self, x, y, yaw, v, raceline):
        """Compute MPC control command"""
        # Get reference trajectory
        ref_traj = self.get_reference_trajectory(x, y, yaw, raceline)
        
        # Initial guess (continue previous control)
        u_init = np.array([self.prev_steer, 0.0] * self.config.N)
        
        # Bounds
        bounds = []
        for _ in range(self.config.N):
            bounds.append((-self.config.max_steer, self.config.max_steer))  # steering
            bounds.append((-0.02, 0.02))  # acceleration
        
        # Optimization
        result = minimize(
            fun=self.cost_function,
            x0=u_init,
            args=(x, y, yaw, v, ref_traj),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-3}
        )
        
        # Extract first control
        u_opt = result.x[:2]
        steer, accel = u_opt
        
        # Update state
        self.prev_steer = steer
        new_speed = v + accel * self.config.dt
        new_speed = np.clip(new_speed, self.config.min_speed, self.config.max_speed)
        self.prev_speed = new_speed
        
        return steer, new_speed
    
    def cost_function(self, u, x0, y0, psi0, v0, ref_traj):
        """MPC cost function"""
        state = np.array([x0, y0, psi0, v0])
        cost = 0.0
        
        for i in range(self.config.N):
            # Control input
            delta = u[2*i]
            a = u[2*i + 1]
            
            # Predict state
            state = self.bicycle_model(state, [delta, a])
            x, y, psi, v = state
            
            # Reference
            x_ref, y_ref, psi_ref = ref_traj[i]
            
            # Position error
            pos_error = (x - x_ref)**2 + (y - y_ref)**2
            
            # Heading error
            heading_error = (psi - psi_ref)**2
            
            # Speed tracking (target max speed)
            speed_error = (v - self.config.max_speed)**2
            
            # Control effort
            steer_effort = delta**2
            accel_effort = a**2
            
            # Steering rate (smoothness)
            if i > 0:
                delta_prev = u[2*(i-1)]
                steer_rate = (delta - delta_prev)**2
            else:
                steer_rate = (delta - self.prev_steer)**2
            
            # Total cost
            cost += (self.config.Q_pos * pos_error +
                    self.config.Q_heading * heading_error +
                    self.config.Q_speed * speed_error +
                    self.config.R_steer * steer_effort +
                    self.config.R_accel * accel_effort +
                    self.config.R_steer_rate * steer_rate)
        
        return cost

def test_mpc():
    """Quick test"""
    # Dummy raceline (circle)
    theta = np.linspace(0, 2*np.pi, 100)
    raceline = np.column_stack([5*np.cos(theta), 5*np.sin(theta)])
    
    # Controller
    mpc = MPCController()
    
    # State
    x, y, yaw, v = 5.0, 0.0, 0.0, 0.04
    
    # Compute control
    steer, speed = mpc.compute_control(x, y, yaw, v, raceline)
    print(f"Steer: {steer:.3f}, Speed: {speed:.3f}")

if __name__ == "__main__":
    test_mpc()
