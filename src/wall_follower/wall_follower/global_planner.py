"""
Global Planner - Receives recorded path, smooths it, computes speeds
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import numpy as np
from scipy.ndimage import uniform_filter1d

class GlobalPlanner(Node):
    def __init__(self):
        super().__init__('global_planner')
        
        # Pubs
        self.pub_path = self.create_publisher(Path, '/global_path', 10)
        self.pub_speeds = self.create_publisher(Float32MultiArray, '/path_speeds', 10)
        
        # Subs
        self.sub_recorded = self.create_subscription(Path, '/recorded_path', self.path_cb, 10)
        
        self.path_ready = False
        self.create_timer(1.0, self.republish)
        self.get_logger().info("GLOBAL PLANNER: Waiting for recorded path...")

    def path_cb(self, msg):
        if self.path_ready or len(msg.poses) < 20:
            return
        
        # Extract points
        pts = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses])
        self.get_logger().info(f"Received {len(pts)} waypoints, smoothing...")
        
        # Smooth the path
        pts[:, 0] = uniform_filter1d(pts[:, 0], size=5, mode='wrap')
        pts[:, 1] = uniform_filter1d(pts[:, 1], size=5, mode='wrap')
        
        # Compute speeds based on curvature
        n = len(pts)
        speeds = np.ones(n) * 0.35  # Max speed
        
        for i in range(n):
            p0, p1, p2 = pts[(i-3)%n], pts[i], pts[(i+3)%n]
            # Curvature from 3 points
            a, b, c = np.linalg.norm(p1-p0), np.linalg.norm(p2-p1), np.linalg.norm(p2-p0)
            area = 0.5 * abs((p1[0]-p0[0])*(p2[1]-p0[1]) - (p2[0]-p0[0])*(p1[1]-p0[1]))
            if area > 1e-6:
                R = (a * b * c) / (4 * area + 1e-6)
                v_max = np.sqrt(0.6 * 9.8 * R)  # mu * g * R
                speeds[i] = np.clip(v_max, 0.15, 0.35)
        
        # Backward pass - brake before corners
        for i in range(n-1, -1, -1):
            prev = (i - 1) % n
            dist = np.linalg.norm(pts[i] - pts[prev])
            max_v = np.sqrt(speeds[i]**2 + 2 * 0.5 * dist)
            speeds[prev] = min(speeds[prev], max_v)
        
        self.global_path = pts
        self.speeds = speeds
        self.path_ready = True
        
        self.get_logger().info(f"GLOBAL PATH READY: {len(pts)} points, speeds {speeds.min():.2f}-{speeds.max():.2f}")
        self.republish()

    def republish(self):
        if not self.path_ready:
            return
        
        # Publish path
        msg = Path()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        for pt in self.global_path:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x, pose.pose.position.y = float(pt[0]), float(pt[1])
            msg.poses.append(pose)
        self.pub_path.publish(msg)
        
        # Publish speeds
        speed_msg = Float32MultiArray()
        speed_msg.data = self.speeds.tolist()
        self.pub_speeds.publish(speed_msg)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(GlobalPlanner())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
