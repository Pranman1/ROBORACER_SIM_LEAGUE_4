"""
Reactive Controller - NEVER STOP (so steering works)
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
import numpy as np

MAX_STEER = 0.5236

class DisparityExtender(Node):
    def __init__(self):
        super().__init__('disparity_extender')
        
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        
        self.pub_throttle = self.create_publisher(Float32, '/autodrive/roboracer_1/throttle_command', 10)
        self.pub_steering = self.create_publisher(Float32, '/autodrive/roboracer_1/steering_command', 10)
        self.sub_lidar = self.create_subscription(LaserScan, '/autodrive/roboracer_1/lidar', self.lidar_callback, qos)
        
        self.count = 0
        self.get_logger().info("Reactive Controller - NEVER STOPS")

    def lidar_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isfinite(ranges), ranges, 10.0)
        ranges = np.clip(ranges, 0.1, 10.0)
        
        n = len(ranges)
        center = n // 2
        
        # Get minimum distances in sectors
        front = np.min(ranges[center-50:center+50])
        front_left = np.min(ranges[center+50:center+150])
        front_right = np.min(ranges[center-150:center-50])
        left = np.min(ranges[center+150:min(n, center+300)])
        right = np.min(ranges[max(0, center-300):center-150])
        
        # Debug
        self.count += 1
        if self.count % 20 == 0:
            self.get_logger().info(f"F={front:.2f} FL={front_left:.2f} FR={front_right:.2f} L={left:.2f} R={right:.2f}")
        
        # === ALWAYS KEEP MOVING (minimum speed 0.05) ===
        # Otherwise steering has no effect!
        
        # Find the MOST OPEN direction
        directions = {
            'front': front,
            'front_left': front_left,
            'front_right': front_right,
            'left': left,
            'right': right
        }
        
        best_dir = max(directions, key=directions.get)
        best_dist = directions[best_dir]
        
        # Steering based on best direction
        if best_dir == 'front':
            steer = 0.0
        elif best_dir == 'front_left':
            steer = MAX_STEER * 0.5
        elif best_dir == 'left':
            steer = MAX_STEER
        elif best_dir == 'front_right':
            steer = -MAX_STEER * 0.5
        elif best_dir == 'right':
            steer = -MAX_STEER
        else:
            steer = 0.0
        
        # BUT also avoid close obstacles
        if front < 0.4:
            # Emergency - turn toward most open side
            if left > right:
                steer = MAX_STEER
            else:
                steer = -MAX_STEER
        elif front_right < 0.4:
            steer = max(steer, MAX_STEER * 0.6)  # Turn left
        elif front_left < 0.4:
            steer = min(steer, -MAX_STEER * 0.6)  # Turn right
        
        # Speed - NEVER ZERO (need motion for steering to work)
        if front < 0.3:
            speed = 0.04  # Crawl but still move
        elif front < 0.5:
            speed = 0.06
        elif front < 0.8:
            speed = 0.08
        elif front < 1.2:
            speed = 0.10
        else:
            speed = 0.15
        
        # Slow down more if turning hard
        if abs(steer) > 0.4:
            speed = min(speed, 0.08)
        
        steer = np.clip(steer, -MAX_STEER, MAX_STEER)
        
        # Publish
        t_msg, s_msg = Float32(), Float32()
        t_msg.data = float(speed)
        s_msg.data = float(steer)
        self.pub_throttle.publish(t_msg)
        self.pub_steering.publish(s_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DisparityExtender()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
