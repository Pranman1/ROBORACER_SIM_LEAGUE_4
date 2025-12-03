import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
import numpy as np

class RobustDriver(Node):
    def __init__(self):
        super().__init__('robust_driver')
        
        # PUBLISHERS
        self.pub_throttle = self.create_publisher(Float32, '/autodrive/roboracer_1/throttle_command', 10)
        self.pub_steering = self.create_publisher(Float32, '/autodrive/roboracer_1/steering_command', 10)
        
        # SUBSCRIBERS
        self.sub_lidar = self.create_subscription(LaserScan, '/autodrive/roboracer_1/lidar', self.lidar_callback, 10)
        
        # SPEEDS - VERY SLOW
        self.SPEED = 0.12            # Constant slow speed
        self.CORNER_SPEED = 0.08     # Slower in corners
        
        # STEERING
        self.MAX_STEER = 0.35        # Max steering (reduced!)
        self.STEER_GAIN = 0.5        # How much to follow target angle
        
        self.get_logger().info("=== ROBUST DRIVER v2 STARTED ===")

    def lidar_callback(self, data):
        # Clean data
        ranges = np.array(data.ranges, dtype=np.float32)
        ranges = np.where(np.isinf(ranges), 10.0, ranges)
        ranges = np.where(np.isnan(ranges), 10.0, ranges)
        ranges = np.clip(ranges, 0.1, 10.0)
        
        num_points = len(ranges)
        center = num_points // 2
        angle_min = data.angle_min
        angle_inc = data.angle_increment
        
        # ONLY look at forward 60 degrees (-30 to +30)
        # This prevents it from steering towards side gaps
        forward_arc = int(num_points * 0.17)  # ~30 degrees each side
        start = center - forward_arc
        end = center + forward_arc
        
        # Get forward ranges only
        forward_ranges = ranges[start:end]
        
        # Find minimum in forward (for safety check)
        forward_min = np.min(forward_ranges)
        
        # Find best direction (furthest point in forward arc)
        # Use smoothing to avoid noise
        window = 7
        smoothed = np.convolve(forward_ranges, np.ones(window)/window, mode='same')
        best_local_idx = np.argmax(smoothed)
        best_global_idx = start + best_local_idx
        
        # Calculate target angle
        target_angle = angle_min + best_global_idx * angle_inc
        
        # REDUCE the steering - don't follow target fully
        steer = target_angle * self.STEER_GAIN
        steer = np.clip(steer, -self.MAX_STEER, self.MAX_STEER)
        
        # Check sides for emergency avoidance
        left_idx = center + int(num_points * 0.15)  # ~27 degrees left
        right_idx = center - int(num_points * 0.15)  # ~27 degrees right
        
        left_dist = np.mean(ranges[left_idx:left_idx+10]) if left_idx < num_points-10 else 10.0
        right_dist = np.mean(ranges[max(0,right_idx-10):right_idx]) if right_idx > 10 else 10.0
        
        # EMERGENCY: If wall close on one side, steer away EARLIER
        if left_dist < 0.75:
            steer = -self.MAX_STEER  # Steer RIGHT
            self.get_logger().warn(f"LEFT WALL CLOSE ({left_dist:.2f}m) - STEERING RIGHT")
        elif right_dist < 0.75:
            steer = self.MAX_STEER   # Steer LEFT
            self.get_logger().warn(f"RIGHT WALL CLOSE ({right_dist:.2f}m) - STEERING LEFT")
        
        # Speed control
        speed = self.SPEED
        if abs(steer) > 0.2 or forward_min < 1.0:
            speed = self.CORNER_SPEED
            
        # Log status
        self.get_logger().info(f"Fwd:{forward_min:.2f}m L:{left_dist:.2f}m R:{right_dist:.2f}m | Steer:{steer:.2f} Speed:{speed}")
        
        # Publish
        t_msg = Float32()
        t_msg.data = float(speed)
        self.pub_throttle.publish(t_msg)
        
        s_msg = Float32()
        s_msg.data = float(steer)
        self.pub_steering.publish(s_msg)

def main(args=None):
    rclpy.init(args=args)
    node = RobustDriver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
