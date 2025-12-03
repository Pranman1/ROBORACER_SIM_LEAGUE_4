import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
import numpy as np

class SlamDriver(Node):
    def __init__(self):
        super().__init__('slam_driver')
        
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        
        # Publishers
        self.pub_throttle = self.create_publisher(Float32, '/autodrive/roboracer_1/throttle_command', 10)
        self.pub_steering = self.create_publisher(Float32, '/autodrive/roboracer_1/steering_command', 10)
        
        # Subscriber
        self.sub_lidar = self.create_subscription(LaserScan, '/autodrive/roboracer_1/lidar', self.lidar_callback, qos)
        
        # Tuning - EXTRA SLOW
        self.SPEED = 0.20
        self.CORNER_SPEED = 0.10
        self.MAX_STEER = 0.60
        
        self.prev_steer = 0.0
        self.count = 0
        
        self.get_logger().info("SIMPLE GAP FOLLOWER STARTED")

    def lidar_callback(self, data):
        self.count += 1
        
        ranges = np.array(data.ranges, dtype=np.float32)
        ranges = np.where(np.isinf(ranges), 10.0, ranges)
        ranges = np.where(np.isnan(ranges), 10.0, ranges)
        ranges = np.clip(ranges, 0.1, 10.0)
        
        num_points = len(ranges)
        center = num_points // 2
        angle_min = data.angle_min
        angle_inc = data.angle_increment
        
        # Forward arc (60 degrees)
        forward_arc = int(num_points * 0.17)
        start = center - forward_arc
        end = center + forward_arc
        forward_ranges = ranges[start:end]
        forward_min = np.min(forward_ranges)
        
        # Find best direction
        window = 7
        smoothed = np.convolve(forward_ranges, np.ones(window)/window, mode='same')
        best_idx = start + np.argmax(smoothed)
        target_angle = angle_min + best_idx * angle_inc
        
        # Steering with smoothing
        steer = target_angle * 0.5
        steer = 0.6 * steer + 0.4 * self.prev_steer
        steer = np.clip(steer, -self.MAX_STEER, self.MAX_STEER)
        self.prev_steer = steer
        
        # Wall avoidance
        left_idx = center + int(num_points * 0.15)
        right_idx = center - int(num_points * 0.15)
        left_dist = np.mean(ranges[left_idx:left_idx+10]) if left_idx < num_points-10 else 10.0
        right_dist = np.mean(ranges[max(0,right_idx-10):right_idx]) if right_idx > 10 else 10.0
        
        # Proportional wall avoidance (not full max)
        if left_dist < 0.85:
            steer = max(steer - 0.15, -self.MAX_STEER)  # Add avoidance, don't override
        if right_dist < 0.85:
            steer = min(steer + 0.15, self.MAX_STEER)
        
        # Speed control
        speed = self.SPEED
        if abs(steer) > 0.2 or forward_min < 1.0:
            speed = self.CORNER_SPEED
        
        # Log every 20th
        if self.count % 20 == 0:
            self.get_logger().info(f"F:{forward_min:.1f} L:{left_dist:.1f} R:{right_dist:.1f} | St:{steer:.2f} Sp:{speed}")
        
        # Publish
        t_msg = Float32()
        t_msg.data = float(speed)
        self.pub_throttle.publish(t_msg)
        
        s_msg = Float32()
        s_msg.data = float(steer)
        self.pub_steering.publish(s_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SlamDriver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
