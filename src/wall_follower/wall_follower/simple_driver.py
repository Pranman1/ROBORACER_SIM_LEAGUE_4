import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32 # <--- NEW MESSAGE TYPE
import math

class GapFollower(Node):
    def __init__(self):
        super().__init__('gap_follower')

        # QoS to ensure we hear the Lidar
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # 1. PUBLISHERS (Separate Gas and Steer)
        self.pub_throttle = self.create_publisher(Float32, '/autodrive/roboracer_1/throttle_command', 10)
        self.pub_steering = self.create_publisher(Float32, '/autodrive/roboracer_1/steering_command', 10)
        
        # 2. SUBSCRIBER
        self.subscription = self.create_subscription(
            LaserScan, 
            '/autodrive/roboracer_1/lidar', 
            self.lidar_callback, 
            qos_profile)
        
        self.get_logger().info("GapFollower Started! Publishing to roboracer_1 topics...")
        
        # TUNING
        self.MAX_THROTTLE = 0.5  # 0.0 to 1.0 (50% power)
        self.MAX_STEER = 0.5     # Radians (approx 30 degrees)

    def lidar_callback(self, scan_data):
        # Clean Data
        ranges = [r if not math.isinf(r) else 0.0 for r in scan_data.ranges]
        
        # Find Gap
        max_distance = 0.0
        max_index = -1
        for i in range(len(ranges)):
            if ranges[i] > max_distance:
                max_distance = ranges[i]
                max_index = i
        
        target_angle = scan_data.angle_min + (max_index * scan_data.angle_increment)
        
        # --- CONTROL LOGIC ---
        
        # 1. Calculate Steering (Clamp it to safety limits)
        steer_cmd = max(min(target_angle, self.MAX_STEER), -self.MAX_STEER)
        
        # 2. Calculate Throttle (Slow down in turns)
        # Simple logic: If steering is 0, throttle is Max. If steering is Max, throttle is reduced.
        throttle_cmd = self.MAX_THROTTLE * (1.0 - abs(steer_cmd) / 1.5)
        
        # --- PUBLISH SEPARATELY ---
        
        # Publish Throttle
        t_msg = Float32()
        t_msg.data = float(throttle_cmd)
        self.pub_throttle.publish(t_msg)
        
        # Publish Steering
        s_msg = Float32()
        s_msg.data = float(steer_cmd)
        self.pub_steering.publish(s_msg)

def main(args=None):
    rclpy.init(args=args)
    node = GapFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()