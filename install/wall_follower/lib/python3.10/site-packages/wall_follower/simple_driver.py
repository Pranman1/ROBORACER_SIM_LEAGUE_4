import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math

class GapFollower(Node):
    def __init__(self):
        super().__init__('gap_follower')
        # 1. PUBLISHER: Sends motor commands
        self.publisher_ = self.create_publisher(Twist, '/autodrive/f1tenth_1/throttle_steering', 10)
        
        # 2. SUBSCRIBER: Listens to the Lidar
        self.subscription = self.create_subscription(LaserScan, '/autodrive/f1tenth_1/lidar', self.lidar_callback, 10)
        
        # TUNING PARAMETERS
        self.MAX_SPEED = 1.5   # Meters/second (Increase if safe)
        self.MAX_STEER = 1.0   # Max steering (approx 45 degrees)

    def lidar_callback(self, scan_data):
        # Clean the Data: Replace 'infinity' with 0.0
        ranges = [r if not math.isinf(r) else 0.0 for r in scan_data.ranges]
        
        # Find the "Deepest" point (The furthest gap)
        max_distance = 0.0
        max_index = -1
        
        # Look for the max range in the array
        for i in range(len(ranges)):
            if ranges[i] > max_distance:
                max_distance = ranges[i]
                max_index = i
        
        # Calculate Steering Angle based on that index
        target_angle = scan_data.angle_min + (max_index * scan_data.angle_increment)
        
        # Create the Drive Command
        cmd = Twist()
        
        # STEERING: Proportional Controller toward the gap
        cmd.angular.z = max(min(target_angle, self.MAX_STEER), -self.MAX_STEER)
        
        # SPEED: Slow down if turning sharp (Safety Controller)
        cmd.linear.x = self.MAX_SPEED * (1.0 - abs(cmd.angular.z) / 2.0)
        
        self.publisher_.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = GapFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()