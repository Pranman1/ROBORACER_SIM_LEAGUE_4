"""
WEIGHTED DRIVER with DEAD-END DETECTION
Steers toward open space, but AVOIDS directions where walls converge (dead ends).
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
import numpy as np

class WeightedDriver(Node):
    def __init__(self):
        super().__init__('weighted_driver')
        
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, 
            history=HistoryPolicy.KEEP_LAST, 
            depth=10
        )
        
        self.pub_throttle = self.create_publisher(Float32, '/autodrive/roboracer_1/throttle_command', 10)
        self.pub_steering = self.create_publisher(Float32, '/autodrive/roboracer_1/steering_command', 10)
        self.sub_lidar = self.create_subscription(LaserScan, '/autodrive/roboracer_1/lidar', self.lidar_callback, qos)
        
        self.SPEED = 0.05
        self.MAX_STEER = 0.5236
        
        self.count = 0
        self.get_logger().info("=" * 50)
        self.get_logger().info("WEIGHTED DRIVER + DEAD-END AVOIDANCE")
        self.get_logger().info("Avoids directions where walls converge!")
        self.get_logger().info("=" * 50)

    def compute_convergence(self, ranges, idx, window=30):
        """
        Check if walls are converging at this direction.
        Look at readings on either side of idx - if they get CLOSER as we go outward,
        that means walls are converging (dead end!).
        
        Returns: convergence score (positive = converging/bad, negative = diverging/good)
        """
        n = len(ranges)
        
        # Get readings on left and right of this direction
        left_start = min(n-1, idx + 10)
        left_end = min(n-1, idx + window)
        right_start = max(0, idx - window)
        right_end = max(0, idx - 10)
        
        if left_end <= left_start or right_end <= right_start:
            return 0.0
        
        left_readings = ranges[left_start:left_end]
        right_readings = ranges[right_start:right_end]
        
        if len(left_readings) < 5 or len(right_readings) < 5:
            return 0.0
        
        # Check trend: are distances DECREASING as we go outward?
        # Decreasing = walls getting closer = converging = dead end
        left_trend = np.mean(left_readings[-5:]) - np.mean(left_readings[:5])  # outer - inner
        right_trend = np.mean(right_readings[:5]) - np.mean(right_readings[-5:])  # outer - inner
        
        # Negative trend = converging (bad), Positive trend = diverging (good)
        convergence = -(left_trend + right_trend) / 2
        
        return convergence

    def lidar_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isfinite(ranges), ranges, 0.1)
        ranges = np.clip(ranges, 0.1, 10.0)
        
        n = len(ranges)
        center = n // 2
        
        # Front 180 degrees
        half_fov = 360
        start = max(0, center - half_fov)
        end = min(n, center + half_fov)
        
        front_ranges = ranges[start:end]
        num_points = len(front_ranges)
        
        # Smooth ranges
        kernel_size = 11
        smoothed = np.convolve(front_ranges, np.ones(kernel_size)/kernel_size, mode='same')
        
        # Create angles array
        angles = np.linspace(-np.pi/2, np.pi/2, num_points)
        
        # Compute weights: distance^2 gives preference to open space
        base_weights = smoothed ** 2
        
        # Now PENALIZE directions with converging walls (dead ends)
        convergence_penalty = np.zeros(num_points)
        check_step = 10  # Check every 10th point for efficiency
        
        for i in range(0, num_points, check_step):
            conv = self.compute_convergence(smoothed, i, window=40)
            # Apply penalty to surrounding indices
            penalty_start = max(0, i - check_step//2)
            penalty_end = min(num_points, i + check_step//2)
            convergence_penalty[penalty_start:penalty_end] = conv
        
        # Apply penalty: reduce weight where convergence is high (dead end)
        # convergence > 0 means converging (bad), multiply weight by small number
        # convergence < 0 means diverging (good), keep or boost weight
        penalty_factor = np.exp(-convergence_penalty * 2)  # e^(-2*conv)
        adjusted_weights = base_weights * penalty_factor
        
        # Also add small forward bias to prefer going straight when options are similar
        forward_bias = np.exp(-angles**2 / 0.5)  # Gaussian centered on forward
        adjusted_weights = adjusted_weights * (1 + 0.3 * forward_bias)
        
        # Weighted average angle
        if np.sum(adjusted_weights) > 0:
            weighted_angle = np.sum(angles * adjusted_weights) / np.sum(adjusted_weights)
        else:
            weighted_angle = 0.0
        
        # Convert to steering
        steer = weighted_angle * 1.5
        steer = np.clip(steer, -self.MAX_STEER, self.MAX_STEER)
        
        # Speed control
        front_dist = np.min(smoothed[num_points//2 - 50:num_points//2 + 50])
        
        if front_dist < 0.5:
            speed = 0.03
        elif front_dist < 1.0:
            speed = 0.04
        else:
            speed = self.SPEED
        
        # Debug
        self.count += 1
        if self.count % 20 == 0:
            best_idx = np.argmax(adjusted_weights)
            best_angle = angles[best_idx]
            max_conv = np.max(convergence_penalty)
            self.get_logger().info(
                f"steer={steer:.2f} angle={np.degrees(weighted_angle):.1f}° | "
                f"best@{np.degrees(best_angle):.1f}° | max_conv={max_conv:.2f} F={front_dist:.2f}"
            )
        
        t_msg = Float32()
        s_msg = Float32()
        t_msg.data = float(speed)
        s_msg.data = float(steer)
        self.pub_throttle.publish(t_msg)
        self.pub_steering.publish(s_msg)


def main(args=None):
    rclpy.init(args=args)
    node = WeightedDriver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
