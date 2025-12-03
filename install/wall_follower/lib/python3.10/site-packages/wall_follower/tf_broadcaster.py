"""
TF Broadcaster - Empty template
"""

import rclpy
from rclpy.node import Node


class TFBroadcaster(Node):
    def __init__(self):
        super().__init__('tf_broadcaster')
        self.get_logger().info("TF Broadcaster started")


def main(args=None):
    rclpy.init(args=args)
    node = TFBroadcaster()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
