"""
Step 1 test script (v2) — confirms we can read BOTH the drone's position
(via /drone/odom) and the pedestrian's position (via the
/gazebo/get_entity_state service) from a single Python script.

Fix from v1: avoids nested spinning by using a plain loop instead of a
ROS timer callback.
"""

import time
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import GetEntityState


class BridgeTestNode(Node):
    def __init__(self):
        super().__init__("bridge_test_node")
        self.drone_pos = None
        self.create_subscription(Odometry, "/drone/odom", self._odom_cb, 10)

        self.state_client = self.create_client(
            GetEntityState, "/gazebo/get_entity_state"
        )
        while not self.state_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for /gazebo/get_entity_state service...")

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self.drone_pos = (p.x, p.y, p.z)

    def get_entity_position(self, name: str):
        req = GetEntityState.Request()
        req.name = name
        future = self.state_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        result = future.result()
        if result is not None and result.success:
            p = result.state.pose.position
            return (p.x, p.y, p.z)
        return None


def main():
    rclpy.init()
    node = BridgeTestNode()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)  # process odom subscription
            ped_pos = node.get_entity_position("pedestrian_1")
            print(f"Drone position: {node.drone_pos}  |  Pedestrian position: {ped_pos}")
            time.sleep(0.9)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
