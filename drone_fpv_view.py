import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

class HumanPOVViewer(Node):
    def __init__(self):
        super().__init__('human_pov_viewer')
        self.bridge = CvBridge()
        # Switched to the back camera topic to check the alternate angle
        self.subscription = self.create_subscription(
            Image,
            '/drone/camera_back/back/image_raw',
            self.image_callback,
            1
        )
        self.get_logger().info('Back Camera POV Active.')

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            frame = cv2.resize(frame, (1280, 960), interpolation=cv2.INTER_LINEAR)
            
            cv2.putText(frame, 'DRONE BACK POV (Scenario View)', (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            cv2.imshow('Human-Aware Drone - Human POV', frame)
            cv2.waitKey(1)
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')

def main(args=None):
    rclpy.init(args=args)
    viewer = HumanPOVViewer()
    try:
        rclpy.spin(viewer)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
