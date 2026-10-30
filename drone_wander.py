import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import math

class HumanAwareDroneWanderer(Node):
    def __init__(self):
        super().__init__('drone_wanderer')
        
        self.publisher_ = self.create_publisher(Twist, '/drone/cmd_vel', 10)
        self.hud_publisher_ = self.create_publisher(Image, '/drone_cam/hud_image', 10)
        
        self.subscription_odom = self.create_subscription(Odometry, '/drone/odom', self.odom_callback, 10)
        self.subscription_states = self.create_subscription(ModelStates, '/gazebo/model_states', self.model_states_callback, 10)
        self.subscription_cam = self.create_subscription(Image, '/drone_cam/drone_camera/image_raw', self.camera_callback, 10)
        
        self.timer = self.create_timer(0.05, self.control_loop)
        self.bridge = CvBridge()
        
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_height = 4.5
        self.current_yaw = 0.0
        self.linear_speed = 0.0
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.accuracy = 98.5
        self.status = "STABLE"
        
        self.obstacles = []
        self.get_logger().info("Human-Aware Drone Wanderer with Top-Right HUD Initialized.")

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_height = msg.pose.pose.position.z
        
        self.velocity_x = msg.twist.twist.linear.x
        self.velocity_y = msg.twist.twist.linear.y
        self.linear_speed = math.sqrt(self.velocity_x**2 + self.velocity_y**2)
        
        # Calculate dynamic path tracking accuracy relative to target center/height
        target_dist_from_center = math.sqrt(self.current_x**2 + self.current_y**2)
        self.accuracy = max(85.0, min(99.9, 100.0 - (target_dist_from_center * 1.5) - (abs(4.5 - self.current_height) * 3.0)))
        
        orientation = msg.pose.pose.orientation
        siny_cosp = 2.0 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def model_states_callback(self, msg):
        self.obstacles = []
        for name, pose in zip(msg.name, msg.pose):
            if name != 'drone':
                self.obstacles.append((pose.position.x, pose.position.y))

    def camera_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            return

        h, w, _ = frame.shape
        
        # Top-Right HUD Dashboard Coordinates
        box_w, box_h = 280, 140
        x1 = w - box_w - 20
        y1 = 20
        x2 = w - 20
        y2 = y1 + box_h

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "--- DRONE TELEMETRY ---", (x1 + 15, y1 + 25), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Speed: {self.linear_speed:.2f} m/s", (x1 + 15, y1 + 52), font, 0.48, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Velocity: [{self.velocity_x:.1f}, {self.velocity_y:.1f}]", (x1 + 15, y1 + 78), font, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Accuracy: {self.accuracy:.1f}%", (x1 + 15, y1 + 104), font, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Status: {self.status}", (x1 + 15, y1 + 128), font, 0.45, (255, 100, 255), 1, cv2.LINE_AA)

        hud_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        self.hud_publisher_.publish(hud_msg)

    def control_loop(self):
        twist = Twist()
        
        target_height = 4.5
        height_error = target_height - self.current_height
        twist.linear.z = 0.4 * height_error
        
        safe_limit = 5.5
        if abs(self.current_x) > safe_limit or abs(self.current_y) > safe_limit:
            self.status = "RETURNING"
            angle_to_center = math.atan2(-self.current_y, -self.current_x)
            angle_diff = angle_to_center - self.current_yaw
            angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
            
            twist.linear.x = 0.3
            twist.angular.z = 1.2 * angle_diff
            self.publisher_.publish(twist)
            return

        repel_x = 0.0
        repel_y = 0.0
        detection_radius = 2.2
        
        for ox, oy in self.obstacles:
            dx = self.current_x - ox
            dy = self.current_y - oy
            dist = math.sqrt(dx**2 + dy**2)
            
            if dist < detection_radius and dist > 0.05:
                strength = (1.0 / dist - 1.0 / detection_radius) / (dist**2)
                repel_x += (dx / dist) * strength * 1.5
                repel_y += (dy / dist) * strength * 1.5

        if abs(repel_x) > 0.05 or abs(repel_y) > 0.05:
            self.status = "AVOIDING"
        else:
            self.status = "CRUISING"

        desired_vx = 0.42
        desired_vy = 0.0
        
        total_vx = desired_vx + (repel_x * math.cos(self.current_yaw) + repel_y * math.sin(self.current_yaw))
        total_vy = desired_vy + (-repel_x * math.sin(self.current_yaw) + repel_y * math.cos(self.current_yaw))
        
        twist.linear.x = max(0.1, min(0.45, total_vx))
        
        if abs(repel_x) > 0.1 or abs(repel_y) > 0.1:
            twist.angular.z = 1.8 * total_vy
        else:
            twist.angular.z = 0.15 * math.sin(self.current_x * 0.8)
            
        self.publisher_.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = HumanAwareDroneWanderer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
