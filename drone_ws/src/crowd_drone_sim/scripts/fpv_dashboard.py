import math
import base64
import tkinter as tk

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy


BG = "#101820"
PANEL = "#182632"
TEXT = "#F1F5F9"
MUTED = "#A8BAC9"
ACCENT = "#38BDF8"
GREEN = "#4ADE80"


class DroneDashboard(Node):
    def __init__(self):
        super().__init__("fpv_dashboard")

        self.bridge = CvBridge()
        self.latest_frame = None
        self.latest_odom = None
        self.camera_error = None

        self.create_subscription(
            Image,
            "/drone/camera_front/front/image_raw",
            self.image_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )

        self.create_subscription(
            Odometry,
            "/drone/odom",
            self.odom_callback,
            10
        )

    def image_callback(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="bgr8"
            )
            self.camera_error = None
        except Exception as exc:
            self.camera_error = str(exc)

    def odom_callback(self, msg):
        self.latest_odom = msg


def make_metric(parent, title, initial="--", unit=""):
    card = tk.Frame(parent, bg=PANEL, padx=14, pady=9)
    tk.Label(
        card, text=title, bg=PANEL, fg=MUTED,
        font=("Arial", 9, "bold")
    ).pack(anchor="w")

    value = tk.Label(
        card, text=initial, bg=PANEL, fg=ACCENT,
        font=("Arial", 19, "bold")
    )
    value.pack(anchor="w")

    if unit:
        tk.Label(
            card, text=unit, bg=PANEL, fg=MUTED,
            font=("Arial", 8)
        ).pack(anchor="w")

    return card, value


def main():
    rclpy.init()
    node = DroneDashboard()

    root = tk.Tk()
    root.title("Human-Aware Drone Navigation | Research Dashboard")
    root.geometry("1280x800")
    root.minsize(800, 550)
    root.configure(bg=BG)

    # Header
    header = tk.Frame(root, bg=BG, padx=18, pady=10)
    header.pack(fill="x")

    tk.Label(
        header, text="HUMAN-AWARE DRONE NAVIGATION",
        bg=BG, fg=TEXT, font=("Arial", 15, "bold")
    ).pack(side="left")

    camera_status = tk.Label(
        header, text="● CONNECTING TO CAMERA",
        bg=BG, fg=ACCENT, font=("Arial", 9, "bold")
    )
    camera_status.pack(side="right")

    # Main FPV view. HUD panels are layered over the camera.
    view = tk.Frame(root, bg="#05090D")
    view.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    video = tk.Label(
        view, bg="#05090D",
        text="Waiting for front camera image...",
        fg=MUTED, font=("Arial", 16)
    )
    video.place(x=0, y=0, relwidth=1, relheight=1)

    # Top-left camera information overlay
    camera_tag = tk.Frame(view, bg=PANEL, padx=12, pady=8)
    camera_tag.place(x=16, y=16)

    tk.Label(
        camera_tag, text="FPV  /  FRONT CAMERA",
        bg=PANEL, fg=TEXT, font=("Arial", 10, "bold")
    ).pack(anchor="w")

    tk.Label(
        camera_tag, text="ROS 2  •  GAZEBO SIMULATION",
        bg=PANEL, fg=MUTED, font=("Arial", 8)
    ).pack(anchor="w", pady=(3, 0))

    # Bottom HUD metric cards
    hud = tk.Frame(view, bg=BG, padx=8, pady=8)
    hud.place(relx=0.5, rely=1.0, anchor="s", y=-16)

    speed_card, speed_value = make_metric(
        hud, "GROUND SPEED", "--", "m/s"
    )
    altitude_card, altitude_value = make_metric(
        hud, "ALTITUDE", "--", "m"
    )
    vx_card, vx_value = make_metric(
        hud, "VELOCITY X", "--", "m/s"
    )
    vy_card, vy_value = make_metric(
        hud, "VELOCITY Y", "--", "m/s"
    )
    vz_card, vz_value = make_metric(
        hud, "VELOCITY Z", "--", "m/s"
    )

    cards = [
        speed_card, altitude_card, vx_card, vy_card, vz_card
    ]
    for i, card in enumerate(cards):
        card.grid(row=0, column=i, padx=3, sticky="nsew")

    footer = tk.Label(
        root,
        text="LIVE TELEMETRY  •  Values are read from drone odometry",
        bg=BG, fg=MUTED, font=("Arial", 9)
    )
    footer.pack(fill="x", padx=14, pady=(0, 7))

    photo = None
    last_size = (0, 0)

    def update_dashboard():
        nonlocal photo, last_size

        if not rclpy.ok():
            root.destroy()
            return

        try:
            rclpy.spin_once(node, timeout_sec=0.001)
        except Exception as exc:
            footer.config(text=f"ROS 2 error: {exc}")

        frame = node.latest_frame
        if frame is not None:
            target_w = max(video.winfo_width(), 320)
            target_h = max(video.winfo_height(), 240)
            h, w = frame.shape[:2]

            scale = min(target_w / w, target_h / h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))

            resized = cv2.resize(frame, (new_w, new_h))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            # Tkinter can display a base64-encoded PPM image.
            ppm = f"P6\n{new_w} {new_h}\n255\n".encode() + rgb.tobytes()
            encoded = base64.b64encode(ppm).decode("ascii")
            photo = tk.
            # Convert the camera frame into a Tkinter-compatible image.
            pil_image = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(pil_image)
	    PhotoImage(data=encoded, format="PPM")

            video.config(image=photo, text="")
            video.image = photo
            camera_status.config(
                text="● CAMERA CONNECTED", fg=GREEN
            )
        else:
            camera_status.config(
                text="● WAITING FOR CAMERA", fg=ACCENT
            )

        if node.camera_error:
            footer.config(text="Camera conversion error: " + node.camera_error)

        odom = node.latest_odom
        if odom is not None:
            p = odom.pose.pose.position
            v = odom.twist.twist.linear

            speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)

            speed_value.config(text=f"{speed:.2f}")
            altitude_value.config(text=f"{p.z:.2f}")
            vx_value.config(text=f"{v.x:.2f}")
            vy_value.config(text=f"{v.y:.2f}")
            vz_value.config(text=f"{v.z:.2f}")

        root.after(50, update_dashboard)

    def close_dashboard():
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close_dashboard)
    root.after(100, update_dashboard)

    try:
        root.mainloop()
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
