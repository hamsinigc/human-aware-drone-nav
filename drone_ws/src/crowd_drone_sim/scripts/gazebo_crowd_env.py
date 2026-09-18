"""
gazebo_crowd_env.py (v3 — observation format aligned to the trained model)

IMPORTANT CHANGE FROM THE PREVIOUS VERSION: the observation layout now
EXACTLY matches CrowdNavEnv (the fast synthetic environment your models
were trained on):
    [drone_x, drone_y, drone_vx, drone_vy, goal_x, goal_y,
     (rel_x, rel_y, vel_x, vel_y, radius) x n_closest]
with n_closest=5 by default -> 6 + 5*5 = 31 dimensions total.

WHY THIS MATTERS: a trained PPO model's neural network has fixed input
dimensions baked in from training. To run ppo_crowdnav_v2.zip (or any
model trained on CrowdNavEnv) against the REAL Gazebo simulation for a
live demo, this environment's observation must be pixel-for-pixel the
same shape and feature order as what the model was trained on — this
version does that. (The 'radius' feature is a constant placeholder,
0.3, matching CrowdNavEnv's default pedestrian radius, since Gazebo
doesn't need to report a literal collision radius for this purpose.)

This is now used for INFERENCE (running a trained policy) rather than
training — training still happens in the fast synthetic environment.
"""

import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import GetEntityState, SetEntityState
from gazebo_msgs.msg import EntityState


class _RosBridge(Node):
    def __init__(self, drone_altitude: float):
        super().__init__("gazebo_crowdnav_bridge")
        self.drone_altitude = drone_altitude
        self.drone_pos = None
        self.drone_vel = None

        self.cmd_pub = self.create_publisher(Twist, "/drone/cmd_vel", 10)
        self.create_subscription(Odometry, "/drone/odom", self._odom_cb, 10)

        self.get_state_client = self.create_client(GetEntityState, "/gazebo/get_entity_state")
        self.set_state_client = self.create_client(SetEntityState, "/gazebo/set_entity_state")
        for client, name in [
            (self.get_state_client, "/gazebo/get_entity_state"),
            (self.set_state_client, "/gazebo/set_entity_state"),
        ]:
            while not client.wait_for_service(timeout_sec=2.0):
                self.get_logger().info(f"Waiting for {name} service...")

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        self.drone_pos = np.array([p.x, p.y])
        self.drone_vel = np.array([v.x, v.y])

    def get_entity_xy(self, name: str):
        req = GetEntityState.Request()
        req.name = name
        future = self.get_state_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        result = future.result()
        if result is not None and result.success:
            p = result.state.pose.position
            return np.array([p.x, p.y])
        return None

    def send_velocity(self, vx: float, vy: float):
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        self.cmd_pub.publish(msg)

    def teleport_drone(self, x: float, y: float):
        req = SetEntityState.Request()
        state = EntityState()
        state.name = "drone"
        state.pose.position.x = float(x)
        state.pose.position.y = float(y)
        state.pose.position.z = float(self.drone_altitude)
        state.pose.orientation.w = 1.0
        state.reference_frame = "world"
        req.state = state
        future = self.set_state_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

    def spin_briefly(self, timeout_sec: float = 0.1):
        rclpy.spin_once(self, timeout_sec=timeout_sec)


class GazeboCrowdNavEnv(gym.Env):
    def __init__(
        self,
        arena_size: float = 10.0,  # matches CrowdNavEnv's training default — keep goals/starts
        # within roughly the same range the policy actually trained on, for reliability
        world_bound_half: float = 8.0,  # CORRECTED: the previous value of 6.5 actually
        # excluded ALL stalls/trees/tent (they sit at distance ~7.0-7.07 from center) —
        # 8.0 comfortably includes that whole main fairground ring while still excluding
        # the isolated giant wheel (distance ~12.73, in its own far corner)
        drone_altitude: float = 4.5,
        drone_max_speed: float = 1.5,
        collision_dist: float = 1.0,
        comfort_dist: float = 1.5,
        dt: float = 0.25,  # FIXED: now matches CrowdNavEnv's training dt exactly
        max_steps: int = 150,
        n_pedestrians: int = 6,
        n_closest: int = 5,  # MATCHES CrowdNavEnv's default — do not change
        ped_radius_placeholder: float = 0.3,  # matches CrowdNavEnv's default ped_radius
    ):
        super().__init__()
        self.arena_size = arena_size
        self.world_bound_half = world_bound_half
        self.drone_altitude = drone_altitude
        self.drone_max_speed = drone_max_speed
        self.collision_dist = collision_dist
        self.comfort_dist = comfort_dist
        self.dt = dt
        self.max_steps = max_steps
        self.n_pedestrians = n_pedestrians
        self.n_closest = n_closest
        self.ped_radius_placeholder = ped_radius_placeholder
        self.pedestrian_names = [f"pedestrian_{i+1}" for i in range(n_pedestrians)]

        # MUST match CrowdNavEnv exactly: 6 + n_closest * 5
        obs_dim = 6 + self.n_closest * 5
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        if not rclpy.ok():
            rclpy.init()
        self.bridge = _RosBridge(drone_altitude)

        self._rng = np.random.default_rng()
        self._prev_ped_positions = {}

    def _get_all_ped_positions(self):
        positions = {}
        for name in self.pedestrian_names:
            pos = self.bridge.get_entity_xy(name)
            if pos is not None:
                positions[name] = pos
        return positions

    def _nearest_k(self, drone_pos, ped_positions, elapsed_dt=None):
        if elapsed_dt is None:
            elapsed_dt = self.dt
        entries = []
        for name, pos in ped_positions.items():
            dist = np.linalg.norm(pos - drone_pos)
            prev = self._prev_ped_positions.get(name)
            vel = (pos - prev) / elapsed_dt if prev is not None else np.zeros(2)
            entries.append((dist, pos - drone_pos, vel))
        entries.sort(key=lambda e: e[0])

        rel_list, vel_list = [], []
        nearest_dist = entries[0][0] if entries else np.inf
        for i in range(self.n_closest):
            if i < len(entries):
                _, rel, vel = entries[i]
            else:
                rel, vel = np.zeros(2), np.zeros(2)
            rel_list.append(rel)
            vel_list.append(vel)
        return rel_list, vel_list, nearest_dist

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        r = self.arena_size / 2
        self.goal = self._rng.uniform(-r, r, size=2)

        # FIXED: randomize the drone's start position (matches how
        # CrowdNavEnv trained — a fixed (0,0) start is a distribution
        # mismatch from what the policy learned). Also retry a few
        # times if the chosen start happens to land too close to a
        # pedestrian (since pedestrians aren't reset between episodes
        # and keep walking, this can otherwise cause an unfair
        # instant collision right at reset).
        max_attempts = 5
        start_pos = np.zeros(2)
        for attempt in range(max_attempts):
            start_pos = self._rng.uniform(-r, r, size=2)
            self.bridge.teleport_drone(start_pos[0], start_pos[1])
            self.bridge.send_velocity(0.0, 0.0)  # FIX: stop any leftover velocity
            # command from the previous episode — planar_move otherwise keeps
            # applying the last cmd_vel it received, causing drift after teleport
            time.sleep(0.3)
            self.bridge.spin_briefly(0.2)

            ped_positions = self._get_all_ped_positions()
            if ped_positions:
                min_dist = min(
                    np.linalg.norm(pos - start_pos) for pos in ped_positions.values()
                )
                print(
                    f"  [reset attempt {attempt+1}] start=({start_pos[0]:.2f},{start_pos[1]:.2f}) "
                    f"nearest_ped_dist={min_dist:.2f}"
                )
                if min_dist > self.collision_dist + 0.5:
                    break  # safe start found
            else:
                print(f"  [reset attempt {attempt+1}] WARNING: no pedestrian data received yet")
                break

        self.step_count = 0
        self._prev_ped_positions = self._get_all_ped_positions()
        actual_drone_pos = self.bridge.drone_pos if self.bridge.drone_pos is not None else start_pos
        print(
            f"  [reset done] commanded_start=({start_pos[0]:.2f},{start_pos[1]:.2f}) "
            f"actual_odom_pos=({actual_drone_pos[0]:.2f},{actual_drone_pos[1]:.2f}) "
            f"goal=({self.goal[0]:.2f},{self.goal[1]:.2f})"
        )
        self._prev_dist_to_goal = np.linalg.norm(self.goal - actual_drone_pos)
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        step_start_time = time.time()
        action = np.clip(action, -1.0, 1.0) * self.drone_max_speed
        self.bridge.send_velocity(action[0], action[1])

        time.sleep(self.dt)
        self.bridge.spin_briefly(0.1)

        drone_pos = self.bridge.drone_pos
        if drone_pos is None:
            drone_pos = np.zeros(2)

        # FIX: keep the drone within the populated area (where pedestrians
        # and fairground objects actually are) instead of letting it drift
        # into the open ground plane beyond the arena — matches how the
        # fast synthetic environment (CrowdNavEnv) also clips drone
        # position to its arena bounds every step.
        half = self.world_bound_half  # hard escape limit, larger than the goal-sampling area
        clamped_pos = np.clip(drone_pos, -half, half)
        if not np.allclose(clamped_pos, drone_pos, atol=1e-3):
            self.bridge.teleport_drone(clamped_pos[0], clamped_pos[1])
            self.bridge.send_velocity(0.0, 0.0)
            self.bridge.spin_briefly(0.1)
            drone_pos = clamped_pos

        ped_positions = self._get_all_ped_positions()
        # FIX: use the ACTUAL measured elapsed time for velocity estimation,
        # not the fixed dt constant — ROS service call overhead makes real
        # elapsed time per step variable, and dividing by a fixed dt when
        # more real time actually passed silently overestimates how fast
        # pedestrians appear to be moving, confusing the policy.
        actual_elapsed = max(time.time() - step_start_time, 1e-3)
        rel_list, vel_list, nearest_dist = self._nearest_k(
            drone_pos, ped_positions, actual_elapsed
        )
        self._prev_ped_positions = ped_positions

        dist_to_goal = np.linalg.norm(self.goal - drone_pos)
        progress_reward = (self._prev_dist_to_goal - dist_to_goal) * 2.0
        self._prev_dist_to_goal = dist_to_goal

        reward = progress_reward - 0.01
        terminated = False
        info = {}

        if dist_to_goal < 0.5:
            reward += 20.0
            terminated = True
            info["result"] = "goal_reached"

        if nearest_dist < self.collision_dist:
            reward -= 20.0
            terminated = True
            info["result"] = "collision"
        elif nearest_dist < self.collision_dist + self.comfort_dist:
            intrusion = (self.collision_dist + self.comfort_dist) - nearest_dist
            reward -= 0.5 * intrusion

        info["nearest_ped_dist"] = nearest_dist
        info["dist_to_goal"] = dist_to_goal

        truncated = self.step_count >= self.max_steps
        if truncated and not terminated:
            info["result"] = "timeout"

        obs = self._get_obs(drone_pos, rel_list, vel_list)
        return obs, reward, terminated, truncated, info

    def _get_obs(self, drone_pos=None, rel_list=None, vel_list=None):
        if drone_pos is None:
            drone_pos = self.bridge.drone_pos if self.bridge.drone_pos is not None else np.zeros(2)
        drone_vel = self.bridge.drone_vel if self.bridge.drone_vel is not None else np.zeros(2)

        if rel_list is None or vel_list is None:
            ped_positions = self._get_all_ped_positions()
            rel_list, vel_list, _ = self._nearest_k(drone_pos, ped_positions)

        # build each pedestrian slot as (rel_x, rel_y, vel_x, vel_y, radius)
        # — EXACTLY matching CrowdNavEnv's 5-feature-per-pedestrian layout
        ped_features = []
        for rel, vel in zip(rel_list, vel_list):
            ped_features.extend(rel)
            ped_features.extend(vel)
            ped_features.append(self.ped_radius_placeholder)

        obs = np.concatenate(
            [drone_pos, drone_vel, self.goal, np.array(ped_features)]
        ).astype(np.float32)
        return obs

    def close(self):
        self.bridge.destroy_node()
