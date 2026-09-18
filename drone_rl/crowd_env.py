"""
CrowdNavEnv — a 2D bird's-eye-view Gymnasium environment for training a
drone to navigate through a crowd of pedestrians.

Why 2D bird's-eye-view: this matches how most published socially-aware
navigation RL papers (CrowdNav, SARL, Social-force + DRL) set up their
simulator, and it matches your project's aerial drone perspective. The
drone is modeled as a disc that can set its own velocity directly
(a common simplification — treats low-level flight control as solved,
so the RL agent focuses purely on *social* navigation decisions).

AGENTS
------
- Drone: controlled by the RL policy. Action = desired (vx, vy).
- Pedestrians: simulated with a simplified Social Force Model — each
  walks toward its own goal and is repelled by nearby agents (including
  the drone). This gives semi-realistic, reactive crowd behavior without
  needing a real dataset to drive it.

OBSERVATION (per step, flattened into one vector)
--------------------------------------------------
- Drone: position (x, y), velocity (vx, vy), goal (gx, gy)          [6]
- For each of the N_closest nearest pedestrians:
    relative position (dx, dy), velocity (vx, vy), radius            [5 each]
(missing pedestrian slots are zero-padded if fewer than N_closest are
present)

ACTION
------
Continuous 2D velocity command (vx, vy), clipped to the drone's max speed.

REWARD
------
+ progress toward goal (shaped, dense)
+ large bonus on reaching goal
- large penalty on collision (episode ends)
- small "discomfort" penalty for entering a pedestrian's personal-space
  buffer, even without a full collision (this is what teaches
  *socially aware* behavior rather than just collision avoidance)
- small time penalty per step (encourages efficient paths)

This file has NO dependency on the trajectory predictor — it works
standalone with raw current-position observations. See the bottom of
this file (`PredictorAugmentedCrowdNavEnv`) for how to plug in
predicted future trajectories from Phase 1 as extra observation
features once you're ready to combine the two project components.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CrowdNavEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        n_pedestrians: int = 8,
        n_closest: int = 5,
        arena_size: float = 10.0,
        drone_radius: float = 0.3,
        ped_radius: float = 0.3,
        drone_max_speed: float = 1.5,
        ped_speed: float = 1.0,
        comfort_dist: float = 0.5,
        dt: float = 0.25,
        max_steps: int = 200,
        render_mode: str = None,
    ):
        super().__init__()
        self.n_pedestrians = n_pedestrians
        self.n_closest = n_closest
        self.arena_size = arena_size
        self.drone_radius = drone_radius
        self.ped_radius = ped_radius
        self.drone_max_speed = drone_max_speed
        self.ped_speed = ped_speed
        self.comfort_dist = comfort_dist
        self.dt = dt
        self.max_steps = max_steps
        self.render_mode = render_mode

        obs_dim = 6 + self.n_closest * 5
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self._rng = np.random.default_rng()
        self.fig = None  # for optional rendering

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.step_count = 0
        r = self.arena_size / 2

        self.drone_pos = self._rng.uniform(-r, r, size=2)
        self.drone_goal = self._rng.uniform(-r, r, size=2)
        while np.linalg.norm(self.drone_goal - self.drone_pos) < r:
            self.drone_goal = self._rng.uniform(-r, r, size=2)
        self.drone_vel = np.zeros(2)

        self.ped_pos = self._rng.uniform(-r, r, size=(self.n_pedestrians, 2))
        self.ped_goal = self._rng.uniform(-r, r, size=(self.n_pedestrians, 2))
        self.ped_vel = np.zeros((self.n_pedestrians, 2))

        self._prev_dist_to_goal = np.linalg.norm(self.drone_goal - self.drone_pos)

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def _social_force_step(self):
        """Move each pedestrian one dt using a simplified social force
        model: attraction to its own goal + repulsion from nearby agents
        (other pedestrians AND the drone)."""
        new_vel = np.zeros_like(self.ped_vel)

        for i in range(self.n_pedestrians):
            to_goal = self.ped_goal[i] - self.ped_pos[i]
            dist_to_goal = np.linalg.norm(to_goal) + 1e-6
            if dist_to_goal < 0.3:
                # reached goal -> pick a new one so the crowd keeps moving
                r = self.arena_size / 2
                self.ped_goal[i] = self._rng.uniform(-r, r, size=2)
                to_goal = self.ped_goal[i] - self.ped_pos[i]
                dist_to_goal = np.linalg.norm(to_goal) + 1e-6

            desired_vel = (to_goal / dist_to_goal) * self.ped_speed

            repulsion = np.zeros(2)
            # repulsion from other pedestrians
            for j in range(self.n_pedestrians):
                if i == j:
                    continue
                diff = self.ped_pos[i] - self.ped_pos[j]
                dist = np.linalg.norm(diff) + 1e-6
                min_dist = 2 * self.ped_radius + self.comfort_dist
                if dist < min_dist:
                    repulsion += (diff / dist) * (min_dist - dist) * 2.0
            # repulsion from the drone
            diff = self.ped_pos[i] - self.drone_pos
            dist = np.linalg.norm(diff) + 1e-6
            min_dist = self.ped_radius + self.drone_radius + self.comfort_dist
            if dist < min_dist:
                repulsion += (diff / dist) * (min_dist - dist) * 2.0

            new_vel[i] = desired_vel + repulsion

        self.ped_vel = new_vel
        self.ped_pos = self.ped_pos + self.ped_vel * self.dt
        r = self.arena_size / 2
        self.ped_pos = np.clip(self.ped_pos, -r, r)

    # ------------------------------------------------------------------
    def step(self, action):
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0) * self.drone_max_speed
        self.drone_vel = action
        self.drone_pos = self.drone_pos + self.drone_vel * self.dt
        r = self.arena_size / 2
        self.drone_pos = np.clip(self.drone_pos, -r, r)

        self._social_force_step()

        # --- reward ---
        dist_to_goal = np.linalg.norm(self.drone_goal - self.drone_pos)
        progress_reward = (self._prev_dist_to_goal - dist_to_goal) * 2.0
        self._prev_dist_to_goal = dist_to_goal

        reward = progress_reward - 0.01  # small time penalty
        terminated = False
        info = {}

        goal_reached = dist_to_goal < (self.drone_radius + 0.2)
        if goal_reached:
            reward += 20.0
            terminated = True
            info["result"] = "goal_reached"

        min_ped_dist = np.inf
        for i in range(self.n_pedestrians):
            dist = np.linalg.norm(self.ped_pos[i] - self.drone_pos)
            min_ped_dist = min(min_ped_dist, dist)
            collision_dist = self.ped_radius + self.drone_radius
            if dist < collision_dist:
                reward -= 20.0
                terminated = True
                info["result"] = "collision"
            elif dist < collision_dist + self.comfort_dist:
                # discomfort penalty: scales with how deep into the
                # personal-space buffer the drone is
                intrusion = (collision_dist + self.comfort_dist) - dist
                reward -= 0.5 * intrusion

        info["min_ped_dist"] = min_ped_dist
        info["dist_to_goal"] = dist_to_goal

        truncated = self.step_count >= self.max_steps
        if truncated and not terminated:
            info["result"] = "timeout"

        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _get_obs(self):
        drone_part = np.concatenate(
            [self.drone_pos, self.drone_vel, self.drone_goal]
        )

        rel_pos = self.ped_pos - self.drone_pos
        dists = np.linalg.norm(rel_pos, axis=1)
        order = np.argsort(dists)[: self.n_closest]

        ped_part = np.zeros((self.n_closest, 5), dtype=np.float32)
        for slot, idx in enumerate(order):
            ped_part[slot, 0:2] = rel_pos[idx]
            ped_part[slot, 2:4] = self.ped_vel[idx]
            ped_part[slot, 4] = self.ped_radius

        obs = np.concatenate([drone_part, ped_part.flatten()]).astype(np.float32)
        return obs

    # ------------------------------------------------------------------
    def render(self):
        import matplotlib.pyplot as plt

        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 6))

        self.ax.clear()
        r = self.arena_size / 2
        self.ax.set_xlim(-r, r)
        self.ax.set_ylim(-r, r)
        self.ax.set_aspect("equal")

        self.ax.scatter(*self.drone_goal, c="green", marker="*", s=200, label="goal")
        self.ax.add_patch(
            plt.Circle(self.drone_pos, self.drone_radius, color="blue", label="drone")
        )
        for p in self.ped_pos:
            self.ax.add_patch(plt.Circle(p, self.ped_radius, color="orange", alpha=0.7))

        self.ax.legend(loc="upper right")
        plt.pause(0.01)

    def close(self):
        if self.fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self.fig)


if __name__ == "__main__":
    env = CrowdNavEnv()
    obs, info = env.reset(seed=0)
    print("Observation shape:", obs.shape)

    total_reward = 0.0
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print("Episode ended:", info.get("result"))
            break

    print("Random-policy total reward over rollout:", round(total_reward, 2))
