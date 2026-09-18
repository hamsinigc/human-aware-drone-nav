"""
PredictionAugmentedCrowdNavEnv — wraps CrowdNavEnv and adds each nearby
pedestrian's PREDICTED FUTURE position (from the trained
TrajectoryTransformer) into the RL observation, on top of their current
position/velocity.

This is the actual "integration" step of the project: it's what lets
the drone anticipate where pedestrians are HEADED, not just react to
where they currently are.

HOW IT WORKS EACH STEP:
1. Maintain a rolling history (last 8 positions) for every pedestrian
   slot, matching the trajectory predictor's expected input length.
2. Feed that history through the trained Transformer to get each
   pedestrian's predicted position several steps into the future.
3. Add that predicted position (relative to the drone) as extra
   features in the observation, for the same nearest-N pedestrians the
   base environment already tracks.

The underlying physics/reward/collision logic is UNCHANGED — only the
information available to the POLICY is richer. This keeps the
comparison ("with prediction" vs "without prediction") fair: both see
the same world, only the observation differs.
"""

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from collections import deque

from model import TrajectoryTransformer
from crowd_env import CrowdNavEnv


class PredictionAugmentedCrowdNavEnv(gym.Wrapper):
    def __init__(
        self,
        base_env: CrowdNavEnv = None,
        checkpoint_path: str = "checkpoint.pt",
        obs_len: int = 8,
        pred_len: int = 12,
        prediction_horizon_step: int = 4,
        device: str = "cpu",
        **base_env_kwargs,
    ):
        """
        prediction_horizon_step: which future timestep's prediction to
        use as the observation feature (e.g. 4 = "where will this
        pedestrian be 4 steps from now", out of the model's pred_len
        total predicted steps). Must be <= pred_len - 1.
        """
        if base_env is None:
            base_env = CrowdNavEnv(**base_env_kwargs)
        super().__init__(base_env)

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.prediction_horizon_step = prediction_horizon_step
        self.device = device

        self.predictor = TrajectoryTransformer(obs_len=obs_len, pred_len=pred_len)
        state_dict = torch.load(checkpoint_path, map_location=device)
        self.predictor.load_state_dict(state_dict)
        self.predictor.eval()
        self.predictor.to(device)

        n_closest = base_env.n_closest
        # extra 2 values (predicted relative x,y) per tracked pedestrian
        base_dim = base_env.observation_space.shape[0]
        new_dim = base_dim + n_closest * 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(new_dim,), dtype=np.float32
        )

        # rolling position history per pedestrian slot
        self._history = None  # set in reset()

    # ------------------------------------------------------------------
    def _init_history(self):
        env = self.env
        self._history = [
            deque(
                [env.ped_pos[i].copy() for _ in range(self.obs_len)],
                maxlen=self.obs_len,
            )
            for i in range(env.n_pedestrians)
        ]

    def _update_history(self):
        env = self.env
        for i in range(env.n_pedestrians):
            self._history[i].append(env.ped_pos[i].copy())

    def _nearest_indices(self):
        env = self.env
        rel_pos = env.ped_pos - env.drone_pos
        dists = np.linalg.norm(rel_pos, axis=1)
        return np.argsort(dists)[: env.n_closest]

    @torch.no_grad()
    def _predict_future_positions(self, indices):
        """Returns predicted future (x,y) for each pedestrian index in
        `indices`, at self.prediction_horizon_step steps ahead."""
        env = self.env
        if len(indices) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        # build (1, n_agents, obs_len, 2) input tensor for just these agents
        histories = np.stack(
            [np.array(self._history[i]) for i in indices], axis=0
        )  # (n_agents, obs_len, 2)
        histories = histories[np.newaxis, ...]  # (1, n_agents, obs_len, 2)
        obs_tensor = torch.tensor(histories, dtype=torch.float32, device=self.device)

        origin = obs_tensor[:, :, 0:1, :]  # first observed frame per agent
        obs_rel = obs_tensor - origin

        mask = torch.ones(1, len(indices), dtype=torch.bool, device=self.device)
        pred_rel = self.predictor(obs_rel, mask)  # (1, n_agents, pred_len, 2)
        pred_abs = pred_rel + origin

        step = min(self.prediction_horizon_step, self.pred_len - 1)
        future_pos = pred_abs[0, :, step, :].cpu().numpy()  # (n_agents, 2)
        return future_pos

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._init_history()
        augmented_obs = self._augment_obs(obs)
        return augmented_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._update_history()
        augmented_obs = self._augment_obs(obs)
        return augmented_obs, reward, terminated, truncated, info

    def _augment_obs(self, base_obs):
        env = self.env
        indices = self._nearest_indices()
        future_pos = self._predict_future_positions(indices)  # (k, 2) absolute

        # pad/truncate to exactly n_closest, and express relative to drone
        pred_features = np.zeros((env.n_closest, 2), dtype=np.float32)
        for slot in range(min(len(indices), env.n_closest)):
            pred_features[slot] = future_pos[slot] - env.drone_pos

        return np.concatenate([base_obs, pred_features.flatten()]).astype(np.float32)


if __name__ == "__main__":
    # smoke test: confirm shapes and that it runs without crashing
    env = PredictionAugmentedCrowdNavEnv(
        checkpoint_path="checkpoint.pt", n_pedestrians=8
    )
    obs, info = env.reset(seed=0)
    print("Base env obs dim would be:", env.env.observation_space.shape)
    print("Augmented obs shape:", obs.shape)

    total_reward = 0.0
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"Episode ended at step {i}:", info.get("result"))
            break

    print("Random-policy total reward:", round(total_reward, 2))
    print("Sanity check passed: prediction-augmented observation pipeline works.")
