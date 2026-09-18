"""
generate_matched_dataset.py

Generates trajectory training data by running crowd_env.py's ACTUAL
social-force pedestrian simulation (instead of the separate synthetic
straight-line dataset from Phase 1). This fixes the distribution
mismatch: the predictor now trains on the exact same kind of pedestrian
motion it will see when deployed inside the RL environment.

The drone follows a simple "move toward goal" heuristic while data is
collected (not a trained policy) — this is a reasonable stand-in since
we mainly need realistic pedestrian reactions to a nearby moving agent,
not the exact behavior of any particular trained policy.

Returns the same (obs, fut, mask) tuple shape as the original
SyntheticCrowdDataset, so no changes are needed to the training loop
logic itself.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from crowd_env import CrowdNavEnv


class CrowdEnvTrajectoryDataset(Dataset):
    def __init__(
        self,
        n_scenes: int = 3000,
        obs_len: int = 8,
        pred_len: int = 12,
        max_agents: int = 8,
        n_pedestrians_range=(3, 8),
        seed: int = 0,
    ):
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.max_agents = max_agents
        self.total_len = obs_len + pred_len
        rng = np.random.default_rng(seed)
        self.scenes = []

        for scene_idx in range(n_scenes):
            n_pedestrians = int(rng.integers(n_pedestrians_range[0], n_pedestrians_range[1] + 1))
            env = CrowdNavEnv(n_pedestrians=n_pedestrians, max_steps=self.total_len + 5)
            env.reset(seed=int(rng.integers(0, 1_000_000)))

            positions = np.zeros((n_pedestrians, self.total_len, 2), dtype=np.float32)
            for t in range(self.total_len):
                # simple goal-seeking heuristic for the drone (not a
                # trained policy) — just needs to be a plausible moving
                # agent so pedestrians react realistically via the
                # social-force repulsion term
                to_goal = env.drone_goal - env.drone_pos
                dist = np.linalg.norm(to_goal) + 1e-6
                action = np.clip(to_goal / dist, -1.0, 1.0)
                env.step(action)  # ignore reward/done — just want the dynamics
                positions[:, t, :] = env.ped_pos

            scene = np.zeros((max_agents, self.total_len, 2), dtype=np.float32)
            mask = np.zeros((max_agents,), dtype=bool)
            n_fill = min(n_pedestrians, max_agents)
            scene[:n_fill] = positions[:n_fill]
            mask[:n_fill] = True

            self.scenes.append((scene, mask))

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene, mask = self.scenes[idx]
        obs = scene[:, : self.obs_len]
        fut = scene[:, self.obs_len :]
        return (
            torch.from_numpy(obs.copy()),
            torch.from_numpy(fut.copy()),
            torch.from_numpy(mask.copy()),
        )


if __name__ == "__main__":
    ds = CrowdEnvTrajectoryDataset(n_scenes=10, max_agents=8)
    obs, fut, mask = ds[0]
    print("obs:", obs.shape, "fut:", fut.shape, "mask:", mask.shape, "n active:", mask.sum().item())
