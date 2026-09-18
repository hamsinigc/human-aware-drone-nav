"""
Dataset utilities.

Two sources supported:

1. `SyntheticCrowdDataset` — generates plausible pedestrian trajectories
   (straight-line walkers + a few "avoidance" interactions) so you can
   train/debug the model immediately with zero external data.

2. `ETHUCYDataset` — loads the standard ETH/UCY benchmark text format
   used by Social-LSTM / Social-GAN / Trajectron++ and most public repos:
       each line: frame_id  agent_id  x  y
   Download any of the standard splits (eth, hotel, univ, zara1, zara2)
   from e.g. https://github.com/StanfordASL/Trajectron-plus-plus or
   https://github.com/agrimgupta92/sgan  (data/ folder) and point
   `data_dir` at the .txt files.

Both return batches shaped (B, N, T, 2) with an agent_mask (B, N) so
scenes with a variable number of agents can be padded into one batch.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


def make_relative(traj):
    """Convert absolute (x,y) to displacement-from-first-obs-frame.
    Helps the model generalize across different absolute positions/scenes.
    Returns (traj_relative, origin) so predictions can be shifted back."""
    origin = traj[..., 0:1, :]  # first observed frame per agent
    return traj - origin, origin


class SyntheticCrowdDataset(Dataset):
    """
    Generates synthetic multi-agent scenes:
      - most agents walk in a straight line with small noise
      - some scenes include a pair of agents on a collision course who
        "swerve" around each other partway through (crude proxy for
        social interaction, enough to sanity-check social attention)
    """

    def __init__(
        self,
        n_scenes: int = 2000,
        obs_len: int = 8,
        pred_len: int = 12,
        max_agents: int = 6,
        dt: float = 0.4,
        seed: int = 0,
    ):
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.max_agents = max_agents
        self.total_len = obs_len + pred_len
        self.dt = dt
        rng = np.random.default_rng(seed)
        self.scenes = []

        for _ in range(n_scenes):
            n_agents = rng.integers(2, max_agents + 1)
            scene = np.zeros((max_agents, self.total_len, 2), dtype=np.float32)
            mask = np.zeros((max_agents,), dtype=bool)

            for a in range(n_agents):
                mask[a] = True
                start = rng.uniform(-5, 5, size=2)
                angle = rng.uniform(0, 2 * np.pi)
                speed = rng.uniform(0.5, 1.5)
                vel = np.array([np.cos(angle), np.sin(angle)]) * speed

                pos = start.copy()
                for t in range(self.total_len):
                    noise = rng.normal(0, 0.02, size=2)
                    # simple avoidance: if two agents are close, nudge apart
                    pos = pos + vel * dt + noise
                    scene[a, t] = pos

            # crude pairwise avoidance pass (post-hoc nudge)
            if n_agents >= 2:
                for t in range(self.total_len):
                    for i in range(n_agents):
                        for j in range(i + 1, n_agents):
                            diff = scene[i, t] - scene[j, t]
                            dist = np.linalg.norm(diff) + 1e-6
                            if dist < 1.0:
                                push = (diff / dist) * (1.0 - dist) * 0.5
                                scene[i, t] += push
                                scene[j, t] -= push

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


class ETHUCYDataset(Dataset):
    """
    Loads ETH/UCY-format .txt files (frame_id, agent_id, x, y) and slices
    them into fixed-length (obs_len + pred_len) windows per scene, grouping
    agents that co-occur in the same frame window.

    Usage:
        ds = ETHUCYDataset(data_dir="data/eth/train", obs_len=8, pred_len=12)
    """

    def __init__(self, data_dir: str, obs_len: int = 8, pred_len: int = 12,
                 max_agents: int = 20, skip: int = 1):
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.total_len = obs_len + pred_len
        self.max_agents = max_agents
        self.samples = []

        txt_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
        if not txt_files:
            raise FileNotFoundError(
                f"No .txt files found in {data_dir}. "
                "Expected ETH/UCY-format files (frame_id agent_id x y)."
            )

        for fp in txt_files:
            data = np.loadtxt(fp)  # columns: frame, agent_id, x, y
            frames = np.unique(data[:, 0])
            frame_data = {f: data[data[:, 0] == f] for f in frames}

            for start_idx in range(0, len(frames) - self.total_len, skip):
                window_frames = frames[start_idx : start_idx + self.total_len]
                agent_ids = None
                for f in window_frames:
                    ids = set(frame_data[f][:, 1])
                    agent_ids = ids if agent_ids is None else agent_ids & ids
                if not agent_ids:
                    continue
                agent_ids = sorted(list(agent_ids))[: self.max_agents]
                n_agents = len(agent_ids)
                if n_agents < 1:
                    continue

                scene = np.zeros((self.max_agents, self.total_len, 2), dtype=np.float32)
                mask = np.zeros((self.max_agents,), dtype=bool)
                for a_idx, aid in enumerate(agent_ids):
                    mask[a_idx] = True
                    for t_idx, f in enumerate(window_frames):
                        row = frame_data[f][frame_data[f][:, 1] == aid][0]
                        scene[a_idx, t_idx] = row[2:4]

                self.samples.append((scene, mask))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scene, mask = self.samples[idx]
        obs = scene[:, : self.obs_len]
        fut = scene[:, self.obs_len :]
        return (
            torch.from_numpy(obs.copy()),
            torch.from_numpy(fut.copy()),
            torch.from_numpy(mask.copy()),
        )


if __name__ == "__main__":
    ds = SyntheticCrowdDataset(n_scenes=10, max_agents=4)
    obs, fut, mask = ds[0]
    print("obs:", obs.shape, "fut:", fut.shape, "mask:", mask.shape)
