# Attention-Based Trajectory Predictor (Phase 1 of the Drone Navigation Project)

This is the **trajectory prediction module** — the first building block of your
"Attention-Based Human Trajectory Prediction for Human-Aware Autonomous Drone
Navigation" project. It's a Transformer that predicts where pedestrians will
walk next, using two forms of attention:

- **Social attention** — agents attend to each other at every timestep
  (captures avoidance, following, group behavior).
- **Temporal attention** — a Transformer encoder-decoder over each agent's
  own time sequence (captures motion patterns like speed and turning).

## Files

| File | Purpose |
|---|---|
| `model.py` | `TrajectoryTransformer` — the model itself |
| `dataset.py` | `SyntheticCrowdDataset` (works out of the box) and `ETHUCYDataset` (real benchmark data) |
| `train.py` | Training loop, ADE/FDE metrics, checkpointing |

## Quickstart (no data download needed)

```bash
pip install torch numpy
python train.py --epochs 20 --data synthetic
```

You should see ADE (Average Displacement Error) and FDE (Final Displacement
Error) fall each epoch — this confirms the model is learning realistic
motion patterns.

## Switching to real benchmark data (ETH/UCY)

1. Download the standard splits (files are plain text: `frame_id agent_id x y`
   per line) from a public repo such as:
   - https://github.com/agrimgupta92/sgan (see `datasets/`)
   - https://github.com/StanfordASL/Trajectron-plus-plus (see `experiments/pedestrians/raw/`)
2. Point the loader at the folder:
   ```bash
   python train.py --epochs 20 --data ethucy --data_dir data/eth/train
   ```

For **Stanford Drone Dataset (SDD)** specifically (recommended since it's
aerial/drone-view, matching your final application), you'll need a small
extra parsing step since SDD's raw format differs slightly (pixel
coordinates + separate `annotations.txt` per scene). Ask me when you're
ready for that and I'll add an `SDDDataset` class the same way.

## What "good" looks like

On real ETH/UCY data with obs_len=8 (3.2s) and pred_len=12 (4.8s), published
baselines score roughly:
- Social-LSTM: ADE ≈ 0.5–1.0m, FDE ≈ 1.0–2.0m (dataset-dependent)
- Social-GAN / Transformer-based models: ADE ≈ 0.3–0.6m, FDE ≈ 0.6–1.2m

Use these as sanity-check targets once you're training on real data (the
synthetic dataset here is only for pipeline debugging — its numbers aren't
comparable).

## Next steps in the overall project (see project roadmap)

1. **(Done here)** Get this predictor training and validating on synthetic
   data, then swap in ETH/UCY or SDD.
2. **Simulator setup** — AirSim, PyBullet, or Gazebo, with a drone agent and
   simulated/scripted pedestrians.
3. **RL environment** — wrap the simulator in a Gym-style `env.step()` /
   `env.reset()` API. State = drone pose + nearby pedestrian positions +
   this model's predicted future trajectories. Action = drone
   velocity/acceleration. Reward = progress − collision − discomfort
   penalties.
4. **RL training** — PPO or SAC (Stable-Baselines3 is the easiest starting
   point) using the frozen, pretrained predictor from step 1 as a feature
   provider inside the state.
5. **Evaluation** — success rate, collision rate, ADE/FDE of predictor,
   social-compliance metrics, comparisons against a no-prediction RL
   baseline.

Ping me when you're ready for step 2 (simulator + Gym environment) or
step 3 (RL policy) and I'll build those out the same way — working code,
tested end to end.
