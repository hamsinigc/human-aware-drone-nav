# Human-Aware Drone Navigation — Team Setup Guide

Attention-based trajectory prediction + PPO reinforcement learning + Gazebo/ROS2
simulation for socially-aware drone navigation through crowds.

## Repo structure

```
.
├── traj_predictor/        # Trajectory prediction Transformer (Windows/Linux, pip venv)
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   └── README.md
│
├── drone_rl/                # RL training in a fast synthetic simulator (pip venv)
│   ├── crowd_env.py
│   ├── train_rl.py
│   ├── prediction_augmented_env.py
│   ├── generate_matched_dataset.py
│   ├── train_predictor_matched.py
│   └── README.md
│
└── drone_ws/                 # ROS2/Gazebo simulation (Linux or WSL2 required)
    └── src/crowd_drone_sim/
        ├── worlds/generate_world.py
        └── scripts/
            ├── gazebo_crowd_env.py
            ├── run_policy_in_gazebo.py
            └── ...
```

## What's NOT in this repo (and how to get it)

Large/generated files are excluded via `.gitignore` to keep the repo small and clonable:

| Excluded | Why | How to get it |
|---|---|---|
| `*/venv/` | Machine-specific | Recreate with `python -m venv venv` (see each folder's README) |
| `*.pt`, `*.zip` (trained models) | Large, regenerable | Either retrain (see below) **or** ask a teammate to share via the shared drive/Git LFS (see below) |
| `traj_predictor/data/` | Large (real ETH/UCY dataset) | Download script / link in `traj_predictor/README.md` |
| `crowd_world.world` | Generated file | Regenerate: `python3 generate_world.py --n_pedestrians 6` |
| `drone_ws/build`, `install`, `log` | ROS2 build artifacts | Regenerated automatically by `colcon build` |

## Option A: Git LFS for trained models (recommended for a small team)

If you want trained `.pt`/`.zip` model checkpoints tracked in the repo itself instead of
shared out-of-band:

```bash
git lfs install
git lfs track "*.pt" "*.zip"
git add .gitattributes
```

Then remove the `*.pt` / `*.zip` lines from `.gitignore` before committing models.
Every teammate needs `git lfs install` once (see https://git-lfs.com).

## Setup — each teammate does this once

### 1. Clone

```bash
git clone <your-repo-url>
cd <repo-name>
```

### 2. Trajectory predictor (`traj_predictor/`)

```bash
cd traj_predictor
python -m venv venv
# Windows:
venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate

pip install torch numpy
python train.py --epochs 20 --data synthetic   # quick sanity check
```

### 3. RL training (`drone_rl/`)

```bash
cd drone_rl
python -m venv venv
venv\Scripts\Activate.ps1   # or source venv/bin/activate

pip install gymnasium stable-baselines3 numpy matplotlib
# CPU-only torch (smaller, avoids GPU-driver issues on shared machines):
pip install torch --index-url https://download.pytorch.org/whl/cpu

python crowd_env.py   # sanity check
```

### 4. Gazebo/ROS2 simulation (`drone_ws/`)

**Requires Linux or WSL2 (Windows) with Ubuntu 22.04 + ROS2 Humble + Gazebo 11.**
See `drone_ws/SETUP.md` for the full install walkthrough (WSL2 setup, ROS2 install,
Gazebo install, common fixes for networking/display issues).

Quick version if ROS2 + Gazebo are already installed:
```bash
mkdir -p drone_ws/src   # if not already present from clone
cd drone_ws
colcon build
source install/setup.bash

cd src/crowd_drone_sim/worlds
python3 generate_world.py --n_pedestrians 6
```

## Running things

**Train the trajectory predictor:**
```bash
cd traj_predictor
python train.py --epochs 20 --data ethucy --data_dir data/eth/train
```

**Train the baseline RL policy (no prediction):**
```bash
cd drone_rl
python train_rl.py --timesteps 1000000
```

**Train RL with trajectory-prediction integrated:**
```bash
python train_rl_with_prediction.py --checkpoint_path checkpoint_matched.pt --timesteps 1000000
```

**Run a trained policy live in Gazebo:**
```bash
# Terminal 1 (WSL/Linux):
gazebo drone_ws/src/crowd_drone_sim/worlds/crowd_world.world

# Terminal 2:
cd drone_ws/src/crowd_drone_sim/scripts
python3 run_policy_in_gazebo.py --model_path ppo_crowdnav_v2.zip --episodes 5
```

## Current best results (for reference)

| Component | Metric |
|---|---|
| Trajectory predictor (real ETH data) | ADE ≈ 0.50m, FDE ≈ 1.0m |
| RL baseline (no prediction) | 95% success, 1/20 collision |
| RL with prediction (matched data) | 90% success, 2/20 collision |
| Gazebo live demo | ~40% success (sim-to-sim gap, documented in project report) |

## Team workflow suggestions

- **Branch per feature/experiment** (e.g. `feature/concert-world`, `experiment/longer-training`) rather than committing directly to `main`.
- **Don't commit trained models casually** — they're large and change every run. Either use Git LFS (Option A above) deliberately, or share final checkpoints via a shared drive and just note the filename/results in your commit message or a shared doc.
- **Anyone touching `drone_ws/`** should pull before starting, since Gazebo world files and ROS2 scripts are easy to accidentally diverge on different machines.
