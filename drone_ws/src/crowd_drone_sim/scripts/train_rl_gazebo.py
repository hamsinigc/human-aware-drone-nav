"""
Train PPO on the REAL Gazebo simulation (single environment - no
parallel copies, since we only have one Gazebo instance running).

Run this with system python3 (ROS2 sourced), with Gazebo already
running your crowd_world.world in another terminal.

Start SMALL first:
    python3 train_rl_gazebo.py --timesteps 500
to confirm nothing crashes, before attempting a longer run.
"""

import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from gazebo_crowd_env import GazeboCrowdNavEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2000)
    parser.add_argument("--save_path", type=str, default="ppo_gazebo_crowdnav")
    args = parser.parse_args()

    env = Monitor(GazeboCrowdNavEnv())

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        gamma=0.99,
        ent_coef=0.01,
    )

    model.learn(total_timesteps=args.timesteps)
    model.save(args.save_path)
    print(f"Training complete. Model saved to {args.save_path}.zip")

    env.close()


if __name__ == "__main__":
    main()


