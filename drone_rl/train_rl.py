"""
Train a PPO policy to navigate the CrowdNavEnv.

Run:
    python train_rl.py --timesteps 200000

This will take a while for good results (crowd navigation RL typically
needs several hundred thousand to a few million timesteps). Start with
a small number (e.g. 20000) just to confirm everything runs, then do a
longer run once you know it works.
"""

import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from crowd_env import CrowdNavEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--n_pedestrians", type=int, default=8)
    parser.add_argument("--save_path", type=str, default="ppo_crowdnav")
    args = parser.parse_args()

    env_kwargs = dict(n_pedestrians=args.n_pedestrians)

    train_env = make_vec_env(
        lambda: CrowdNavEnv(**env_kwargs), n_envs=args.n_envs
    )
    eval_env = Monitor(CrowdNavEnv(**env_kwargs))

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./eval_logs/",
        eval_freq=max(5000 // args.n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    model.learn(total_timesteps=args.timesteps, callback=eval_callback)
    model.save(args.save_path)
    print(f"Training complete. Model saved to {args.save_path}.zip")

    # quick evaluation summary
    print("\nRunning 20 evaluation episodes...")
    results = {"goal_reached": 0, "collision": 0, "timeout": 0}
    for ep in range(20):
        obs, _ = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
        results[info.get("result", "timeout")] += 1

    print("Evaluation over 20 episodes:", results)
    success_rate = results["goal_reached"] / 20
    print(f"Success rate: {success_rate * 100:.1f}%")


if __name__ == "__main__":
    main()
