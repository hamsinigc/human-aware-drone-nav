"""
Train PPO on the PREDICTION-AUGMENTED environment — this is the "with
trajectory prediction" condition, to compare against your existing
baseline (plain CrowdNavEnv, no prediction).

Run:
    python train_rl_with_prediction.py --timesteps 1000000

This mirrors train_rl.py almost exactly (same PPO hyperparameters, same
evaluation procedure) so the comparison between "with" and "without"
prediction is fair — the only thing that changes is the environment's
observation, not the training setup.
"""

import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from prediction_augmented_env import PredictionAugmentedCrowdNavEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--n_pedestrians", type=int, default=8)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pt")
    parser.add_argument("--save_path", type=str, default="ppo_crowdnav_with_prediction")
    args = parser.parse_args()

    env_kwargs = dict(n_pedestrians=args.n_pedestrians)

    def make_env():
        return PredictionAugmentedCrowdNavEnv(
            checkpoint_path=args.checkpoint_path, **env_kwargs
        )

    train_env = make_vec_env(make_env, n_envs=args.n_envs)
    eval_env = Monitor(make_env())

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model_with_prediction/",
        log_path="./eval_logs_with_prediction/",
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
    print(f"Success rate WITH prediction: {success_rate * 100:.1f}%")
    print("Compare this against your existing baseline (ppo_crowdnav_v2.zip): 95.0%")


if __name__ == "__main__":
    main()
