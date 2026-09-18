"""
run_policy_in_gazebo.py

Loads a trained PPO model and runs it LIVE against the real Gazebo
simulation — this is what actually makes the drone autonomously
navigate around the crowd for a demo.

Run this with system python3 (ROS2 sourced), with Gazebo already
running your world (fair or concert) in another terminal.

Usage:
    python3 run_policy_in_gazebo.py --model_path ppo_crowdnav_v2.zip --episodes 5

Watch the Gazebo window while this runs — the drone should visibly
move toward its goal while reacting to nearby pedestrians.
"""

import argparse
from stable_baselines3 import PPO

from gazebo_crowd_env import GazeboCrowdNavEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="ppo_crowdnav_v2.zip",
        help="Path to a trained model .zip (must have been trained on CrowdNavEnv's "
             "31-dim observation format — e.g. ppo_crowdnav.zip or ppo_crowdnav_v2.zip)",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--n_pedestrians", type=int, default=6)
    parser.add_argument(
        "--arena_size", type=float, default=10.0,
        help="Goal/start sampling range (side length). Default 10.0 matches training "
             "exactly, for best reliability. Increase (e.g. 16-18) to make goals spawn "
             "further out toward landmarks like the giant wheel — more visually varied "
             "for a demo, but distances further outside what the policy trained on may "
             "reduce success rate.",
    )
    parser.add_argument(
        "--world_bound_half", type=float, default=8.0,
        help="Hard limit the drone can never cross, regardless of goal sampling range — "
             "keeps it from ever drifting into empty ground beyond your objects/pedestrians.",
    )
    args = parser.parse_args()

    print(f"Loading model from {args.model_path} ...")
    model = PPO.load(args.model_path)

    env = GazeboCrowdNavEnv(
        n_pedestrians=args.n_pedestrians,
        arena_size=args.arena_size,
        world_bound_half=args.world_bound_half,
    )

    results = {"goal_reached": 0, "collision": 0, "timeout": 0}

    for ep in range(1, args.episodes + 1):
        print(f"\n=== Episode {ep}/{args.episodes} ===")
        obs, info = env.reset()
        print(f"Goal set to: ({env.goal[0]:.2f}, {env.goal[1]:.2f})")

        done = False
        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            print(
                f"  step {step}: dist_to_goal={info.get('dist_to_goal', 0):.2f} "
                f"nearest_ped_dist={info.get('nearest_ped_dist', 0):.2f}"
            )

        result = info.get("result", "unknown")
        results[result] = results.get(result, 0) + 1
        print(f"Episode {ep} ended: {result} (after {step} steps)")

    print("\n=== Demo summary ===")
    print(results)
    env.close()


if __name__ == "__main__":
    main()
