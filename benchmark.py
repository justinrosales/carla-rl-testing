#!/usr/bin/env python3
# ==============================================================================
# CARLA RL Benchmark Framework
# ==============================================================================
# A centralized "one-click" benchmarking script that trains and evaluates
# multiple reinforcement learning algorithms on the CARLA gym environment.
#
# Supported algorithms (all from Stable Baselines v2):
#   DQN, A2C, ACER, PPO1, ACKTR, TRPO
#
# Usage:
#   python benchmark.py                           # Run all algorithms
#   python benchmark.py --algorithms DQN,A2C      # Run specific algorithms
#   python benchmark.py --timesteps 30000         # Custom training length
#   python benchmark.py --help                    # See all options
#
# Compatible with Nautilus CARLA deployment (headless via -RenderOffScreen).
# ==============================================================================

import argparse
import json
import os
import sys
import time
import traceback
from collections import OrderedDict
from datetime import datetime

import gym
import gym_carla
import numpy as np

# ---------------------------------------------------------------------------
# Stable Baselines v2 algorithm imports
# ---------------------------------------------------------------------------
from stable_baselines import DQN, A2C, ACER, PPO1, ACKTR, TRPO
from stable_baselines.deepq.policies import MlpPolicy as DQN_MlpPolicy
from stable_baselines.common.policies import MlpPolicy


# ==============================================================================
# Algorithm Registry
# ==============================================================================
# Each entry defines the SB2 class, its policy class, and tuned hyperparameters.
# These are designed for the CARLA discrete action space (9 actions).
# ==============================================================================

ALGORITHM_REGISTRY = OrderedDict({

    "DQN": {
        "class": DQN,
        "policy": DQN_MlpPolicy,
        "description": "Deep Q-Network with experience replay and target network",
        "hyperparams": {
            "policy_kwargs": dict(layers=[256, 256, 128]),
            "learning_rate": 1e-4,
            "buffer_size": 50000,
            "learning_starts": 500,
            "batch_size": 64,
            "exploration_fraction": 0.3,
            "exploration_final_eps": 0.05,
            "target_network_update_freq": 1000,
            "gamma": 0.99,
            "train_freq": 4,
            "double_q": True,
        },
    },

    "A2C": {
        "class": A2C,
        "policy": MlpPolicy,
        "description": "Advantage Actor-Critic (synchronous, deterministic)",
        "hyperparams": {
            "policy_kwargs": dict(net_arch=[256, 256]),
            "learning_rate": 7e-4,
            "n_steps": 5,
            "gamma": 0.99,
            "ent_coef": 0.01,
            "vf_coef": 0.25,
            "max_grad_norm": 0.5,
        },
    },

    "ACER": {
        "class": ACER,
        "policy": MlpPolicy,
        "description": "Actor-Critic with Experience Replay (off-policy A2C variant)",
        "hyperparams": {
            "policy_kwargs": dict(net_arch=[256, 256]),
            "learning_rate": 7e-4,
            "n_steps": 20,
            "gamma": 0.99,
            "replay_ratio": 4,
            "ent_coef": 0.01,
        },
    },

    "PPO1": {
        "class": PPO1,
        "policy": MlpPolicy,
        "description": "Proximal Policy Optimization (MPI-based implementation)",
        "hyperparams": {
            "policy_kwargs": dict(net_arch=[256, 256]),
            "gamma": 0.99,
            "timesteps_per_actorbatch": 256,
            "clip_param": 0.2,
            "entcoeff": 0.01,
            "optim_epochs": 4,
            "optim_batchsize": 64,
            "lam": 0.95,
        },
    },

    "ACKTR": {
        "class": ACKTR,
        "policy": MlpPolicy,
        "description": "Actor-Critic using Kronecker-Factored Trust Region",
        "hyperparams": {
            "policy_kwargs": dict(net_arch=[256, 256]),
            "learning_rate": 0.25,
            "n_steps": 20,
            "gamma": 0.99,
            "ent_coef": 0.01,
            "vf_coef": 0.25,
            "max_grad_norm": 0.5,
        },
    },

    "TRPO": {
        "class": TRPO,
        "policy": MlpPolicy,
        "description": "Trust Region Policy Optimization",
        "hyperparams": {
            "policy_kwargs": dict(net_arch=[256, 256]),
            "gamma": 0.99,
            "timesteps_per_batch": 1024,
            "max_kl": 0.01,
            "lam": 0.98,
            "entcoeff": 0.0,
            "cg_iters": 10,
            "cg_damping": 0.1,
            "vf_stepsize": 1e-3,
            "vf_iters": 5,
        },
    },

})


# ==============================================================================
# Environment Configuration
# ==============================================================================

def get_env_params(port=2000, town="Town03"):
    """Return the standard CARLA gym environment parameters."""
    return {
        "number_of_vehicles": 1,
        "number_of_walkers": 0,
        "display_size": 256,
        "max_past_step": 1,
        "dt": 0.1,
        "discrete": True,
        "discrete_acc": [-2.0, 0.0, 2.0],
        "discrete_steer": [-0.2, 0.0, 0.2],
        "continuous_accel_range": [-3.0, 3.0],
        "continuous_steer_range": [-0.3, 0.3],
        "ego_vehicle_filter": "vehicle.lincoln*",
        "port": port,
        "town": town,
        "max_time_episode": 500,
        "max_waypt": 12,
        "obs_range": 32,
        "lidar_bin": 0.125,
        "d_behind": 12,
        "out_lane_thres": 2.0,
        "desired_speed": 6,
        "max_ego_spawn_times": 200,
        "display_route": True,
    }


# ==============================================================================
# Evaluation
# ==============================================================================

def evaluate_model(model, env, n_episodes=3, max_steps_per_ep=500):
    """
    Evaluate a trained model for a fixed number of episodes.

    Returns:
        dict with keys: episode_rewards, episode_lengths, mean_reward,
                        std_reward, mean_length
    """
    episode_rewards = []
    episode_lengths = []

    for ep in range(n_episodes):
        obs = env.reset()
        total_reward = 0.0
        steps = 0

        for step in range(max_steps_per_ep):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            if done:
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"    Episode {ep + 1}/{n_episodes}: "
              f"reward={total_reward:.2f}, steps={steps}")

    return {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
    }


# ==============================================================================
# Training Pipeline
# ==============================================================================

def train_algorithm(algo_name, algo_config, env_params, timesteps,
                    eval_episodes, output_dir, tb_dir):
    """
    Train and evaluate a single algorithm.

    Returns:
        dict with training and evaluation results, or None on failure.
    """
    print(f"\n{'='*70}")
    print(f"  ALGORITHM: {algo_name}")
    print(f"  {algo_config['description']}")
    print(f"{'='*70}")

    algo_output = os.path.join(output_dir, algo_name)
    os.makedirs(algo_output, exist_ok=True)

    env = None
    try:
        # ---- Create environment ----
        print(f"  [{algo_name}] Creating CARLA environment...")
        env = gym.make("carla-v0", params=env_params)

        # ---- Instantiate model ----
        print(f"  [{algo_name}] Building model with hyperparameters:")
        for k, v in algo_config["hyperparams"].items():
            print(f"    {k}: {v}")

        model = algo_config["class"](
            algo_config["policy"],
            env,
            verbose=1,
            tensorboard_log=tb_dir,
            **algo_config["hyperparams"],
        )

        # ---- Train ----
        print(f"\n  [{algo_name}] Training for {timesteps} timesteps...")
        t_start = time.time()
        model.learn(
            total_timesteps=timesteps,
            tb_log_name=f"{algo_name}_benchmark",
        )
        train_wall_time = time.time() - t_start
        print(f"  [{algo_name}] Training complete in {train_wall_time:.1f}s")

        # ---- Save model ----
        model_path = os.path.join(algo_output, f"{algo_name.lower()}_carla")
        model.save(model_path)
        print(f"  [{algo_name}] Model saved to {model_path}")

        # ---- Evaluate ----
        print(f"  [{algo_name}] Evaluating for {eval_episodes} episodes...")
        eval_results = evaluate_model(model, env, n_episodes=eval_episodes)

        result = {
            "algorithm": algo_name,
            "description": algo_config["description"],
            "timesteps": timesteps,
            "train_wall_time_s": round(train_wall_time, 2),
            "model_path": model_path,
            "hyperparams": {k: str(v) for k, v in
                           algo_config["hyperparams"].items()},
            **eval_results,
        }

        print(f"  [{algo_name}] Mean reward: {eval_results['mean_reward']:.2f}"
              f" ± {eval_results['std_reward']:.2f}")
        return result

    except Exception as e:
        print(f"\n  [ERROR] {algo_name} failed: {e}")
        traceback.print_exc()
        return {
            "algorithm": algo_name,
            "error": str(e),
            "timesteps": timesteps,
        }

    finally:
        # Always clean up the CARLA environment
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


# ==============================================================================
# Results Output
# ==============================================================================

def print_results_table(results):
    """Print a formatted comparison table of benchmark results."""
    print(f"\n{'='*70}")
    print("  BENCHMARK RESULTS SUMMARY")
    print(f"{'='*70}\n")

    header = f"  {'Algorithm':<10} {'Mean Reward':>12} {'Std':>8} " \
             f"{'Mean Ep Len':>12} {'Train Time':>12} {'Status':>8}"
    print(header)
    print(f"  {'-'*64}")

    for r in results:
        if "error" in r:
            print(f"  {r['algorithm']:<10} {'—':>12} {'—':>8} "
                  f"{'—':>12} {'—':>12} {'FAILED':>8}")
        else:
            print(f"  {r['algorithm']:<10} "
                  f"{r['mean_reward']:>12.2f} "
                  f"{r['std_reward']:>8.2f} "
                  f"{r['mean_length']:>12.1f} "
                  f"{r['train_wall_time_s']:>10.1f}s "
                  f"{'OK':>8}")

    print(f"  {'-'*64}\n")

    # Identify best algorithm (by mean reward)
    successful = [r for r in results if "error" not in r]
    if successful:
        best = max(successful, key=lambda r: r["mean_reward"])
        print(f"  Best performing: {best['algorithm']} "
              f"(mean reward: {best['mean_reward']:.2f})\n")


def save_results(results, output_dir):
    """Save benchmark results to a JSON file."""
    results_path = os.path.join(output_dir, "results.json")
    # Convert episode_rewards/lengths lists to serialisable form
    serialisable = []
    for r in results:
        entry = dict(r)
        if "episode_rewards" in entry:
            entry["episode_rewards"] = [float(x)
                                        for x in entry["episode_rewards"]]
        if "episode_lengths" in entry:
            entry["episode_lengths"] = [int(x)
                                        for x in entry["episode_lengths"]]
        serialisable.append(entry)

    with open(results_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": serialisable,
        }, f, indent=2)

    print(f"  Results saved to {results_path}")
    return results_path


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="CARLA RL Benchmark — train & compare multiple algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py                              # all algorithms
  python benchmark.py --algorithms DQN,A2C,PPO1    # selected algorithms
  python benchmark.py --timesteps 30000 --eval-episodes 5
  python benchmark.py --port 2000 --town Town03
        """,
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        default="all",
        help="Comma-separated list of algorithms to run, or 'all' "
             f"(choices: {', '.join(ALGORITHM_REGISTRY.keys())}). "
             "Default: all",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=15000,
        help="Training timesteps per algorithm (default: 15000)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=3,
        help="Number of evaluation episodes per algorithm (default: 3)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=2000,
        help="CARLA server port (default: 2000)",
    )
    parser.add_argument(
        "--town",
        type=str,
        default="Town03",
        help="CARLA town/map name (default: Town03)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results",
        help="Directory for saved models and results (default: "
             "./benchmark_results)",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=str,
        default="./tensorboard",
        help="TensorBoard log directory (default: ./tensorboard)",
    )
    return parser.parse_args()


# ==============================================================================
# Main
# ==============================================================================

def main():
    args = parse_args()

    # ---- Resolve algorithm selection ----
    if args.algorithms.lower() == "all":
        selected = list(ALGORITHM_REGISTRY.keys())
    else:
        selected = [a.strip().upper() for a in args.algorithms.split(",")]
        invalid = [a for a in selected if a not in ALGORITHM_REGISTRY]
        if invalid:
            print(f"Error: Unknown algorithm(s): {', '.join(invalid)}")
            print(f"Available: {', '.join(ALGORITHM_REGISTRY.keys())}")
            sys.exit(1)

    # ---- Setup ----
    os.makedirs(args.output_dir, exist_ok=True)
    env_params = get_env_params(port=args.port, town=args.town)

    print("\n" + "=" * 70)
    print("  CARLA RL BENCHMARK")
    print("=" * 70)
    print(f"  Algorithms  : {', '.join(selected)}")
    print(f"  Timesteps   : {args.timesteps}")
    print(f"  Eval eps    : {args.eval_episodes}")
    print(f"  CARLA port  : {args.port}")
    print(f"  Town        : {args.town}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"  TB dir      : {args.tensorboard_dir}")
    print("=" * 70)

    # ---- Run benchmark ----
    results = []
    total_start = time.time()

    for i, algo_name in enumerate(selected, 1):
        print(f"\n  >>> Running algorithm {i}/{len(selected)}: {algo_name}")
        result = train_algorithm(
            algo_name=algo_name,
            algo_config=ALGORITHM_REGISTRY[algo_name],
            env_params=env_params,
            timesteps=args.timesteps,
            eval_episodes=args.eval_episodes,
            output_dir=args.output_dir,
            tb_dir=args.tensorboard_dir,
        )
        results.append(result)

    total_time = time.time() - total_start

    # ---- Output ----
    print_results_table(results)
    save_results(results, args.output_dir)

    successful = sum(1 for r in results if "error" not in r)
    print(f"  Benchmark complete: {successful}/{len(selected)} algorithms "
          f"succeeded in {total_time:.1f}s total.\n")


if __name__ == "__main__":
    main()
