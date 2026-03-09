# This file is modified from <https://github.com/cjy1992/gym-carla.git>:
# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu)
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import gym
import gym_carla
import carla
from stable_baselines import DQN
from stable_baselines.deepq.policies import MlpPolicy

def main():
  # parameters for the gym_carla environment
  params = {
    'number_of_vehicles': 1,
    'number_of_walkers': 0,
    'display_size': 256,  # screen size of bird-eye render
    'max_past_step': 1,  # the number of past steps to draw
    'dt': 0.1,  # time interval between two frames
    'discrete': True,  # whether to use discrete control space
    'discrete_acc': [-2.0, 0.0, 2.0],  # gentler acceleration for smoother control
    'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
    'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
    'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
    'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
    'port': 2000,  # connection port
    'town': 'Town03',  # which town to simulate
    'max_time_episode': 500,  # shorter episodes for faster iteration
    'max_waypt': 12,  # maximum number of waypoints
    'obs_range': 32,  # observation range (meter)
    'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
    'd_behind': 12,  # distance behind the ego vehicle (meter)
    'out_lane_thres': 2.0,  # threshold for out of lane
    'desired_speed': 6,  # lower target speed for easier initial learning
    'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    'display_route': True,  # whether to render the desired route
  }

  # Set gym-carla environment
  env = gym.make('carla-v0', params=params)

  # Custom network architecture: wider layers for high-dimensional obs
  policy_kwargs = dict(layers=[256, 256, 128])

  # Fine-tuned DQN hyperparameters
  model = DQN(
    MlpPolicy,
    env,
    verbose=1,
    policy_kwargs=policy_kwargs,
    learning_rate=1e-4,           # slower learning for stability
    buffer_size=50000,            # replay buffer size
    learning_starts=500,          # start learning earlier (short episodes)
    batch_size=64,                # larger batches for smoother gradients
    exploration_fraction=0.3,     # explore for 30% of training
    exploration_final_eps=0.05,   # keep some randomness to avoid local optima
    target_network_update_freq=1000,  # more stable target network
    gamma=0.99,                   # discount factor
    train_freq=4,                 # update every 4 steps for stability
    double_q=True,                # reduce overestimation bias
    tensorboard_log="./tensorboard/"
  )

  # Train with more timesteps
  model.learn(total_timesteps=15000, tb_log_name="DQN_finetuned")

  # Save the trained model
  model.save("dqn_carla_finetuned")
  print("Model saved to dqn_carla_finetuned.zip")

  # Evaluate the trained model (limited steps to avoid infinite loop)
  obs = env.reset()
  for i in range(500):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    print(f"Step {i}, Reward: {rewards:.2f}")
    if dones:
      print(f"Episode ended at step {i}")
      obs = env.reset()

  print("Evaluation complete.")

if __name__ == '__main__':
  main()
