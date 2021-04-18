import gym
import gym_px4_sitl
import numpy as np
import warnings
warnings.filterwarnings('ignore')  #  using tensorflow 1.14 and numpy 1.17 will cause some warning
import tensorflow as tf

from utils.td3_new import TD3
from utils.custom_policy import CustomPolicy
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

env = gym.make('px4_sitl_avoidance-v0')

model = TD3.load('model_pure_bc')
model.env = env

# 加入action 噪声
n_actions = env.action_space.shape[-1]
model.action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# 强化学习
model.learn(total_timesteps=100000, log_interval=1)

model.save('model_bc_rl')