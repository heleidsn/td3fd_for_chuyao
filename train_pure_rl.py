import gym
import gym_px4_sitl
import numpy as np
import warnings
warnings.filterwarnings('ignore')  #  using tensorflow 1.14 and numpy 1.17 will cause some warning
import tensorflow as tf

# from utils.td3_new import TD3
from stable_baselines.td3 import TD3
from utils.dataset import ExpertDataset
from utils.custom_policy import CustomPolicy, CustomPolicyVGGBigV3
from stable_baselines.ddpg.noise import NormalActionNoise
from stable_baselines.common.callbacks import EvalCallback

import datetime
import os

# init folders
now = datetime.datetime.now()
now_string = now.strftime('%Y_%m_%d_%H_%M_' + 'vel_gap_crash_3m')
file_path = 'logs_final/' + now_string
log_path = file_path + '/logs'
model_path = file_path + '/models'
os.makedirs(log_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

import os
os.environ['OPENAI_LOG_FORMAT'] = 'stdout,log,csv,tensorboard'
os.environ['OPENAI_LOGDIR'] = log_path + 'openai/'
from stable_baselines.logger import configure

configure()

env = gym.make('px4_sitl_avoidance-v0')

n_actions = env.action_space.shape[-1]
action_noise1 = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3(CustomPolicyVGGBigV3, env, action_noise=action_noise1)
model.tensorboard_log = log_path
model.verbose = 1
model.learning_starts = 1000

total_timesteps = 200000

# create eval_callback
eval_freq = 5000
n_eval_episodes = 10
eval_callback = EvalCallback(env, best_model_save_path= file_path + '/eval',
                        log_path= file_path + '/eval', eval_freq=eval_freq, n_eval_episodes=n_eval_episodes,
                        deterministic=True, render=False)

model.learn(total_timesteps=total_timesteps, log_interval=1, callback=eval_callback)

# model save
model_name = now_string + '_' + str(total_timesteps)
model.save(model_path + '/' + model_name)
