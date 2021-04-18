import gym
import gym_px4_sitl
import numpy as np
import warnings
warnings.filterwarnings('ignore')      #  using tensorflow 1.14 and numpy 1.17 will cause some warning
import tensorflow as tf

from utils.td3_new import TD3
from utils.dataset import ExpertDataset
from utils.custom_policy import CustomPolicy


env = gym.make('px4_sitl_avoidance-v0')


model = TD3.load('model_expert')     # 加载expert model，主要是为了使用replay buffer中的数据进行监督学习
model.env = env

model.pretrain(total_timesteps=5000) # 进行BC训练（可以指定步数）

model.save('model_pure_bc')          # 保存纯BC的模型(不需要保存replay buffer)
