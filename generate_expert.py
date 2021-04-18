import gym
import gym_px4_sitl
import numpy as np
import warnings
warnings.filterwarnings('ignore')  #  using tensorflow 1.14 and numpy 1.17 will cause some warning
import tensorflow as tf

from utils.td3_new import TD3                   # 加载修改后的TD3
from utils.custom_policy import CustomPolicy    # 加载自定义的模型

env = gym.make('px4_sitl_avoidance-v0')         # 自定义gym环境

model = TD3(CustomPolicy, env, verbose=1)       # 创建TD3模型

model.generate_expert_replay_buffer(total_episode_num=50)   # 生成expert demo数据 （共50episode）

model.save('model_expert', save_replay_buffer=True) # 保存模型（包括replay_buffer)
