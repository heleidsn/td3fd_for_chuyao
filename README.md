# td3fd_for_chuyao
TD3fD framework

## 使用

1. `generate_expert.py`

将expert的操作记录在replay buffer中用于后续计算

1. `pretrain.py`

利用replay buffer中的数据进行监督学习

3. `bc_rl_train.py`

加载监督学习过后的网络，在环境中进行强化学习

4. `train_pure_rl.py`

只使用强化学习来训练（初始随机模型），用来和td3fd算法进行对比

## 总体思路

1. 配置好训练环境
2. 利用expert生成demo数据（包括obs，action，done等所有信息）
3. 利用expert的demo数据进行behavior cloning（也就是监督学习）
4. 加载上面得到的BC之后的模型，在环境中进行强化学习

## TD3fD部分（TODO）

前面主要介绍了一个训练的大概框架，还没有到TD3fD部分，其实TD3fD最核心的部分就是一个自定义的loss function，需要对stable baselines中的东西进行一点修改，下次再进行介绍

## 说明

以上代码为删减后，不能直接运行，只作为framework的参考，TD3fD部分可以先不实现，直接Behavior Cloning走起，最后还可以和TD3fD结果进行对比

