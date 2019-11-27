from active_learner.al import ActiveLearner
from active_learner.skill_model import skill_model
from stable_baselines import PPO2, DQN
from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.deepq.policies import MlpPolicy
import numpy as np
import gym
import multiprocessing as mp
from sklearn.gaussian_process.kernels import RBF
from stable_baselines.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
from stable_baselines import results_plotter

# policy_kwargs={'net_arch': [8, dict(pi=[16, 16])]}
path = "/home/lixiaopu/active_learner_model/0.1-2-5/"
init_task_index = 4
model = ActiveLearner(id_num=5, task_param_name='masspole', task_min=0.1, task_max=2, algorithm=PPO2,
                      nminibatches=4, max_reward=200, reward_threshold=190, policy=MlpPolicy,
                      policy_kwargs={'net_arch': [dict(pi=[32, 32])]}, need_vec_env=True)

"""contextual environments"""
task_params = model.get_task_params()
env = []
for i in task_params:
    env.append(gym.make('CartPole-v0', masspole=i))

"""active learning"""
# model.initialization(env[0],10000,path+"test3/")
# model.learning_rate_model("0.1-5-5/init_model/", '5_4')
# model.learning_rate_model(path + "init_model/", '1_4')
# model.learning_rate_model(path + "test/" + "init_model/", '1_16')
# model.learning_rate_model(path + "test2/" + "init_model/", '5_16')
# plt.show()
# model.evaluate_skill_model_from_file(env, other_path, need_render=True)

# model.random_run(env, 0, RBF(1, (1, 1)), path+"random1/", 50000, 10000)
model.run(env, 0, RBF(1, (1, 1)), path+"1/", 50000, 10000, 0.01)
# model.run(env, 4, RBF(1, (1, 1)), path+"3/", 50000, 10000, 0.01)
