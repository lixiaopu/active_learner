import numpy as np
from active_learner.al import ActiveLearner
from stable_baselines import PPO2, DQN, TD3, TRPO
from stable_baselines.common.policies import MlpPolicy
#from stable_baselines.td3.policies import MlpPolicy
from stable_baselines.deepq.policies import MlpPolicy
import gym
import sys
from sklearn.gaussian_process.kernels import RBF
import gym_al
import matplotlib.pyplot as plt
# policy_kwargs={'net_arch': [8, dict(pi=[16, 16])]}
path = sys.argv[1]

model = ActiveLearner(id_num=5, task_param_name='gravity', task_min=0.001, task_max=0.005, algorithm=DQN,
                      max_reward=-70, reward_threshold=-100, policy=MlpPolicy, need_vec_env=True)


task_params = model.get_task_params()
env = []
for i in task_params:
    env.append(gym.make('MountainCarAl-v2', gravity=i))

#model.initialization(env[0],100000,path)
#model.learning_rate_model(path,'')
#model.evaluate_one_model_from_file(env[0],path+'best_model.pkl',need_render=True)
'''
#model.initialization(env[0],100000,path+str(0.0025)+'/')
tasks = np.linspace(0.0001,0.01,10)
reward_set = []
for i in tasks:
    env1 = gym.make('MountainCar-v0', gravity=i)
    #model.initialization(env1,100000,path+str(i)+'/')
    reward = model.evaluate_one_model_from_file(env1,path+'0.0025/init_model/best_model.pkl',need_render=False)
    reward_set.append(reward)
plt.figure()
plt.plot(tasks,reward_set)
plt.show()
'''

perf_set = []
result_set = []

for i in range(5):
    perf, result = model.random_run(env, 2, RBF(1, (1, 1)), RBF(1, (1, 1)), path + "al"+str(i+1)+"/", 100000, 10000)
    perf_set.append(perf)
    result_set.append(result)
    np.save(path + "al_sp.npy", perf_set)
    np.save(path + "al_rs.npy", result_set)
