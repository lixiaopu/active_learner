import numpy as np
from active_learner.al import ActiveLearner
from stable_baselines import PPO2, DQN, TD3,TRPO,DDPG
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.td3.policies import MlpPolicy
import gym
from sklearn.gaussian_process.kernels import RBF
import gym_al
import matplotlib.pyplot as plt
# TD3 TRPO DDPG
# move average with 100
path = "/home/lixiaopu/active_learner_model/pendulum2/"
model = ActiveLearner(id_num=5, task_param_name='m', task_min=0.1, task_max=2, algorithm=TD3,
                      max_reward=-100, reward_threshold=-200, policy=MlpPolicy,need_vec_env=True)


task_params = model.get_task_params()
env = []
for i in task_params:
    env.append(gym.make('Pendulum-v0', m=i))

perf_set = []
result_set = []


for i in range(5):
    perf, result = model.run(env, 2, RBF(1, (1, 1)), RBF(1, (1, 1)), path + "al"+str(i+1)+"/", 100000, 10000,0.01)
    perf_set.append(perf)
    result_set.append(result)
    np.save(path + "al_sp.npy", perf_set)
    np.save(path + "al_rs.npy", result_set)
'''
tasks = np.linspace(0.1,10,30)
reward_set = []
for i in tasks:
    env1 = gym.make('Pendulum-v0', m=i)
    #model.learning_rate_model(path+"al1/init_model/",'')
    reward = model.evaluate_one_model_from_file(env1,path+'al1/init_model/best_model.pkl',need_render=False)
    reward_set.append(reward)
plt.figure()
plt.plot(tasks,reward_set)
plt.title("Pendulum m=1")
plt.show()

env1 = gym.make('Pendulum-v0', m=2)
model.learn_from_file(env1,path+'al1/',100000)
'''
