from active_learner.al import ActiveLearner
from stable_baselines import PPO2, DQN
from stable_baselines.common.policies import MlpPolicy
import gym
from sklearn.gaussian_process.kernels import RBF
import gym_al

# policy_kwargs={'net_arch': [8, dict(pi=[16, 16])]}
path = "/home/lixiaopu/active_learner_model/test/"
init_task_index = 4
model = ActiveLearner(id_num=5, task_param_name='masspole', task_min=0.1, task_max=2, algorithm=PPO2,
                      nminibatches=4, max_reward=200, reward_threshold=190, policy=MlpPolicy,
                      policy_kwargs={'net_arch': [dict(pi=[32, 32])]}, need_vec_env=True)

"""contextual environments"""
task_params = model.get_task_params()
env = []
for i in task_params:
    env.append(gym.make('CartPoleAl-v2', masspole=i))

"""active learning"""
model.run(env, 0, RBF(1, (1, 1)), RBF(1, (1, 1)), path+"1/", 50000, 10000, 0.01)


