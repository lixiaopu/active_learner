# Active Learner

Active Learner is to automatically learn optimal policies for a set of task parameters.
During the learning process, Active Learner will always select the most promising task
parameter which maximizes performance improvement over entire task set.

## Installation
Active Learner requires python3 (>=3.5) 

Required Python packages:
- `Stabe-Baselines` https://stable-baselines.readthedocs.io/en/master/guide/install.html
- `gym` https://github.com/openai/gym/blob/master/README.rst
- `numpy`
- `scikit-learn`

Note: Stabe-Baselines supports Tensorflow versions from 1.8.0 to 1.14.0. Support for Tensorflow 2 API is planned.

### Install using pip
- `pip install .`

## Example

```python
from active_learner.al import ActiveLearner
from stable_baselines import PPO2, DQN
from stable_baselines.common.policies import MlpPolicy
import gym
from sklearn.gaussian_process.kernels import RBF

path = "/home/username/active_learner_model/1/"
init_task_index = 4
model = ActiveLearner(id_num=5, task_param_name='masspole', task_min=0.1, task_max=5, algorithm=PPO2,
                      nminibatches=4, max_reward=200, reward_threshold=190, policy=MlpPolicy,
                      policy_kwargs={'net_arch': [dict(pi=[32, 32])]}, need_vec_env=True)

# contextual environments
task_params = model.get_task_params()
env = []
for i in task_params:
    env.append(gym.make('CartPole-v0', masspole=i))

# active learning process
model.run(env, init_task_index, RBF(1, (1, 1)), path, 100000, 10000, 0.01)
```

## Citing the Project
