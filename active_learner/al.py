# __all__ = ['ActiveLearner', 'func', 'moving_average', 'plot_results']

import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.vec_env import DummyVecEnv
import scipy.optimize
from active_learner.reward_model import reward_model, update_reward_model
from active_learner.skill_model import skill_model


def func(x, a, b):
    return a * np.exp(b / x)


def func1(t, a, b, c):
    """define a function for learning rate model"""
    r = a * np.exp(b / t) + c
    return r


def moving_average(values, window):
    """Smooth values by doing a moving average"""
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def add_noise(params, noise_coef):
    """Mutate parameters by adding normal noise to them"""
    return dict((name, param + noise_coef * np.random.normal(size=param.shape))
                for name, param in params.items())


def evaluate(env, model, need_render=False):
    """Return mean episode_rewards (sum of episodic rewards) for the given model"""
    episode_rewards = []
    for _ in range(10):
        reward_sum = 0
        done = False
        obs = env.reset()
        while not done:
            if need_render:
                env.render()
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            reward_sum += reward
        episode_rewards.append(reward_sum)
    env.close()
    return np.mean(episode_rewards)


class ActiveLearner(object):
    """
    Active Learning algorithm
    :param id_num: (int) The total number of tasks
    :param task_param_name: (str) The name of the task variable
    :param task_min: (float) The minimum task parameter
    :param task_max: (float) The maximum task parameter
    :param algorithm: (RLModel or str) The RL model to use (PPO2, DDPG, ...)
    :param policy: (Policy or str) The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param max_reward: (int) The maximum reward that an episode can consist of
    :param reward_threshold: (float) The reward threshold before the task is considered solved
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param need_vec_env: (bool) Whether or not to use a vectorized environment
    """

    def __init__(self, id_num, task_param_name, task_min, task_max, algorithm, policy, nminibatches, max_reward,
                 reward_threshold,
                 policy_kwargs=None, need_vec_env=False):
        # policy_kwargs={'net_arch': [8, ]}
        self.init_path = ''
        self.path = ''
        self.id_num = id_num
        self.task_param_name = task_param_name
        self.task_min = task_min
        self.task_max = task_max
        self.algorithm = algorithm
        self.policy = policy
        self.need_vec_env = need_vec_env
        self.nminibatches = nminibatches
        self.max_reward = max_reward
        self.reward_threshold = reward_threshold
        self.policy_kwargs = policy_kwargs
        self.task_range = np.linspace(self.task_min, self.task_max, self.id_num)
        self.vec_env = []
        self.n_steps = 1
        self.best_mean_reward = -np.inf

    def callback(self, _locals, _globals):
        """
        Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)

        :param _locals: (dict)
        :param _globals: (dict)
        """
        # global n_steps, best_mean_reward
        # Evaluate policy training performance
        x, y = ts2xy(load_results(self.path), 'timesteps')
        # print(x[-1])
        # Print stats every 1000 calls
        if len(x) > 0 and x[-1] > 500 * self.n_steps:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print(
                "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward,
                                                                                         mean_reward))

            # New best model, you could save the agent here
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(self.path + 'best_model.pkl')
            self.n_steps += 1
        return True

    def get_task_params(self):
        """print and return all task parameters"""
        print('{:^10}'.format('env'), '{:^20}'.format('task parameter'))
        print("-" * 30)
        for i in range(self.id_num):
            print('{:^10}'.format("env_" + str(i)), '{:^20}'.format(self.task_range[i]))
        return self.task_range

    def get_vec_envs(self, env):
        """return the vectorized environments"""
        for i in range(self.id_num):
            self.vec_env.append(DummyVecEnv([lambda: env[i]]))
        return self.vec_env

    def read_parameter(self, path):
        """read the policy parameter from a file"""
        model = self.algorithm.load(path)
        model_params = model.get_parameters()
        # print(dict((name, param.shape) for name, param in model_params.items()))
        model_params = dict((key, value) for key, value in model_params.items()
                            if ("/pi" in key or "/shared" in key))
        return model_params

    def learning_rate_model(self, path, title):
        """
        plot the smoothed learning rate model and return the fitting result

        :param path: (str) the save location of the results to plot
        :param title: (str) the title of the task to plot
        """
        x, y = ts2xy(load_results(path), 'timesteps')
        y = moving_average(y, window=20)
        # Truncate x
        x = x[len(x) - len(y):]
        a, b, c = scipy.optimize.curve_fit(func1, x, y)[0]
        x_fit = np.arange(0, np.max(x) + 100000)
        y_fit = func1(x_fit, a, b, c)
        for i in range(len(y_fit)):
            if y_fit[i] > self.max_reward:
                y_fit[i] = self.max_reward
        plt.figure()
        plt.plot(x_fit, y_fit, '-', label='fit')
        plt.plot(x, y, '--', label='sample')
        plt.xlabel('Number of Timesteps')
        plt.ylabel('Rewards')
        plt.title(title + " Smoothed")
        plt.legend(loc='best')
        plt.show(block=False)
        plt.pause(3)
        plt.close()
        return x_fit, y_fit

    def initialization(self, env, total_timesteps, path):
        """
        Return a trained model and model path for the initial task
        """
        # self.path = path + "learned_model/"
        self.init_path = path + "init_model/"
        self.path = self.init_path
        self.n_steps = 1
        self.best_mean_reward = -np.inf
        os.makedirs(self.init_path, exist_ok=True)
        env = Monitor(env, self.init_path, allow_early_resets=True)
        if self.need_vec_env:
            env = DummyVecEnv([lambda: env])
        model = self.algorithm(self.policy, env, verbose=0, policy_kwargs=self.policy_kwargs,
                               nminibatches=self.nminibatches)
        model.learn(total_timesteps=total_timesteps, callback=self.callback)
        # model.save(path + 'init_model.pkl')
        model_path = self.init_path + 'best_model.pkl'
        model_params = self.read_parameter(model_path)
        env.close()
        return model_params

    def evaluate_skill_model_from_weight(self, w, env, need_render=False):
        """load and evaluate skill model"""
        result = []
        env_new = []
        if self.need_vec_env:
            env_new = self.get_vec_envs(env)
        for i in range(self.id_num):
            model = self.algorithm(self.policy, env_new[i], verbose=0, policy_kwargs=self.policy_kwargs)
            model.load_parameters(w[i], exact_match=False)
            reward = evaluate(env_new[i], model, need_render)
            print("Evaluate task %.2f: mean reward is %.2f" % (self.task_range[i], reward))
            result.append(reward)
        return result

    def evaluate_skill_model_with_noise(self, w, env, noise_coef, need_render=False):
        """load and evaluate skill model with noise from weight"""
        env_new = []
        new_w = []
        new_r = []
        if self.need_vec_env:
            env_new = self.get_vec_envs(env)
        for i in range(self.id_num):
            result = []
            model = self.algorithm(self.policy, env_new[i], verbose=0, policy_kwargs=self.policy_kwargs)
            for j in range(10):
                w0 = add_noise(w[i], noise_coef)
                model.load_parameters(w0, exact_match=False)
                reward = evaluate(env_new[i], model, need_render)
                result.append((w0, reward))
            top_candidates = sorted(result, key=lambda x: x[1], reverse=True)[:3]
            mean_params = dict(
                (name, np.stack([top_candidate[0][name] for top_candidate in top_candidates]).mean(0))
                for name in w[0].keys()
            )
            mean_reward = sum(top_candidate[1] for top_candidate in top_candidates) / 3.0
            print("Evaluate task %.2f with noise: mean reward is %.2f" % (self.task_range[i], mean_reward))
            new_w.append(mean_params)
            new_r.append(mean_reward)
        return new_w, new_r

    def learn_from_weight(self, w, env, path, learn_steps):
        """return a trained model params for one task parameter"""
        self.path = path + "learned_model/"
        os.makedirs(self.path, exist_ok=True)
        self.n_steps = 1
        self.best_mean_reward = -np.inf
        env = Monitor(env, self.path, allow_early_resets=True)
        if self.need_vec_env:
            env = DummyVecEnv([lambda: env])
        model = self.algorithm(self.policy, env, verbose=0, policy_kwargs=self.policy_kwargs)
        model.load_parameters(w, exact_match=False)
        model.set_env(env)
        model.learn(total_timesteps=learn_steps, callback=self.callback)
        os.remove(self.path + "monitor.csv")
        # model.save(path + 'learned_model_' + str(index) + '.pkl')
        # model_path = path + 'learned_model_' + str(index) + '.pkl'
        model_path = self.path + 'best_model.pkl'
        model_params = self.read_parameter(model_path)
        del model
        env.close()
        return model_params

    def run_skill_model(self, env, kernel, learning_interval, task_range_sample, model_params):
        """predict next task parameter based on the current skill model"""
        w = skill_model(kernel, task_range_sample, self.task_range, model_params)
        t, r_t = self.learning_rate_model(self.init_path, "Learning Curve")
        reward = self.evaluate_skill_model_from_weight(w, env)
        r_pred, smooth_task_range, smooth_r_pred = reward_model(self.task_range, reward, kernel)
        current_pref = sum(r_pred)
        r_newpred = []
        for i in range(self.id_num):
            if r_t[-1] < r_pred[i]:
                r_newpred.append(r_pred[i] + 0.1 * r_pred[i])
            else:
                for j in t:
                    if r_t[j] >= r_pred[i]:
                        t_pred = j + learning_interval
                        if t_pred > t[-1]:
                            r_newpred.append(r_t[-1])
                        else:
                            r_newpred.append(r_t[t_pred])
                        break
        r_newpred = np.array(r_newpred)
        r_newpred[r_newpred > self.max_reward] = self.max_reward
        plt.figure()
        plt.ion()
        index = update_reward_model(kernel, self.task_range, r_pred, r_newpred, learning_interval, smooth_task_range,
                                    smooth_r_pred, self.max_reward, self.reward_threshold)
        plt.ioff()
        plt.show(block=False)
        plt.pause(1)
        plt.close()
        return w, index, current_pref, r_pred

    def run_skill_model_with_noise(self, env, kernel, learning_interval, task_range_sample, model_params, noise_coef):
        """predict next task parameter based on the current skill model (evaluate skill model with noise)"""
        w = skill_model(kernel, task_range_sample, self.task_range, model_params)
        t, r_t = self.learning_rate_model(self.init_path, "Learning Curve")
        weight, reward = self.evaluate_skill_model_with_noise(w, env, noise_coef)
        r_pred, smooth_task_range, smooth_r_pred = reward_model(self.task_range, reward, kernel)
        current_pref = sum(r_pred)
        r_newpred = []
        for i in range(self.id_num):
            if r_t[-1] < r_pred[i]:
                r_newpred.append(r_pred[i] + 0.1 * r_pred[i])
            else:
                for j in t:
                    if r_t[j] >= r_pred[i]:
                        t_pred = j + learning_interval
                        if t_pred > t[-1]:
                            r_newpred.append(r_t[-1])
                        else:
                            r_newpred.append(r_t[t_pred])
                        break
        r_newpred = np.array(r_newpred)
        r_newpred[r_newpred > self.max_reward] = self.max_reward
        plt.figure()
        plt.ion()
        index = update_reward_model(kernel, self.task_range, r_pred, r_newpred, learning_interval, smooth_task_range,
                                    smooth_r_pred, self.max_reward, self.reward_threshold)
        plt.ioff()
        plt.show(block=False)
        plt.pause(1)
        plt.close()
        return weight, index, current_pref, r_pred

    def save_skill_model(self, w, env):
        """save skill model with weight for each task parameter"""
        env_new = []
        if self.need_vec_env:
            env_new = self.get_vec_envs(env)
        for i in range(self.id_num):
            model = self.algorithm(self.policy, env_new[i], verbose=0, policy_kwargs=self.policy_kwargs)
            model.load_parameters(w[i], exact_match=False)
            model.save(self.path + 'env_' + str(i) + '_model.pkl')

    def evaluate_skill_model_from_file(self, env, need_render=False):
        """load and evaluate skill model from file"""
        result = []
        env_new = []
        if self.need_vec_env:
            env_new = self.get_vec_envs(env)
        for i in range(self.id_num):
            model = self.algorithm.load(self.path + 'env_' + str(i) + '_model.pkl')
            reward = evaluate(env_new[i], model, need_render)
            print("Evaluate task %d (%s %.2f): mean reward of 10 episodes is %.2f" % (i, self.task_param_name,
                                                                                      self.task_range[i], reward))
            result.append(reward)
        return result

    def run(self, env, init_task_index, kernel, path, init_learning_timesteps, learning_interval, noise_coef):
        """run one active learning process"""
        perf = []
        tasks = [self.task_range[init_task_index]]
        model_params = []
        model_params1 = self.initialization(env[init_task_index], total_timesteps=init_learning_timesteps,
                                            path=path)
        model_params.append(model_params1)
        for i in range(100):
            print('---------active learning process ' + str(i + 1) + '---------')
            w, next_task_index1, current_perf, r_pred = self.run_skill_model_with_noise(env, kernel, learning_interval,
                                                                                        tasks, model_params, noise_coef)
            perf.append(current_perf)
            print(perf)
            if all(r > self.reward_threshold for r in r_pred):
                print('Congratulations! Active learning finished at process ' + str(i))
                self.save_skill_model(w, env)
                '''
                plt.figure()
                plt.title('Skill performance')
                plt.plot(perf)
                plt.xlabel('Active Learning Process')
                plt.ylabel('Skill Performance')
                plt.show()
                '''
                break
            # print('success')
            # perf.append(current_perf)
            model_params3 = self.learn_from_weight(w[next_task_index1], env[next_task_index1], path,
                                                   learning_interval)
            if self.task_range[next_task_index1] in tasks:
                index = tasks.index(self.task_range[next_task_index1])
                model_params[index] = model_params3
            else:
                tasks.append(self.task_range[next_task_index1])
                model_params.append(model_params3)

        result = self.evaluate_skill_model_from_file(env, need_render=True)
        return perf, result

    def run_without_noise(self, env, init_task_index, kernel, path, init_learning_timesteps, learning_interval):
        """run one active learning process"""
        perf = []
        tasks = [self.task_range[init_task_index]]
        model_params = []
        model_params1 = self.initialization(env[init_task_index], total_timesteps=init_learning_timesteps,
                                            path=path)
        model_params.append(model_params1)
        for i in range(100):
            print('---------active learning process ' + str(i + 1) + '---------')
            w, next_task_index1, current_perf, r_pred = self.run_skill_model(env, kernel, learning_interval,
                                                                             tasks, model_params)
            perf.append(current_perf)
            print(perf)
            if all(r > self.reward_threshold for r in r_pred):
                print('Congratulations! Active learning finished at process ' + str(i))
                self.save_skill_model(w, env)
                '''
                plt.figure()
                plt.title('Skill performance')
                plt.plot(perf)
                plt.xlabel('Active Learning Process')
                plt.ylabel('Skill Performance')
                plt.show()
                '''
                break
            # print('success')
            # perf.append(current_perf)
            model_params3 = self.learn_from_weight(w[next_task_index1], env[next_task_index1], path,
                                                   learning_interval)
            if self.task_range[next_task_index1] in tasks:
                index = tasks.index(self.task_range[next_task_index1])
                model_params[index] = model_params3
            else:
                tasks.append(self.task_range[next_task_index1])
                model_params.append(model_params3)

        result = self.evaluate_skill_model_from_file(env, need_render=True)
        return perf, result

    def random_run(self, env, init_task_index, kernel, path, init_learning_timesteps, learning_interval, noise_coef):
        """run one active learning process"""
        random_task = np.random.randint(5, size=25)
        print(random_task)
        perf = []
        tasks = [self.task_range[init_task_index]]
        model_params = []
        model_params1 = self.initialization(env[init_task_index], total_timesteps=init_learning_timesteps,
                                            path=path)
        model_params.append(model_params1)
        for i in range(25):
            print('---------active learning process ' + str(i + 1) + '---------')
            w, next_task_index1, current_perf, r_pred = self.run_skill_model_with_noise(env, kernel, learning_interval,
                                                                                        tasks, model_params, noise_coef)
            perf.append(current_perf)
            print("random next task:" + str(random_task[i]))
            print(perf)
            if all(r > self.reward_threshold for r in r_pred) or i == 24:
                print('Congratulations! Active learning finished at process ' + str(i))
                self.save_skill_model(w, env)
                '''
                plt.figure()
                plt.title('Skill performance')
                plt.plot(perf)
                plt.xlabel('Active Learning Process')
                plt.ylabel('Skill Performance')
                plt.show()
                '''
                break
            # print('success')
            # perf.append(current_perf)
            model_params3 = self.learn_from_weight(w[random_task[i]], env[random_task[i]], path,
                                                   learning_interval)
            if self.task_range[random_task[i]] in tasks:
                index = tasks.index(self.task_range[random_task[i]])
                model_params[index] = model_params3
            else:
                tasks.append(self.task_range[random_task[i]])
                model_params.append(model_params3)

        result = self.evaluate_skill_model_from_file(env, need_render=True)
        return perf, result
