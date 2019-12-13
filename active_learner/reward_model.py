import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


def reward_model(task_range, reward, kernel):
    """reward model based on the current skill model"""
    kernel = kernel
    # task_params = np.array([task_params]).T
    smooth_task_range = np.array([np.linspace(task_range[0], task_range[-1], len(task_range) * 100)]).T
    task_range = np.array([task_range]).T
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gp.fit(task_range, reward)
    r_pred, sigma1 = gp.predict(task_range, return_std=True)
    smooth_r_pred, sigma2 = gp.predict(smooth_task_range, return_std=True)
    '''
    plt.figure()
    plt.plot(task_range, reward, 'b.', label='Observations')
    plt.plot(smooth_task_range, smooth_r_pred, 'b-', label='Prediction')
    plt.title('Reward Model')
    plt.xlabel('Task Param')
    plt.ylabel('Reward')
    plt.legend(loc='lower center')
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    '''
    return r_pred, smooth_task_range, smooth_r_pred


def update_reward_model(kernel, task_range_1, r_pred, r_newpred, learning_interval, smooth_task_range, smooth_r_pred,
                        max_reward, reward_threshold):
    """predicted reward model based on the assumption that the reward across task parameters changes smoothly"""
    task_num = len(task_range_1)
    task_range = np.array([task_range_1]).T
    print('-' * 100)
    print('Evaluated reward:', r_pred)
    print('Predicted reward:', r_newpred)
    print('-' * 100)
    print('New prediction')
    r_integrate_set = []
    for i in range(task_num):
        #plt.clf()
        kernel = kernel
        # kernel = Dot(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0))**2
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
        '''
        plt.plot(smooth_task_range, smooth_r_pred, 'b-', label='Prediction')
        plt.plot(task_range, r_newpred, 'b*', label='Prediction after ' + str(learning_interval) + ' timesteps')
        plt.xlabel('Task Param')
        plt.ylabel('Reward')
        plt.tick_params(labelsize=20)
        plt.title('Predicted Reward Model')
        '''
        new_reward = [0 for i in range(task_num)]
        if abs(r_pred[i] - max_reward) < abs(max_reward - reward_threshold):
            new_reward = r_pred
        else:
            for j in range(task_num):
                if i == j:
                    new_reward[j] = r_newpred[j]
                else:
                    k = abs(j - i)
                    new_reward[j] = r_pred[j] + abs(r_newpred[j] - r_pred[j]) * np.exp(-0.5 * k ** 2)
        print("%.2f" % task_range_1[i], new_reward)
        # plt.plot(task_range,new_reward,'r*',label='')
        gp.fit(task_range, new_reward)
        r_pred1, sigma1 = gp.predict(task_range, return_std=True)
        smooth_r_pred1, sigma2 = gp.predict(smooth_task_range, return_std=True)
        # plt.ylim(0,200)
        '''
        plt.plot(task_range[i], r_newpred[i], 'y*', ms=20, label='Next Task')
        plt.plot(smooth_task_range, smooth_r_pred1, 'r-', label=u'New Prediction')
        plt.legend(loc='lower center')
        plt.pause(1)
        '''
        r_integrate = np.trapz(r_pred1)
        # r_integrate = np.trapz(smooth_r_pred1,smooth_task_range.ravel())
        r_integrate_set.append(r_integrate)
    print('next task:')
    print("env %d %.2f" % (np.argmax(r_integrate_set), task_range_1[np.argmax(r_integrate_set)]))
    return np.argmax(r_integrate_set)


