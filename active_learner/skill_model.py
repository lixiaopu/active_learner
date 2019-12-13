import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor


def skill_model(kernel, task_params, task_range, model_params):
    """return skill model with Gaussian process regression"""
    task_params_num = len(task_params)
    task_range_num = len(task_range)
    task_params = np.array([task_params]).T
    task_range = np.array([task_range]).T
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    key = []
    new_w = []
    params_shape = []
    params_shape_ravel = []
    x = 0
    for name, param in model_params[0].items():
        key.append(name)
        v = []
        params_shape.append(model_params[0][name].shape)
        x += model_params[0][name].ravel().shape[0]
        params_shape_ravel.append(x)
        for i in range(task_params_num):
            v.append(model_params[i][name].ravel())
        for n in range(v[0].shape[0]):
            gp.fit(task_params, np.array(v).T[n])
            w_pred, sigma = gp.predict(task_range, return_std=True)
            #if n == 1:
                #plt.figure()
                #plt.plot(task_params, np.array(v).T[n],'r*')
                #plt.plot(task_range, w_pred, '-')
                #plt.show()
            new_w.append(w_pred)
    new_w = np.array(new_w).T
    new_w3 = []
    for m in range(task_range_num):

        new_w2 = []
        new_w1 = np.split(new_w[m], params_shape_ravel[:-1])
        #print(new_w1)
        for a in range(len(new_w1)):
            new_w2.append(np.array(new_w1[a]).reshape(params_shape[a]))
        new_w3.append(dict((name, param)
                           for name, param in zip(key, new_w2)))
    return new_w3



