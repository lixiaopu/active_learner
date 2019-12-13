import numpy as np
import matplotlib.pyplot as plt

path = "/home/lixiaopu/active_learner_model/pendulum_random/"

a1 = np.load(path+"al_ts.npy", allow_pickle=True)


def fun(x, n):
    x_axis = np.linspace(0,23,24)
    sp_l = []
    for i in x:
        sp_l.append(len(i))

    for j in range(n):
        for ii in range(25-sp_l[j]):
            x[j].append(x[j][-1])
    plt.figure()
    for m in range(n):
        ll = []
        for k in range(24):
            ll.append(x[m][k+1]-x[m][k])
        plt.bar(x_axis,ll,width=0.2)
        x_axis = x_axis+0.2
    plt.show()


fun(a1,1)


