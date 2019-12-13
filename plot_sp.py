import numpy as np
import matplotlib.pyplot as plt

path = ""
a1 = np.load(path+"al_sp.npy", allow_pickle=True)

def fun(x, n):
    sp_l = []
    sp_mean = []
    sp_std = []
    for i in x:
        sp_l.append(len(i))

    for j in range(n):
        for ii in range(25-sp_l[j]):
            x[j].append(x[j][-1])
    for k in range(25):
        ll = []
        for m in range(n):
            ll.append(x[m][k])
        #ll = [x[0][k],x[1][k],x[2][k],x[3][k],x[4][k]]
        sp_mean.append(np.mean(ll))
        sp_std.append(np.std(ll))
    x_axis = np.linspace(0,len(sp_mean)-1,len(sp_mean))
    return sp_mean, sp_std, x_axis


m1, s1, l1 = fun(a1,5)

plt.figure()
plt.plot(l1, m1, 'r', label="")
plt.errorbar(l1, m1, ecolor="r", color='r', yerr=s1, fmt="o")
plt.legend(loc="lower right")
plt.title("")

plt.show()
