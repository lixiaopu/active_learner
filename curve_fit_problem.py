import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from stable_baselines.results_plotter import load_results, ts2xy


def func(x, a, b, c):
    return a * np.exp(-b * x) + c


def func1(t, a, b, c):
    """define a function for learning rate model"""
    r = a * np.exp(b / t) + c
    return r


def moving_average(values, window):
    """Smooth values by doing a moving average"""
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

path = "/home/lixiaopu/active_learner_model/test/"
xdata, ydata = ts2xy(load_results(path + "init_model/"), 'timesteps')
ydata = moving_average(ydata, window=20)
xdata = xdata[len(xdata) - len(ydata):]


plt.plot(xdata, ydata, 'b-', label='data')


try:
    popt, pcov = curve_fit(func1, xdata, ydata, p0=(10, -10, 10))
except RuntimeError:
    popt, pcov = curve_fit(func1, xdata, ydata, p0=(-10, -10, 10))


plt.plot(xdata, func1(xdata, *popt), 'r-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))





plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

