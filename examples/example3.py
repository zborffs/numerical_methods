import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp
import time


def arenstorf(t, x_vector, mu, mu_prime):
    """
    https://www.johndcook.com/blog/2020/02/08/arenstorf-orbit/
    :param t:
    :param x_vector:
    :param mu:
    :param mu_prime:
    :return:
    """
    # x, y, xdot, ydot
    x = x_vector[0]
    y = x_vector[1]
    xdot = x_vector[2]
    ydot = x_vector[3]
    D1 = float(np.power(((x + mu)**2 + + y**2), 3/2))
    D2 = float(np.power(((x - mu_prime)**2 + y**2), 3/2))
    return np.array([
        xdot,
        ydot,
        x + 2 * ydot - mu_prime * (x + mu) / D1 - mu * (x - mu_prime) / D2,
        y - 2 * xdot - mu_prime * y / D1 - mu * y / D2
    ])


mu = 0.012277471
mu_prime = 1 - mu
epsilon = 1e-12
y0 = np.array([0.994, 0.0, 0.0, -2.001585106])
f = lambda t, x: arenstorf(t, x, mu, mu_prime)

# independent variables
tmaxes = [17.0652, 100.0]
methods = ['RK45', 'DOP853', 'Radau']

# dependent variables
duration = {}
for method in methods:
    for tmax in tmaxes:
        start = time.time()
        sol = solve_ivp(f, (0, tmax), y0, method=method, atol=epsilon, rtol=epsilon)
        stop = time.time()
        if method in duration:
            v = duration[method]
            v.append((tmax, stop - start))
            duration[method] = v
        else:
            duration[method] = [(tmax, stop - start)]
        plt.figure()
        plt.plot(sol.y[0,:], sol.y[1,:])
        plt.title(f"Arenstorf orbit T={tmax}, method={method}")
        plt.xlabel("y1")
        plt.ylabel("y2")
        # plt.xlim((-1.5, 1.5))
        # plt.ylim((-1.5, 1.5))
        plt.grid()
        plt.savefig(f'/Users/zachbortoff/Documents/Documents - Zach’s MacBook Pro (2)/school/csmc661/numerical_methods/figures/example3/{tmax}_{method}.png')
        # plt.show()


plt.figure()
for method in methods:
    tmax_dur = np.array(duration[method])
    tmax = tmax_dur[:,0]
    dur = tmax_dur[:,1]
    plt.plot(tmax, np.log(dur), label=f'{method}')
plt.legend()
plt.xlabel("Tmax")
plt.ylabel("log(CPU Time)")
plt.grid()
plt.savefig(f"/Users/zachbortoff/Documents/Documents - Zach’s MacBook Pro (2)/school/csmc661/numerical_methods/figures/example3/tmax_vs_cpu_time.png")
# plt.show()
