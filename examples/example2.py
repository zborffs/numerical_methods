import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp
import time


f_param = lambda t, x, mu: np.array([x[1], mu * ((1 - x[0]**2) * x[1]) - x[0]])

# independent variables
mus = [int(10), int(100), int(1000)]
epsilons = [1e-6, 1e-9, 1e-12]
methods = ['RK45', 'LSODA']

# dependent variables
duration = {}

y0 = np.array([2.0, 0.0])
tmax = 1000.0
for epsilon in epsilons:
    for method in methods:
        for mu in mus:
            f = lambda t, x: f_param(t, x, mu)
            start = time.time()
            sol = solve_ivp(f, (0, tmax), y0, method=method, atol=epsilon, rtol=epsilon)
            stop = time.time()
            if method in duration:
                if mu in duration[method]:
                    v = duration[method][mu]
                    v.append((epsilon, stop - start))
                    duration[method][mu] = v
                else:
                    v = duration[method]
                    v[mu] = [(epsilon, stop - start)]
                    duration[method] = v
            else:
                duration[method] = { mu: [(epsilon, stop - start)] }
            plt.figure()
            plt.plot(sol.y[0,:], sol.y[1,:])
            plt.title(f"{method}, epsilon={epsilon}, mu={mu}")
            plt.xlabel("y1")
            plt.ylabel("y2")
            plt.grid()
            # plt.axis('equal')
            plt.savefig(f'/Users/zachbortoff/Documents/Documents - Zach’s MacBook Pro (2)/school/csmc661/numerical_methods/figures/example2/{epsilon}_{method}_{mu}.png')
            # plt.show()


plt.figure()
for method in methods:
    for mu in mus:
        eps_dur = np.array(duration[method][mu])
        eps = eps_dur[:,0]
        dur = eps_dur[:,1]
        plt.plot(-np.log(eps), np.log(dur), label=f'{method}, mu={mu}')
plt.legend()
plt.xlabel("-log(epsilon)")
plt.ylabel("log(CPU Time)")
plt.grid()
plt.savefig(f"/Users/zachbortoff/Documents/Documents - Zach’s MacBook Pro (2)/school/csmc661/numerical_methods/figures/example2/epsilon_vs_cpu_time.png")
# plt.show()

