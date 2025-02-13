"""

Use the method of undetermined coefficients to determine a0, a1, b0, b1 that make the linear two-step explicit method
consistent of as high order as possible:

u_n+1  - a0 u_n - a1 u_n-1 = h (b0 f_n + b1 f_n-1)

"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.integrate import RK45
import sympy as sym


# manually Taylor expanded each of the terms. then i collected terms and created the following linear system of eqns

# symbolically determine the coefficients that make the reqd method consistent
h = sym.Symbol('h')
A = sym.Matrix([
    [-1, -1, 0, 0],
    [0, h, -h, -h],
    [0, -h**2/2, 0, h**2],
    [0, h**3/6, 0, -h**3/2]
])
b = sym.Matrix([-1, -h, -h**2/2, -h**3/6])
coeffs = A.inv() @ b
coeffs = np.array(coeffs).astype(np.float64)
a0 = coeffs[0]
a1 = coeffs[1]
b0 = coeffs[2]
b1 = coeffs[3]
print(f"a0 = {a0}\na1 = {a1}\nb0 = {b0}\nb1 = {b1}")



# (c) Apply this method to the 2D gravity problem w/ a unit-circle solution
f = lambda t, x: np.array([x[2], x[3], -x[0] / (x[0]**2 + x[1]**2), -x[1] / (x[0]**2 + x[1]**2)])
x0 = np.array([1, 0, 0, 1])
T = 4 * np.pi
N = [20, 40, 80]
x = []

# solve the problem using RK4 first to seed the 2-step method
for Ni in N:
    h = 2 * np.pi / Ni
    t = np.linspace(0, T, 2*Ni)
    xi = np.zeros((x0.shape[0], 2*Ni))
    xi[:, 0] = x0

    # use rk45 to determine the 1
    rk45 = RK45(f, 0, x0, T, first_step=h, max_step=h)
    rk45.step()
    xi[:, 1] = rk45.y

    # integrate using the method given
    for ii in range(2, t.shape[0]):
        xi[:, ii] = a0 * xi[:, ii-1] + a1 * xi[:, ii-2] + h * (b0 * f(t[ii], xi[:, ii-1]) + b1 * f(t[ii], xi[:, ii-2]))
    x.append(xi)
    plt.plot(xi[0, :], xi[1, :])
    plt.xlim((-1.5, 1.5))
    plt.ylim((-1.5, 1.5))
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"N={Ni}, Norm @ 4pi = {np.linalg.norm(xi[:, -1])}")
    plt.savefig(f'/Users/zachbortoff/Documents/Documents - Zach’s MacBook Pro (2)/school/csmc661/numerical_methods/figures/unstable_{Ni}.png')
    plt.show()
    print(f"N = {Ni}, ||y|| = {np.linalg.norm(xi[:, -1])}")


# (d)
f = lambda t, x: np.array([x[2], x[3], -x[0] / (x[0]**2 + x[1]**2), -x[1] / (x[0]**2 + x[1]**2)])
x0 = np.array([1, 0, 0, 1])
T = 8 * np.pi
N = [20, 40, 80]
x = []

# solve the problem using RK4 first to seed the 2-step method
for Ni in N:
    h = 2 * np.pi / Ni
    t = np.linspace(0, T, 4 * Ni)
    xi = np.zeros((x0.shape[0], 4 * Ni))
    xi[:, 0] = x0

    # integrate using the method given
    for ii in range(1, t.shape[0]):
        xi[:, ii] = xi[:, ii-1] + h * f(t[ii] + h / 2, xi[:, ii-1] + h / 2 * f(t[ii], xi[:, ii-1]))
    x.append(xi)
    plt.plot(xi[0, :], xi[1, :])
    plt.xlim((-1.5, 1.5))
    plt.ylim((-1.5, 1.5))
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"N={Ni}, Norm @ 8pi = {np.linalg.norm(xi[:, -1])}")
    plt.savefig(f'/Users/zachbortoff/Documents/Documents - Zach’s MacBook Pro (2)/school/csmc661/numerical_methods/figures/stable_{Ni}.png')
    plt.show()
    print(f"N = {Ni}, ||y|| = {np.linalg.norm(xi[:, -1])}")