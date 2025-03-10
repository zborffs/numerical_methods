
import matplotlib.pyplot as plt
import numpy as np


def hamiltonian(u, v, x, y):
    return 1/2 * u**2 + 1/2 * v**2 - 1 / np.sqrt(x**2 + y**2)

def nabla_q(q):
    x = q[0]
    y = q[1]
    return np.array([x/(x**2 + y**2)**(3/2), y/(x**2 + y**2)**(3/2)])

def nabla_p(p):
    u = p[0]
    v = p[1]
    return np.array([u, v])

def sv_method(x, h):
    pn = x[0:2]
    qn = x[2:4]
    pn_12 = pn - h/2 * nabla_q(x[2:4])
    qn_1 = qn + h/2 * (nabla_p(pn_12) + nabla_p(pn_12))
    pn_1 = pn_12 - h/2 * nabla_q(qn_1)
    return np.hstack((pn_1, qn_1))



p0 = np.array([0, 1/2])
q0 = np.array([2, 0])
x0 = np.hstack([p0, q0])
a = 4/3
T = 2 * np.pi * np.sqrt(a**3)
print(f"Period: {T}")
h = 0.01 * T
n = int(round(10 * T / h))
x = np.zeros((x0.shape[0], n))
x[:, 0] = x0
H = np.zeros((n,))
H[0] = hamiltonian(x0[0], x0[1], x0[2], x0[3])
t = np.linspace(0, T, n)

for k in range(1, n):
    x[:, k] = sv_method(x[:, k-1], h)
    H[k] = hamiltonian(x[0, k], x[1, k], x[2, k], x[3, k])

e = 1/2
xmax = a * (1 + e)
xmin = -a * (1 - e)
plt.plot(x[2, :], x[3, :])
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.xlim((xmin, xmax))
plt.axis('equal')
plt.show()

plt.plot(t, H)
plt.grid()
plt.xlabel("time")
plt.ylabel("Hamiltonian")

plt.show()


