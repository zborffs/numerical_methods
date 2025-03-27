import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, block_diag, kron, identity, vstack, hstack
from scipy.sparse.linalg import spsolve

# Parameters
M = 100  # x-points (periodic)
N = 50   # y-points (excluding y=0)
Lx = 2 * np.pi
Ly = 2
hx = Lx / M
hy = Ly / (N + 1)

# Grid
x = np.linspace(-np.pi, np.pi, M, endpoint=False)
y = np.linspace(hy, 2, N)  # y=0 excluded (Dirichlet)

# Source term
f = np.zeros((M, N))
for i in range(M):
    if -np.pi/2 <= x[i] <= np.pi/2:
        f[i, :] = -np.cos(x[i])

# Sparse matrix setup
# X-direction: periodic circulant matrix
diag_x = [-2/hx**2 * np.ones(M), 1/hx**2 * np.ones(M), 1/hx**2 * np.ones(M)]
offsets_x = [0, 1, -1]
Dx = diags(diag_x, offsets_x, shape=(M, M)).tolil()
Dx[0, -1] = 1/hx**2  # Periodic BC
Dx[-1, 0] = 1/hx**2

# Y-direction: tridiagonal
diag_y = 1/hy**2 * np.ones(N)
offsets_y = [-1, 0, 1]
Dy = diags([diag_y, -2/hy**2 * np.ones(N), diag_y], offsets_y, shape=(N, N)).tolil()

# Modify top row for Neumann BC
Dy[-1, -2] = 2/hy**2  # Due to Neumann: du/dy=0 at y=2

# Build block matrix
A = kron(Dy, identity(M)) + kron(identity(N), Dx)

# Flatten source term
F = f.flatten()

# Solve
u = spsolve(A.tocsr(), F)

# Reshape and plot
U = u.reshape(N, M).T
X, Y = np.meshgrid(x, y, indexing='ij')

plt.contourf(X, Y, U, levels=50)
plt.colorbar(label='Temperature')
plt.xlabel('x')
plt.ylabel('y')
# plt.title('Stationary Temperature Distribution')
plt.show()

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, U, alpha=0.5)

plt.show()