import numpy as np
import matplotlib.pyplot as plt

def compute_beta(alpha0):
    beta_1 = 1/3 - alpha0/12
    beta0 = 4/3 + (2*alpha0)/3
    beta1 = 1/3 + (5*alpha0)/12
    return beta_1, beta0, beta1

def plot_ras(alphas, title):
    plt.figure()
    theta = np.linspace(0, 2*np.pi, 1000)
    z = np.exp(1j * theta)
    for alpha0 in alphas:
        beta_1, beta0, beta1 = compute_beta(alpha0)
        rho = z**2 + alpha0*z - 1 - alpha0
        sigma = beta_1 * z**2 + beta0 * z + beta1
        hlambda = rho / sigma
        plt.plot(hlambda.real, hlambda.imag, label=f'α₀={alpha0:.1f}')
    plt.xlabel('Re(hλ)')
    plt.ylabel('Im(hλ)')
    plt.title(title)
    plt.legend()
    plt.grid()

# Plot for α₀ = -1.8 to -1.1
alphas1 = np.arange(-1.8, -1.0, 0.1)
plot_ras(alphas1, 'RAS Boundaries for α₀ = -1.8 to -1.1')
plt.show()

# Plot for α₀ = -1.0 to -0.1
alphas2 = np.arange(-1.0, 0.0, 0.1)
plot_ras(alphas2, 'RAS Boundaries for α₀ = -1.0 to -0.1')
plt.show()