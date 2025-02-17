import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from typing import Optional


class ExplicitRungeKuttaMethod:
    def __init__(self, name, a, b, c):
        # a,b,c encode Butcher array
        self.name = name
        self.a = a
        self.b = b
        self.c = c

    def stability_function(self):
        s = self.a.shape[0]
        I = np.eye(s)
        e = np.ones(s)
        return lambda z: 1 + z * np.dot(self.b, np.linalg.solve(I - z * self.a, e))

    def __repr__(self):
        ret = f"{self.name}\n"

        # write down stability function using sympy and simplify
        s = self.a.shape[0]
        I = np.eye(s)
        e = np.ones(s)
        z = sp.symbols('z')
        A_inv = sp.Matrix(I) - z * sp.Matrix(self.a)
        A_inv = A_inv.inv()
        b_vec = sp.Matrix(self.b)
        e_vec = sp.Matrix(e)
        R_z = 1 + z * (b_vec.T * A_inv * e_vec)[0]
        ret += f"Stability function R(z): \"{sp.simplify(R_z)}\"\n"

        # write down butcher's array
        ret += "Butcher's Array:\n"
        for ii in range(s):
            ret += f"{self.c[ii]:.2f} | "
            for jj in range(s):
                ret += f"{self.a[ii, jj]:.2f} "
            ret += "\n"
        ret += "-----+-----\n     | "
        for ii in range(s):
            ret += f"{self.b[ii]:.2f} "
        ret += "\n"
        return ret

    def roots(self):
        """
        Find the roots of the characteristic polynomial representation of the stability function.

        Parameters:
        A (np.ndarray): Matrix A of the Butcher array.
        b (np.ndarray): Vector b of the Butcher array.

        Returns:
        list: Roots of the characteristic polynomial.
        """
        s = self.a.shape[0]
        I = np.eye(s)
        e = np.ones(s)

        z = sp.symbols('z')
        A_inv = sp.Matrix(I) - z * sp.Matrix(self.a)
        A_inv = A_inv.inv()
        b_vec = sp.Matrix(self.b)
        e_vec = sp.Matrix(e)

        R_z = 1 + z * (b_vec.T * A_inv * e_vec)[0]
        R_poly = sp.simplify(R_z)

        roots = sp.solve(R_poly, z)
        return roots

    def stability_region(self, show=True, save_path=Optional[str]):
        """
        Plot the absolute stability region for the given Butcher array (A, b, c).

        Parameters:
        A (np.ndarray): Matrix A of the Butcher array.
        b (np.ndarray): Vector b of the Butcher array.
        title (str): Title of the plot.
        """
        R = self.stability_function()

        # Create a grid of points in the complex plane
        x = np.linspace(-5, 5, 400)
        y = np.linspace(-5, 5, 400)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y

        # Evaluate the stability function on the grid
        R_values = np.vectorize(R)(Z)

        # Plot the contour where |R(z)| = 1 and fill the interior
        plt.figure(figsize=(8, 6))
        plt.contourf(X, Y, np.abs(R_values), levels=[0, 1], colors=['blue'], alpha=0.3)
        plt.contour(X, Y, np.abs(R_values), levels=[1], colors='blue')
        plt.axhline(0, color='black', lw=0.5)
        plt.axvline(0, color='black', lw=0.5)
        plt.xlabel('Re(z)')
        plt.ylabel('Im(z)')
        plt.title(f'Absolute Stability Region of {self.name}')
        plt.grid(True)
        if save_path is not None:
            plt.savefig(save_path)
        if show:
            plt.show()




class ForwardEuler(ExplicitRungeKuttaMethod):
    def __init__(self):
        super().__init__("Forward Euler", np.array([[0]]), np.array([1]), np.array([0]))


class ExplicitMidpoint(ExplicitRungeKuttaMethod):
    def __init__(self):
        super().__init__("Explicit Midpoint", np.array([[0, 0],[1/2, 0]]), np.array([0, 1]), np.array([0, 1/2]))


class KuttasMethod(ExplicitRungeKuttaMethod):
    def __init__(self):
        super().__init__("Kutta's Method", np.array([[0, 0, 0],[1/2, 0, 0], [-1, 2, 0]]), np.array([1/6, 2/3, 1/6]), np.array([0, 1/2, 1]))


class RK4(ExplicitRungeKuttaMethod):
    def __init__(self):
        super().__init__("Standard 4-Stage, Fourth Order Runge-Kutta Method", np.array([[0, 0, 0, 0],[1/2, 0, 0, 0], [0, 1/2, 0, 0], [0, 0, 1, 0]]), np.array([1/6, 1/3, 1/3, 1/6]), np.array([0, 1/2, 1/2, 1]))


class DOPRI5(ExplicitRungeKuttaMethod):
    def __init__(self):
        super().__init__("Embedded Runge-Kutta Method: Dormand and Prince 5(4)", np.array([[0, 0, 0, 0, 0, 0, 0],[1/5, 0, 0, 0, 0, 0, 0], [3/40, 9/40, 0, 0, 0, 0, 0], [44/45, -56/15, 32/9, 0, 0, 0, 0], [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0], [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0], [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]]), np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]), np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1]))