import numpy as np
import pandas as pd
import scipy
from scipy import stats
import math
from scipy import stats
class your_distribution(stats.rv_continuous):
    def _pdf(self, x):
        return np.exp(-x**2/2.)/np.sqrt(2.0*np.pi)

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dsimension.

    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N
def test_2d_visualization():
    size = 100
    sigma_x = 6.
    sigma_y = 2.

    x = np.linspace(-10, 10, size)
    y = np.linspace(-10, 10, size)

    x, y = np.meshgrid(x, y)
    z = (1 / (2 * np.pi * sigma_x * sigma_y) * np.exp(-(x ** 2 / (2 * sigma_x ** 2)
                                                        + y ** 2 / (2 * sigma_y ** 2))))

    plt.contourf(x, y, z, cmap='Blues')
    plt.colorbar()
    plt.show()
def test_3d_visualization():
    # Our 2-dimensional distribution will be over variables X and Y
    N = 60
    X = np.linspace(-3, 3, N)
    Y = np.linspace(-3, 4, N)
    X, Y = np.meshgrid(X, Y)

    # Mean vector and covariance matrix
    mu = np.array([0., 1.])
    Sigma = np.array([[1., -0.5], [-0.5, 1.5]])

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    # The distribution on the variables X, Y packed into pos.
    Z = multivariate_gaussian(pos, mu, Sigma)

    # Create a surface plot and projected filled contour plot under it.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                    cmap=cm.viridis)

    cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)

    # Adjust the limits, ticks and view angle
    ax.set_zlim(-0.15, 0.2)
    ax.set_zticks(np.linspace(0, 0.2, 5))
    ax.view_init(27, -21)

    plt.show()
if __name__ == "__main__":
    distribution = your_distribution(name = "dis_1")
    print(distribution.rvs(size = 100))
