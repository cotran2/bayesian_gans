import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D



def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

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


class Distribution():
    """
    Distribution class for type 1 or type 2
    """
    def __init__(self, size = 100, sigma_1 = 0.1, sigma_2 = 0.1, mu_1 = 1, mu_2 = 1, int_start = -1,int_end = 1):
        self.size = int(size)
        self.sigma_1 = float(sigma_1)
        self.sigma_2 = float(sigma_2)
        self.mu_1 = float(mu_1)
        self.mu_2 = float(mu_2)
        self.start = float(int_start)
        self.end = float(int_end)
    def distribution_1(self, alpha = 1):
        x_1 = np.linspace(self.start, self.end, self.size)
        x_2 = np.linspace(self.start, self.end, self.size)

        x_1, x_2 = np.meshgrid(x_1, x_2)
        z = (1 / (2 * np.pi * self.sigma_1 * self.sigma_2)) * np.exp(-(np.sqrt(x_1 ** 2 + x_2 ** 2)- self.mu_1
                                                                       ) ** 2 / (2 * self.sigma_1 ** 2)
                                                                      - (x_2 - self.mu_2) ** 2 / (2 * self.sigma_2 ** 2))

        return x_1,x_2,z
    def distribution_2(self, alpha = 1):
        x_1 = np.linspace(self.start, self.end, self.size)
        x_2 = np.linspace(self.start, self.end, self.size)
        x_1, x_2 = np.meshgrid(x_1, x_2)
        sigma_1 = 1/10
        sigma_2 = np.sqrt(10)
        z = (1 / (2 * np.pi * sigma_1 * sigma_2))*np.exp(-(x_2 - x_1**2) ** 2 /(2*sigma_1**2) -
                                                          (x_1 - 1) ** 2 / (2* sigma_2 ** 2))

        return x_1, x_2, z
    def visualize(self, type = 1):
        if type == 1:
            x_1,x_2,z = self.distribution_1()
        elif type == 2:
            x_1, x_2, z = self.distribution_2()
        plt.contourf(x_1, x_2, z, cmap='Blues')
        plt.colorbar()
        plt.show()
if __name__ == "__main__":
    d = Distribution()
    d.visualize(type = 2)