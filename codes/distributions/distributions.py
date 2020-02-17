import numpy as np
import matplotlib.pyplot as plt

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

    def distribution_1(self, z, alpha = 1):
        z = np.reshape(z, [z.shape[0], 2])
        x_1, x_2 = z[:, 0], z[:, 1]
        norm = (1 / (2 * np.pi * self.sigma_1 * self.sigma_2))
        exp1 = (np.sqrt(x_1 ** 2 + x_2 ** 2)- self.mu_1) ** 2 / (2 * self.sigma_1 ** 2)
        exp2 = (x_2 - self.mu_2) ** 2 / (2 * self.sigma_2 ** 2)
        return x_1,x_2,norm*(np.exp(-exp1-exp2))
    def distribution_2(self, z, alpha = 1):
        z = np.reshape(z, [z.shape[0], 2])
        x_1, x_2 = z[:, 0], z[:, 1]
        sigma_1 = 1/10
        sigma_2 = np.sqrt(10)
        return x_1, x_2, (1 / (2 * np.pi * sigma_1 * sigma_2))*np.exp(-(x_2 - x_1**2) ** 2 /(2*sigma_1**2) -
                                                          (x_1 - 1) ** 2 / (2* sigma_2 ** 2))
    def gaussian(self, z, alpha = 1):
        z = np.reshape(z, [z.shape[0], 2])
        x_1, x_2 = z[:, 0], z[:, 1]
        sigma_1 = 1
        sigma_2 = 1

        return x_1, x_2, (1 / (2 * np.pi * sigma_1 * sigma_2)) * np.exp(-(x_1 ) ** 2 / (2 * sigma_1 ** 2) -
                                                           (x_2) ** 2 / (2 * sigma_2 ** 2))
    def visualize(self, options = 1):

        if options == 1:
            r = np.linspace(-5, 5, 1000)
            z = np.array(np.meshgrid(r, r)).transpose(1, 2, 0)
            z = np.reshape(z, [z.shape[0] * z.shape[1], -1])
            x_1,x_2,z = self.distribution_1(z)
        elif options == 2:
            r = np.linspace(-5, 5, 1000)
            z = np.array(np.meshgrid(r, r)).transpose(1, 2, 0)
            z = np.reshape(z, [z.shape[0] * z.shape[1], -1])
            x_1, x_2, z = self.distribution_2(z)
        elif options == "sampling_1":
            samples = self.metropolis_hastings(self.distribution_1)
        elif options == "sampling_2":
            samples = self.metropolis_hastings(self.distribution_2)
        if type(options) == int:
            plt.contourf(x_1, x_2, z, cmap='Blues')
            plt.colorbar()
            plt.show()
        else:
            plt.hexbin(samples[:, 0], samples[:, 1], cmap='rainbow')
            plt.gca().set_aspect('equal', adjustable='box')
            plt.xlim([-3, 3])
            plt.ylim([-3, 3])
            plt.show()

    def metropolis_hastings(self, density_function, sampling_size = 100000):
        """
        mtropolis hasting algorithms for sampling
        :param density_function:
        :param sampling_size:
        :return: samples
        """
        burning_size = 100
        sampling_size += burning_size
        x0 = np.array([[0, 0]])
        xt = x0
        samples = []
        count = 0
        while count < sampling_size:
            xt_candidate = np.array([np.random.multivariate_normal(xt[0], np.eye(2))])
            accept_prob = (density_function(xt_candidate)[-1]) / (density_function(xt)[-1])
            if np.random.uniform(0, 1) < accept_prob:
                xt = xt_candidate
            samples.append([xt[:, 0], xt[:, 1]])
            count = len(np.unique(np.array(samples), axis=0))
	    if count%100 == 0:
	        print(count)
        samples = np.unique(np.array(samples), axis=0)
        return samples[burning_size:]


if __name__ == "__main__":
    d = Distribution()
    d.visualize(options = 'sampling_1')
