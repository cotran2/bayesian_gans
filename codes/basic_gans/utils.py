import tensorflow as tf
from tensorflow.keras import datasets
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate


def get_data_mnist():
    """
    get dataset
    :param dataset: one of [cifar10,cifar100,mnist]
    :return: corresponding dataset
    """
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = tf.image.per_image_standardization(x_train)
    x_test = tf.image.per_image_standardization(x_test)
    x_train = tf.reshape(x_train, (len(x_train), 28 * 28))
    x_test = tf.reshape(x_test, (len(x_test), 28 * 28))
    x_train = tf.cast(x_train,tf.float32)
    x_test = tf.cast(x_test, tf.float32)
    y_train = tf.cast(y_train, tf.int32)
    y_test = tf.cast(y_test, tf.int32)

    return x_train,y_train,x_test,y_test


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


def inverse_transform_sampling(data, n_bins, n_samples):
    hist, bin_edges = np.histogram(data, bins=n_bins, density=True)
    cum_values = np.zeros(bin_edges.shape)
    cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
    inv_cdf = interpolate.interp1d(cum_values, bin_edges)
    r = np.random.rand(n_samples)
    return inv_cdf(r)