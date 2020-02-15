import tensorflow as tf
from tensorflow.keras import datasets
import numpy as np
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath('../../'))
from codes.distributions import distributions as dist
def distribution_1(z, alpha = 1):
    sigma_1 = 0.1
    sigma_2 = 0.1
    mu_1 = 1
    mu_2 = 1
    z = np.reshape(z, [z.shape[0], 2])
    x_1, x_2 = z[:, 0], z[:, 1]
    norm = (1 / (2 * np.pi * sigma_1 * sigma_2))
    exp1 = (np.sqrt(x_1 ** 2 + x_2 ** 2)- mu_1) ** 2 / (2 * sigma_1 ** 2)
    exp2 = (x_2 - mu_2) ** 2 / (2 * sigma_2 ** 2)
    z = norm*(np.exp(-exp1-exp2))
    return z

def distribution_2( z, alpha = 1):
    z = np.reshape(z, [z.shape[0], 2])
    x_1, x_2 = z[:, 0], z[:, 1]
    sigma_1 = 1/10
    sigma_2 = np.sqrt(10)
    z = (1 / (2 * np.pi * sigma_1 * sigma_2))*np.exp(-(x_2 - x_1**2) ** 2 /(2*sigma_1**2) -
                                                      (x_1 - 1) ** 2 / (2* sigma_2 ** 2))

    return z

def gaussian(z, alpha = 1):
    z = np.reshape(z, [z.shape[0], 2])
    x_1, x_2 = z[:, 0], z[:, 1]
    sigma_1 = 1
    sigma_2 = 1
    z = (1 / (2 * np.pi * sigma_1 * sigma_2)) * np.exp(-(x_1 ) ** 2 / (2 * sigma_1 ** 2) -
                                                       (x_2) ** 2 / (2 * sigma_2 ** 2))

    return z
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

def get_data_distribution(params):
    """
    get data set with customized distributions
    :param params:
    :return:
    """
    x_train,y_train,x_test,y_test = [],[],[],[]
    d = dist.Distribution()
    for _ in range(params.n_samples):
        x_train.append(np.random.multivariate_normal(mean=[0,0],cov=np.eye(2),size=params.sampling_size))
        if params.dataset == "1":
            func = distribution_1

        elif params.dataset == "2":
            func = distribution_2
        y_train.append(d.metropolis_hastings(func, sampling_size=params.sampling_size))
    x_train = tf.convert_to_tensor(np.array(x_train), np.float32)
    y_train = tf.convert_to_tensor(np.array(y_train), np.float32)
    return x_train,y_train,x_test,y_test