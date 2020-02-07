import tensorflow as tf
from tensorflow.keras import datasets
import numpy as np
from tqdm import tqdm
from codes.distributions import distributions as dist

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
    for _ in tqdm(range(params.n_samples)):
        x_train.append(np.random.multivariate_normal(mean = [0,0],
                                                     cov = np.eye(2),
                                                     size = params.sampling_size))
        if params.dataset == "1":
            y_train.append(d.metropolis_hastings(d.distribution_1,sampling_size= params.sampling_size))
        elif params.dataset == "2":
            y_train.append(d.metropolis_hastings(d.distribution_2,sampling_size= params.sampling_size))
    x_train = tf.convert_to_tensor(np.array(x_train), np.float32)
    y_train = tf.convert_to_tensor(np.array(y_train), np.float32)
    return x_train,y_train,x_test,y_test