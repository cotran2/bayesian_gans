from dcgan import *
import tensorflow as tf
from utils import *
import numpy as np
import matplotlib.pyplot as plt

class HyperParameters():
    epochs = 100
    dataset = 'mnist'
    batch_size = 100
    noise_size = 10
    seed = 1234

if __name__ ==  "__main__":
    params = HyperParameters

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    discriminator = Discriminator()
    generator = Generator()

    if params.dataset == "mnist":
        x_train, y_train, x_test, y_test = get_data_mnist()
        print('Train data shape : {}'.format(x_train.shape))
    else:
        x_train, y_train, x_test, y_test = get_data_mnist()
        print('Train data shape : {}'.format(x_train.shape))
        print(x_train.shape)
    full_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(
        8192, seed=params.seed).batch(params.batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(
        8192, seed=params.seed).batch(params.batch_size)

    epoch = 0
    while epoch < params.epochs:
        for x,y in full_dataset:
            training_step(generator = generator,
                          discriminator = discriminator,
                          images=x,
                          batch_size = params.batch_size,
                          noise_size = params. noise_size)
        epoch += 1
        print(epoch)
    fake_image = generator(np.random.uniform(-1, 1, size=(1, 100)))
    plt.imshow(tf.reshape(fake_image, shape=(28, 28)), cmap="gray")