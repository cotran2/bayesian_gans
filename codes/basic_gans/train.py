from dcgan import *
import tensorflow as tf
from utils import *
import numpy as np
import matplotlib.pyplot as plt

class HyperParameters():
    epochs = 1000
    dataset = '2'
    batch_size = 10
    noise_size = 100
    seed = 1234
    n_samples = 10
    sampling_size = 10000
    status = 'train'
if __name__ ==  "__main__":

    params = HyperParameters
    tf.random.set_seed(params.seed)
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    if params.dataset == "mnist":
        x_train, y_train, x_test, y_test = get_data_mnist()
        print('Train data shape : {}'.format(x_train.shape))
        discriminator = Discriminator()
        generator = Generator()
    else:
        if params.n_samples>=1000:
            x_train = np.load('x_train.npy')
            y_train = np.load('y_train.npy')
            np.save('x_train.npy',x_train)
            np.save('y_train.npy',y_train)
        else:
            x_train, y_train, x_test, y_test = get_data_distribution(params)
            print('Train data shape : {}'.format(x_train.shape))
            print(x_train.shape)
        discriminator = Discriminator(hidden_units = 4, output_units =2)
        generator = Generator(random_noise_size = 2, hidden_units = 4, output_units = 2)
        params.noise_size = 2
    full_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(
        8192, seed=params.seed).batch(params.batch_size)
    if params.status == "train":
        epoch = 0
        while epoch < params.epochs:
            for x,y in full_dataset:
                training_step(generator = generator,
                              discriminator = discriminator,
                              images= y,
                              noise = x,
                              batch_size = params.batch_size,
                              noise_size = params.noise_size)
            epoch += 1
            print(epoch)
        if params.dataset == "mnist":
            fake_image = generator(np.random.uniform(-1, 1, size=(1, 100)))
            plt.imshow(tf.reshape(fake_image, shape=(28, 28)), cmap="gray")
        else:
            random_sample = x
            fake_image = generator(random_sample)
            np.save('generated_sample.npy',fake_image)
            np.save('random_sample.npy',random_sample)