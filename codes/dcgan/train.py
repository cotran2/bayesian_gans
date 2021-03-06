from dcgan import *
import tensorflow as tf
from utils import *
import os
import datetime
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

class HyperParameters():
    epochs = 1000
    dataset = '2'
    batch_size = 100
    noise_size = 2
    seed = 1234
    n_samples = 1000
    sampling_size = 500
    # train/ not train
    training_status = 'train'
    # save/ load
    data_status = 'save'
    # write/ not
    writing_status = 'not'
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
        if params.data_status=="load":
            x_train = np.load('x_train_{}.npy'.format(params.dataset))
            y_train = np.load('y_train_{}.npy'.format(params.dataset))
        elif params.data_status=="save":
            x_train, y_train, x_test, y_test = get_data_distribution(params)
            print('Train data shape : {}'.format(x_train.shape))
            print(x_train.shape)
            np.save('x_train_{}.npy'.format(params.dataset), x_train)
            np.save('y_train_{}.npy'.format(params.dataset), y_train)
        params.noise_size = 2
        x_train = np.reshape(x_train, (params.sampling_size * params.n_samples, params.noise_size))
        y_train = np.reshape(y_train, (params.sampling_size * params.n_samples, params.noise_size))
        discriminator = Discriminator(hidden_units = 4, output_units =params.noise_size)
        generator = Generator(random_noise_size = params.noise_size, hidden_units = 4, output_units = params.noise_size)
        print("Loading inputs with shape: {}".format(x_train.shape))
    full_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(
        8192, seed=params.seed).batch(params.batch_size)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    epoch_gen_loss_avg = tf.keras.metrics.Mean()
    epoch_dis_loss_avg = tf.keras.metrics.Mean()

    if not os.path.exists(train_log_dir):
        os.makedirs(train_summary_writer)
    if params.training_status == "train":
        for epoch in tqdm(range(params.epochs)):
            for x,y in full_dataset:
                gen_loss, dis_loss = training_step(generator = generator,
                              discriminator = discriminator,
                              images= y,
                              noise = x,
                              batch_size = params.batch_size,
                              noise_size = params.noise_size)
            if params.writing_status == "write":
                with train_summary_writer.as_default():
                    tf.summary.scalar('gen_loss', epoch_gen_loss_avg.result(), step=epoch)
                    tf.summary.scalar('dis_loss', epoch_dis_loss_avg.result(), step=epoch)
        if params.dataset == "mnist":
            fake_image = generator(np.random.uniform(-1, 1, size=(1, 100)))
            plt.imshow(tf.reshape(fake_image, shape=(28, 28)), cmap="gray")
        else:
            random_sample = x
            fake_image = generator(random_sample)
            np.save('generated_sample_{}.npy'.format(params.dataset),fake_image)
            np.save('random_sample_{}.npy'.format(params.dataset),random_sample)
