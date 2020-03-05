from __future__ import print_function, division

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import os
import matplotlib
import sys
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

class GAN():
    def __init__(self, params):
        self.params = params
        self.img_shape = (self.params.noise_size,)

        gen_optimizer = Adam(0.01, 0.5)
        dis_optimizer = Adam(0.01, 0.5)
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=dis_optimizer,
                                   metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=gen_optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.params.noise_size,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=gen_optimizer)

    def build_generator(self):

        noise_shape = (self.params.noise_size,)

        model = Sequential()

        model.add(Dense(8, input_shape=noise_shape))
        model.add(LeakyReLU(alpha=0.1))
        #         model.add(Dense(8))
        #         model.add(LeakyReLU(alpha=0.1))
        #         model.add(Dense(4))
        #         model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization())
        model.add(Dense(int(self.params.noise_size), activation='linear'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(8, input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, dataset):
        params = self.params
        # Load the dataset
        X_train = dataset
        X_train = np.reshape(X_train, (int(params.n_samples * params.sampling_size), 2))
        # Rescale -1 to 1
        for epoch in range(params.epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], params.batch_size)
            imgs = X_train[idx]

            noise = np.random.multivariate_normal(mean=[0, 0], cov=np.eye(2), size=(params.batch_size))
            noise = np.reshape(noise, (params.batch_size, params.noise_size))
            # noise = np.random.uniform(0,1,size = (half_batch,params.noise_size))
            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((params.batch_size, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((params.batch_size, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.multivariate_normal(mean=[0, 0], cov=np.eye(2), size=(params.batch_size))
            noise = np.reshape(noise, (params.batch_size, params.noise_size))
            # noise = np.random.uniform(0,1,size = (params.batch_size,params.noise_size))
            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * params.batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress

            # If at save interval => save generated image samples
            if epoch % params.save_interval == 0:
                self.save_imgs(epoch)
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

    def save_imgs(self,epoch):
        params = self.params
        noise = np.random.multivariate_normal(mean=[0, 0], cov=np.eye(2), size=(params.batch_size))
        print(noise.shape)
        # noise = np.random.uniform(0,1,size = (params.batch_size,params.noise_size))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.hexbin(gen_imgs[:, 0], gen_imgs[:, 1], C= gen_imgs.squeeze(), cmap='rainbow')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])
        fig.savefig(self.params.img_path+"/{}.png".format(epoch), dpi=800, bbox_inches='tight')
        plt.close(fig)
