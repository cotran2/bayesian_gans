from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pathlib import Path

import tensorflow as tf
from absl import flags
from tensorflow import random
from tensorflow.python.keras import layers
from tensorflow.python.keras import metrics
from tensorflow.python.keras import models

import ops
from utils import img_merge
from utils import pbar
from utils import save_image_grid



class WGAN_GP:
    def __init__(self,pararms):
        self.z_dim = pararms.z_size
        self.epochs = pararms.epochs
        self.batch_size = pararms.batch_size
        self.image_size = pararms.image_size
        self.n_critic = pararms.n_critic
        self.grad_penalty_weight = pararms.g_penalty
        self.total_images = pararms.total_num_examples
        self.g_opt = ops.AdamOptWrapper(learning_rate=pararms.g_lr)
        self.d_opt = ops.AdamOptWrapper(learning_rate=pararms.d_lr)
        self.G = self.build_generator()
        self.D = self.build_discriminator()
        self.params = pararms
        self.G.summary()
        self.D.summary()


    def train_uniform_gausian(self):
        x_real = random.normal((self.params.n_samples, 1, 1, self.z_dim))
        g_train_loss = metrics.Mean()
        d_train_loss = metrics.Mean()
        for epoch in range(self.epochs):
            for _ in range(self.n_critic):
                self.train_d(x_real)
                d_loss = self.train_d(x_real)
                d_train_loss(d_loss)
            g_loss = self.train_g()
            g_train_loss(g_loss)
            self.train_g()
            g_train_loss.reset_states()
            d_train_loss.reset_states()

    def train(self, dataset):
        z = tf.constant(random.normal((self.params.n_samples, 1, 1, self.z_dim)))
        g_train_loss = metrics.Mean()
        d_train_loss = metrics.Mean()

        for epoch in range(self.epochs):
            for batch in dataset:
                for _ in range(self.n_critic):
                    self.train_d(batch)
                    d_loss = self.train_d(batch)
                    d_train_loss(d_loss)

                g_loss = self.train_g()
                g_train_loss(g_loss)
                self.train_g()


            g_train_loss.reset_states()
            d_train_loss.reset_states()


            samples = self.generate_samples(z)
            image_grid = img_merge(samples, n_rows=8).squeeze()
            save_image_grid(image_grid, epoch + 1)
    @tf.function
    def train_g(self):
        z = random.uniform((self.batch_size, 1, 1, self.z_dim))
        with tf.GradientTape() as t:
            x_fake = self.G(z, training=True)
            fake_logits = self.D(x_fake, training=True)
            loss = ops.g_loss_fn(fake_logits)
        grad = t.gradient(loss, self.G.trainable_variables)
        self.g_opt.apply_gradients(zip(grad, self.G.trainable_variables))
        return loss

    @tf.function
    def train_d(self,x_real):
        z = random.uniform((self.batch_size, 1, 1, self.z_dim))
        with tf.GradientTape() as t:
            x_fake = self.G(z, training=True)
            fake_logits = self.D(x_fake, training=True)
            real_logits = self.D(x_real, training=True)
            cost = ops.d_loss_fn(fake_logits, real_logits)
            gp = self.gradient_penalty(partial(self.D, training=True), x_real, x_fake)
            cost += self.grad_penalty_weight * gp
        grad = t.gradient(cost, self.D.trainable_variables)
        self.d_opt.apply_gradients(zip(grad, self.D.trainable_variables))
        return cost

    def gradient_penalty(self, f, real, fake):
        alpha = random.uniform([self.batch_size, 1, 1, 1], 0., 1.)
        diff = fake - real
        inter = real + (alpha * diff)
        with tf.GradientTape() as t:
            t.watch(inter)
            pred = f(inter)
        grad = t.gradient(pred, [inter])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))
        gp = tf.reduce_mean((slopes - 1.) ** 2)
        return gp


    def build_generator(self):
        dim = self.image_size
        mult = dim // 8

        x = inputs = layers.Input((1, 1, self.z_dim))
        x = ops.UpConv2D(dim * mult, 4, 1, 'valid')(x)
        x = ops.BatchNorm()(x)
        x = layers.ReLU()(x)

        while mult > 1:
            x = ops.UpConv2D(dim * (mult // 2))(x)
            x = ops.BatchNorm()(x)
            x = layers.ReLU()(x)

            mult //= 2

        x = ops.UpConv2D(3)(x)
        x = layers.Activation('tanh')(x)
        return models.Model(inputs, x, name='Generator')

    def build_discriminator(self):
        dim = self.image_size
        mult = 1
        i = dim // 2

        x = inputs = layers.Input((dim, dim, 3))
        x = ops.Conv2D(dim)(x)
        x = ops.LeakyRelu()(x)

        while i > 4:
            x = ops.Conv2D(dim * (2 * mult))(x)
            x = ops.LayerNorm(axis=[1, 2, 3])(x)
            x = ops.LeakyRelu()(x)

            i //= 2
            mult *= 2

        x = ops.Conv2D(1, 4, 1, 'valid')(x)
        return models.Model(inputs, x, name='Discriminator')