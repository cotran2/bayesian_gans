import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

class Generator(keras.Model):

    def __init__(self, random_noise_size=100, hidden_units = 128, output_units = 784):
        super().__init__(name='generator')
        # units
        self.random_noise_size = random_noise_size
        self.hidden_units = hidden_units
        self.output_units = output_units
        # layers
        self.input_layer = keras.layers.Dense(units=random_noise_size)
        self.dense_1 = keras.layers.Dense(units=self.hidden_units)
        self.leaky_1 = keras.layers.LeakyReLU(alpha=0.01)
        self.dense_2 = keras.layers.Dense(units=self.hidden_units)
        self.leaky_2 = keras.layers.LeakyReLU(alpha=0.01)
        self.dense_3 = keras.layers.Dense(units=self.hidden_units)
        self.leaky_3 = keras.layers.LeakyReLU(alpha=0.01)
        self.output_layer = keras.layers.Dense(units=self.output_units, activation="tanh")
        # optimizers
        self.opt = tf.keras.optimizers.Adam()
    def call(self, input_tensor):
        ## Definition of Forward Pass
        x = self.input_layer(input_tensor)
        x = self.dense_1(x)
        x = self.leaky_1(x)
        x = self.dense_2(x)
        x = self.leaky_2(x)
        x = self.dense_3(x)
        x = self.leaky_3(x)
        return self.output_layer(x)

    def generate_noise(self, batch_size, random_noise_size):
        if random_noise_size==2:
            return np.random.multivariate_normal(mean=[0,0],cov=np.eye(2),size=batch_size)
        else:
            return np.random.uniform(-1, 1, size=(batch_size, random_noise_size))


class Discriminator(keras.Model):
    def __init__(self, hidden_units = 128, output_units = 784):
        super().__init__(name="discriminator")
        #units
        self.hidden_units = hidden_units
        self.output_units = output_units
        # layers
        self.input_layer = keras.layers.Dense(units=self.output_units)
        self.dense_1 = keras.layers.Dense(units=self.hidden_units)
        self.leaky_1 = keras.layers.LeakyReLU(alpha=0.01)
        self.dense_2 = keras.layers.Dense(units=self.hidden_units)
        self.leaky_2 = keras.layers.LeakyReLU(alpha=0.01)
        self.dense_3 = keras.layers.Dense(units=self.hidden_units)
        self.leaky_3 = keras.layers.LeakyReLU(alpha=0.01)
        self.logits = keras.layers.Dense(units=1)  # This neuron tells us if the input is fake or real
        # Optimizers
        self.opt = tf.keras.optimizers.Adam()
    def call(self, input_tensor):
        ## Definition of Forward Pass
        x = self.input_layer(input_tensor)
        x = self.dense_1(x)
        x = self.leaky_1(x)
        x = self.leaky_2(x)
        x = self.leaky_3(x)
        x = self.leaky_3(x)
        x = self.logits(x)
        return x


def generator_objective(dx_of_gx):
    # Labels are true here because generator thinks he produces real images.
    return cross_entropy(tf.ones_like(dx_of_gx), dx_of_gx)


@tf.function()
def training_step(generator: Generator, discriminator: Discriminator, images: np.ndarray, batch_size = 100, noise_size = 10, k_loop = 1):
    for _ in range(k_loop):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            noise = generator.generate_noise(batch_size, noise_size)
            g_z = generator(noise)
            d_x_true = discriminator(images)  # Trainable?
            d_x_fake = discriminator(g_z)  # dx_of_gx

            discriminator_loss = discriminator_objective(d_x_true, d_x_fake)
            # Adjusting Gradient of Discriminator
            gradients_of_discriminator = disc_tape.gradient(discriminator_loss, discriminator.trainable_variables)
            discriminator.opt.apply_gradients(zip(gradients_of_discriminator,
                                                        discriminator.trainable_variables))  # Takes a list of gradient and variables pairs

            generator_loss = generator_objective(d_x_fake)
            # Adjusting Gradient of Generator
            gradients_of_generator = gen_tape.gradient(generator_loss, generator.trainable_variables)
            generator.opt.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))


def discriminator_objective(d_x, g_z, smoothing_factor=0.9):
    """
    d_x = real output
    g_z = fake output
    """
    real_loss = cross_entropy(tf.ones_like(d_x) * smoothing_factor,
                              d_x)  # If we feed the discriminator with real images, we assume they all are the right pictures --> Because of that label == 1
    fake_loss = cross_entropy(tf.zeros_like(g_z),
                              g_z)  # Each noise we feed in are fakes image --> Because of that labels are 0

    total_loss = real_loss + fake_loss

    return total_loss