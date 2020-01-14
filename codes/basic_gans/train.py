from dcgan import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras




if __name__ ==  "__main__":
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    discriminator = Discriminator()
    generator = Generator()