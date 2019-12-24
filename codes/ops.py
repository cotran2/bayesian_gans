from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from tensorflow import optimizers
from tensorflow import reduce_mean
from tensorflow.python.keras import layers


class Conv2D(layers.Layer):
    def __init__(self, filters, kernel_size=4, strides=2, padding='same'):
        super(Conv2D, self).__init__()
        self.conv_op = layers.Conv2D(filters=filters,
                                     kernel_size=kernel_size,
                                     strides=strides,
                                     padding=padding,
                                     use_bias=False,
                                     kernel_initializer='he_normal')

    def call(self, inputs, **kwargs):
        return self.conv_op(inputs)


class UpConv2D(layers.Layer):
    def __init__(self, filters, kernel_size=4, strides=2, padding='same'):
        super(UpConv2D, self).__init__()
        self.up_conv_op = layers.Conv2DTranspose(filters=filters,
                                                 kernel_size=kernel_size,
                                                 strides=strides,
                                                 padding=padding,
                                                 use_bias=False,
                                                 kernel_initializer='he_normal')

    def call(self, inputs, **kwargs):
        return self.up_conv_op(inputs)


class BatchNorm(layers.Layer):
    def __init__(self, epsilon=1e-4, axis=-1, momentum=0.99):
        super(BatchNorm, self).__init__()
        self.batch_norm = layers.BatchNormalization(epsilon=epsilon,
                                                    axis=axis,
                                                    momentum=momentum)

    def call(self, inputs, **kwargs):
        return self.batch_norm(inputs)


class LayerNorm(layers.Layer):
    def __init__(self, epsilon=1e-4, axis=-1):
        super(LayerNorm, self).__init__()
        self.layer_norm = layers.LayerNormalization(epsilon=epsilon, axis=axis)

    def call(self, inputs, **kwargs):
        return self.layer_norm(inputs)


class LeakyRelu(layers.Layer):
    def __init__(self, alpha=0.2):
        super(LeakyRelu, self).__init__()
        self.leaky_relu = layers.LeakyReLU(alpha=alpha)

    def call(self, inputs, **kwargs):
        return self.leaky_relu(inputs)


class AdamOptWrapper(optimizers.Adam):
    def __init__(self,
                 learning_rate=1e-4,
                 beta_1=0.,
                 beta_2=0.9,
                 epsilon=1e-4,
                 amsgrad=False,
                 **kwargs):
        super(AdamOptWrapper, self).__init__(learning_rate, beta_1, beta_2, epsilon,
                                             amsgrad, **kwargs)


def d_loss_fn(f_logit, r_logit):
    f_loss = reduce_mean(f_logit)
    r_loss = reduce_mean(r_logit)
    return f_loss - r_loss


def g_loss_fn(f_logit):
    f_loss = -reduce_mean(f_logit)
    return f_loss