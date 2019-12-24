from wgan_gp import *

class HyperParameters:
    z_dim = 64
    epochs = 1000
    batch_size = 1
    image_size = 64
    n_critic = 5
    grad_penalty_weight = 10
    total_images = 1
    g_lr = .0001
    d_lr = .0001