from gan import *
import tensorflow as tf
from utils import *
import os
from tqdm import tqdm
import numpy as np

class HyperParameters():
    epochs = 20000
    dataset = '2'
    batch_size = 1000
    noise_size = 2
    seed = 1234
    n_samples = 1000
    sampling_size = 500
    # train/ not train
    training_status = 'train'
    # save/ load
    data_status = 'load'
    # write/ not
    writing_status = 'not'
    save_interval = 200
if __name__ ==  "__main__":
    params = HyperParameters
    cwd = os.path.dirname(os.path.dirname(os.getcwd()))
    params.data_path = cwd + '/data'
    params.result_path = cwd + '/results/gan'
    params.img_path = cwd + '/results/gan/img/distribution_{}'.format(params.dataset)
    tf.random.set_seed(params.seed)
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    if not os.path.exists(params.result_path):
        os.makedirs(params.result_path)
    if not os.path.exists(params.img_path):
        os.makedirs(params.img_path)
    if params.data_status=="load":
        x_train = np.load(params.data_path+'/x_train_{}.npy'.format(params.dataset))
        y_train = np.load(params.data_path+'/y_train_{}.npy'.format(params.dataset))
    elif params.data_status=="save":
        x_train, y_train, x_test, y_test = get_data_distribution(params)
        print('Train data shape : {}'.format(x_train.shape))
        print(x_train.shape)
        np.save('x_train_{}.npy'.format(params.dataset), x_train)
        np.save('y_train_{}.npy'.format(params.dataset), y_train)
        params.noise_size = 2
    print("Loading inputs with shape: {}".format(y_train.shape))
    if params.training_status == "train":
        gan = GAN(params)
        gan.train(y_train)
