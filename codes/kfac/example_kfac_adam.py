import tensorflow as tf
import tensorflow_datasets as tfds
import math
import kfac
from utils import *

if __name__ == "___main__":
  TRAINING_SIZE = 40000
  VALIDATION_SIZE = 10000
  TEST_SIZE = 10000
  SEED = 20190524

  num_training_steps = 7500
  batch_size = 1000
  layers = tf.keras.layers

  # We take the ceiling because we do not drop the remainder of the batch
  compute_steps_per_epoch = lambda x: int(math.ceil(1. * x / batch_size))
  steps_per_epoch = compute_steps_per_epoch(TRAINING_SIZE)
  val_steps = compute_steps_per_epoch(VALIDATION_SIZE)


  optimizer_name = 'kfac'  # 'kfac' or 'adam'

  # Best Hyperparameters from the Random Search
  if optimizer_name == 'kfac':
    init_learning_rate = 0.22721400059936694
    final_learning_rate = 1e-04
    init_damping = 0.28872127217018184
    final_damping = 1e-6
    momentum = 1 - 0.018580394981260295
    lr_decay_rate = 1 - 0.001090107322908028
    damping_decay_rate = 1 - 0.0002870880729016523
  elif optimizer_name == 'adam':
    init_learning_rate = 2.24266320779
    final_learning_rate = 1e-4
    init_epsilon = 0.183230038808
    final_epsilon = 1e-8
    momentum = 1 - 0.0296561513388
    lr_decay_rate = 1 - 0.000610416031571
    epsilon_decay_rate = 1 - 0.000212682338199
  else:
    raise ValueError('Ensure optimizer_name is kfac or adam')



  tf.reset_default_graph()
  tf.set_random_seed(SEED)

  data, info = get_input_pipeline(batch_size=batch_size,
                                  seed=SEED,
                                  repeat_validation=True,
                                  use_augmentation=True)

  model = resnet_v2(input_shape=info['input_shape'],
                    depth=20,
                    num_classes=info['num_classes'])

  loss = 'sparse_categorical_crossentropy'

  training_callbacks = [
      kfac.keras.callbacks.ExponentialDecay(hyperparameter='learning_rate',
                                            init_value=init_learning_rate,
                                            final_value=final_learning_rate,
                                            decay_rate=lr_decay_rate)
  ]

  if optimizer_name == 'kfac':
    opt = kfac.keras.optimizers.Kfac(learning_rate=init_learning_rate,
                                     damping=init_damping,
                                     model=model,
                                     loss=loss,
                                     momentum=momentum,
                                     seed=SEED)
    training_callbacks.append(kfac.keras.callbacks.ExponentialDecay(
        hyperparameter='damping',
        init_value=init_damping,
        final_value=final_damping,
        decay_rate=damping_decay_rate))

  elif optimizer_name == 'adam':
    opt = tf.keras.optimizers.Adam(learning_rate=init_learning_rate,
                                   beta_1=momentum,
                                   epsilon=init_epsilon)
    training_callbacks.append(kfac.keras.callbacks.ExponentialDecay(
        hyperparameter='epsilon',
        init_value=init_epsilon,
        final_value=final_epsilon,
        decay_rate=epsilon_decay_rate))

  else:
    raise ValueError('optimizer_name must be "adam" or "kfac"')

  model.compile(loss=loss, optimizer=opt, metrics=['acc'])
  history = model.fit(x=data['train'],
                      epochs=num_training_steps//steps_per_epoch,
                      steps_per_epoch=steps_per_epoch,
                      validation_data=data['validation'],
                      validation_steps=val_steps,
                      callbacks=training_callbacks)

