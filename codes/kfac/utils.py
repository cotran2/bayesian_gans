import tensorflow as tf
import tensorflow_datasets as tfds
import math
import kfac

def _parse_fn(x):
    image, label = x['image'], x['label']
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.int32)
    image = image / 127.5 - 1
    return image, label


def _augment_image(image, crop_amount, seed=None):
    # Random Brightness, Contrast, Jpeg Quality, Hue, and Saturation did not
    # seem to work well as augmentations for our training specifications
    input_shape = image.shape.as_list()
    cropped_size = [input_shape[0] - crop_amount,
                  input_shape[1] - crop_amount,
                  input_shape[2]]
    flipped = tf.image.random_flip_left_right(image, seed)
    cropped = tf.image.random_crop(flipped, cropped_size, seed)
    return tf.image.pad_to_bounding_box(image=cropped,
                                      offset_height=crop_amount // 2,
                                      offset_width=crop_amount // 2,
                                      target_height=input_shape[0],
                                      target_width=input_shape[1])


def _get_raw_data():
  # We split the training data into training and validation ourselves for
  # hyperparameter tuning.
  training_pct = int(100.0 * TRAINING_SIZE / (TRAINING_SIZE + VALIDATION_SIZE))
  train_split = tfds.Split.TRAIN.subsplit(tfds.percent[:training_pct])
  validation_split = tfds.Split.TRAIN.subsplit(tfds.percent[training_pct:])

  train_data, info = tfds.load('cifar10:3.*.*', with_info=True, split=train_split)
  val_data = tfds.load('cifar10:3.*.*', split=validation_split)
  test_data = tfds.load('cifar10:3.*.*', split='test')

  input_shape = info.features['image'].shape
  num_classes = info.features['label'].num_classes
  info = {'input_shape': input_shape, 'num_classes': num_classes}
  return info, train_data, val_data, test_data


def get_input_pipeline(batch_size=None,
                       use_augmentation=True,
                       seed=None,
                       crop_amount=6,
                       drop_remainder=False,
                       repeat_validation=True):
  """Creates CIFAR10 Data Pipeline.

  Args:
    batch_size (int): Batch size used for training.
    use_augmentation (bool): If true, applies random horizontal flips and crops
      then pads to images.
    seed (int): Random seed used for augmentation operations.
    crop_amount (int): Number of pixels to crop from the height and width of the
      image. So, the cropped image will be [height - crop_amount, width -
      crop_amount, channels] before it is padded to restore its original size.
    drop_remainder (bool): Whether to drop the remainder of the batch. Needs to
      be true to work on TPUs.
    repeat_validation (bool): Whether to repeat the validation set. Test set is
      never repeated.

  Returns:
    A tuple with an info dict (with input_shape (tuple) and number of classes
    (int)) and data dict (train_data (tf.DatasetAdapter), validation_data,
    (tf.DatasetAdapter) and test_data (tf.DatasetAdapter))
  """
  info, train_data, val_data, test_data = _get_raw_data()

  if not batch_size:
    batch_size = max(TRAINING_SIZE, VALIDATION_SIZE, TEST_SIZE)

  train_data = train_data.map(_parse_fn).shuffle(8192, seed=seed).repeat()
  if use_augmentation:
    train_data = train_data.map(
        lambda x, y: (_augment_image(x, crop_amount, seed), y))
  train_data = train_data.batch(
      min(batch_size, TRAINING_SIZE), drop_remainder=drop_remainder)
  train_data = train_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  val_data = val_data.map(_parse_fn)
  if repeat_validation:
    val_data = val_data.repeat()
  val_data = val_data.batch(
      min(batch_size, VALIDATION_SIZE), drop_remainder=drop_remainder)
  val_data = val_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  # Don't repeat test data because it is only used once to evaluate at the end.
  test_data = test_data.map(_parse_fn)
  if repeat_validation:
    test_data = test_data.repeat()
  test_data = test_data.batch(
      min(batch_size, TEST_SIZE), drop_remainder=drop_remainder)
  test_data = test_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  data = {'train': train_data, 'validation': val_data, 'test': test_data}
  return data, info


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
  """2D Convolution-Batch Normalization-Activation stack builder.

  Based on https://keras.io/examples/cifar10_resnet/.

  Args:
    inputs (tensor): input tensor from input image or previous layer
    num_filters (int): Conv2D number of filters
    kernel_size (int): Conv2D square kernel dimensions
    strides (int): Conv2D square stride dimensions
    activation (string): activation name
    batch_normalization (bool): whether to include batch normalization
    conv_first (bool): conv-bn-activation (True) or bn-activation-conv (False)

  Returns:
    x (tensor): tensor as input to the next layer
  """
  conv = layers.Conv2D(num_filters,
                       kernel_size=kernel_size,
                       strides=strides,
                       padding='same',
                       kernel_initializer='he_normal',
                       kernel_regularizer=tf.keras.regularizers.l2(1e-4))

  x = inputs
  if conv_first:
    x = conv(x)
    if batch_normalization:
      x = layers.BatchNormalization()(x)
    if activation is not None:
      x = layers.Activation(activation)(x)
  else:
    if batch_normalization:
      x = layers.BatchNormalization()(x)
    if activation is not None:
      x = layers.Activation(activation)(x)
    x = conv(x)
  return x


def resnet_v2(input_shape, depth, num_classes=10):
  """ResNet Version 2 Model builder [b].

    Based on https://keras.io/examples/cifar10_resnet/.

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    Args:
      input_shape (tuple/list): shape of input image tensor
      depth (int): number of core convolutional layers
      num_classes (int): number of classes (CIFAR10 has 10)

    Returns:
      model (Model): Keras model instance
    """
  if (depth - 2) % 9 != 0:
    raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
  # Start model definition.
  num_filters_in = 16
  num_res_blocks = int((depth - 2) / 9)

  inputs = tf.keras.Input(shape=input_shape)
  # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
  x = resnet_layer(inputs=inputs, num_filters=num_filters_in, conv_first=True)

  # Instantiate the stack of residual units
  for stage in range(3):
    for res_block in range(num_res_blocks):
      activation = 'relu'
      batch_normalization = True
      strides = 1
      if stage == 0:
        num_filters_out = num_filters_in * 4
        if res_block == 0:  # first layer and first stage
          activation = None
          batch_normalization = False
      else:
        num_filters_out = num_filters_in * 2
        if res_block == 0:  # first layer but not first stage
          strides = 2  # downsample

      # bottleneck residual unit
      y = resnet_layer(inputs=x,
                       num_filters=num_filters_in,
                       kernel_size=1,
                       strides=strides,
                       activation=activation,
                       batch_normalization=batch_normalization,
                       conv_first=False)
      y = resnet_layer(inputs=y, num_filters=num_filters_in, conv_first=False)
      y = resnet_layer(inputs=y,
                       num_filters=num_filters_out,
                       kernel_size=1,
                       conv_first=False)
      if res_block == 0:
        # linear projection residual shortcut connection to match
        # changed dims
        x = resnet_layer(inputs=x,
                         num_filters=num_filters_out,
                         kernel_size=1,
                         strides=strides,
                         activation=None,
                         batch_normalization=False)
      x = layers.Add()([x, y])

    num_filters_in = num_filters_out

  # Add classifier on top.
  # v2 has BN-ReLU before Pooling
  x = layers.BatchNormalization()(x)
  x = layers.Activation('relu')(x)
  x = layers.AveragePooling2D(pool_size=8)(x)
  y = layers.Flatten()(x)
  outputs = layers.Dense(num_classes,
                         activation='softmax',
                         kernel_initializer='he_normal')(y)

  # Instantiate model.
  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  return model

