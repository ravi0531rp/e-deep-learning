import tensorflow as tf
import tensorflow_addons as tfa


def multiple_augs():
    data_augmentation = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
      tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
      tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
      tf.keras.layers.experimental.preprocessing.RandomContrast(0.2),
    ])
    return data_augmentation

def cutout_augmenatation(image, label):
    img = tfa.image.random_cutout(image, (108,108), constant_values = 0)
    return img, label