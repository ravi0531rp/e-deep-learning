import tensorflow as tf
from roof_condition.data.data_augmentations import multiple_augs, cutout_augmenatation
from roof_condition.data.data_utils import load_data


def data_generator(features, labels, batch_size= 64, aug=False, train=True):
    dataset = tf.data.Dataset.from_tensor_slices((features,labels))
    autotune = tf.data.experimental.AUTOTUNE
    dataset = dataset.map(load_data, num_parallel_calls=autotune)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
    data_augmentation = multiple_augs()
    if aug:
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y), 
                              num_parallel_calls=autotune)
        dataset = dataset.map(cutout_augmenatation, num_parallel_calls=autotune)
    if train:
        dataset = dataset.repeat()
    dataset = dataset.prefetch(autotune)
    return dataset