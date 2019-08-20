import tensorflow as tf
from sparse_logistic_regression import SparseLogisticRegression
import os
import re


def _parse_function(example_proto):
    # Parse the input tf.Example proto using the dictionary below.
    feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'time': tf.io.FixedLenFeature([], tf.int64),
        'duration': tf.io.FixedLenFeature([], tf.int64),
        'index': tf.io.VarLenFeature(tf.string),
    }
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    return parsed_features['label'], parsed_features['index']


def create_dataset(filenames):
    raw_dataset = tf.data.TFRecordDataset(filenames)
    parsed_dataset = raw_dataset.map(_parse_function)
    return parsed_dataset


def hash_function(x):
    x = int(x)
    x = x % (2 ** 16 - 1)  # depend on the memory
    return x


def get_batch(dataset, batch_size):
    labels, train_id, train_val = [], [], []
    for (label, id) in dataset.take(batch_size):
        labels.append(label.numpy())
        train_id.append(list(map(hash_function, id.values.numpy())))
        train_val.append([1.0] * len(id.values))
    labels = tf.cast(tf.reshape(labels, (-1, 1)), tf.float64)
    return train_id, train_val, labels


def get_all_filenames(path='./2019070500'):
    filenames = []
    for tfrecord_file in os.listdir(path):
        if re.match('.*tfrecord$', tfrecord_file):
            filenames.append(os.path.join(path, tfrecord_file))
    return filenames


filenames = get_all_filenames('./data')
parsed_dataset = create_dataset(filenames)
batch_size = 256
print(filenames)
slr = SparseLogisticRegression(feature_dim=2 ** 16 - 1, iteration=100)
# if 'parameters.txt' in os.listdir('.'):
#     slr.load_model()
for _ in range(1):
    train_id, train_val, labels = get_batch(parsed_dataset, batch_size)
    slr.fit(train_id, train_val, labels)
slr.save_model()

test_id, test_val, test_labels = get_batch(parsed_dataset, 1200)
pred = slr.predict(test_id, test_val)
tf.print('the accuarcy is {}'.format(tf.reduce_mean(tf.cast(tf.equal(pred, test_labels), dtype=tf.float64))))
