import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


def _init64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


mnist = input_data.read_data_sets(
    "../data/MNIST_data/", dtype=tf.uint8, one_hot=True)

images = mnist.train.images
labels = mnist.train.labels
pixels = images.shape[1]

num_examples = mnist.train.num_examples

# 输出TFRecord文件的地址
filename = "./records/output.tfrecords"

writer = tf.python_io.TFRecordWriter(filename)
for index in range(num_examples):
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'pixels': _init64_feature(pixels),
        'label': _init64_feature(np.argmax(labels[index])),
        'image_raw': _bytes_feature(image_raw)
    }))

    writer.write(example.SerializeToString())
writer.close()
