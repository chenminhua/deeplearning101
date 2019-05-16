import tensorflow as tf
# pip install tensorflow-datasets
import tensorflow_datasets as tfds

# tfds works in both Eager and Graph modes
tf.enable_eager_execution()

# See available datasets
print(tfds.list_builders())

# mnist = tfds.builder('mnist')
# # 默认下载到 ~/tensorflow_datasets下面
# mnist.download_and_prepare()

datasets = tfds.load("mnist")
train_dataset, test_dataset = datasets["train"], datasets["test"]
assert isinstance(train_dataset, tf.data.Dataset)

print(tfds.as_numpy(train_dataset))
