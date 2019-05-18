from tensorflow.examples.tutorials.mnist import input_data
from keras.models import load_model
import numpy as np
import os

save_dir = "./model/mnist_cnn"
model_name = 'cnn_mnist.h5'
model_path = os.path.join(save_dir, model_name)
mnist_model = load_model(model_path)

mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot=True)
X_test = mnist.test.images.reshape(10000, 28, 28, 1)
Y_test = mnist.test.labels

loss_and_metrics = mnist_model.evaluate(X_test, Y_test, verbose=2)
print("test loss:", loss_and_metrics[0])
print("test accuracy:", loss_and_metrics[1])
