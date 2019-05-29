import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

import tflearn.datasets.mnist as mnist

trainX, trainY, testX, testY = mnist.load_data(
    data_dir="../data/MNIST_data", one_hot=True)

trainX = trainX.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])

x = input_data(shape=[None, 28, 28, 1], name='x')

conv1 = conv_2d(x, 32, 5, activation='relu')
pool1 = max_pool_2d(conv1, 2)

conv2 = conv_2d(pool1, 64, 5, activation='relu')
pool2 = max_pool_2d(conv2, 2)

full1 = fully_connected(pool2, 500, activation='relu')  # 構建全連接層
full2 = fully_connected(full1, 10, activation='softmax')

opt = regression(full2, optimizer='sgd', learning_rate=0.01,
                 loss='categorical_crossentropy')

model = tflearn.DNN(opt, tensorboard_verbose=0)
model.fit(trainX, trainY, n_epoch=20, validation_set=(
    [testX, testY]), show_metric=True)
