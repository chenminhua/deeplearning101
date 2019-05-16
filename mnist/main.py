from keras.datasets import mnist
import tensorflow as tf
import sklearn.model_selection
import numpy as np
from keras.utils import np_utils

n_classes = 10
(x_train, y_train), (x_test, y_test) = mnist.load_data('mnist.npz')
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
y_train = np_utils.to_categorical(y_train, n_classes)
y_test = np_utils.to_categorical(y_test, n_classes)
x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(
    x_train, y_train, test_size=1.0/12.0)
print("training data x size {}".format(x_train.shape))
print("training data y size {}".format(y_train.shape))
print("validation data x size {}".format(x_val.shape))
print("validation data y size {}".format(y_val.shape))
print("test data x size {}".format(x_test.shape))
print("test data y size {}".format(y_test.shape))

print(y_val)


def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500
BATCH_SIZE = 100

# 基础学习率
LEARNING_RATE_BASE = 0.8
# 学习率的衰减率
LEARNING_RATE_DECAY = 0.99
# 正则系数
REGULARIZATION_RATE = 0.0001
# 训练轮数
TRAINING_STEPS = 30000
# 滑动平均衰减率
MOVING_AVERAGE_DECAY = 0.99


def train():

    # 定义存储训练轮数的变量
    global_step = tf.Variable(0, trainable=False)

    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类。
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)

    # 在所有代表神经网络参数的变量上使用滑动平均。
    # tf.trainable_variables 返回的就是GraphKeys.TRAINABLE_VARIABLES中的元素。
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算使用了滑动平均后的前向传播
    average_y = inference(x, variable_averages, weights1,
                          biases1, weights2, biases2)

    # 计算交叉熵
    # tf.nn.sparse_softmax_cross_entropy_with_logits函数的第一个参数是一个一维数组，
    # 第二个参数是一个数字标签
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1))

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy + regularization

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        55000/BATCH_SIZE,
        LEARNING_RATE_DECAY)

    train_step = tf.train.GradientDescentOptimizer(
        learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name="train")

    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        validate_feed = {x: x_val, y_: y_val}
        test_feed = {x: x_test, y_: y_test}

        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After {} training steps, validation accuracy is {}".format(
                    i, validate_acc))
            # todo
            xs, ys = next_batch(BATCH_SIZE, x_train, y_train)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        # 训练结束后在测试集上检测
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After {} training steps, test accuracy is {}".format(
            TRAINING_STEPS, test_acc))


train()
