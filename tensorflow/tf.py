import tensorflow as tf

from numpy.random import RandomState

batch_size = 8

with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, shape=[None, 2], name="x-input")
    y_ = tf.placeholder(tf.float32, shape=[None, 1], name="y-input")

# 定义神经网络的参数以及神经网络前向传播过程
with tf.name_scope("layer-1"):
    w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
    a = tf.matmul(x, w1)

with tf.name_scope("layer-2"):
    w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))
    y = tf.matmul(a, w2)

with tf.name_scope("loss"):
    # 定义损失函数和反向传播算法
    cross_entropy = -tf.reduce_mean(
        y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 生成一个模拟数据集
rdm = RandomState()
dataset_size = 128
X = rdm.rand(dataset_size, 2)

Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    writer = tf.summary.FileWriter('./summary/tf', sess.graph)
    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        sess.run(train_step, feed_dict = {x: X[start: end], y_: Y[start: end]})
        
        if i % 1000 == 0:
            total_cross_entropy = sess.run(
                cross_entropy, feed_dict={x: X, y_: Y})
        print("After {} training steps, cross entropy on all data is {}".format(
            i, total_cross_entropy))
    print(sess.run(w1))
    print(sess.run(w2))

writer.close()
