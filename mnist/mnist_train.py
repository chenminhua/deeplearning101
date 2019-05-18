import mnist_inference
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot=True)


# 配置训练时参数
training_epochs = 15
REGULARIZATION_RATE = 0.0001
BATCH_SIZE = 100

# 模型保存的路径和文件名称
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "model.ckpt"


def train():

    # 构建网络，定义输入与输出
    x = tf.placeholder(tf.float32, [None, mnist_inference.n_input])
    y = tf.placeholder(tf.float32, [None, mnist_inference.n_output])
    global_step = tf.Variable(0, trainable=False)

    # # 正则化项
    # regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # regularization = regularizer(weight_h1) + regularizer(weight_h2)
    pred = mnist_inference.inference(x)
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=pred, labels=y))
    loss = cross_entropy_loss
    optimizer = tf.train.AdamOptimizer(
        learning_rate=0.001).minimize(loss, global_step=global_step)
    init_op = tf.global_variables_initializer()

    # acc
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)
        writer = tf.summary.FileWriter('../summary/mnist_mlp', sess.graph)

        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples /
                              BATCH_SIZE)  # batch size为100
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
                _, cost, step = sess.run([optimizer, loss, global_step],
                                         feed_dict={x: batch_x, y: batch_y})
                avg_cost += cost / total_batch
            print('epoch:{}, step {}, cost in training set={}, acc in Validation set {}'.format(
                epoch+1, step, avg_cost, accuracy.eval({x: mnist.validation.images, y: mnist.validation.labels})))
            saver.save(sess, os.path.join(MODEL_SAVE_PATH,
                                          MODEL_NAME), global_step=global_step)
        # 训练完成后在测试集上测试acc

        print("Accuracy:", accuracy.eval(
            {x: mnist.test.images, y: mnist.test.labels}))

    writer.close()


if __name__ == '__main__':
    train()
