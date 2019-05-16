import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot=True)

n_input = 784   # 输入层神经元个数
n_output = 10   # 输出层神经元个数
n_hidden_1 = 256  # 第一个隐藏层神经元个数
n_hidden_2 = 256  # 第二个隐藏层神经元个数
n_classes = 10
training_epochs = 15
REGULARIZATION_RATE = 0.0001

# 网络参数
weight_h1 = tf.Variable(tf.random_normal([n_input, n_hidden_1]))
weight_h2 = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))
weight_out = tf.Variable(tf.random_normal([n_hidden_2, n_output]))

biases_b1 = tf.Variable(tf.random_normal([n_hidden_1]))
biases_b2 = tf.Variable(tf.random_normal([n_hidden_2]))
biases_out = tf.Variable(tf.random_normal([n_output]))

# 构建网络
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])

l1 = tf.nn.relu(tf.matmul(x, weight_h1) + biases_b1)
l2 = tf.nn.relu(tf.matmul(l1, weight_h2) + biases_b2)
pred = tf.matmul(l2, weight_out) + biases_out

cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=pred, labels=y))

regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
regularization = regularizer(weight_h1) + regularizer(weight_h2)
loss = cross_entropy_loss + regularization
optimizer = tf.train.AdamOptimizer(
    learning_rate=0.001).minimize(loss)

init_op = tf.global_variables_initializer()

# acc
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.Session() as sess:
    sess.run(init_op)
    writer = tf.summary.FileWriter('../summary/mnist_mlp', sess.graph)

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/100)  # batch size为100
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(100)
            _, c = sess.run([optimizer, loss], feed_dict={
                            x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print('epoch:{}, cost in training set={}, acc in Validation set {}'.format(
            epoch+1, avg_cost, accuracy.eval({x: mnist.validation.images, y: mnist.validation.labels})))

    # 训练完成后在测试集上测试acc

    print("Accuracy:", accuracy.eval(
        {x: mnist.test.images, y: mnist.test.labels}))

writer.close()
