import tensorflow as tf

n_input = 784   # 输入层神经元个数
n_output = 10   # 输出层神经元个数
n_hidden_1 = 256  # 第一个隐藏层神经元个数
n_hidden_2 = 256  # 第二个隐藏层神经元个数


def inference(input_tensor):
    with tf.variable_scope('l1'):
        weight_h1 = tf.get_variable(
            "weight_h1", [n_input, n_hidden_1], initializer=tf.random_normal_initializer())
        biases_b1 = tf.get_variable(
            "biases_b1", [n_hidden_1], initializer=tf.random_normal_initializer())
        l1 = tf.nn.relu(tf.matmul(input_tensor, weight_h1) + biases_b1)

    with tf.variable_scope('l2'):
        weight_h2 = tf.get_variable(
            "weight_h2", [n_hidden_1, n_hidden_2], initializer=tf.random_normal_initializer())
        biases_b2 = tf.get_variable(
            "biases_b2", [n_hidden_2], initializer=tf.random_normal_initializer())
        l2 = tf.nn.relu(tf.matmul(l1, weight_h2) + biases_b2)

    with tf.variable_scope("output_layer"):
        weight_out = tf.get_variable(
            "weight_out", [n_hidden_2, n_output], initializer=tf.random_normal_initializer())
        biases_out = tf.get_variable(
            "biases_out", [n_output], initializer=tf.random_normal_initializer())
        pred = tf.matmul(l2, weight_out) + biases_out
    return pred
