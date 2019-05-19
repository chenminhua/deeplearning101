import tensorflow as tf

lstm_size = 100
# 一行代码实现lstm
lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

batch_size = 100
# 将lstm中的状态初始化为全0数组
state = lstm.zero_state(batch_size, tf.float32)

loss = 0.0
num_steps = 1000
for i in range(num_steps):
    if i > 0:
        tf.get_variable_scope().reuse_variables()
    lstm_output, state = lstm(current_input, state)
    final_output = fully_connected(lstm_output)
    loss += calc_loss(final_output, expected_output)
