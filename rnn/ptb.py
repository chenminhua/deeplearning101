import reader
import tensorflow as tf
DATA_PATH = "../data/ptb"
HIDDEN_SIZE = 200    # 隐藏层规模
NUM_LAYERS = 2       # LSTM 结构的层数
VOCAB_SIZE = 10000   # 词典规模

LEARNING_RATE = 1.0     # 学习速率
TRAIN_BATCH_SIZE = 20   # 训练数据batch大小
TRAIN_NUM_STEP = 35     # 训练数据截断长度

EVAL_BATCH_SIZE = 1     # 测试数据batch大小
EVAL_NUM_STEP = 1       # 测试数据截断长度
NUM_EPOCH = 2           # 使用训练数据的轮数
KEEP_PROB = 0.5         # 节点不被dropout的概率
MAX_GRAD_NORM = 5       # 用于控制梯度膨胀的参数

train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)
print(len(train_data))
print(train_data[:100])

# 将训练数据组织成batch大小为4，截断长度为5的数组
result = reader.ptb_iterator(train_data, 4, 5)
x, y = result.__next__()
print(x)
print(y)


class PTBModel():
    def __init__(self, is_training, batch_size, num_steps):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        if is_training:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=KEEP_PROB)
            cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * NUM_LAYERS)

            # 初始化最初的状态，也就是全零的向量
            self.initial_state = cell.zero_state(batch_size, tf.float32)
            # 将单词id转换为单词向量
            embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])
