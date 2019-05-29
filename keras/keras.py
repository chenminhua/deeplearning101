# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 重构数据至4维（样本，像素X，像素Y，通道）
x_train = x_train.reshape(x_train.shape+(1,))
x_test = x_test.reshape(x_test.shape+(1,))

x_train, x_test = x_train / 255.0, x_test / 255.0

# 数据标签
label_train = tf.keras.utils.to_categorical(y_train, 10)
label_test = tf.keras.utils.to_categorical(y_test, 10)

# 建立LeNet-5模型
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(6, kernel_size=(5, 5), strides=(
        1, 1), activation='tanh', padding='valid'),
    tf.keras.layers.AveragePooling2D(pool_size=(
        2, 2), strides=(2, 2), padding='valid'),
    tf.keras.layers.Conv2D(16, kernel_size=(5, 5), strides=(
        1, 1), activation='tanh', padding='valid'),
    tf.keras.layers.AveragePooling2D(pool_size=(
        2, 2), strides=(2, 2), padding='valid'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation='tanh'),
    tf.keras.layers.Dense(84, activation='tanh'),
    tf.keras.layers.Dense(10, activation='softmax'),
])

# 编译模型，使用SGD优化器
model.compile(optimizer='SGD',
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

# 学习20轮，使用20%数据交叉验证
records = model.fit(x_train, label_train, epochs=20, validation_split=0.2)

# 预测
y_pred = np.argmax(model.predict(x_test), axis=1)
print("prediction accuracy: {}".format(1.0*sum(y_pred == y_test)/len(y_test)))

# 绘制结果
plt.plot(records.history['loss'], label='training set loss')
plt.plot(records.history['val_loss'], label='validation set loss')
plt.ylabel('categorical cross-entropy')
plt.xlabel('epoch')
plt.legend()
plt.show()
