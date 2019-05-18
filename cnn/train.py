import tensorflow.gfile as gfile
import os
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot=True)
X_train = mnist.train.images.reshape(55000, 28, 28, 1)
Y_train = mnist.train.labels
X_val = mnist.validation.images.reshape(5000, 28, 28, 1)
Y_val = mnist.validation.labels

n_classes = 10

# 使用keras sequential model 定义 mnist cnn
model = Sequential()

# 第一个卷积层
model.add(Conv2D(filters=32, kernel_size=(3, 3),
                 activation="relu", input_shape=(28, 28, 1)))
# 第二个卷积层
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))

# 池化层
model.add(MaxPooling2D(pool_size=(2, 2)))

# drop out 层
model.add(Dropout(0.25))

# 上面的都属于特征提取，现在用flatten层将提取出来的特征摊平后输入全连接网络
model.add(Flatten())

# 全连接层
model.add(Dense(128, activation='relu'))

# dropout 50%
model.add(Dropout(0.5))

# softmax输出层
model.add(Dense(n_classes, activation="softmax"))

# 查看mnist cnn模型网络结构
model.summary()

for layer in model.layers:
    print("layer output structure {}".format(
        layer.get_output_at(0).get_shape().as_list()))

# 编译模型
model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'], optimizer='adam')

# 训练模型
history = model.fit(X_train,
                    Y_train,
                    batch_size=128,
                    epochs=3,
                    verbose=2,
                    validation_data=(X_val, Y_val))


fig = plt.figure()
plt.subplot(2, 1, 1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2, 1, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.tight_layout()

plt.show()

# 保存模型
save_dir = "./model/mnist_cnn"

if gfile.Exists(save_dir):
    gfile.DeleteRecursively(save_dir)
gfile.MakeDirs(save_dir)

model_name = 'cnn_mnist.h5'
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
