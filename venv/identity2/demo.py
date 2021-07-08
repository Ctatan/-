import matplotlib.pyplot as plt
# 支持中文
import path as path

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import random, pathlib
import numpy as np
from tensorflow.keras import layers, models

np.random.seed(1)

# 设置随机种子尽可能使结果可以重现
import tensorflow as tf

tf.random.set_seed(1)
data_dir = "C:/python/DM_identify/venv/verification_code/identify_code/captcha"
data_dir = pathlib.Path(data_dir)

list_images = list(data_dir.glob('*'))
all_images_path = [str(path) for path in list_images]
# print(all_images_path)

all_label_names = [path.split("\\")[7].split(".")[0] for path in all_images_path]
print(all_label_names)

image_count = len(all_images_path)
print("图片数量：", image_count)

plt.figure(figsize=(10, 5))

for i in range(20):
    plt.subplot(5, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(0)

    #     显示图片
    images = plt.imread(all_images_path[i])
    plt.imshow(images)
    #     显示标签
    print(all_label_names[i])
    plt.xlabel(all_label_names[i])
# plt.show()

# 标签数字化
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
char_set = number + alphabet
# print(char_set)
char_set_len = len(char_set)
# print(char_set_len)
label_name_len = len(all_label_names[0])


# print(label_name_len)


# 将字符串数字化，将标签名称用二维数组表示
def text2vec(text):
    vector = np.zeros([label_name_len, char_set_len])
    # print("adfasd",vector)
    for i, c in enumerate(text):
        # print('下标索引值:',i,"索引内容",c)
        # index（）方法返回存在的子字符串的最开始的索引值
        idx = char_set.index(c)

        # print(idx)
        vector[i][idx] = 1.0
    return vector


all_labels = [text2vec(i) for i in all_label_names]


# print(text2vec(all_label_names[0]))

# 预处理函数
def preprocess_image(image):
    # 将.jpeg图片进行解码，得到像素值
    image_jpeg = tf.image.decode_jpeg(image, channels=1)
    image_size = tf.image.resize(image_jpeg, [50, 200])
    return image_size / 255.0


def load_and_preprocess_image(path):
    image_path = tf.io.read_file(path)
    return preprocess_image(image_path)


# 加载数据
# 构建tf.data.Dataset最简单的方法就是使用from_tensor_slices方法。
# tf.data.experimental.AUTOTUNE 自动设置为最大的可用线程数，机器算力拉满
AUTOTUNE = tf.data.experimental.AUTOTUNE
path_ds = tf.data.Dataset.from_tensor_slices(all_images_path)
print(path_ds)
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
print(image_ds)
label_ds = tf.data.Dataset.from_tensor_slices(all_labels)
print(label_ds)

image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
print("测试YI :", image_label_ds)

train_ds = image_label_ds.take(1000)  # 前1000个数据用来训练模型
val_ds = image_label_ds.skip(1000)  # 剩下的数据用来测试

# 配置数据
BATCH_SIZE = 16

train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
print("测试", train_ds)

val_ds = val_ds.batch(BATCH_SIZE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
print("测试二", val_ds)

# 构建网络模型
model = models.Sequential([

    # 卷积层1，卷积核是3x3，relu激活函数
    layers.Conv2D(32, (3, 3), input_shape=(50, 200, 1)),
    layers.ReLU(),
    layers.MaxPooling2D((2, 2)),  # 池化层1，2x2层采样，下降维度跟突出特征

    #卷积层2
    layers.Conv2D(64, (3, 3)),  # 卷积层是2，卷积核是3x3
    layers.ReLU(),
    layers.MaxPooling2D((2, 2)),  # 池化层2,2x2采样，降维跟突出特征

    # #卷积层3
    # layers.Conv2D(128,(3,3)),
    # layers.ReLU(),
    # layers.MaxPooling2D((2,2)),  #降维

    # Flatten层，连接卷积层和全连接层
    layers.Flatten(),
    layers.Dense(1000, activation='relu'),  # 全连接层，特征进一步提取

    layers.Dense(label_name_len * char_set_len),
    layers.Reshape([label_name_len, char_set_len]),
    layers.Softmax()  # 输出层，输出预期结果
])

# 打印网络结构
print(model.summary())
model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

epochs = 4

his = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)
print(his.history.keys())
# 模型评估
acc = his.history['accuracy']
val_acc = his.history['val_accuracy']

loss = his.history['loss']
val_loss = his.history['val_loss']

plt.plot(loss)

epochs_range = range(epochs)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='训练时的准确率')
plt.plot(epochs_range, val_acc, label='测试的准确率')
plt.legend(loc='lower right')
plt.title('训练和测试的准确率的折线图')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, acc, label='训练时的误差率')
plt.plot(epochs_range, val_acc, label='测试的误差率')
plt.legend(loc='upper right')
plt.title('训练和测试的误差率的折线图')
plt.show()

# 保存模型
model.save('models/demo1.h5')
print('已经保存')
