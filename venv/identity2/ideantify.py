# 导入数据
import matplotlib.pyplot as plt
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import random,pathlib

# 设置随机种子尽可能使结果可以重现
import numpy as np
np.random.seed(1)

# 设置随机种子尽可能使结果可以重现
import tensorflow as tf
tf.random.set_seed(1)
data_dir = "C:/python/DM_identify/venv/verification_code/identify_code/captcha"
data_dir = pathlib.Path(data_dir)  #创建path对象



# path.glob():获取路径下的所有符合filename的文件，返回一个generator
# list(): 方法用于将元组或字符串转换为列表。
list_images = list(data_dir.glob('*'))
all_image_paths = [str(path) for path in list_images]

# 打乱数据
random.shuffle(all_image_paths)

# 获取数据标签
all_label_names = [path.split("\\")[7].split(".")[0] for path in all_image_paths]

image_count = len(all_image_paths)
print("图片总数为：",image_count)

plt.figure(figsize=(10, 5))

for i in range(20):
    plt.subplot(5, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    # 显示图片
    images = plt.imread(all_image_paths[i])
    plt.imshow(images)
    # 显示标签
    print(all_label_names[i])
    plt.xlabel(all_label_names[i])

plt.show()

# 标签数字化
number   = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
char_set       = number + alphabet
char_set_len   = len(char_set)
label_name_len = len(all_label_names[0])


# 将字符串数字化
def text2vec(text):
    vector = np.zeros([label_name_len, char_set_len])
    for i, c in enumerate(text):
        idx = char_set.index(c)
        vector[i][idx] = 1.0
    return vector

all_labels = [text2vec(i) for i in all_label_names]

# 预处理函数
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, [50, 200])
    return image/255.0

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

# 加载数据
# 构建 tf.data.Dataset 最简单的方法就是使用 from_tensor_slices 方法。
AUTOTUNE = tf.data.experimental.AUTOTUNE

path_ds  = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
label_ds = tf.data.Dataset.from_tensor_slices(all_labels)

image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
print("测试一：",image_label_ds)

train_ds = image_label_ds.take(1000)  # 前1000个batch
val_ds   = image_label_ds.skip(1000)  # 跳过前1000，选取后面的

# 配置数据
BATCH_SIZE = 32

train_ds = train_ds.batch(BATCH_SIZE)
print("asdfasdfa",train_ds)
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE) #使用prefetch()可显著减少空闲时间：

val_ds = val_ds.batch(BATCH_SIZE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE) #使用prefetch()可显著减少空闲时间：
print("测试二：",val_ds)





