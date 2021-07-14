import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

import identify_code.ideantify as idt
from keras import models

# 加载模型

new_model = models.load_model("models/final_model.h5")


def vec2text(vec):
    text = []
    for i, c in enumerate(vec):
        text.append(idt.char_set[c])
    return "".join(text)


idt.plt.figure(figsize=(10, 8))
# 图形的宽为10高为8

for images, labels in idt.val_ds.take(1):
    success = 0
    count = 16
    for i in range(16):
        idt.plt.subplot(4, 4, i + 1)
        idt.plt.xticks([])
        idt.plt.yticks([])
        idt.plt.grid(False)
        # 显示图片

        idt.plt.imshow(images[i])

        # 需要给图片增加一个维度
        img_array = tf.expand_dims(images[i], 0)

        # 使用模型预测验证码
        predictions = new_model.predict(img_array)

        # 真实值
        data_y = vec2text(np.argmax(labels, axis=2)[i])
        # 预测值
        prediction_value = vec2text(np.argmax(predictions, axis=2)[0])
        idt.plt.xlabel("真实值:" + data_y)
        idt.plt.title("预测值:" + prediction_value)
        idt.plt.axis("on")

        if data_y.upper() == prediction_value.upper():
            print("y预测=", prediction_value, "y实际=", data_y, "预测成功")
            success += 1
        else:
            print("y预测=", prediction_value, "y实际=", data_y, "预测失败！！！！！")
    print("预测", count, "次", "成功率=", success / count)

    # # 增加混淆矩阵
    # cm = confusion_matrix(data_y,prediction_value);
    # print(cm)

idt.plt.show()
