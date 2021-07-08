import numpy as np
import tensorflow as tf
import identify_code.ideantify as idt
from keras import models


# 加载模型

new_model = models.load_model("models/crack_captcha.h5")


def vec2text(vec):
    """
    还原标签（向量->字符串）
    """
    text = []
    for i, c in enumerate(vec):
        text.append(idt.char_set[c])
    return "".join(text)


idt.plt.figure(figsize=(10, 8))
# 图形的宽为10高为8


for images, labels in idt.val_ds.take(1):
    for i in range(30):
        ax = idt.plt.subplot(6, 5, i + 1)
        # 显示图片

        idt.plt.imshow(images[i])

        # 需要给图片增加一个维度
        img_array = tf.expand_dims(images[i], 0)

        # 使用模型预测验证码
        predictions = new_model.predict(img_array)
        idt.plt.title(vec2text(np.argmax(predictions, axis=2)[0]))
        idt.plt.axis("off")
idt.plt.show()

