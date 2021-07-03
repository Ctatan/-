import numpy as np
from tensorflow.keras import datasets, layers, models
import identify_code.ideantify as idt
import tensorflow as tf

model = models.Sequential([

    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 200, 1)),  # 卷积层1，卷积核3*3
    layers.MaxPooling2D((2, 2)),  # 池化层1，2*2采样
    layers.Conv2D(64, (3, 3), activation='relu'),  # 卷积层2，卷积核3*3
    layers.MaxPooling2D((2, 2)),  # 池化层2，2*2采样

    layers.Flatten(),  # Flatten层，连接卷积层与全连接层
    layers.Dense(1000, activation='relu'),  # 全连接层，特征进一步提取

    layers.Dense(idt.label_name_len * idt.char_set_len),
    layers.Reshape([idt.label_name_len, idt.char_set_len]),
    layers.Softmax()  # 输出层，输出预期结果
])
# 打印网络结构
print(model.summary())

model.compile(optimizer="adam",
              loss='categorical_crossentropy',
              metrics=['accuracy'])

epochs = 200

history = model.fit(
    idt.train_ds,
    validation_data=idt.val_ds,
    epochs=epochs
)

# 模型评估
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

idt.plt.figure(figsize=(12, 4))
idt.plt.subplot(1, 2, 1)

idt.plt.plot(epochs_range, acc, label='Training Accuracy')
idt.plt.plot(epochs_range, val_acc, label='Validation Accuracy')
idt.plt.legend(loc='lower right')
idt.plt.title('Training and Validation Accuracy')

idt.plt.subplot(1, 2, 2)
idt.plt.plot(epochs_range, loss, label='Training Loss')
idt.plt.plot(epochs_range, val_loss, label='Validation Loss')
idt.plt.legend(loc='upper right')
idt.plt.title('Training and Validation Loss')
idt.plt.show()
# 保存模型
model.save('models/crack_captcha.model')



