import numpy as np
from tensorflow.keras import layers, models
import identify_code.ideantify as idt

model = models.Sequential([

    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(50, 200, 1)),  # 卷积层1，卷积核3*3
    layers.ReLU(),
    layers.MaxPooling2D((2, 2)),  # 池化层1，2*2采样

    layers.Conv2D(128, (3, 3), activation='relu'),  # 卷积层2，卷积核3*3
    layers.ReLU(),
    layers.MaxPooling2D((2, 2)),  # 池化层2，2*2采样

    layers.Conv2D(256, (3, 3), activation='relu'),  # 卷积层2，卷积核3*3
    layers.ReLU(),
    layers.MaxPooling2D((2, 2)),  # 池化层2，2*2采样

    layers.Flatten(),  # Flatten层，连接卷积层与全连接层
    layers.Dense(1024, activation='relu'),  # 全连接层，特征进一步提取

    layers.Dense(idt.label_name_len * idt.char_set_len),
    layers.Reshape([idt.label_name_len, idt.char_set_len]),
    layers.Softmax()  # 输出层，输出预期结果
])
# 打印网络结构
print(model.summary())

# model.compile()方法用于在配置训练方法时，告知训练时用的优化器、损失函数和准确率评测标准
model.compile(optimizer="adam",
              loss='categorical_crossentropy',
              metrics=['accuracy'])

epochs = 20
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

idt.plt.plot(epochs_range, acc, label='训练时的准确率')
idt.plt.plot(epochs_range, val_acc, label='测试的准确率')
idt.plt.legend(loc='lower right')
idt.plt.title('训练和测试的准确率的折线图')

idt.plt.subplot(1, 2, 2)
idt.plt.plot(epochs_range, loss, label='训练时的误差率')
idt.plt.plot(epochs_range, val_loss, label='测试的误差率')
idt.plt.legend(loc='upper right')
idt.plt.title('训练和测试的误差率的折线图')
idt.plt.show()

# 保存模型
model.save('models/final_model.h5')
