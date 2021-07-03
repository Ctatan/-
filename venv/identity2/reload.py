import numpy as np
from keras import models
import tensorflow as tf
import identify_code.ideantify as idt

# 加载模型
from identify_code.netsModel import model

new_model = models.load_model("models/crack_captcha.model")
