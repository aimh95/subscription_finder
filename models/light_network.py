import keras.layers
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class LightNet(tf.keras.Model):
    def __init__(self):
        super(LightNet, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, padding="same", activation="relu")
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=1, strides=1, padding="same", activation="relu")
        self.conv3 = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="same", activation="relu")
        self.conv4 = tf.keras.layers.Conv2D(128, kernel_size=1, strides=1, padding="same", activation="relu")
        self.conv5 = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="same", activation="relu")

        self.midcheck = tf.keras.layers.Conv2D(1, kernel_size=3, strides=1, padding="same", activation="sigmoid")

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        mid_check = self.midcheck(x)
        return mid_check

class LightNetImageFeatureExtract(tf.keras.Model):
    def __init__(self):
        super(LightNetImageFeatureExtract, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, padding="same", activation="relu")
        self.baseblock1 = BaseBlock(dwconv_ch=32, dwconv_stride=1, conv_ch=64)
        self.baseblock2 = BaseBlock(dwconv_ch=64, dwconv_stride=1, conv_ch=128)
        self.baseblock3 = BaseBlock(dwconv_ch=128, dwconv_stride=1, conv_ch=128)
        self.midcheck = tf.keras.layers.Conv2D(1, kernel_size=3, strides=1, padding="same", activation="sigmoid")

    def call(self, x):
        x = self.conv1(x)
        x = self.baseblock1(x)
        x = self.baseblock2(x)
        x = self.baseblock3(x)
        mid_check = self.midcheck(x)
        return mid_check


class LightNetClassifier(tf.keras.Model):
    def __init__(self):
        super(LightNetClassifier, self).__init__()
        self.baseblock4 = BaseBlock(dwconv_ch=128, dwconv_stride=2, conv_ch=256)
        self.baseblock5 = BaseBlock(dwconv_ch=256, dwconv_stride=1, conv_ch=256)
        self.baseblock6 = BaseBlock(dwconv_ch=256, dwconv_stride=2, conv_ch=512)
        self.block_sequence = tf.keras.models.Sequential()
        for i in range(5):
            self.block_sequence.add(BaseBlock(dwconv_ch=512, dwconv_stride=1, conv_ch=512))
        self.avg_pooling = tf.keras.layers.GlobalAvgPool2D()
        self.dense = tf.keras.layers.Dense(1024, activation="relu")
        self.classifier = tf.keras.layers.Dense(1, activation="simoid")

    def call(self, x):
        x = self.baseblock4(x)
        x = self.baseblock5(x)
        x = self.baseblock6(x)
        x = self.block_sequence(x)
        x = self.avg_pooling(x)
        x = self.dense(x)
        x = self.classifier(x)
        return x

class BaseBlock(tf.keras.Model):
    def __init__(self, dwconv_ch, dwconv_stride, conv_ch):
        super(BaseBlock, self).__init__()
        self.conv_dw = tf.keras.layers.DepthwiseConv2D(kernel_size=dwconv_ch, strides=dwconv_stride, padding="same")
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(conv_ch, kernel_size=1, strides=1, padding="same")
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()

    def call(self, x):
        x = self.conv_dw(x)
        # x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.conv(x)
        # x = self.batch_norm2(x)
        x = self.relu2(x)
        return x
