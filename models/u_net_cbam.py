import keras.layers
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class U_Net_CBAM(tf.keras.Model):
    def __init__(self):
        super(U_Net_CBAM, self).__init__()
        self.encoder_block1 = EncoderConvBlock(filters=16)
        self.encoder_block2 = EncoderConvBlock(filters=32)
        self.encoder_block3 = EncoderConvBlock(filters=32)

        self.decoder_block1 = DecoderConvBlock(filters=32)
        self.concat1 = keras.layers.Concatenate(axis=3)
        self.decoder_block2 = DecoderConvBlock(filters=32)
        self.concat2 = keras.layers.Concatenate(axis=3)
        self.decoder_block3 = DecoderConvBlock(filters=16)

        self.recon1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", strides=(1, 1), activation="relu")
        self.recon2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", strides=(1, 1), activation="relu")
        self.recon3 = tf.keras.layers.Conv2D(filters = 1, kernel_size=(3,3), padding="same", strides=(1, 1), activation="sigmoid")

    def call(self, x):
        enc1_x = self.encoder_block1(x)
        enc2_x = self.encoder_block2(enc1_x)
        enc3_x = self.encoder_block3(enc2_x)

        dec3_x = self.decoder_block1(enc3_x)
        dec3_x = self.concat1([dec3_x, enc2_x])
        dec2_x = self.decoder_block2(dec3_x)
        dec2_x = self.concat1([dec2_x, enc1_x])
        dec1_x = self.decoder_block3(dec2_x)

        x = self.recon1(dec1_x)
        x = self.recon2(x)
        x = self.recon3(x)

        return x

class EncoderConvBlock(tf.keras.Model):
    def __init__(self, filters=64):
        super(EncoderConvBlock, self).__init__()
        self.initial_block = tf.keras.layers.Conv2D(filters=filters, kernel_size=(7, 7), padding="same", strides=(1,1), activation="relu")
        self.conv1 = tf.keras.layers.Conv2D(filters = filters, kernel_size=(3,3), padding="same", strides=(1, 1), activation="relu")
        self.conv2 = tf.keras.layers.Conv2D(filters = filters, kernel_size=(3,3), padding="same", strides=(1, 1), activation="relu")
        self.conv3 = tf.keras.layers.Conv2D(filters = filters, kernel_size=(3,3), padding="same", strides=(1, 1), activation="relu")
        self.cbam = CBAM(filters)
        self.maxpool = tf.keras.layers.MaxPool2D((2,2))

    def call(self, x):
        residual = self.initial_block(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.cbam(x)
        x = residual + x
        return self.maxpool(x)


class DecoderConvBlock(tf.keras.Model):
    def __init__(self, filters=64):
        super(DecoderConvBlock, self).__init__()
        self.initial_block = tf.keras.layers.Conv2D(filters=filters, kernel_size=(7, 7), padding="same", strides=(1,1), activation="relu")
        self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding="same", strides=(1, 1),
                                            activation="relu")
        self.conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding="same", strides=(1, 1),
                                            activation="relu")
        self.conv3 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding="same", strides=(1, 1),
                                            activation="relu")
        self.cbam = CBAM(filters)
        self.upconv = tf.keras.layers.UpSampling2D((2,2))

    def call(self, x):
        residual = self.initial_block(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.cbam(x)
        x = residual+x
        return self.upconv(x)


class ChannelGate(tf.keras.Model):
    def __init__(self, channels, reduction_ratio=16, pool_types=["avg", "max"]):
        super(ChannelGate, self).__init__()
        self.channels = channels
        self.flatten = tf.keras.layers.Flatten()
        self.relu = tf.keras.layers.ReLU()
        self.dense = tf.keras.layers.Dense(channels)

        self.global_avg_pool = tf.keras.layers.GlobalAvgPool2D()
        self.global_max_pool = tf.keras.layers.GlobalMaxPool2D()

        self.pool_types = pool_types

    def call(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == "avg":
                avg_pool = self.global_avg_pool(x)
                channel_att_raw = self.flatten(avg_pool)
                channel_att_raw = self.relu(channel_att_raw)
                channel_att_raw = self.dense(channel_att_raw)

            elif pool_type == "max":
                max_pool = self.global_max_pool(x)
                channel_att_raw = self.flatten(max_pool)
                channel_att_raw = self.relu(channel_att_raw)
                channel_att_raw = self.dense(channel_att_raw)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum+channel_att_raw
        scale = tf.expand_dims(tf.expand_dims(tf.keras.activations.sigmoid(channel_att_sum), axis=1), axis=1)
        return x * scale


class SpatialGate(tf.keras.Model):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.spatial = tf.keras.layers.Conv2D(1, kernel_size=kernel_size, strides=1, padding="same")

    def call(self, x):
        x_out = self.spatial(x)
        scale = tf.keras.activations.sigmoid(x_out)
        return x*scale

class CBAM(tf.keras.Model):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types = ["avg", "max"], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def call(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out