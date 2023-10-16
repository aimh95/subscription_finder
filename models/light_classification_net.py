import keras.layers
import tensorflow as tf

class Light_Classification_Net(tf.keras.Model):
    def __init__(self):
        super(Light_Classification_Net, self).__init__()
        self.conv = tf.keras.layers.Conv2D(64, kernel_size=7, strides=1, padding="same", activation="relu")

        self.light_base_block1_1 = Light_Base_Block(64)
        self.light_base_block1_2 = Light_Base_Block(64)

        self.avgpool1 = tf.keras.layers.AvgPool2D((2,2))
        self.conv1 = tf.keras.layers.Conv2D(128, kernel_size=7, strides=1, padding="same", activation="relu")

        self.light_base_block2_1 = Light_Base_Block(128)
        self.light_base_block2_2 = Light_Base_Block(128)
        self.avgpool2 = tf.keras.layers.AvgPool2D((2,2))
        self.conv2 = tf.keras.layers.Conv2D(256, kernel_size=7, strides=1, padding="same", activation="relu")

        self.light_base_block3_1 = Light_Base_Block(256)
        self.light_base_block3_2 = Light_Base_Block(256)
        self.global_average_pooling = tf.keras.layers.GlobalAvgPool2D()
        self.fully_connected = tf.keras.layers.Dense(1024, activation="relu")
        self.output_layer = tf.keras.layers.Dense(1, activation="sigmoid")


    def call(self, x):
        x = self.conv(x)
        # x = self.light_base_block(x)
        x = self.light_base_block1_1(x)
        x = self.light_base_block1_2(x)
        x = self.avgpool1(x)
        x = self.conv1(x)
        x = self.light_base_block2_1(x)
        x = self.light_base_block2_2(x)
        x = self.avgpool2(x)
        x = self.conv2(x)
        x = self.light_base_block3_1(x)
        x = self.light_base_block3_2(x)
        x = self.global_average_pooling(x)
        x = self.fully_connected(x)
        x = self.output_layer(x)
        return x


class Light_Base_Block(tf.keras.Model):
    def __init__(self, channel):
        super(Light_Base_Block, self).__init__()
        self.conv_pw1 = tf.keras.layers.Conv2D(channel, kernel_size=1, strides=1, padding="same")
        self.relu1 = tf.keras.layers.ReLU()
        self.conv_dw2 = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="same")
        self.relu2 = tf.keras.layers.ReLU()
        self.conv_pw3 = tf.keras.layers.Conv2D(channel, kernel_size=1, strides=1, padding="same")
        self.relu3 = tf.keras.layers.ReLU()
        self.cbam = CBAM(channel)

    def call(self, x):
        residual = x
        out = self.relu1(self.conv_pw1(x))
        out = self.relu2(self.conv_dw2(out))
        out = self.relu3(self.conv_pw3(out))
        out = self.cbam(out)
        out = residual + out
        return out

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