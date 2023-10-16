import keras.layers
import tensorflow as tf

class Light_U_Net(tf.keras.Model):
    def __init__(self):
        super(Light_U_Net, self).__init__()
        self.enter_layer = tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, padding="same", activation="relu")

        self.encoder_block1 = EncoderConvBlock(channel=32)
        self.encoder_block2 = EncoderConvBlock(channel=64)
        self.encoder_block3 = EncoderConvBlock(channel=128)

        self.decoder_block1 = DecoderConvBlock(channel=128)
        self.concat1 = keras.layers.Concatenate(axis=3)
        self.decoder_block2 = DecoderConvBlock(channel=64)
        self.concat2 = keras.layers.Concatenate(axis=3)
        self.decoder_block3 = DecoderConvBlock(channel=32)

        self.recon1 = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), padding="same", strides=(1, 1), activation="relu")
        self.recon2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), padding="same", strides=(1, 1), activation="relu")
        self.recon3 = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), padding="same", strides=(1, 1), activation="sigmoid")

    def call(self, x):
        x = self.enter_layer(x)
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
    def __init__(self, channel=32):
        super(EncoderConvBlock, self).__init__()
        self.conv_dw1 = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="same")
        self.relu1_1 = tf.keras.layers.ReLU()
        self.conv_pw1 = tf.keras.layers.Conv2D(channel, kernel_size=1, strides=1, padding="same")
        self.relu1_2 = tf.keras.layers.ReLU()

        self.conv_dw2 = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="same")
        self.relu2_1 = tf.keras.layers.ReLU()
        self.conv_pw2 = tf.keras.layers.Conv2D(channel, kernel_size=1, strides=1, padding="same")
        self.relu2_2 = tf.keras.layers.ReLU()

        self.conv_dw3 = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, padding="same")
        self.relu3_1 = tf.keras.layers.ReLU()
        self.conv_pw3 = tf.keras.layers.Conv2D(channel, kernel_size=1, strides=1, padding="same")
        self.relu3_2 = tf.keras.layers.ReLU()

    def call(self, x):
        x = self.conv_dw1(x)
        x = self.relu1_1(x)
        x = self.conv_pw1(x)
        x = self.relu1_2(x)

        x = self.conv_dw2(x)
        x = self.relu2_1(x)
        x = self.conv_pw2(x)
        x = self.relu2_2(x)

        x = self.conv_dw3(x)
        x = self.relu3_1(x)
        x = self.conv_pw3(x)
        x = self.relu3_2(x)

        return x


class DecoderConvBlock(tf.keras.Model):
    def __init__(self, channel=32):
        super(DecoderConvBlock, self).__init__()
        self.conv_dw1 = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="same")
        self.relu1_1 = tf.keras.layers.ReLU()
        self.conv_pw1 = tf.keras.layers.Conv2D(channel, kernel_size=1, strides=1, padding="same")
        self.relu1_2 = tf.keras.layers.ReLU()

        self.conv_dw2 = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="same")
        self.relu2_1 = tf.keras.layers.ReLU()
        self.conv_pw2 = tf.keras.layers.Conv2D(channel, kernel_size=1, strides=1, padding="same")
        self.relu2_2 = tf.keras.layers.ReLU()

        self.conv_dw3 = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="same")
        self.relu3_1 = tf.keras.layers.ReLU()
        self.conv_pw3 = tf.keras.layers.Conv2D(channel, kernel_size=1, strides=1, padding="same")
        self.relu3_2 = tf.keras.layers.ReLU()

        self.upconv = tf.keras.layers.UpSampling2D((2, 2))

    def call(self, x):
        x = self.conv_dw1(x)
        x = self.relu1_1(x)
        x = self.conv_pw1(x)
        x = self.relu1_2(x)

        x = self.conv_dw2(x)
        x = self.relu2_1(x)
        x = self.conv_pw2(x)
        x = self.relu2_2(x)

        x = self.conv_dw3(x)
        x = self.relu3_1(x)
        x = self.conv_pw3(x)
        x = self.relu3_2(x)

        return self.upconv(x)


