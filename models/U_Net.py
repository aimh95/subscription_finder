import keras.layers
import tensorflow as tf


class U_Net(tf.keras.Model):
    def __init__(self):
        super(U_Net, self).__init__()
        self.encoder_block1 = EncoderConvBlock(filters=64)
        self.encoder_block2 = EncoderConvBlock(filters=64)
        self.encoder_block3 = EncoderConvBlock(filters=64)

        self.decoder_block1 = DecoderConvBlock(filters=64)
        self.concat1 = keras.layers.Concatenate()
        self.decoder_block2 = DecoderConvBlock(filters=64)
        self.concat2 = keras.layers.Concatenate()
        self.decoder_block3 = DecoderConvBlock(filters=64)

        self.recon = tf.keras.layers.Conv2D(filters = 1, kernel_size=(3,3), padding="same", strides=(1, 1), activation="sigmoid")

    def call(self, x):
        enc1_x = self.encoder_block1(x)
        enc2_x = self.encoder_block2(enc1_x)
        enc3_x = self.encoder_block3(enc2_x)

        dec3_x = self.decoder_block1(enc3_x)
        # dec3_x = self.concat1(dec3_x, enc2_x)
        dec2_x = self.decoder_block2(dec3_x)
        # dec2_x = self.concat1(dec2_x, enc1_x)
        dec1_x = self.decoder_block3(dec2_x)

        x = self.recon(dec1_x)

        return x

class EncoderConvBlock(tf.keras.Model):
    def __init__(self, filters=64):
        super(EncoderConvBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters = filters, kernel_size=(3,3), padding="same", strides=(1, 1), activation="relu")
        self.conv2 = tf.keras.layers.Conv2D(filters = filters, kernel_size=(3,3), padding="same", strides=(1, 1), activation="relu")
        self.conv3 = tf.keras.layers.Conv2D(filters = filters, kernel_size=(3,3), padding="same", strides=(1, 1), activation="relu")
        self.maxpool = tf.keras.layers.MaxPool2D((2,2))

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.maxpool(x)


class DecoderConvBlock(tf.keras.Model):
    def __init__(self, filters=64):
        super(DecoderConvBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding="same", strides=(1, 1),
                                            activation="relu")
        self.conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding="same", strides=(1, 1),
                                            activation="relu")
        self.conv3 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding="same", strides=(1, 1),
                                            activation="relu")
        self.upconv = tf.keras.layers.UpSampling2D((2,2))

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.upconv(x)


