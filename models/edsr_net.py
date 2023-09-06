import tensorflow as tf

class edsr_cnn_model(tf.keras.Model):
    def __init__(self):
        super(edsr_cnn_model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters = 32, kernel_size=(3,3), padding="same", strides=(1, 1), activation="relu")
        self.conv2 = tf.keras.layers.Conv2D(filters = 32, kernel_size=(3,3), padding="same", strides=(1, 1))

        self.conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", strides=(1, 1), activation="relu")
        self.conv4 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", strides=(1, 1))

        self.conv5 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", strides=(1, 1), activation="relu")
        self.conv6 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", strides=(1, 1))

        self.recon = tf.keras.layers.Conv2D(filters = 3, kernel_size=(3,3), padding="same", strides=(1, 1), activation="relu")

    def call(self, x):
        res_x1 = self.conv1(x)
        x = self.conv2(res_x1)
        x = x+res_x1

        res_x2 = self.conv3(x)
        x = self.conv4(res_x2)
        x = x+res_x2

        res_x3 = self.conv5(x)
        x = self.conv6(res_x3)
        x = x+res_x3

        x = x+res_x1

        x = self.recon(x)
        return x