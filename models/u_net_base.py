import tensorflow as tf

class unet_base(tf.keras.Model):
    def __init__(self):
        super(unet_base, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters = 32, kernel_size=(3,3), padding="same", strides=(1, 1), activation="relu")
        self.conv2 = tf.keras.layers.Conv2D(filters = 32, kernel_size=(3,3), padding="same", strides=(1, 1), activation="relu")
        self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2))

        self.conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", strides=(1, 1), activation="relu")
        self.conv4 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", strides=(1, 1), activation="relu")
        self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

        self.conv5 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", strides=(1, 1), activation="relu")
        self.conv6 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", strides=(1, 1), activation="relu")
        self.maxpool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

        self.flatten = tf.keras.layers.Flatten()
        self.fcn1 = tf.keras.layers.Dense(128, activation="relu")
        self.fcn2 = tf.keras.layers.Dense(64, activation="relu")
        self.fcn3 = tf.keras.layers.Dense(32, activation="relu")
        self.fcn4 = tf.keras.layers.Dense(1, activation="sigmoid")



    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.fcn1(x)
        x = self.fcn2(x)
        x = self.fcn3(x)
        x = self.fcn4(x)
        return x