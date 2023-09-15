import tensorflow as tf
import os
class Callback(tf.keras.callbacks.Callback):
    def __init__(self, model, file_path):
        self.model = model
        self.file_path = file_path

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(os.path.join(self.file_path, str(epoch)), save_format='tf')
