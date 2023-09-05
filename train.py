import os.path

import tensorflow as tf
from utils.training_dataset_gen import custom_dataloader
from models.u_net_base import unet_base
import datetime
# from keras.losses import binary_crossentropy

# fashion_mnist = tf.keras.datasets.fashion_mnist

# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
model = unet_base()
(train_x, train_labels) = custom_dataloader()
# check_point_path = "./check_point/simple_cnn/cp.ckpt"
check_point_path = "./check_point/unet_basic/cp.ckpt"
check_point_dir = os.path.dirname(check_point_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=check_point_path, save_weights_only=True, verbose=1)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics = "accuracy")

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(train_x,train_labels,  batch_size=32, epochs=100, callbacks=[tensorboard_callback, cp_callback])
model.save_weights("./check_point/")