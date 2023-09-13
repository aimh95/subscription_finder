import os.path

import tensorflow as tf
from utils.single_char_distribution import custom_dataloader
# from models.simple_cnn_network import simple_cnn_model
from models.real_u_net_relu import U_Net
from utils.loss_function import l1_loss
import datetime
from keras.losses import binary_crossentropy, mean_squared_error
import numpy as np

model = U_Net()
train_x, train_y = custom_dataloader(batch_size=1)
# check_point_path = "./check_point/simple_cnn/cp.ckpt"
check_point_path = "./check_point/real_u_net_datagen_point_spread/cp.ckpt"
check_point_dir = os.path.dirname(check_point_path)


# if os.path.isfile(check_point_path):
#     model.load_weights(check_point_path)
lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_steps=100, decay_rate=0.92, staircase=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=check_point_path, save_weights_only=True, verbose=1)

model.compile(optimizer=optimizer, loss=mean_squared_error)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x=train_x, y=train_y, batch_size=32, epochs=1000, callbacks=[tensorboard_callback, cp_callback])
model.save_weights("./check_point/")