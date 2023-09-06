import os.path

import tensorflow as tf
from utils.YOLO_BASED_training_dataset_gen import custom_dataloader
# from models.simple_cnn_network import simple_cnn_model
from models.edsr_net import edsr_cnn_model
from utils.loss_function import l1_loss
import datetime

model = edsr_cnn_model()
train_x, train_y = custom_dataloader(batch_size=4)
# check_point_path = "./check_point/simple_cnn/cp.ckpt"
check_point_path = "./check_point/edsr_cnn/cp.ckpt"
check_point_dir = os.path.dirname(check_point_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=check_point_path, save_weights_only=True, verbose=1)

model.compile(optimizer="adam", loss=l1_loss)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x=train_x, y=train_y, batch_size=32, epochs=10000, callbacks=[tensorboard_callback, cp_callback])
model.save_weights("./check_point/")