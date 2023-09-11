import os.path

import tensorflow as tf
from utils.YOLO_BASED_training_dataset_gen import custom_dataloader
# from models.simple_cnn_network import simple_cnn_model
from models.real_u_net import U_Net
from utils.loss_function import l1_loss
import datetime
from keras.losses import binary_crossentropy

model = U_Net()
train_x, train_y = custom_dataloader(batch_size=1, n_sep=1)
# check_point_path = "./check_point/simple_cnn/cp.ckpt"
check_point_path = "./check_point/real_u_net_dataset_update/cp.ckpt"
check_point_dir = os.path.dirname(check_point_path)

if os.path.isfile(check_point_path):
    model.load_weights(check_point_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=check_point_path, save_weights_only=True, verbose=1)

model.compile(optimizer="adam", loss=binary_crossentropy)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x=train_x, y=train_y, batch_size=32, epochs=50, callbacks=[tensorboard_callback, cp_callback])
model.save_weights("./check_point/")