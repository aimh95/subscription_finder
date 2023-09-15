import os.path
import os
import keras.models
import tensorflow as tf
# from utils.single_char_distribution import custom_dataloader
from utils.final_training_dataset_gen import custom_dataloader
from data_gen import total_text
# from models.simple_cnn_network import simple_cnn_model
from models.real_u_net import U_Net
from utils.loss_function import l1_loss
import datetime
from keras.losses import binary_crossentropy, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt




model = U_Net()
train_x, train_y = custom_dataloader(data_path="./datasets/train_dataset/indoor",batch_size=1)
# data_loder = total_text.DataLoader(dataset_path="./datasets/train_dataset/totaltext")
# train_x, train_y = data_loder.data_gen()

# check_point_path = "./check_point/simple_cnn/cp.ckpt"



finetuning_check_point_path = "./check_point/real_u_net_61_0100"
fine_tune_at = 62

# check_point_path = "./check_point/real_u_net_realworld_dataset_finetuning/cp.ckpt"
check_point_path = "./check_point/real_u_net_realworld_dataset_finetuning/cp.ckpt"
check_point_dir = os.path.dirname(check_point_path)


model.compile(optimizer="adam", loss="binary_crossentropy")
model = tf.keras.models.load_model("./MyModel_tf")

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=check_point_path, save_weights_only=True, verbose=1)


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x=train_x, y=train_y, batch_size=8, epochs=1, callbacks=[tensorboard_callback, cp_callback])
# model.save(os.path.join(check_point_path, "u_net_real_world_10.h5"))
model.save('./MyModel_tf',save_format='tf')

