import os.path
import os
import keras.models
import tensorflow as tf
# from utils.single_char_distribution import custom_dataloader
from utils.final_training_dataset_gen import custom_dataloader
from data_gen import custom_data, total_text
# from models.simple_cnn_network import simple_cnn_model
from models.real_u_net import U_Net
from utils.loss_function import l1_loss
import datetime
from keras.losses import binary_crossentropy, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from utils.callback import Callback

model = U_Net()

data_version = 1
initial_epoch = 0


finetuning_weight_path = os.path.join("weight_path/real_u_net_light_totaltext_100epochs", str(initial_epoch))
finetuning_weight_path = "None"
finetuning_weight_dir = os.path.dirname(finetuning_weight_path)



weight_save_path = "weight_path/real_u_net_totaltext_200epochs_load"
weight_save_dir = os.path.dirname(weight_save_path)

if os.path.isdir(finetuning_weight_path):
    print("data loaded from", finetuning_weight_path)
    opt = tf.keras.optimizers.Adam(learning_rate=5e-4)
    model = tf.keras.models.load_model(finetuning_weight_path)
    loss = tf.keras.losses.binary_crossentropy
    model.compile(optimizer=opt, loss=loss)
else:
    opt = tf.keras.optimizers.Adam(learning_rate=5e-4)
    loss = tf.keras.losses.binary_crossentropy
    model.compile(optimizer=opt, loss=loss)


if data_version == 0:
    data_loader = custom_data.CustomDataLoader(dataset_path="./datasets/train_dataset/indoor")

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    cb = Callback(model=model, file_path=weight_save_path)
    checkpoint_path = os.path.join(weight_save_path, "cp-{epoch:04d}.ckpt")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    model.fit(data_loader, initial_epoch=initial_epoch, epochs=150, callbacks=[tensorboard_callback, cb, cp_callback])


elif data_version == 1:
    data_loader = total_text.TotalTextDataLoader(dataset_path="./datasets/train_dataset/totaltext")
    # train_x, train_y = data_loader.data_gen()

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    cb = Callback(model=model, file_path=weight_save_path)
    checkpoint_path = os.path.join(weight_save_path, "cp-{epoch:04d}.ckpt")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    model.fit(data_loader, initial_epoch=initial_epoch, batch_size = 32, epochs=200, callbacks=[tensorboard_callback, cb, cp_callback])



