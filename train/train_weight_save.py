import os.path
import os
import tensorflow.keras.models
import tensorflow as tf

import utils.loss_function
from data_gen import custom_data, total_text
from models.light_u_net import Light_U_Net
# from models.light_network import LightNet
from models.real_u_net import U_Net
from models.u_net_cbam import U_Net_CBAM
import datetime
from utils.callback import MyCallback
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.tools import freeze_graph

import warnings
warnings.filterwarnings('ignore')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# model = U_Net()
model = U_Net_CBAM()

DATA_VERSION = 0
INITIAL_EPOCH = 633
FINAL_EPOCH = 800


finetuning_weight_path = os.path.join("../weight_path/u_net_cbam_craft_epochs", str(INITIAL_EPOCH-1)+".h5")
# finetuning_weight_path = "None"
finetuning_weight_dir = os.path.dirname(finetuning_weight_path)


weight_save_path = "../weight_path/u_net_cbam_craft_epochs"
weight_save_dir = os.path.dirname(weight_save_path)

if os.path.isfile(finetuning_weight_path):
    print("pretrained model loaded from", finetuning_weight_path)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.build((1, 240, 240, 3))
    model.load_weights(finetuning_weight_path)
    loss = tf.keras.losses.binary_crossentropy
    model.compile(optimizer=opt, loss=loss)
else:
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    loss = tf.keras.losses.binary_crossentropy
    model.compile(optimizer=opt, loss=loss)
    pass

if DATA_VERSION == 0:
    print("custom dataloader loaded")
    data_loader = custom_data.CustomDataLoader(target_resolution = (256, 256),batch_size=32, dataset_path="../datasets/train_dataset/indoor", heatmap_flag="gaussian")
    # model.build((1, 240, 240, 3))
    # model.summary()
    log_dir = "logs/fit/" + model.name
    # loss = utils.loss_function.objective_loss
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    cb = MyCallback(model=model, file_path=weight_save_path)
    model.fit(data_loader, initial_epoch=INITIAL_EPOCH, epochs=FINAL_EPOCH, callbacks=[cb])
    pass

elif DATA_VERSION == 1:
    # data_loader = total_text.TotalTextDataLoader(dataset_path="./datasets/train_dataset/totaltext")
    data_loader = total_text.TotalTextDataLoader(dataset_path="../datasets/train_dataset/totaltext")
    # model.build((1, 240, 240, 3))
    # model.summary()
    log_dir = "logs/fit/" + model.name +"attempt_2"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    cb = MyCallback(model=model, file_path=weight_save_path)
    # checkpoint_path = os.path.join(weight_save_path, "cp-{epoch:04d}.h5")
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1)
    print("run")
    model.fit(data_loader, epochs=FINAL_EPOCH, verbose=1, callbacks=[cb])



