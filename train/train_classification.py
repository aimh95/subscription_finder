import os.path
import os
import keras.models
import tensorflow as tf
from data_gen import classification_datagen
from models.light_classification_net import Light_Classification_Net
import datetime
from utils.callback import MyCallback
from keras.callbacks import ModelCheckpoint
from tensorflow.python.tools import freeze_graph

# model = U_Net()
model = Light_Classification_Net()

data_version = 1
initial_epoch = 0


finetuning_weight_path = os.path.join("./weight_path/u_net_totaltext_100epochs", str(initial_epoch-1)+".h5")
finetuning_weight_path = "None"
finetuning_weight_dir = os.path.dirname(finetuning_weight_path)


weight_save_path = "./weight_path/lightclassificationnet_totaltext_100epochs"
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

if data_version == 0:
    print("custom dataloader loaded")
    data_loader = classification_datagen.TotalTextClassificationDataLoader(crop_size = (440, 160), dataset_path="./datasets/train_dataset/sample/train_x")
    model.build((1, 240, 240, 3))
    model.summary()
    log_dir = "logs/fit/" + model.name
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    cb = MyCallback(model=model, file_path=weight_save_path, data=data_loader)
    model.fit(data_loader, initial_epoch=initial_epoch, epochs=101, callbacks=[tensorboard_callback, cb])
    pass

elif data_version == 1:
    # data_loader = total_text.TotalTextDataLoader(dataset_path="./datasets/train_dataset/totaltext")
    data_loader = classification_datagen.TotalTextClassificationDataLoader(crop_size=(256, 256), dataset_path="../datasets/train_dataset/totaltext")
    model.build((1, 256, 256, 3))
    model.summary()
    log_dir = "logs/fit/" + model.name
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    cb = MyCallback(model=model, file_path=weight_save_path)
    checkpoint_path = os.path.join(weight_save_path, "cp-{epoch:04d}.h5")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1)
    print("run")
    model.fit(data_loader, batch_size=32, initial_epoch=initial_epoch, epochs=100)


