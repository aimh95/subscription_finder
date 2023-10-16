import os.path
import os
import keras.models
import tensorflow as tf
from data_gen import custom_data, total_text
from models.light_network import LightNetClassifier, LightNetImageFeatureExtract
import datetime
from utils.callback import Callback
from utils.custom_utils import grad

model_subtitle_mapping = LightNetImageFeatureExtract()
model_classifier = LightNetClassifier()

initial_epoch = 0


weight_save_path = "../weight_path/lightnet_totaltext"
weight_save_dir = os.path.dirname(weight_save_path)

data_loader = total_text.TotalTextDataLoader(dataset_path="../datasets/train_dataset/totaltext")
# train_x, train_y = data_loader.data_gen()

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
cb = Callback(model=model_subtitle_mapping, file_path=os.path.join(weight_save_path, "mapping"))
checkpoint_path = os.path.join(weight_save_path, "cp-{epoch:04d}.ckpt")

train_loss_results = []
train_accuracy_results = []

optimizer_mapping = tf.keras.optimizers.Adam(learning_rate=1e-3)
optimizer_classifier = tf.keras.optimizers.Adam(learning_rate=0.04)
loss_mapping = tf.keras.losses.binary_crossentropy
loss_classifier = tf.keras.losses.binary_crossentropy

num_epochs = 500

for epoch in range(num_epochs):
  epoch_loss_avg = tf.keras.metrics.Mean()
  epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

  # Training loop - using batches of 32
  for x, y in ds_train_batch:
    # Optimize the model
    mapping_loss_value, mapping_grads = grad(model_subtitle_mapping, x, y, mapping_epoch_loss_avg)
    optimizer_mapping.apply_gradients(zip(mapping_grads, model_subtitle_mapping.trainable_variables))
    epoch_loss_avg.update_state(mapping_loss_value)
    epoch_accuracy.update_state(y, model_subtitle_mapping(x, training=True))

    classifier_loss_value, classifier_grads = grad(model_classifier, x, y, classifier_epoch_loss_avg)
    optimizer_classifier.apply_gradients(zip(classifier_grads, model_classifier.trainable_variables))
    classifier_epoch_loss_avg.update_state(classifier_loss_value)
    epoch_accuracy.update_state(y, model_subtitle_mapping(x, training=True))

  # End epoch
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())

  if epoch % 50 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))