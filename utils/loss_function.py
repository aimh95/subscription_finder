import tensorflow as tf
import numpy as np

def l1_loss(y_true, y_pred):
    # l = sig(abs(yi = fyi))
    err = y_true - y_pred
    loss = tf.math.reduce_mean(tf.math.abs(err))
    return loss